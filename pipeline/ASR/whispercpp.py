import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import soundfile as sf
import torch


class WhisperCppASR:
    """
    Whisper.cpp backend via CLI invocation.

    Notes:
    - We run whisper.cpp on a temporary WAV file.
    - We request JSON output and use `-ml 1` (max segment length = 1 token/word-ish)
      to approximate word-level timestamps suitable for our existing pipeline
      (expects chunks: [{text, timestamp(start,end)}]).
    """

    def __init__(
        self,
        model: str = "tiny.en",
        device: str = "cpu",
        language: Optional[str] = "en",
        binary_path: Optional[str] = None,
        model_dir: Optional[str] = None,
    ) -> None:
        self.device = device
        self.language = language

        self.binary_path = binary_path or os.environ.get("WHISPERCPP_BIN", "whisper-cli")
        self.model_dir = model_dir or os.environ.get("WHISPERCPP_MODEL_DIR", "/models")
        self.model_path = self._resolve_model(model)

    def _resolve_model(self, model: str) -> str:
        # If a full path was provided, use it.
        if model and (model.endswith(".bin") or model.endswith(".gguf") or "/" in model):
            return model

        # Map common names to whisper.cpp filenames.
        # NOTE: these are the canonical ggml .bin names hosted by ggerganov/whisper.cpp.
        mapping = {
            "tiny": "ggml-tiny.bin",
            "tiny.en": "ggml-tiny.en.bin",
            "base": "ggml-base.bin",
            "base.en": "ggml-base.en.bin",
            "small": "ggml-small.bin",
            "small.en": "ggml-small.en.bin",
            "medium": "ggml-medium.bin",
            "medium.en": "ggml-medium.en.bin",
            "large-v2": "ggml-large-v2.bin",
            "large-v3": "ggml-large-v3.bin",
        }
        fname = mapping.get(model or "", None)
        if fname is None:
            # fall back to env-provided direct path
            env_path = os.environ.get("WHISPERCPP_MODEL_PATH")
            if env_path:
                return env_path
            raise ValueError(f"Unknown whisper.cpp model name '{model}'. Provide a path or known name.")

        return str(Path(self.model_dir) / fname)

    def _write_temp_wav(self, audio: torch.Tensor, sr: int = 16000) -> Path:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().to("cpu").float().squeeze()
            audio_np = audio.numpy()
        else:
            audio_np = audio

        fd, out_name = tempfile.mkstemp(prefix="wcpp_", suffix=".wav")
        os.close(fd)
        out_path = Path(out_name)
        sf.write(out_path, audio_np, sr)
        return out_path

    def _parse_json(self, json_path: Path) -> Dict[str, Any]:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def transcribe(
        self,
        audio: torch.Tensor,
        return_timestamps: bool = True,
        language: Optional[str] = None,
        word_timestamps: bool = True,
        **decode_kwargs,
    ) -> Dict[str, Any]:
        # whisper.cpp expects 16k mono wav. Caller already provides 16k.
        wav_path = None
        out_prefix = None
        try:
            wav_path = self._write_temp_wav(audio, sr=16000)
            out_prefix = Path(tempfile.mkdtemp(prefix="wcpp_out_")) / "out"

            lang = language or self.language
            cmd: List[str] = [
                self.binary_path,
                "-m",
                self.model_path,
                "-f",
                str(wav_path),
                "-oj",              # JSON output
                "-of",
                str(out_prefix),    # output prefix
            ]

            # Prefer word splitting (doesn't guarantee word-level timestamps,
            # but usually improves segmentation for streaming).
            if word_timestamps:
                cmd += ["-sow"]

            if lang:
                cmd += ["-l", str(lang)]

            # Streaming-friendly: transcribe only a slice of the file.
            offset_ms = decode_kwargs.get("offset_ms")
            duration_ms = decode_kwargs.get("duration_ms")
            if isinstance(offset_ms, (int, float)) and offset_ms > 0:
                cmd += ["-ot", str(int(offset_ms))]
            if isinstance(duration_ms, (int, float)) and duration_ms > 0:
                cmd += ["-d", str(int(duration_ms))]

            # Optional whisper.cpp built-in VAD (simplifies server-side hacks).
            if decode_kwargs.get("vad") is True:
                cmd += ["--vad"]
                vt = decode_kwargs.get("vad_threshold")
                if isinstance(vt, (int, float)):
                    cmd += ["-vt", str(float(vt))]
                vspd = decode_kwargs.get("vad_min_speech_ms")
                if isinstance(vspd, (int, float)):
                    cmd += ["-vspd", str(int(vspd))]
                vsd = decode_kwargs.get("vad_min_silence_ms")
                if isinstance(vsd, (int, float)):
                    cmd += ["-vsd", str(int(vsd))]
                vpad = decode_kwargs.get("vad_speech_pad_ms")
                if isinstance(vpad, (int, float)):
                    cmd += ["-vp", str(int(vpad))]

            # Control how much text context whisper.cpp keeps internally
            max_context = decode_kwargs.get("max_context")
            if isinstance(max_context, int):
                cmd += ["-mc", str(max_context)]

            # Optional prompt (used for streaming stability)
            prompt = decode_kwargs.get("initial_prompt") or decode_kwargs.get("prompt")
            if prompt:
                cmd += ["--prompt", str(prompt)]
                # Keep prompt prepended across calls (more stable for streaming).
                if decode_kwargs.get("carry_initial_prompt", True):
                    cmd += ["--carry-initial-prompt"]

            # Optional beam size
            beam_size = decode_kwargs.get("beam_size")
            if isinstance(beam_size, int) and beam_size > 0:
                cmd += ["-bs", str(beam_size)]

            best_of = decode_kwargs.get("best_of")
            if isinstance(best_of, int) and best_of > 0:
                cmd += ["-bo", str(best_of)]

            # Run
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            json_path = Path(str(out_prefix) + ".json")
            data = self._parse_json(json_path)

            # whisper.cpp `-oj` output contains `transcription` as a list of items:
            # { offsets: {from,to} (ms), text: str, ... }
            segs = data.get("transcription") or data.get("segments") or []
            chunks: List[Dict[str, Any]] = []
            full_text_parts: List[str] = []
            for seg in segs:
                txt = (seg.get("text") or "").strip()
                if not txt:
                    continue
                # whisper.cpp sometimes emits a placeholder for silence
                if txt == "[BLANK_AUDIO]":
                    continue
                if "offsets" in seg and isinstance(seg["offsets"], dict):
                    start = float(seg["offsets"].get("from", 0.0) or 0.0) / 1000.0
                    end = float(seg["offsets"].get("to", 0.0) or 0.0) / 1000.0
                else:
                    start = float(seg.get("start", 0.0) or 0.0)
                    end = float(seg.get("end", start) or start)
                full_text_parts.append(txt)
                if return_timestamps:
                    chunks.append({"text": txt, "timestamp": (start, end)})

            result: Dict[str, Any] = {"text": " ".join(full_text_parts).strip()}
            if return_timestamps and chunks:
                result["chunks"] = chunks
            return result
        finally:
            try:
                if wav_path is not None:
                    wav_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                if out_prefix is not None:
                    # remove output json + any sidecar files
                    for ext in [".json", ".srt", ".txt", ".vtt"]:
                        p = Path(str(out_prefix) + ext)
                        p.unlink(missing_ok=True)
                    # remove temp dir
                    out_prefix.parent.rmdir()
            except Exception:
                pass

    def transcribe_segments(
        self,
        audio: torch.Tensor,
        speech_segments: list[dict[str, float]],
        sample_rate: int = 16000,
        return_timestamps: bool = True,
        language: Optional[str] = None,
        word_timestamps: bool = True,
        **decode_kwargs,
    ) -> list[dict]:
        """
        Offline helper used by /process: transcribe only VAD speech segments.

        Returns a list of dicts:
          { text, start, end, chunks: [{text, timestamp}] }

        We run whisper.cpp on each segment separately and then shift timestamps
        back into the original audio timeline.
        """
        # Ensure mono 1D
        if isinstance(audio, torch.Tensor):
            audio_t = audio
            while audio_t.dim() > 1:
                audio_t = audio_t.squeeze(0)
        else:
            audio_t = torch.as_tensor(audio, dtype=torch.float32)

        results: list[dict] = []
        for seg in speech_segments:
            start_time = float(seg.get("start", 0.0) or 0.0)
            end_time = float(seg.get("end", start_time) or start_time)
            if end_time <= start_time:
                continue

            start_samp = int(start_time * sample_rate)
            end_samp = int(end_time * sample_rate)
            if end_samp <= start_samp:
                continue

            seg_audio = audio_t[start_samp:end_samp]
            # Skip extremely short
            if seg_audio.numel() < int(0.10 * sample_rate):
                continue

            tr = self.transcribe(
                seg_audio,
                return_timestamps=return_timestamps,
                language=language,
                word_timestamps=word_timestamps,
                **decode_kwargs,
            )

            out = {
                "text": (tr.get("text") or "").strip(),
                "start": start_time,
                "end": end_time,
            }

            if return_timestamps and "chunks" in tr:
                adjusted = []
                for ch in (tr.get("chunks") or []):
                    ts = ch.get("timestamp") or (None, None)
                    if not (isinstance(ts, (list, tuple)) and len(ts) == 2):
                        continue
                    s0, s1 = ts
                    if s0 is None and s1 is None:
                        continue
                    s0 = float(s0 or 0.0) + start_time
                    s1 = float(s1 or s0) + start_time
                    adjusted.append({"text": ch.get("text", ""), "timestamp": (s0, s1)})
                if adjusted:
                    out["chunks"] = adjusted

            results.append(out)

        return results


