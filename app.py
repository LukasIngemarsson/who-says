import os
import time
import tempfile
import subprocess
import base64
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from flask import Flask, jsonify, request, send_from_directory, send_file
from loguru import logger
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn.functional as F

import warnings
from utils import load_audio_from_file, match_frequency
warnings.filterwarnings("ignore")

# -------------------------------------------------
# Try importing the WhoSays pipeline
# -------------------------------------------------
try:
    from main import WhoSays
    PIPELINE_AVAILABLE = True
except ImportError:
    logger.warning("Could not import WhoSays from main.py. Running in mock mode.")
    PIPELINE_AVAILABLE = False

load_dotenv(".env")

static_folder_path = os.environ.get("FLASK_STATIC_FOLDER", "../frontend/dist")
static_folder_path = os.path.abspath(static_folder_path)

app = Flask(__name__, static_folder=static_folder_path, static_url_path='')

# -------------------------------------------------
# Initialize pipeline
# -------------------------------------------------
if PIPELINE_AVAILABLE:
    logger.info("Loading WhoSays pipeline... This may take a moment.")
    pipeline = WhoSays()
    logger.info("Pipeline loaded successfully. Server is ready.")
else:
    pipeline = None

# -------------------------------------------------
# Global state
# -------------------------------------------------

# Live ASR per session:
#   session_id -> {
#       "buffer": Tensor on pipeline device (1D),
#       "cursor": int,
#       "last_asr_time": float,
#       "last_word_end": float,
#       "last_snippet": str,
#   }
SESSION_ASR_STATE: Dict[str, Dict[str, Any]] = {}

# Per-session speaker timeline in wall-clock seconds:
#   session_id -> [
#       {"start": float, "end": float | None, "speaker": str},
#       ...
#   ]
SESSION_SPEAKER_TIMELINE: Dict[str, List[Dict[str, Any]]] = {}

# Global rolling / speech buffers for diarization (per process, not per session)
ROLLING_BUFFER: Optional[torch.Tensor] = None  # ~1s of audio
SPEECH_BUFFER: Optional[torch.Tensor] = None   # up to ~2.5s for embedding

# Known speakers
EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)
KNOWN_SPEAKERS: Dict[str, torch.Tensor] = {}  # name -> 1D normed embedding (CPU)

# Streaming config (whisper.cpp-first)
#
# NOTE: Frontend chunk size can vary depending on backend. For all gating logic
# we use *seconds* rather than "frames" to avoid mismatches.
ASR_MIN_NEW_SEC = 0.5          # minimum new audio (seconds) before ASR
MIN_ASR_INTERVAL_SEC = 0.0     # throttle ASR calls (seconds)
MAX_ASR_BUFFER_SEC = 12.0       # cap per-session ASR buffer (seconds)
WCPP_CONTEXT_SEC = 0.0          # decode this much audio before cursor for stability

# Minimal de-dup: guard against jittery re-emits across calls
CROSS_TAIL_DUP_PAD_SEC = 0.0

# Rolling prompt (used as whisper.cpp --prompt for streaming stability)
ASR_PROMPT_MAX_CHARS = 240
USE_INITIAL_PROMPT = False

# Diarization: how much speech to accumulate for embedding
MIN_SPEECH_SEC = 0.0

# Diarization state
CURRENT_SPEAKER: Optional[str] = None
CURRENT_CONFIDENCE: float = 0.0
LAST_SWITCH_TIME: float = 0.0
SWITCH_COOLDOWN: float = 0.0   # seconds

SIM_CONF_FLOOR = 0.05
SIM_CONF_CEIL = 0.50

# -------------------------------------------------
# Tuning presets
# -------------------------------------------------
TUNING_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        # Streaming
        "asr_min_new_sec": 0.50,
        "cross_tail_dup_pad_sec": 0.03,
        "min_asr_interval_sec": 0.45,
        "max_asr_buffer_sec": 12.0,
        "wcpp_context_sec": 2.0,
        "use_initial_prompt": True,
        "asr_prompt_max_chars": 240,

        # ASR (whisper.cpp CLI wrapper supports these)
        "beam_size": 1,
        "best_of": 1,

        # VAD (used both for diarization VAD and whisper.cpp --vad)
        "vad_threshold": 0.55,
        "vad_min_speech_ms": 150,
        "vad_min_silence_ms": 150,
        "vad_speech_pad_ms": 40,
    },

    "clean_speech": {
        "asr_min_new_sec": 0.50,
        "cross_tail_dup_pad_sec": 0.02,
        "min_asr_interval_sec": 0.40,
        "max_asr_buffer_sec": 12.0,
        "wcpp_context_sec": 2.0,
        "use_initial_prompt": True,
        "asr_prompt_max_chars": 240,

        "beam_size": 1,
        "best_of": 1,

        "vad_threshold": 0.50,
        "vad_min_speech_ms": 120,
        "vad_min_silence_ms": 120,
        "vad_speech_pad_ms": 30,
    },

    "noisy_room": {
        "asr_min_new_sec": 0.75,
        "cross_tail_dup_pad_sec": 0.04,
        "min_asr_interval_sec": 0.55,
        "max_asr_buffer_sec": 12.0,
        "wcpp_context_sec": 2.5,
        "use_initial_prompt": True,
        "asr_prompt_max_chars": 240,

        "beam_size": 1,
        "best_of": 1,

        "vad_threshold": 0.60,
        "vad_min_speech_ms": 200,
        "vad_min_silence_ms": 220,
        "vad_speech_pad_ms": 50,
    },

    # Preset optimized for the built-in `thetestsound.wav` script:
    # - keep prompt enabled
    # - slightly slower cadence + larger new-audio window for stability
    "thetestsound_script": {
        # Streaming
        "asr_min_new_sec": 0.60,
        "min_asr_interval_sec": 0.45,
        "max_asr_buffer_sec": 12.0,
        "cross_tail_dup_pad_sec": 0.03,
        "wcpp_context_sec": 2.5,
        "use_initial_prompt": True,
        "asr_prompt_max_chars": 240,

        # ASR
        "beam_size": 3,
        "best_of": 3,

        # VAD
        "vad_threshold": 0.55,
        "vad_min_speech_ms": 150,
        "vad_min_silence_ms": 150,
        "vad_speech_pad_ms": 40,
    },
}

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def similarity_to_confidence(score: float,
                             floor: float = SIM_CONF_FLOOR,
                             ceil: float = SIM_CONF_CEIL) -> float:
    """Map cosine similarity into [0,1] for UI purposes."""
    x = (score - floor) / (ceil - floor)
    x = max(0.0, min(1.0, x))
    return float(x)


def squash_adjacent_short_repeats(text: str, max_token_len: int = 4) -> str:
    """
    Collapse adjacent repeated short tokens (e.g. "okay okay" -> "okay")
    to reduce stuttery duplicates from streaming ASR.
    """
    if not text:
        return text

    tokens = text.split()
    if len(tokens) <= 1:
        return text

    out: List[str] = []
    prev_norm: Optional[str] = None

    for tok in tokens:
        norm = re.sub(r"[^\w]+", "", tok).lower()
        if norm and prev_norm == norm and len(norm) <= max_token_len:
            continue
        out.append(tok)
        prev_norm = norm

    return " ".join(out)


def _is_prompt_worthy(snippet: str) -> bool:
    """
    Keep the rolling `initial_prompt` clean.
    Feeding partial/jittery fragments into the prompt can make the next decode
    latch onto them and produce junk tokens later (e.g. stray 'or...' / 'is...').
    """
    s = (snippet or "").strip()
    if not s:
        return False
    if "..." in s:
        return False

    toks = [t for t in s.split() if t.strip()]
    # Very short snippets are often boundary artifacts.
    if len(toks) < 3:
        return False

    # Prefer complete sentences; otherwise require a longer phrase.
    if s[-1] in ".!?":
        return True
    return len(toks) >= 6


def save_embedding(name: str, tensor: torch.Tensor) -> None:
    try:
        file_path = EMBEDDINGS_DIR / f"{name}.pt"
        torch.save(tensor.cpu(), file_path)
        logger.info(f"Saved embedding for {name} to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save embedding for {name}: {e}")


def load_embeddings() -> None:
    KNOWN_SPEAKERS.clear()
    try:
        for file_path in EMBEDDINGS_DIR.glob("*.pt"):
            name = file_path.stem
            tensor = torch.load(file_path, map_location="cpu").squeeze()
            if tensor.dim() != 1:
                tensor = tensor.flatten()
            tensor = F.normalize(tensor, p=2, dim=0)
            KNOWN_SPEAKERS[name] = tensor
        logger.info(f"Loaded {len(KNOWN_SPEAKERS)} embeddings from {EMBEDDINGS_DIR}")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj


def convert_to_wav(input_path: str) -> str:
    """Convert any input audio to 16kHz mono WAV using ffmpeg."""
    try:
        output_fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(output_fd)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            output_path,
        ]
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
        logger.info(f"Converted {input_path} -> {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode()
        logger.error(f"FFmpeg conversion failed: {msg}")
        raise RuntimeError(f"Could not convert audio file: {msg}")


def _current_tuning_snapshot() -> Dict[str, Any]:
    asr_cfg: Dict[str, Any] = {}
    vad_cfg: Dict[str, Any] = {}
    if PIPELINE_AVAILABLE and pipeline is not None:
        asr = getattr(pipeline, "asr", None)
        vad = getattr(pipeline, "vad", None)

        if asr is not None:
            asr_cfg = {
                "beam_size": int(getattr(asr, "beam_size", 1)),
                "best_of": int(getattr(asr, "best_of", 1)),
                "temperature": float(getattr(asr, "temperature", 0.0)),
                "compression_ratio_threshold": float(
                    getattr(asr, "compression_ratio_threshold", 2.4)
                ),
                "log_prob_threshold": float(
                    getattr(asr, "log_prob_threshold", -1.0)
                ),
                "no_speech_threshold": float(
                    getattr(asr, "no_speech_threshold", 0.8)
                ),
                "chunk_length": int(getattr(asr, "chunk_length", 15)),
                "patience": float(getattr(asr, "patience", 1.0)),
                "repetition_penalty": float(getattr(asr, "repetition_penalty", 1.0)),
                "no_repeat_ngram_size": int(getattr(asr, "no_repeat_ngram_size", 0)),
                "condition_on_previous_text": bool(
                    getattr(asr, "condition_on_previous_text", True)
                ),
            }

        if vad is not None:
            vad_cfg = {
                "threshold": float(getattr(vad, "threshold", 0.65)),
                "min_speech_duration_ms": int(
                    getattr(vad, "min_speech_duration_ms", 200)
                ),
                "min_silence_duration_ms": int(
                    getattr(vad, "min_silence_duration_ms", 150)
                ),
                "speech_pad_ms": int(getattr(vad, "speech_pad_ms", 80)),
            }

    return {
        "streaming": {
            "asr_min_new_sec": ASR_MIN_NEW_SEC,
            "min_asr_interval_sec": MIN_ASR_INTERVAL_SEC,
            "max_asr_buffer_sec": MAX_ASR_BUFFER_SEC,
            "wcpp_context_sec": WCPP_CONTEXT_SEC,
            "min_speech_sec": MIN_SPEECH_SEC,
            "cross_tail_dup_pad_sec": CROSS_TAIL_DUP_PAD_SEC,
            "use_initial_prompt": USE_INITIAL_PROMPT,
            "asr_prompt_max_chars": ASR_PROMPT_MAX_CHARS,
        },
        "asr": asr_cfg,
        "vad": vad_cfg,
    }

# -------------------------------------------------
# Speaker timeline helpers
# -------------------------------------------------
def _speaker_at_time(session_id: str, t: float, fallback: str) -> str:
    timeline = SESSION_SPEAKER_TIMELINE.get(session_id, [])
    for seg in reversed(timeline):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", t))
        if start <= t <= end:
            return seg.get("speaker") or fallback
    return fallback


def _assign_words_to_speakers(
    session_id: str,
    snippet_obj: Any,
    fallback_speaker: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    if not isinstance(snippet_obj, dict):
        text = (snippet_obj or "").strip() if snippet_obj is not None else ""
        if not text:
            return "", []
        return text, [{"speaker": fallback_speaker, "text": text}]

    text = (snippet_obj.get("text") or "").strip()
    words = snippet_obj.get("words") or []
    duration = float(snippet_obj.get("duration") or 0.0)

    if not text or not words:
        return text, [{"speaker": fallback_speaker, "text": text}] if text else ("", [])

    segments: List[Dict[str, Any]] = []

    for w in words:
        tok = (w.get("text") or "").strip()
        if not tok:
            continue

        start_rel = w.get("start")
        end_rel = w.get("end")
        if start_rel is None and end_rel is None:
            continue

        if "start_abs" in w and "end_abs" in w:
            start_abs = float(w["start_abs"])
            end_abs = float(w["end_abs"])
            center_abs = 0.5 * (start_abs + end_abs)
        else:
            tail_end = time.time()
            start_rel = float(start_rel or 0.0)
            end_rel = float(end_rel or start_rel)
            center_rel = 0.5 * (start_rel + end_rel)
            center_abs = tail_end - max(0.0, duration - center_rel)

        speaker = _speaker_at_time(session_id, center_abs, fallback_speaker)

        if segments and segments[-1]["speaker"] == speaker:
            segments[-1]["words"].append(tok)
        else:
            segments.append({"speaker": speaker, "words": [tok]})

    transcript_segments: List[Dict[str, Any]] = []
    for seg in segments:
        seg_text = " ".join(seg["words"]).strip()
        if not seg_text:
            continue
        transcript_segments.append(
            {"speaker": seg["speaker"] or fallback_speaker, "text": seg_text}
        )

    return text, transcript_segments

# -------------------------------------------------
# Incremental ASR with micro-tail
# -------------------------------------------------
def get_live_snippet_for_session(session_id: str, sr: int) -> Optional[Dict[str, Any]]:
    """
    Incremental ASR: decode only the new audio since the last cursor.
    No multi-second tail is re-decoded, which avoids recomposition loops
    and timestamp jitter on past audio.
    """
    if not PIPELINE_AVAILABLE or pipeline is None:
        return None

    state = SESSION_ASR_STATE.get(session_id)
    if state is None:
        return None

    buffer: torch.Tensor = state["buffer"]
    cursor: int = int(state["cursor"])
    t0: float = float(state.get("t0", 0.0))

    if buffer.shape[0] == 0 or cursor >= buffer.shape[0]:
        return None

    # New audio region (incremental)
    new_audio = buffer[cursor:]
    min_new_samples = int(float(ASR_MIN_NEW_SEC) * float(sr))
    if new_audio.shape[0] < min_new_samples:
        return None

    # -------------------------------------------------
    # whisper.cpp-first live path:
    # - decode only the NEW slice (offset/duration) using whisper.cpp CLI
    # - rely on whisper.cpp built-in VAD and internal context
    # - keep only minimal time-based de-dup
    # -------------------------------------------------
    # whisper.cpp-only live path
    now = time.time()
    last_asr_time = float(state.get("last_asr_time", 0.0))
    if now - last_asr_time < float(MIN_ASR_INTERVAL_SEC):
        return None

    use_prompt = bool(USE_INITIAL_PROMPT)
    initial_prompt = (state.get("prompt") or "").strip() if use_prompt else ""

    # Decode a little context *before* the cursor to stabilize boundaries.
    # We still only EMIT words that fall after the cursor.
    ctx_samples = int(max(0.0, float(WCPP_CONTEXT_SEC)) * float(sr))
    slice_start = max(0, cursor - ctx_samples)
    slice_audio = buffer[slice_start:]

    slice_start_sec = slice_start / float(sr)
    new_region_start_abs = t0 + (cursor / float(sr))
    new_region_end_abs = t0 + (buffer.shape[0] / float(sr))

    # Use the same VAD knobs for diarization AND whisper.cpp internal VAD.
    # This keeps the UI tuning panel meaningful in whisper.cpp mode.
    vad_threshold = float(getattr(getattr(pipeline, "vad", None), "threshold", 0.50))
    vad_min_speech_ms = int(getattr(getattr(pipeline, "vad", None), "min_speech_duration_ms", 200))
    vad_min_silence_ms = int(getattr(getattr(pipeline, "vad", None), "min_silence_duration_ms", 150))
    vad_speech_pad_ms = int(getattr(getattr(pipeline, "vad", None), "speech_pad_ms", 30))

    asr_cfg = getattr(getattr(pipeline, "config", None), "asr", None)
    beam_size = int(getattr(asr_cfg, "beam_size", 3)) if asr_cfg is not None else 3
    best_of = int(getattr(asr_cfg, "best_of", 3)) if asr_cfg is not None else 3

    try:
        with torch.inference_mode():
            asr_result = pipeline.asr.transcribe(
                slice_audio,
                return_timestamps=True,
                word_timestamps=True,
                # whisper.cpp VAD:
                vad=True,
                vad_threshold=vad_threshold,
                vad_min_speech_ms=vad_min_speech_ms,
                vad_min_silence_ms=vad_min_silence_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
                # context:
                initial_prompt=initial_prompt if initial_prompt else None,
                carry_initial_prompt=True,
                # search
                beam_size=beam_size,
                best_of=best_of,
            )
    except Exception as e:
        logger.error(f"ASR error (whispercpp) in get_live_snippet_for_session: {e}")
        return None

    if isinstance(asr_result, dict):
        chunks = asr_result.get("chunks") or []
    else:
        chunks = []

    if not chunks:
        # Consume the audio slice so we don't reprocess it forever.
        state["cursor"] = buffer.shape[0]
        state["last_asr_time"] = now
        return None

    # Convert chunks (segment-level) into pseudo word-level timestamps by
    # distributing the segment duration across its tokens.
    last_word_end_abs = float(state.get("last_word_end", float("-inf")))
    # Safety: if last_word_end got poisoned by a bad segment, don't let it block new emissions.
    if last_word_end_abs > new_region_start_abs + 1.0:
        last_word_end_abs = new_region_start_abs

    words: List[Dict[str, Any]] = []
    for ch in chunks:
        text = (ch.get("text") or "").strip()
        ts = ch.get("timestamp") or (None, None)
        if not text or text == "[BLANK_AUDIO]":
            continue
        start_rel, end_rel = ts
        if start_rel is None and end_rel is None:
            continue
        start_rel = float(start_rel or 0.0)
        end_rel = float(end_rel or start_rel)

        # The whisper.cpp wrapper returns timestamps relative to the provided audio.
        # Shift from slice-time -> buffer-time coordinates.
        start_rel = start_rel + slice_start_sec
        end_rel = end_rel + slice_start_sec

        start_abs = t0 + start_rel
        end_abs = t0 + end_rel
        if end_abs <= start_abs:
            continue
        # Ensure we're only emitting within the new region
        if end_abs <= new_region_start_abs:
            continue
        if start_abs >= new_region_end_abs:
            continue

        toks = [t for t in text.split() if t.strip()]
        if not toks:
            continue
        seg_dur = max(1e-3, (end_abs - start_abs))
        step = seg_dur / float(len(toks))
        for i, tok in enumerate(toks):
            w_start_abs = start_abs + (i * step)
            w_end_abs = start_abs + ((i + 1) * step)
            # Minimal cross-call dedupe
            if w_start_abs <= last_word_end_abs + CROSS_TAIL_DUP_PAD_SEC:
                continue
            words.append({
                "text": tok,
                "start": float(w_start_abs - t0),
                "end": float(w_end_abs - t0),
                "start_abs": float(w_start_abs),
                "end_abs": float(w_end_abs),
            })

    if not words:
        state["cursor"] = buffer.shape[0]
        state["last_asr_time"] = now
        return None

    snippet = " ".join(w["text"] for w in words).strip()
    if not snippet:
        state["cursor"] = buffer.shape[0]
        state["last_asr_time"] = now
        return None

    # Simple snippet-level dedupe
    last_snippet = (state.get("last_snippet") or "").strip()
    if snippet == last_snippet:
        state["cursor"] = buffer.shape[0]
        state["last_asr_time"] = now
        state["last_word_end"] = float(max(w["end_abs"] for w in words))
        return None

    state["cursor"] = buffer.shape[0]
    state["last_asr_time"] = now
    state["last_word_end"] = float(max(w["end_abs"] for w in words))
    state["last_snippet"] = snippet

    # Only add good snippets into the prompt (keeps whisper.cpp stable).
    if use_prompt and _is_prompt_worthy(snippet):
        prompt = ((state.get("prompt") or "").strip() + " " + snippet).strip()
        if len(prompt) > ASR_PROMPT_MAX_CHARS:
            prompt = prompt[-ASR_PROMPT_MAX_CHARS:].lstrip()
        state["prompt"] = prompt

    return {
        "text": snippet,
        "words": words,
        "duration": float(slice_audio.shape[0] / float(sr)),
    }

# -------------------------------------------------
# Flask routes
# -------------------------------------------------
@app.route("/")
def index():
    logger.info(f"Serving {app.static_folder}")
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def catch_all(path: str):
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


@app.route("/testaudio/thetestsound.wav", methods=["GET"])
def serve_testsound():
    """Serve the built-in test WAV for UI-based streaming tests."""
    wav_path = os.path.join(os.getcwd(), "thetestsound.wav")
    if not os.path.exists(wav_path):
        return jsonify({"error": "thetestsound.wav not found on server"}), 404
    return send_file(wav_path, mimetype="audio/wav")


@app.route("/status", methods=["GET"])
def status():
    asr_backend = None
    try:
        if PIPELINE_AVAILABLE and pipeline is not None:
            asr_type = getattr(getattr(pipeline, "config", None), "asr", None)
            asr_type = getattr(asr_type, "asr_type", None)
            asr_backend = getattr(asr_type, "value", None) or (str(asr_type) if asr_type is not None else None)
    except Exception:
        asr_backend = None
    return jsonify({
        "status": "WhoSays server is running.",
        "pipeline_loaded": PIPELINE_AVAILABLE,
        "known_speakers": list(KNOWN_SPEAKERS.keys()),
        "asr_backend": asr_backend,
    })

# -------------------------------------------------
# /tuning
# -------------------------------------------------
@app.route("/tuning", methods=["GET", "POST"])
def tuning():
    global ASR_MIN_NEW_SEC, MIN_SPEECH_SEC, CROSS_TAIL_DUP_PAD_SEC
    global MIN_ASR_INTERVAL_SEC, MAX_ASR_BUFFER_SEC
    global WCPP_CONTEXT_SEC
    global USE_INITIAL_PROMPT, ASR_PROMPT_MAX_CHARS

    if request.method == "GET":
        return jsonify(_current_tuning_snapshot())

    data = request.get_json(silent=True) or {}

    # preset handling
    if "preset" in data:
        preset_name = data["preset"]
        if preset_name not in TUNING_PRESETS:
            return jsonify({"error": f"Unknown preset '{preset_name}'"}), 400
        preset = TUNING_PRESETS[preset_name]

        if "asr_min_new_sec" in preset:
            ASR_MIN_NEW_SEC = max(0.0, float(preset["asr_min_new_sec"]))
        if "min_asr_interval_sec" in preset:
            MIN_ASR_INTERVAL_SEC = max(0.0, float(preset["min_asr_interval_sec"]))
        if "max_asr_buffer_sec" in preset:
            MAX_ASR_BUFFER_SEC = max(0.0, float(preset["max_asr_buffer_sec"]))
        if "wcpp_context_sec" in preset:
            WCPP_CONTEXT_SEC = max(0.0, float(preset["wcpp_context_sec"]))
        if "min_speech_sec" in preset:
            MIN_SPEECH_SEC = max(0.0, float(preset["min_speech_sec"]))
        if "cross_tail_dup_pad_sec" in preset:
            CROSS_TAIL_DUP_PAD_SEC = max(0.0, float(preset["cross_tail_dup_pad_sec"]))
        if "use_initial_prompt" in preset:
            USE_INITIAL_PROMPT = bool(preset["use_initial_prompt"])
        if "asr_prompt_max_chars" in preset:
            ASR_PROMPT_MAX_CHARS = max(0, int(preset["asr_prompt_max_chars"]))

        if PIPELINE_AVAILABLE and pipeline is not None:
            asr = getattr(pipeline, "asr", None)
            vad = getattr(pipeline, "vad", None)

            if asr:
                if "beam_size" in preset:
                    asr.beam_size = int(preset["beam_size"])
                if "best_of" in preset:
                    asr.best_of = int(preset["best_of"])

            if vad:
                if "vad_threshold" in preset:
                    vad.threshold = float(preset["vad_threshold"])
                if "vad_min_speech_ms" in preset:
                    vad.min_speech_duration_ms = int(preset["vad_min_speech_ms"])
                if "vad_min_silence_ms" in preset:
                    vad.min_silence_duration_ms = int(preset["vad_min_silence_ms"])
                if "vad_speech_pad_ms" in preset:
                    vad.speech_pad_ms = int(preset["vad_speech_pad_ms"])

        return jsonify({"message": f"Applied preset '{preset_name}'", "settings": preset})

    # Manual tuning
    try:
        if "asr_min_new_sec" in data:
            ASR_MIN_NEW_SEC = max(0.0, float(data["asr_min_new_sec"]))
        if "min_asr_interval_sec" in data:
            MIN_ASR_INTERVAL_SEC = max(0.0, float(data["min_asr_interval_sec"]))
        if "max_asr_buffer_sec" in data:
            MAX_ASR_BUFFER_SEC = max(0.0, float(data["max_asr_buffer_sec"]))
        if "wcpp_context_sec" in data:
            WCPP_CONTEXT_SEC = max(0.0, float(data["wcpp_context_sec"]))
        if "use_initial_prompt" in data:
            USE_INITIAL_PROMPT = bool(data["use_initial_prompt"])
        if "asr_prompt_max_chars" in data:
            ASR_PROMPT_MAX_CHARS = max(0, int(data["asr_prompt_max_chars"]))
        if "min_speech_sec" in data:
            MIN_SPEECH_SEC = max(0.0, float(data["min_speech_sec"]))
        if "cross_tail_dup_pad_sec" in data:
            CROSS_TAIL_DUP_PAD_SEC = max(0.0, float(data["cross_tail_dup_pad_sec"]))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid streaming tuning values"}), 400

    if PIPELINE_AVAILABLE and pipeline is not None:
        asr = getattr(pipeline, "asr", None)
        vad = getattr(pipeline, "vad", None)

        if asr:
            try:
                if "beam_size" in data:
                    asr.beam_size = int(data["beam_size"])
                if "best_of" in data:
                    asr.best_of = int(data["best_of"])
            except (TypeError, ValueError):
                return jsonify({"error": "Invalid ASR tuning values"}), 400

        if vad:
            try:
                if "vad_threshold" in data:
                    vad.threshold = float(data["vad_threshold"])
                if "vad_min_speech_ms" in data:
                    vad.min_speech_duration_ms = int(data["vad_min_speech_ms"])
                if "vad_min_silence_ms" in data:
                    vad.min_silence_duration_ms = int(data["vad_min_silence_ms"])
                if "vad_speech_pad_ms" in data:
                    vad.speech_pad_ms = int(data["vad_speech_pad_ms"])
            except (TypeError, ValueError):
                return jsonify({"error": "Invalid VAD tuning values"}), 400

    return jsonify(_current_tuning_snapshot())

# -------------------------------------------------
# /upload_embeddings - enroll speakers
# -------------------------------------------------
@app.route("/upload_embeddings", methods=["POST"])
def upload_embeddings():
    if not PIPELINE_AVAILABLE:
        name = request.form.get("name", "Unknown")
        return jsonify({"message": f"Mock enrollment for {name}", "vector_size": 0})

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        speaker_name = request.form.get("name")
        if not speaker_name:
            return jsonify({"error": "Speaker 'name' is required"}), 400

        suffix = Path(str(file.filename)).suffix or ".webm"
        temp_file_path = None
        wav_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                file.save(temp_file)
                temp_file_path = temp_file.name

            wav_path = convert_to_wav(temp_file_path)

            waveform, sr = load_audio_from_file(wav_path, sr=16000)
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.float().squeeze()

            device = pipeline.config.device
            waveform = waveform.to(device)

            with torch.inference_mode():
                embedding_tensor = pipeline.embedder.embed(waveform, sr)

            while embedding_tensor.dim() > 1:
                embedding_tensor = embedding_tensor.squeeze(0)

            embedding_tensor = F.normalize(embedding_tensor, p=2, dim=0)

            KNOWN_SPEAKERS[speaker_name] = embedding_tensor.cpu()
            save_embedding(speaker_name, embedding_tensor)

            return jsonify({
                "message": f"Enrolled {speaker_name}",
                "vector_size": str(tuple(embedding_tensor.shape)),
            })

        except Exception as e:
            logger.error(f"Error during upload_embeddings: {e}")
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

    except Exception as e:
        logger.error(f"Error in upload_embeddings: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# /identify_speaker - live diarization + ASR
# -------------------------------------------------
@app.route("/identify_speaker", methods=["POST"])
def identify_speaker():
    global ROLLING_BUFFER, SPEECH_BUFFER
    global CURRENT_SPEAKER, CURRENT_CONFIDENCE, LAST_SWITCH_TIME

    if not PIPELINE_AVAILABLE:
        return jsonify({"error": "Pipeline not available"}), 503

    try:
        if "audio_data" not in request.form:
            return jsonify({"error": "No audio_data provided"}), 400

        audio_data_b64 = request.form.get("audio_data")
        sample_rate = int(request.form.get("sample_rate", 16000))
        session_id = request.form.get("session_id", "default")

        try:
            audio_bytes = base64.b64decode(audio_data_b64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

            waveform = torch.from_numpy(audio_array).float().squeeze()
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)

            sr = sample_rate
            target_sr = pipeline.config.sr
            if sr != target_sr:
                wf_2d = waveform.unsqueeze(0)
                wf_2d = match_frequency(wf_2d, sr, sr=target_sr)
                waveform = wf_2d.squeeze(0)
                sr = target_sr

            device = pipeline.config.device
            waveform = waveform.to(device)

        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return jsonify({"error": "Invalid audio data"}), 400

        # --- ASR buffer per session ---
        if session_id not in SESSION_ASR_STATE:
            SESSION_ASR_STATE[session_id] = {
                "buffer": torch.zeros(0, device=device),
                "cursor": 0,
                # Absolute time (seconds) corresponding to buffer[0]
                "t0": 0.0,
                "last_asr_time": 0.0,
                "last_word_end": float("-inf"),
                "last_snippet": "",
                "prompt": "",
            }
        asr_state = SESSION_ASR_STATE[session_id]
        asr_state["buffer"] = torch.cat([asr_state["buffer"], waveform.detach()])

        # Cap ASR buffer (seconds) and keep an absolute time offset so
        # timestamps remain monotonic even when we trim old samples.
        MAX_ASR_BUFFER = int(sr * float(MAX_ASR_BUFFER_SEC))
        if asr_state["buffer"].shape[0] > MAX_ASR_BUFFER:
            overflow = asr_state["buffer"].shape[0] - MAX_ASR_BUFFER
            asr_state["cursor"] = max(0, asr_state["cursor"] - overflow)
            asr_state["t0"] = float(asr_state.get("t0", 0.0)) + (overflow / float(sr))
            asr_state["buffer"] = asr_state["buffer"][-MAX_ASR_BUFFER:]

        # --- Global rolling buffer (1s) + speech buffer (2.5s) ---
        if ROLLING_BUFFER is None:
            ROLLING_BUFFER = torch.zeros(sr, device=device)
        if SPEECH_BUFFER is None:
            SPEECH_BUFFER = torch.zeros(0, device=device)

        chunk = waveform
        if chunk.shape[0] >= sr:
            ROLLING_BUFFER = chunk[-sr:]
        else:
            needed = sr - chunk.shape[0]
            ROLLING_BUFFER = torch.cat([ROLLING_BUFFER[-needed:], chunk])

        vad_input = ROLLING_BUFFER

        # VAD for diarization
        try:
            with torch.inference_mode():
                speech_segments = pipeline.vad(vad_input)
        except Exception as e:
            logger.error(f"VAD error in identify_speaker: {e}")
            return jsonify({"error": "VAD failure"}), 500

        debug_mode = request.args.get("debug") == "1"
        debug_payload = None
        if debug_mode:
            try:
                rms = float(torch.sqrt(torch.mean(vad_input.float() ** 2)).detach().cpu())
                mx = float(torch.max(torch.abs(vad_input.float())).detach().cpu())
            except Exception:
                rms, mx = None, None
            debug_payload = {
                "vad_threshold": float(getattr(pipeline.vad, "threshold", -1.0)),
                "vad_min_speech_duration_ms": int(getattr(pipeline.vad, "min_speech_duration_ms", -1)),
                "vad_min_silence_duration_ms": int(getattr(pipeline.vad, "min_silence_duration_ms", -1)),
                "vad_speech_pad_ms": int(getattr(pipeline.vad, "speech_pad_ms", -1)),
                "vad_segments": speech_segments,
                "vad_input_rms": rms,
                "vad_input_absmax": mx,
                "vad_input_samples": int(vad_input.shape[0]),
                "sr": int(sr),
            }

        if not speech_segments:
            # If diarization-VAD misses speech, do NOT skip ASR.
            # Let the ASR-side VAD decide, so we don't drop words.
            snippet_obj = get_live_snippet_for_session(session_id, sr)
            live_text, transcript_segments = _assign_words_to_speakers(
                session_id,
                snippet_obj,
                CURRENT_SPEAKER or "Unknown",
            )
            #live_text = squash_adjacent_short_repeats(live_text)
            #for seg in transcript_segments:
                #seg["text"] = squash_adjacent_short_repeats(seg.get("text") or "")

            ui_conf = similarity_to_confidence(CURRENT_CONFIDENCE)
            resp = {
                "speaker": CURRENT_SPEAKER,
                "has_speech": bool(live_text),
                "confidence": ui_conf,
                "transcript": live_text,
            }
            if transcript_segments:
                resp["transcript_segments"] = transcript_segments
            if debug_payload is not None:
                resp["debug"] = debug_payload
            return jsonify(resp)

        # Build speech buffer
        speech_portions: List[torch.Tensor] = []
        for seg in speech_segments:
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            seg_audio = vad_input[start_sample:end_sample]
            if seg_audio.shape[0] > 0:
                speech_portions.append(seg_audio)

        if speech_portions:
            new_speech = torch.cat(speech_portions)
        else:
            new_speech = torch.zeros(0, device=device)

        SPEECH_BUFFER = torch.cat([SPEECH_BUFFER, new_speech])

        MAX_SPEECH_SAMPLES = int(2.5 * sr)
        if SPEECH_BUFFER.shape[0] > MAX_SPEECH_SAMPLES * 1.5:
            SPEECH_BUFFER = SPEECH_BUFFER[-MAX_SPEECH_SAMPLES:]

        MIN_SPEECH_SAMPLES = int(max(1.0, float(MIN_SPEECH_SEC)) * float(sr))
        if SPEECH_BUFFER.shape[0] < MIN_SPEECH_SAMPLES:
            ui_conf = similarity_to_confidence(CURRENT_CONFIDENCE)
            snippet_obj = get_live_snippet_for_session(session_id, sr)
            live_text = snippet_obj.get("text") if isinstance(snippet_obj, dict) else (snippet_obj or "")
            return jsonify({
                "speaker": CURRENT_SPEAKER,
                "has_speech": True,
                "confidence": ui_conf,
                "message": "Collecting more speech...",
                "transcript": live_text,
            })

        # Compute embedding
        with torch.inference_mode():
            emb = pipeline.embedder.embed(SPEECH_BUFFER, sr)
        while emb.dim() > 1:
            emb = emb.squeeze(0)
        emb = F.normalize(emb, p=2, dim=0)

        best_speaker = None
        best_score = -1.0
        second_best = -1.0

        if KNOWN_SPEAKERS:
            for name, ref_cpu in KNOWN_SPEAKERS.items():
                ref = F.normalize(ref_cpu.to(device), p=2, dim=0)
                score = torch.dot(emb, ref).item()
                if score > best_score:
                    second_best = best_score
                    best_score = score
                    best_speaker = name
                elif score > second_best:
                    second_best = score

        now = time.time()
        MIN_CONFIDENCE = 0.30
        MARGIN_THRESHOLD = 0.05

        margin = best_score - second_best if second_best > -1.0 else best_score

        confident_speaker: Optional[str] = None
        if best_speaker is not None and best_score >= MIN_CONFIDENCE and margin >= MARGIN_THRESHOLD:
            confident_speaker = best_speaker

        if confident_speaker is not None:
            if CURRENT_SPEAKER is None:
                CURRENT_SPEAKER = confident_speaker
                CURRENT_CONFIDENCE = best_score
                LAST_SWITCH_TIME = now
            else:
                if confident_speaker != CURRENT_SPEAKER:
                    if now - LAST_SWITCH_TIME >= SWITCH_COOLDOWN:
                        CURRENT_SPEAKER = confident_speaker
                        CURRENT_CONFIDENCE = best_score
                        LAST_SWITCH_TIME = now
                else:
                    ALPHA = 0.3
                    CURRENT_CONFIDENCE = ALPHA * best_score + (1 - ALPHA) * CURRENT_CONFIDENCE
                    LAST_SWITCH_TIME = now
        logger.log(f"Speaker: {CURRENT_SPEAKER}{best_score}")
        # Speaker timeline
        timeline = SESSION_SPEAKER_TIMELINE.setdefault(session_id, [])
        if CURRENT_SPEAKER is not None:
            if timeline and timeline[-1]["speaker"] == CURRENT_SPEAKER:
                timeline[-1]["end"] = now
            else:
                if timeline and timeline[-1].get("end") is None:
                    timeline[-1]["end"] = now
                timeline.append({"start": now, "end": now, "speaker": CURRENT_SPEAKER})

            WINDOW_SEC = 30.0
            cutoff = now - WINDOW_SEC
            while timeline and timeline[0].get("end") is not None and timeline[0]["end"] < cutoff:
                timeline.pop(0)

        ui_conf = similarity_to_confidence(CURRENT_CONFIDENCE)

        snippet_obj = get_live_snippet_for_session(session_id, sr)
        live_text, transcript_segments = _assign_words_to_speakers(
            session_id,
            snippet_obj,
            CURRENT_SPEAKER or "Unknown",
        )

        #live_text = squash_adjacent_short_repeats(live_text)
        #for seg in transcript_segments:
        #    seg["text"] = squash_adjacent_short_repeats(seg.get("text") or "")

        resp: Dict[str, Any] = {
            "speaker": CURRENT_SPEAKER or "Unknown",
            "has_speech": bool(live_text),
            "confidence": ui_conf,
            "transcript": live_text,
        }
        if transcript_segments:
            resp["transcript_segments"] = transcript_segments

        return jsonify(resp)

    except Exception as e:
        logger.error(f"Error in identify_speaker: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# /correct_speaker (disabled in live mode)
# -------------------------------------------------
@app.route("/correct_speaker", methods=["POST"])
def correct_speaker():
    return jsonify({
        "error": "Speaker corrections are disabled during recording. Please enroll speakers before starting."
    }), 400

# -------------------------------------------------
# Startup
# -------------------------------------------------
if PIPELINE_AVAILABLE:
    load_embeddings()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
