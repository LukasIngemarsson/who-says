#!/usr/bin/env python3
"""batch_stream_eval.py

Run the live streaming endpoint (/identify_speaker) on many audio files and
compute repetition/junk/fragmentation metrics.

This is meant to detect overfitting to a single "testsound" clip.

Usage:
  python batch_stream_eval.py --base-url http://localhost:8000 \
    --files thetestsound.wav samples/examples/single_speaker_sample.wav \
    --chunk-sec 1.0 --sleep-sec 0.0

Notes:
- Converts input audio to 16kHz mono WAV with ffmpeg (so MP3/M4A/OGG work).
- Uses a fresh session_id per file.
"""

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import soundfile as sf


def _norm_tok(tok: str) -> str:
    tok = (tok or "").strip().lower()
    tok = re.sub(r"^\W+|\W+$", "", tok)
    return tok


def convert_to_wav_16k_mono(in_path: Path) -> Path:
    """Convert any audio file to a temp 16kHz mono wav."""
    in_path = in_path.resolve()
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    fd, out_name = tempfile.mkstemp(prefix="eval_", suffix=".wav")
    os.close(fd)
    out_path = Path(out_name)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(in_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def load_wav_float32(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    # keep in [-1,1] if possible
    mx = float(np.max(np.abs(audio))) if audio.size else 0.0
    if mx > 1.5:  # very likely int-like scaling
        audio = audio / mx
    return audio, int(sr)


def post_chunk(url: str, chunk: np.ndarray, sr: int, session_id: str, timeout_sec: float) -> Dict[str, Any]:
    b64 = base64.b64encode(chunk.tobytes()).decode("ascii")
    r = requests.post(
        url,
        data={
            "audio_data": b64,
            "sample_rate": str(sr),
            "session_id": session_id,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout_sec,
    )
    r.raise_for_status()
    return r.json()


def tokens_from_text(text: str) -> List[str]:
    toks = [_norm_tok(t) for t in (text or "").split()]
    return [t for t in toks if t]


def adj_dup_rate(toks: List[str]) -> float:
    if len(toks) < 2:
        return 0.0
    dup = 0
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            dup += 1
    return dup / max(1, (len(toks) - 1))


def trigram_repeat_rate(toks: List[str]) -> float:
    if len(toks) < 6:
        return 0.0
    trigs = [tuple(toks[i : i + 3]) for i in range(0, len(toks) - 2)]
    seen = set()
    rep = 0
    for t in trigs:
        if t in seen:
            rep += 1
        else:
            seen.add(t)
    return rep / max(1, len(trigs))


@dataclass
class FileResult:
    file: str
    session_id: str
    duration_sec: float
    responses: int
    has_speech_rate: float
    text: str
    total_words: int
    adj_dup_rate: float
    trigram_repeat_rate: float
    ellipsis_token_rate: float
    short_snippet_rate: float


def run_one_file(
    base_url: str,
    file_path: Path,
    chunk_sec: float,
    sleep_sec: float,
    timeout_sec: float,
) -> Tuple[FileResult, List[Dict[str, Any]]]:
    identify_url = base_url.rstrip("/") + "/identify_speaker"

    tmp_wav = None
    try:
        tmp_wav = convert_to_wav_16k_mono(file_path)
        audio, sr = load_wav_float32(tmp_wav)

        chunk_size = int(round(chunk_sec * sr))
        if chunk_size <= 0:
            raise ValueError("chunk_sec must be > 0")

        session_id = f"eval_{file_path.stem}_{int(time.time()*1000)}"
        responses: List[Dict[str, Any]] = []
        snippets: List[str] = []
        short_snips = 0
        speech_count = 0

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if chunk.size == 0:
                continue
            data = post_chunk(identify_url, chunk, sr, session_id, timeout_sec)
            responses.append({
                "chunk_index": int(i // chunk_size),
                "chunk_samples": int(chunk.size),
                **data,
            })

            if data.get("has_speech"):
                speech_count += 1

            text = (data.get("transcript") or "").strip()
            if text:
                snippets.append(text)
                if len(tokens_from_text(text)) <= 2:
                    short_snips += 1

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        full_text = " ".join(snippets).strip()
        toks = tokens_from_text(full_text)
        ellipsis = 0
        for t in (full_text or "").split():
            if "..." in t:
                ellipsis += 1

        res = FileResult(
            file=str(file_path),
            session_id=session_id,
            duration_sec=float(len(audio) / float(sr)) if sr > 0 else 0.0,
            responses=len(responses),
            has_speech_rate=(speech_count / max(1, len(responses))),
            text=full_text,
            total_words=len(toks),
            adj_dup_rate=adj_dup_rate(toks),
            trigram_repeat_rate=trigram_repeat_rate(toks),
            ellipsis_token_rate=(ellipsis / max(1, len((full_text or "").split()))),
            short_snippet_rate=(short_snips / max(1, len(snippets))),
        )
        return res, responses
    finally:
        if tmp_wav is not None:
            try:
                tmp_wav.unlink(missing_ok=True)
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--chunk-sec", type=float, default=1.0)
    ap.add_argument("--sleep-sec", type=float, default=0.0)
    ap.add_argument("--timeout-sec", type=float, default=120.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--include-raw", action="store_true")
    args = ap.parse_args()

    tuning = None
    try:
        tuning = requests.get(args.base_url.rstrip("/") + "/tuning", timeout=30).json()
    except Exception:
        tuning = None

    results: List[Dict[str, Any]] = []
    for f in args.files:
        p = Path(f)
        fr, raw = run_one_file(
            args.base_url,
            p,
            chunk_sec=args.chunk_sec,
            sleep_sec=args.sleep_sec,
            timeout_sec=args.timeout_sec,
        )
        row: Dict[str, Any] = {
            **fr.__dict__,
        }
        if args.include_raw:
            row["raw"] = raw
        results.append(row)

    # Aggregate
    def avg(key: str) -> float:
        vals = [float(r.get(key) or 0.0) for r in results]
        return float(sum(vals) / max(1, len(vals)))

    out_obj = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "base_url": args.base_url,
        "chunk_sec": args.chunk_sec,
        "sleep_sec": args.sleep_sec,
        "tuning": tuning,
        "files": results,
        "summary": {
            "n": len(results),
            "avg_has_speech_rate": avg("has_speech_rate"),
            "avg_adj_dup_rate": avg("adj_dup_rate"),
            "avg_trigram_repeat_rate": avg("trigram_repeat_rate"),
            "avg_ellipsis_token_rate": avg("ellipsis_token_rate"),
            "avg_short_snippet_rate": avg("short_snippet_rate"),
        },
    }

    out_path: Path
    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path("results") / "streaming_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"eval_{out_obj['timestamp']}.json"

    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(json.dumps(out_obj["summary"], indent=2))


if __name__ == "__main__":
    main()

