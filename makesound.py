#!/usr/bin/env python3
"""
Chunked sender for /identify_speaker.

Reads an audio file, converts to mono 16 kHz float32, and sends ~1 second
chunks to the running server via /identify_speaker with a fixed session_id.

Dependencies: soundfile, numpy, scipy, requests
    pip install soundfile numpy scipy requests

Usage:
    python makesound.py --file thetestsound.wav \
        --url http://localhost:8000/identify_speaker \
        --session debug_session_chunked \
        --chunk-sec 1.0
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly


def load_and_prepare(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    print(f"Loaded: sr={sr}, dtype={audio.dtype}, shape={audio.shape}")

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr)
        sr = target_sr
        print(f"Resampled to {sr} Hz")

    # To float32 in [-1, 1]
    if np.issubdtype(audio.dtype, np.integer):
        maxv = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / maxv
    else:
        audio = audio.astype(np.float32)

    print(f"Prepared: sr={sr}, dtype={audio.dtype}, min={audio.min():.4f}, max={audio.max():.4f}")
    return audio, sr


def post_chunk(url: str, chunk: np.ndarray, sr: int, session_id: str, timeout_sec: float):
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
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)


def main():
    p = argparse.ArgumentParser(description="Chunked sender for /identify_speaker")
    p.add_argument("--file", required=True, type=Path, help="Path to WAV/MP3/OGG file")
    p.add_argument("--url", default="http://localhost:8000/identify_speaker", help="Endpoint URL")
    p.add_argument("--session", default="debug_session_chunked", help="Session ID")
    p.add_argument("--chunk-sec", type=float, default=1.0, help="Chunk length in seconds (at 16 kHz)")
    p.add_argument("--target-sr", type=int, default=16000, help="Target sample rate")
    p.add_argument("--timeout-sec", type=float, default=120.0, help="HTTP timeout per chunk")
    p.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep between chunks (seconds)")
    args = p.parse_args()

    audio, sr = load_and_prepare(args.file, target_sr=args.target_sr)
    chunk_size = int(args.chunk_sec * sr)
    if chunk_size <= 0:
        print("chunk size must be > 0", file=sys.stderr)
        sys.exit(1)

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) == 0:
            continue
        post_chunk(args.url, chunk, sr, args.session, args.timeout_sec)
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()


