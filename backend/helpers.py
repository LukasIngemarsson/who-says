"""
General helper/utility functions for the WhoSays backend.
"""
import os
import re
import subprocess
import tempfile
from typing import Any, List, Optional

import numpy as np
from loguru import logger

from backend.config import SIM_CONF_FLOOR, SIM_CONF_CEIL


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


def is_prompt_worthy(snippet: str) -> bool:
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


def make_serializable(obj: Any) -> Any:
    """Convert numpy arrays and nested structures to JSON-serializable types."""
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
