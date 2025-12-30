"""
Incremental ASR (Automatic Speech Recognition) logic for the WhoSays backend.
"""
import time
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

import backend.config as cfg
from backend.helpers import is_prompt_worthy


def get_live_snippet_for_session(session_id: str, sr: int) -> Optional[Dict[str, Any]]:
    """
    Incremental ASR: decode only the new audio since the last cursor.
    No multi-second tail is re-decoded, which avoids recomposition loops
    and timestamp jitter on past audio.
    """
    if not cfg.PIPELINE_AVAILABLE or cfg.pipeline is None:
        return None

    state = cfg.SESSION_ASR_STATE.get(session_id)
    if state is None:
        return None

    buffer: torch.Tensor = state["buffer"]
    cursor: int = int(state["cursor"])
    t0: float = float(state.get("t0", 0.0))

    if buffer.shape[0] == 0 or cursor >= buffer.shape[0]:
        return None

    # New audio region (incremental)
    new_audio = buffer[cursor:]
    min_new_samples = int(max(1.0, float(cfg.ASR_MIN_NEW_SEC)) * float(sr))
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
    if now - last_asr_time < float(cfg.MIN_ASR_INTERVAL_SEC):
        return None

    use_prompt = bool(cfg.USE_INITIAL_PROMPT)
    initial_prompt = (state.get("prompt") or "").strip() if use_prompt else ""

    # Decode a little context *before* the cursor to stabilize boundaries.
    # We still only EMIT words that fall after the cursor.
    ctx_samples = int(max(0.0, float(cfg.WCPP_CONTEXT_SEC)) * float(sr))
    slice_start = max(0, cursor - ctx_samples)
    slice_audio = buffer[slice_start:]

    slice_start_sec = slice_start / float(sr)
    new_region_start_abs = t0 + (cursor / float(sr))
    new_region_end_abs = t0 + (buffer.shape[0] / float(sr))

    # Use the same VAD knobs for diarization AND whisper.cpp internal VAD.
    # This keeps the UI tuning panel meaningful in whisper.cpp mode.
    vad_threshold = float(getattr(getattr(cfg.pipeline, "vad", None), "threshold", 0.50))
    vad_min_speech_ms = int(getattr(getattr(cfg.pipeline, "vad", None), "min_speech_duration_ms", 200))
    vad_min_silence_ms = int(getattr(getattr(cfg.pipeline, "vad", None), "min_silence_duration_ms", 150))
    vad_speech_pad_ms = int(getattr(getattr(cfg.pipeline, "vad", None), "speech_pad_ms", 30))

    asr_cfg = getattr(getattr(cfg.pipeline, "config", None), "asr", None)
    beam_size = int(getattr(asr_cfg, "beam_size", 3)) if asr_cfg is not None else 3
    best_of = int(getattr(asr_cfg, "best_of", 3)) if asr_cfg is not None else 3

    try:
        with torch.inference_mode():
            asr_result = cfg.pipeline.asr.transcribe(
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
            if w_start_abs <= last_word_end_abs + cfg.CROSS_TAIL_DUP_PAD_SEC:
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
    if use_prompt and is_prompt_worthy(snippet):
        prompt = ((state.get("prompt") or "").strip() + " " + snippet).strip()
        if len(prompt) > cfg.ASR_PROMPT_MAX_CHARS:
            prompt = prompt[-cfg.ASR_PROMPT_MAX_CHARS:].lstrip()
        state["prompt"] = prompt

    return {
        "text": snippet,
        "words": words,
        "duration": float(slice_audio.shape[0] / float(sr)),
    }
