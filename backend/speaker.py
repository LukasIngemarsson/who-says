"""
Speaker timeline, embedding management, and word attribution for the WhoSays backend.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger

import backend.config as cfg


def save_embedding(name: str, tensor: torch.Tensor) -> None:
    """Save a speaker embedding to disk."""
    try:
        file_path = cfg.EMBEDDINGS_DIR / f"{name}.pt"
        torch.save(tensor.cpu(), file_path)
        logger.info(f"Saved embedding for {name} to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save embedding for {name}: {e}")


def load_embeddings() -> None:
    """Load all speaker embeddings from disk."""
    cfg.KNOWN_SPEAKERS.clear()
    try:
        for file_path in cfg.EMBEDDINGS_DIR.glob("*.pt"):
            name = file_path.stem
            tensor = torch.load(file_path, map_location="cpu").squeeze()
            if tensor.dim() != 1:
                tensor = tensor.flatten()
            tensor = F.normalize(tensor, p=2, dim=0)
            cfg.KNOWN_SPEAKERS[name] = tensor
        logger.info(f"Loaded {len(cfg.KNOWN_SPEAKERS)} embeddings from {cfg.EMBEDDINGS_DIR}")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")


def speaker_at_time(session_id: str, t: float, fallback: str) -> str:
    """Get the speaker active at a given wall-clock time."""
    timeline = cfg.SESSION_SPEAKER_TIMELINE.get(session_id, [])
    for seg in reversed(timeline):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", t))
        if start <= t <= end:
            return seg.get("speaker") or fallback
    return fallback


def get_overlap_speakers_at_time(session_id: str, t: float) -> Optional[str]:
    """
    Check if timestamp falls within any overlap segment.
    Returns combined speaker string (e.g., "Johan, Lukas") if in overlap, else None.

    Note: t is in audio-relative time (seconds from start of audio).
    Overlap segments are stored in wall-clock time.
    We convert using session_start_time.
    """
    timeline = cfg.SESSION_OVERLAP_TIMELINE.get(session_id, [])
    if not timeline:
        return None

    # Get session start time to convert audio time to wall-clock time
    asr_state = cfg.SESSION_ASR_STATE.get(session_id)
    if asr_state is None:
        return None

    session_start = asr_state.get("session_start_time", 0.0)
    wall_clock_t = session_start + t

    # Add padding to overlap boundaries to catch words at edges
    # SOD detection can be slightly delayed, so extend start earlier
    START_PADDING = 0.25  # seconds to extend overlap start earlier
    END_PADDING = 0.1     # seconds to extend overlap end later

    for seg in timeline:
        start = float(seg.get("start", 0.0)) - START_PADDING
        end = float(seg.get("end", start)) + END_PADDING
        speakers = seg.get("speakers", [])
        logger.debug(f"[overlap-check] audio_t={t:.2f}s -> wall_clock_t={wall_clock_t:.2f}, overlap_seg=[{start:.2f}, {end:.2f}], speakers={speakers}")
        if start <= wall_clock_t <= end:
            if len(speakers) >= 2:
                logger.info(f"[overlap-match] Word at t={t:.2f}s matches overlap [{start:.2f}, {end:.2f}] -> {speakers}")
                return ", ".join(speakers)
    return None


def assign_words_to_speakers(
    session_id: str,
    snippet_obj: Any,
    fallback_speaker: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Assign words from an ASR snippet to speakers based on timeline.
    Returns (text, transcript_segments).
    """
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

        # Check if word falls within overlap first
        overlap_speakers = get_overlap_speakers_at_time(session_id, center_abs)
        if overlap_speakers:
            speaker = overlap_speakers
            logger.debug(f"[overlap] Word '{tok}' at audio_time={center_abs:.2f}s assigned to overlap speakers: {overlap_speakers}")
        else:
            speaker = speaker_at_time(session_id, center_abs, fallback_speaker)

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


def get_speakers_in_range(
    session_id: str,
    start_time: float,
    end_time: float,
    lookback_sec: float = 2.0,
) -> List[str]:
    """
    Get list of speakers who were active in the given time range.
    Uses SESSION_SPEAKER_TIMELINE.

    Since the timeline only tracks one speaker at a time, we also look back
    a few seconds to find recent speakers who might be involved in an overlap.
    """
    timeline = cfg.SESSION_SPEAKER_TIMELINE.get(session_id, [])
    speakers = set()

    # Extend the search window to catch recent speakers
    extended_start = start_time - lookback_sec

    for seg in timeline:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))

        # Check for overlap with extended window
        if seg_start < end_time and seg_end > extended_start:
            speaker = seg.get("speaker")
            if speaker:
                speakers.add(speaker)

    return list(speakers)
