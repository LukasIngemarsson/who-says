"""
Tuning presets loading and management for the WhoSays backend.
"""
import json
from typing import Any, Dict

from loguru import logger

import backend.config as cfg


def load_tuning_presets(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Load tuning presets from JSON file.
    Automatically reloads if file has been modified.
    """
    if not cfg.TUNING_PRESETS_FILE.exists():
        logger.warning(f"Tuning presets file not found: {cfg.TUNING_PRESETS_FILE}")
        return cfg.TUNING_PRESETS

    try:
        file_mtime = cfg.TUNING_PRESETS_FILE.stat().st_mtime
        if not force_reload and cfg.TUNING_PRESETS and file_mtime <= cfg.TUNING_PRESETS_LAST_MODIFIED:
            return cfg.TUNING_PRESETS

        with open(cfg.TUNING_PRESETS_FILE, "r") as f:
            data = json.load(f)

        # Filter out metadata keys (those starting with _)
        presets = {
            k: {pk: pv for pk, pv in v.items() if not pk.startswith("_")}
            for k, v in data.items()
            if not k.startswith("_") and isinstance(v, dict)
        }

        cfg.TUNING_PRESETS = presets
        cfg.TUNING_PRESETS_LAST_MODIFIED = file_mtime
        logger.info(f"Loaded {len(presets)} tuning presets from {cfg.TUNING_PRESETS_FILE}")
        return cfg.TUNING_PRESETS

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in tuning presets file: {e}")
        return cfg.TUNING_PRESETS
    except Exception as e:
        logger.error(f"Error loading tuning presets: {e}")
        return cfg.TUNING_PRESETS


def get_current_tuning_snapshot() -> Dict[str, Any]:
    """Get a snapshot of current tuning configuration."""
    asr_cfg: Dict[str, Any] = {}
    vad_cfg: Dict[str, Any] = {}

    if cfg.PIPELINE_AVAILABLE and cfg.pipeline is not None:
        asr = getattr(cfg.pipeline, "asr", None)
        vad = getattr(cfg.pipeline, "vad", None)

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
            "asr_min_new_sec": cfg.ASR_MIN_NEW_SEC,
            "min_asr_interval_sec": cfg.MIN_ASR_INTERVAL_SEC,
            "max_asr_buffer_sec": cfg.MAX_ASR_BUFFER_SEC,
            "wcpp_context_sec": cfg.WCPP_CONTEXT_SEC,
            "min_speech_sec": cfg.MIN_SPEECH_SEC,
            "cross_tail_dup_pad_sec": cfg.CROSS_TAIL_DUP_PAD_SEC,
            "use_initial_prompt": cfg.USE_INITIAL_PROMPT,
            "asr_prompt_max_chars": cfg.ASR_PROMPT_MAX_CHARS,
        },
        "asr": asr_cfg,
        "vad": vad_cfg,
        "overlap": {
            "enabled": cfg.OVERLAP_DETECTION_ENABLED,
            "buffer_sec": cfg.OVERLAP_BUFFER_SEC,
            "detection_interval_sec": cfg.OVERLAP_DETECTION_INTERVAL_SEC,
            "min_duration_sec": cfg.OVERLAP_MIN_DURATION_SEC,
        },
    }
