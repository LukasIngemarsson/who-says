"""
Global configuration, constants, and state for the WhoSays backend.
"""
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# -------------------------------------------------
# Pipeline availability
# -------------------------------------------------
PIPELINE_AVAILABLE = False
pipeline = None

def init_pipeline():
    """Initialize the WhoSays pipeline. Call once at startup."""
    global PIPELINE_AVAILABLE, pipeline
    try:
        from main import WhoSays
        PIPELINE_AVAILABLE = True
        from loguru import logger
        logger.info("Loading WhoSays pipeline... This may take a moment.")
        pipeline = WhoSays()
        logger.info("Pipeline loaded successfully. Server is ready.")
    except ImportError:
        from loguru import logger
        logger.warning("Could not import WhoSays from main.py. Running in mock mode.")
        PIPELINE_AVAILABLE = False
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

# Track all unique speakers identified in each session (for overlap display)
SESSION_IDENTIFIED_SPEAKERS: Dict[str, set] = {}

# Global rolling / speech buffers for diarization (per process, not per session)
ROLLING_BUFFER: Optional[torch.Tensor] = None  # ~1s of audio
SPEECH_BUFFER: Optional[torch.Tensor] = None   # up to ~2.5s for embedding

# Known speakers
EMBEDDINGS_DIR = Path(__file__).parent.parent / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)
KNOWN_SPEAKERS: Dict[str, torch.Tensor] = {}  # name -> 1D normed embedding (CPU)

# -------------------------------------------------
# Speaker Overlap Detection (SOD) State
# -------------------------------------------------

# Per-session overlap detection buffer:
#   session_id -> {
#       "buffer": Tensor on pipeline device (1D),
#       "buffer_start_time": float,  # Absolute time when buffer started
#       "last_detection_time": float,
#   }
SESSION_OVERLAP_STATE: Dict[str, Dict[str, Any]] = {}

# Per-session overlap timeline (detected overlaps):
#   session_id -> [
#       {
#           "start": float,      # Absolute wall-clock time
#           "end": float,
#           "speakers": List[str],  # Speakers involved (from diarization)
#           "duration": float,
#           "confidence": float,    # Detection confidence if available
#       },
#       ...
#   ]
SESSION_OVERLAP_TIMELINE: Dict[str, List[Dict[str, Any]]] = {}

# Global SOD detector instance (lazy-loaded)
SOD_DETECTOR: Optional[Any] = None
SOD_DETECTOR_LOCK = threading.Lock()

# Speaker Overlap Separation (SOS) for separating overlapping speech
SOS_SEPARATOR: Optional[Any] = None
SOS_SEPARATOR_LOCK = threading.Lock()

# Per-session separated audio storage:
#   session_id -> {
#       (start, end): {speaker_idx: waveform_tensor, ...},
#       ...
#   }
SESSION_SEPARATED_AUDIO: Dict[str, Dict[tuple, Dict[int, torch.Tensor]]] = {}

# Overlap detection configuration
OVERLAP_BUFFER_SEC: float = 0.6           # How much audio to buffer before detection
OVERLAP_DETECTION_INTERVAL_SEC: float = 1.0  # Minimum interval between detections
OVERLAP_MIN_DURATION_SEC: float = 0.1     # Minimum overlap duration to report
OVERLAP_DETECTION_ENABLED: bool = True    # Global toggle

# Streaming config (whisper.cpp-first)
#
# NOTE: Frontend chunk size can vary depending on backend. For all gating logic
# we use *seconds* rather than "frames" to avoid mismatches.
ASR_MIN_NEW_SEC = 0.40          # minimum new audio (seconds) before ASR
MIN_ASR_INTERVAL_SEC = 0.50     # throttle ASR calls (seconds) - should be >= ASR_MIN_NEW_SEC
MAX_ASR_BUFFER_SEC = 12.0       # cap per-session ASR buffer (seconds)
WCPP_CONTEXT_SEC = 2.0          # decode this much audio before cursor for stability

# Minimal de-dup: guard against jittery re-emits across calls
CROSS_TAIL_DUP_PAD_SEC = 0.03

# Rolling prompt (used as whisper.cpp --prompt for streaming stability)
ASR_PROMPT_MAX_CHARS = 240
USE_INITIAL_PROMPT = True

# Diarization: how much speech to accumulate for embedding
# Lower = finer speaker timeline granularity (better word attribution)
# Higher = more stable speaker detection (more audio context)
MIN_SPEECH_SEC = 0.5  # require 0.5s of speech before identifying speaker
SPEAKER_WARMUP_SEC = 1.0  # require more audio at start before first identification (reduced for faster feedback)

# Diarization state
CURRENT_SPEAKER: Optional[str] = None
CURRENT_CONFIDENCE: float = 0.0
LAST_SWITCH_TIME: float = 0.0
SWITCH_COOLDOWN: float = 1.0   # seconds - prevent rapid speaker flickering

# No-speech verification: require two consecutive frames with no speech before confirming
PREV_NO_SPEECH_DETECTED: bool = False

SIM_CONF_FLOOR = 0.05
SIM_CONF_CEIL = 0.35

# -------------------------------------------------
# Tuning presets (loaded from JSON file)
# -------------------------------------------------
TUNING_PRESETS_FILE = Path(__file__).parent.parent / "tuning_presets.json"
TUNING_PRESETS: Dict[str, Dict[str, Any]] = {}
TUNING_PRESETS_LAST_MODIFIED: float = 0.0
