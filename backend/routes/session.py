"""
Session management routes for the WhoSays backend.
"""
from flask import Blueprint, jsonify, request
from loguru import logger

import backend.config as cfg

session_bp = Blueprint('session', __name__)


@session_bp.route("/reset_session", methods=["POST"])
def reset_session():
    """Reset all state for a session."""
    session_id = request.form.get("session_id", "default")

    # Clear per-session state
    cfg.SESSION_ASR_STATE.pop(session_id, None)
    cfg.SESSION_SPEAKER_TIMELINE.pop(session_id, None)
    cfg.SESSION_OVERLAP_STATE.pop(session_id, None)
    cfg.SESSION_OVERLAP_TIMELINE.pop(session_id, None)
    cfg.SESSION_IDENTIFIED_SPEAKERS.pop(session_id, None)
    cfg.SESSION_SEPARATED_AUDIO.pop(session_id, None)

    # Optionally reset global state if requested
    if request.form.get("reset_global") == "1":
        cfg.ROLLING_BUFFER = None
        cfg.SPEECH_BUFFER = None
        cfg.CURRENT_SPEAKER = None
        cfg.CURRENT_CONFIDENCE = 0.0
        cfg.LAST_SWITCH_TIME = 0.0
        cfg.PREV_NO_SPEECH_DETECTED = False
        logger.info(f"Reset global buffers for session {session_id}")

    return jsonify({"message": f"Session {session_id} reset"})
