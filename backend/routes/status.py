"""
Status endpoint for the WhoSays backend.
"""
from flask import Blueprint, jsonify

import backend.config as cfg

status_bp = Blueprint('status', __name__)


@status_bp.route("/status", methods=["GET"])
def status():
    asr_backend = None
    try:
        if cfg.PIPELINE_AVAILABLE and cfg.pipeline is not None:
            asr_type = getattr(getattr(cfg.pipeline, "config", None), "asr", None)
            asr_type = getattr(asr_type, "asr_type", None)
            asr_backend = getattr(asr_type, "value", None) or (str(asr_type) if asr_type is not None else None)
    except Exception:
        asr_backend = None
    return jsonify({
        "status": "WhoSays server is running.",
        "pipeline_loaded": cfg.PIPELINE_AVAILABLE,
        "known_speakers": list(cfg.KNOWN_SPEAKERS.keys()),
        "asr_backend": asr_backend,
    })
