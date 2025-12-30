"""
Overlap detection configuration and status routes for the WhoSays backend.
"""
from flask import Blueprint, jsonify, request

import backend.config as cfg
from backend.overlap import get_recent_overlaps

overlap_bp = Blueprint('overlap', __name__)


@overlap_bp.route("/overlap_status", methods=["GET"])
def overlap_status():
    """Get current overlap detection status and recent overlaps."""
    session_id = request.args.get("session_id", "default")
    since = float(request.args.get("since", 0.0))

    recent_overlaps = get_recent_overlaps(session_id, since)

    return jsonify({
        "enabled": cfg.OVERLAP_DETECTION_ENABLED,
        "buffer_sec": cfg.OVERLAP_BUFFER_SEC,
        "detection_interval_sec": cfg.OVERLAP_DETECTION_INTERVAL_SEC,
        "min_duration_sec": cfg.OVERLAP_MIN_DURATION_SEC,
        "recent_overlaps": recent_overlaps,
        "total_overlaps": len(cfg.SESSION_OVERLAP_TIMELINE.get(session_id, [])),
    })


@overlap_bp.route("/overlap_config", methods=["GET", "POST"])
def overlap_config():
    """Get or set overlap detection configuration."""
    if request.method == "GET":
        return jsonify({
            "enabled": cfg.OVERLAP_DETECTION_ENABLED,
            "buffer_sec": cfg.OVERLAP_BUFFER_SEC,
            "detection_interval_sec": cfg.OVERLAP_DETECTION_INTERVAL_SEC,
            "min_duration_sec": cfg.OVERLAP_MIN_DURATION_SEC,
        })

    data = request.get_json(silent=True) or {}

    if "enabled" in data:
        cfg.OVERLAP_DETECTION_ENABLED = bool(data["enabled"])

    if "buffer_sec" in data:
        cfg.OVERLAP_BUFFER_SEC = max(1.0, min(10.0, float(data["buffer_sec"])))

    if "detection_interval_sec" in data:
        cfg.OVERLAP_DETECTION_INTERVAL_SEC = max(0.5, min(10.0, float(data["detection_interval_sec"])))

    if "min_duration_sec" in data:
        cfg.OVERLAP_MIN_DURATION_SEC = max(0.0, min(1.0, float(data["min_duration_sec"])))
        # Reset detector to pick up new min_duration
        cfg.SOD_DETECTOR = None

    return jsonify({
        "message": "Configuration updated",
        "enabled": cfg.OVERLAP_DETECTION_ENABLED,
        "buffer_sec": cfg.OVERLAP_BUFFER_SEC,
        "detection_interval_sec": cfg.OVERLAP_DETECTION_INTERVAL_SEC,
        "min_duration_sec": cfg.OVERLAP_MIN_DURATION_SEC,
    })
