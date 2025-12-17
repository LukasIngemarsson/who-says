"""
Tuning configuration routes for the WhoSays backend.
"""
from flask import Blueprint, jsonify, request

import backend.config as cfg
from backend.tuning import load_tuning_presets, get_current_tuning_snapshot

tuning_bp = Blueprint('tuning', __name__)


@tuning_bp.route("/tuning", methods=["GET", "POST"])
def tuning():
    # Hot-reload presets if file changed
    load_tuning_presets()

    if request.method == "GET":
        snapshot = get_current_tuning_snapshot()
        snapshot["available_presets"] = list(cfg.TUNING_PRESETS.keys())
        return jsonify(snapshot)

    data = request.get_json(silent=True) or {}

    # preset handling
    if "preset" in data:
        preset_name = data["preset"]
        if preset_name not in cfg.TUNING_PRESETS:
            return jsonify({
                "error": f"Unknown preset '{preset_name}'",
                "available_presets": list(cfg.TUNING_PRESETS.keys())
            }), 400
        preset = cfg.TUNING_PRESETS[preset_name]

        if "asr_min_new_sec" in preset:
            cfg.ASR_MIN_NEW_SEC = max(0.05, float(preset["asr_min_new_sec"]))
        if "min_asr_interval_sec" in preset:
            cfg.MIN_ASR_INTERVAL_SEC = max(0.0, float(preset["min_asr_interval_sec"]))
        if "max_asr_buffer_sec" in preset:
            cfg.MAX_ASR_BUFFER_SEC = max(1.0, float(preset["max_asr_buffer_sec"]))
        if "wcpp_context_sec" in preset:
            cfg.WCPP_CONTEXT_SEC = max(0.0, float(preset["wcpp_context_sec"]))
        if "min_speech_sec" in preset:
            cfg.MIN_SPEECH_SEC = max(0.05, float(preset["min_speech_sec"]))
        if "cross_tail_dup_pad_sec" in preset:
            cfg.CROSS_TAIL_DUP_PAD_SEC = max(0.0, float(preset["cross_tail_dup_pad_sec"]))
        if "use_initial_prompt" in preset:
            cfg.USE_INITIAL_PROMPT = bool(preset["use_initial_prompt"])
        if "asr_prompt_max_chars" in preset:
            cfg.ASR_PROMPT_MAX_CHARS = max(0, int(preset["asr_prompt_max_chars"]))

        if cfg.PIPELINE_AVAILABLE and cfg.pipeline is not None:
            asr = getattr(cfg.pipeline, "asr", None)
            vad = getattr(cfg.pipeline, "vad", None)

            if asr:
                if "beam_size" in preset:
                    asr.beam_size = int(preset["beam_size"])
                if "best_of" in preset:
                    asr.best_of = int(preset["best_of"])

            if vad:
                if "vad_threshold" in preset:
                    vad.threshold = float(preset["vad_threshold"])
                if "vad_min_speech_ms" in preset:
                    vad.min_speech_duration_ms = int(preset["vad_min_speech_ms"])
                if "vad_min_silence_ms" in preset:
                    vad.min_silence_duration_ms = int(preset["vad_min_silence_ms"])
                if "vad_speech_pad_ms" in preset:
                    vad.speech_pad_ms = int(preset["vad_speech_pad_ms"])

        # Overlap detection settings from preset
        if "overlap_detection_enabled" in preset:
            cfg.OVERLAP_DETECTION_ENABLED = bool(preset["overlap_detection_enabled"])
        if "overlap_buffer_sec" in preset:
            cfg.OVERLAP_BUFFER_SEC = max(1.0, min(10.0, float(preset["overlap_buffer_sec"])))
        if "overlap_detection_interval_sec" in preset:
            cfg.OVERLAP_DETECTION_INTERVAL_SEC = max(0.5, min(10.0, float(preset["overlap_detection_interval_sec"])))
        if "overlap_min_duration_sec" in preset:
            cfg.OVERLAP_MIN_DURATION_SEC = max(0.0, min(1.0, float(preset["overlap_min_duration_sec"])))
            cfg.SOD_DETECTOR = None  # Reset to pick up new min_duration

        return jsonify({"message": f"Applied preset '{preset_name}'", "settings": preset})

    # Manual tuning
    try:
        if "asr_min_new_sec" in data:
            cfg.ASR_MIN_NEW_SEC = max(0.05, float(data["asr_min_new_sec"]))
        if "min_asr_interval_sec" in data:
            cfg.MIN_ASR_INTERVAL_SEC = max(0.0, float(data["min_asr_interval_sec"]))
        if "max_asr_buffer_sec" in data:
            cfg.MAX_ASR_BUFFER_SEC = max(1.0, float(data["max_asr_buffer_sec"]))
        if "wcpp_context_sec" in data:
            cfg.WCPP_CONTEXT_SEC = max(0.0, float(data["wcpp_context_sec"]))
        if "use_initial_prompt" in data:
            cfg.USE_INITIAL_PROMPT = bool(data["use_initial_prompt"])
        if "asr_prompt_max_chars" in data:
            cfg.ASR_PROMPT_MAX_CHARS = max(0, int(data["asr_prompt_max_chars"]))
        if "min_speech_sec" in data:
            cfg.MIN_SPEECH_SEC = max(0.05, float(data["min_speech_sec"]))
        if "cross_tail_dup_pad_sec" in data:
            cfg.CROSS_TAIL_DUP_PAD_SEC = max(0.0, float(data["cross_tail_dup_pad_sec"]))
        # Overlap detection manual tuning
        if "overlap_detection_enabled" in data:
            cfg.OVERLAP_DETECTION_ENABLED = bool(data["overlap_detection_enabled"])
        if "overlap_buffer_sec" in data:
            cfg.OVERLAP_BUFFER_SEC = max(1.0, min(10.0, float(data["overlap_buffer_sec"])))
        if "overlap_detection_interval_sec" in data:
            cfg.OVERLAP_DETECTION_INTERVAL_SEC = max(0.5, min(10.0, float(data["overlap_detection_interval_sec"])))
        if "overlap_min_duration_sec" in data:
            cfg.OVERLAP_MIN_DURATION_SEC = max(0.0, min(1.0, float(data["overlap_min_duration_sec"])))
            cfg.SOD_DETECTOR = None  # Reset to pick up new min_duration
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid streaming tuning values"}), 400

    if cfg.PIPELINE_AVAILABLE and cfg.pipeline is not None:
        asr = getattr(cfg.pipeline, "asr", None)
        vad = getattr(cfg.pipeline, "vad", None)

        if asr:
            try:
                if "beam_size" in data:
                    asr.beam_size = int(data["beam_size"])
                if "best_of" in data:
                    asr.best_of = int(data["best_of"])
            except (TypeError, ValueError):
                return jsonify({"error": "Invalid ASR tuning values"}), 400

        if vad:
            try:
                if "vad_threshold" in data:
                    vad.threshold = float(data["vad_threshold"])
                if "vad_min_speech_ms" in data:
                    vad.min_speech_duration_ms = int(data["vad_min_speech_ms"])
                if "vad_min_silence_ms" in data:
                    vad.min_silence_duration_ms = int(data["vad_min_silence_ms"])
                if "vad_speech_pad_ms" in data:
                    vad.speech_pad_ms = int(data["vad_speech_pad_ms"])
            except (TypeError, ValueError):
                return jsonify({"error": "Invalid VAD tuning values"}), 400

    return jsonify(get_current_tuning_snapshot())


@tuning_bp.route("/tuning/presets", methods=["GET"])
def tuning_presets_list():
    """List all available tuning presets."""
    load_tuning_presets()  # Hot-reload
    return jsonify({
        "presets": list(cfg.TUNING_PRESETS.keys()),
        "file": str(cfg.TUNING_PRESETS_FILE),
        "details": {
            name: {k: v for k, v in preset.items()}
            for name, preset in cfg.TUNING_PRESETS.items()
        }
    })


@tuning_bp.route("/tuning/reload", methods=["POST"])
def tuning_presets_reload():
    """Force reload tuning presets from JSON file."""
    load_tuning_presets(force_reload=True)
    return jsonify({
        "message": "Tuning presets reloaded",
        "presets": list(cfg.TUNING_PRESETS.keys()),
        "file": str(cfg.TUNING_PRESETS_FILE)
    })
