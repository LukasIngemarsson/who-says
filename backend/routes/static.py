"""
Static file serving routes for the WhoSays backend.
"""
import os

from flask import Blueprint, send_from_directory, send_file, current_app

static_bp = Blueprint('static', __name__)


@static_bp.route("/")
def index():
    from loguru import logger
    logger.info(f"Serving {current_app.static_folder}")
    return send_from_directory(current_app.static_folder, "index.html")


@static_bp.route("/<path:path>")
def catch_all(path: str):
    file_path = os.path.join(current_app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(current_app.static_folder, path)
    return send_from_directory(current_app.static_folder, "index.html")


@static_bp.route("/testaudio/thetestsound.wav", methods=["GET"])
def serve_testsound():
    """Serve the built-in test WAV for UI-based streaming tests."""
    from flask import jsonify
    wav_path = os.path.join(os.getcwd(), "thetestsound.wav")
    if not os.path.exists(wav_path):
        return jsonify({"error": "thetestsound.wav not found on server"}), 404
    return send_file(wav_path, mimetype="audio/wav")
