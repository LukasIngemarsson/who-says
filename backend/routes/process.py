import os
import tempfile
from pathlib import Path

from flask import Blueprint, jsonify, request
from loguru import logger

import backend.config as cfg
from backend.helpers import convert_to_wav, make_serializable

process_bp = Blueprint('process', __name__)

# -------------------------------------------------
# /process - full file diarization / ASR
# -------------------------------------------------
@process_bp.route('/process', methods=['POST'])
def process_audio():
    if not cfg.PIPELINE_AVAILABLE:
        return jsonify({"error": "Pipeline not available. Server running in mock mode."}), 503

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No 'file' part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        num_speakers = request.form.get('num_speakers', 2, type=int)

        temp_file_path = None
        wav_path = None

        try:
            suffix = Path(str(file.filename)).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                file.save(temp_file)
                temp_file_path = temp_file.name

            wav_path = convert_to_wav(temp_file_path)
            logger.info(f"Processing converted file '{wav_path}' with {num_speakers} speakers")
            logger.info(f"Using {len(cfg.KNOWN_SPEAKERS)} enrolled speakers: {list(cfg.KNOWN_SPEAKERS.keys())}")

            pipeline_device = cfg.pipeline.config.device
            known_speakers_on_device = {
                name: emb.to(pipeline_device)
                for name, emb in cfg.KNOWN_SPEAKERS.items()
            }

            result = cfg.pipeline(
                wav_path,
                num_speakers=num_speakers,
                include_timing=True,
                known_speakers=known_speakers_on_device
            )

            return jsonify(make_serializable(result))

        except Exception as e:
            logger.error(f"Error during pipeline processing: {e}")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

    except Exception as e:
        logger.error(f"Unhandled error in /process: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

