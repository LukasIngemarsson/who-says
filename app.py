import os
import tempfile
import subprocess
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from loguru import logger
from dotenv import load_dotenv
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore")

try:
    from main import WhoSays
    PIPELINE_AVAILABLE = True
except ImportError:
    logger.warning("Could not import WhoSays from main.py. Running in mock mode.")
    PIPELINE_AVAILABLE = False

load_dotenv(".env")

static_folder_path = os.environ.get("FLASK_STATIC_FOLDER", "../frontend/dist")
static_folder_path = os.path.abspath(static_folder_path)
 
app = Flask(__name__, static_folder=static_folder_path, static_url_path='')

if PIPELINE_AVAILABLE:
    logger.info("Loading WhoSays pipeline... This may take a moment.")
    pipeline = WhoSays()
    logger.info("Pipeline loaded successfully. Server is ready.")
else:
    pipeline = None

KNOWN_SPEAKERS = {}

EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

def save_embedding(name, tensor):
    try:
        file_path = EMBEDDINGS_DIR / f"{name}.pt"
        torch.save(tensor.cpu(), file_path)
        logger.info(f"Saved embedding for {name} to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save embedding for {name}: {e}")

def load_embeddings():
    KNOWN_SPEAKERS.clear()
    try:
        for file_path in EMBEDDINGS_DIR.glob("*.pt"):
            name = file_path.stem
            tensor = torch.load(file_path, map_location='cpu')
            # Ensure tensor is 1D (squeeze any extra dimensions)
            tensor = tensor.squeeze()
            if tensor.dim() != 1:
                logger.warning(f"Embedding for {name} has unexpected shape {tensor.shape}, flattening")
                tensor = tensor.flatten()
            KNOWN_SPEAKERS[name] = tensor
        logger.info(f"Loaded {len(KNOWN_SPEAKERS)} embeddings from {EMBEDDINGS_DIR}")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")

def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj

def convert_to_wav(input_path):
    try:
        output_fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(output_fd)
        
        command = [
            'ffmpeg', 
            '-y',
            '-i', input_path,
            '-ac', '1',
            '-ar', '16000',
            output_path
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        logger.info(f"Converted {input_path} to {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        raise Exception(f"Could not convert audio file: {e.stderr.decode()}")

if PIPELINE_AVAILABLE:
    load_embeddings()

@app.route('/')
def index():
    logger.info(f"Serving {app.static_folder}")
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def catch_all(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "WhoSays server is running.",
        "pipeline_loaded": PIPELINE_AVAILABLE,
        "known_speakers": list(KNOWN_SPEAKERS.keys())
    })

@app.route('/process', methods=['POST'])
def process_audio():
    if not PIPELINE_AVAILABLE:
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

            result = pipeline(
                wav_path, 
                num_speakers=num_speakers, 
                include_timing=True, 
                known_speakers=KNOWN_SPEAKERS
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
    
@app.route('/upload_embeddings', methods=['POST'])
def upload_embeddings():
    if not PIPELINE_AVAILABLE:
        name = request.form.get('name', 'Unknown')
        return jsonify({"message": f"Mock enrollment for {name}", "vector_size": 0})

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        speaker_name = request.form.get('name')

        if not speaker_name:
            return jsonify({"error": "Speaker 'name' is required"}), 400
        
        suffix = Path(str(file.filename)).suffix or ".webm"
        temp_file_path = None
        wav_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                file.save(temp_file)
                temp_file_path = temp_file.name

            wav_path = convert_to_wav(temp_file_path)
            logger.info(f"Generating embedding for speaker: {speaker_name} from {wav_path}")
            
            embedding_tensor = pipeline.get_reference_embedding(wav_path)
            # Ensure embedding is 1D and on CPU before saving
            embedding_tensor = embedding_tensor.squeeze().cpu()
            if embedding_tensor.dim() != 1:
                embedding_tensor = embedding_tensor.flatten()
            
            KNOWN_SPEAKERS[speaker_name] = embedding_tensor
            logger.info(f"Enrolled {speaker_name}. Total known speakers: {len(KNOWN_SPEAKERS)}")
            
            shape = embedding_tensor.shape if hasattr(embedding_tensor, 'shape') else "unknown"
            save_embedding(speaker_name, embedding_tensor)

            return jsonify({
                "message": f"Successfully enrolled speaker: {speaker_name}",
                "vector_size": shape
            })

        except Exception as e:
            logger.error(f"Pipeline error during embedding generation: {e}")
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

    except Exception as e:
        logger.error(f"Error in upload_embeddings: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)