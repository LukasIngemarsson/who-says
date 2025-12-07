import os
import tempfile
import subprocess
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from loguru import logger
from dotenv import load_dotenv
import numpy as np
import threading
import torch
import time

import warnings
warnings.filterwarnings("ignore")

# Import main logic
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

# Initialize pipeline lazily
pipeline = None
pipeline_loading = False
pipeline_lock = threading.Lock()

def load_pipeline():
    """Load the pipeline in the background."""
    global pipeline, pipeline_loading
    with pipeline_lock:
        if pipeline is None and not pipeline_loading:
            pipeline_loading = True
            try:
                logger.info("Loading WhoSays pipeline... This may take a moment.")
                pipeline = WhoSays()
                logger.info("Pipeline loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load pipeline: {e}")
                pipeline = None
            finally:
                pipeline_loading = False

def get_pipeline():
    """Get pipeline, loading it if necessary."""
    global pipeline
    if pipeline is None:
        if not PIPELINE_AVAILABLE:
            return None
        if pipeline_loading:
            # Wait for background loading to complete
            while pipeline_loading:
                time.sleep(0.1)  # Fixed: was threading.Event().wait(0.1)
        else:
            # Load synchronously if not already loading
            load_pipeline()
    return pipeline

KNOWN_SPEAKERS = {}

EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

def save_embedding(name, tensor):
    """Save a single embedding to disk."""
    try:
        file_path = EMBEDDINGS_DIR / f"{name}.pt"
        torch.save(tensor.cpu(), file_path)
        logger.info(f"Saved embedding for {name} to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save embedding for {name}: {e}")

def load_embeddings():
    """Load all embeddings from disk."""
    KNOWN_SPEAKERS.clear()
    try:
        for file_path in EMBEDDINGS_DIR.glob("*.pt"):
            name = file_path.stem
            tensor = torch.load(file_path)
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
    """
    Converts any audio file to a standard PCM WAV file using ffmpeg.
    Returns the path to the new wav file.
    """
    try:
        # Create a temp file for the output
        output_fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(output_fd)
        
        # Run ffmpeg command: -i input -ac 1 (mono) -ar 16000 (16kHz) output.wav
        # 16kHz mono is the standard for most speech embedding models (like wav2vec, pyannote)
        command = [
            'ffmpeg', 
            '-y', # Overwrite output
            '-i', input_path,
            '-ac', '1', # Convert to mono
            '-ar', '16000', # Convert to 16kHz
            output_path
        ]
        
        # Run process, suppress stdout/stderr unless error
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        logger.info(f"Converted {input_path} to {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        # If conversion fails, we might try returning the original path or raising error
        raise Exception(f"Could not convert audio file: {e.stderr.decode()}")

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
    pipeline = get_pipeline()
    if not pipeline:
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
            # 1. Save original file
            suffix = Path(str(file.filename)).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                file.save(temp_file)
                temp_file_path = temp_file.name
            
            # 2. Convert to WAV (Standardize input)
            wav_path = convert_to_wav(temp_file_path)
            
            logger.info(f"Processing converted file '{wav_path}' with {num_speakers} speakers")

            # 3. Run Pipeline on WAV file
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
            # Clean up both original and converted files
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

    except Exception as e:
        logger.error(f"Unhandled error in /process: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    
@app.route('/upload_embeddings', methods=['POST'])
def upload_embeddings():
    pipeline = get_pipeline()
    if not pipeline:
        name = request.form.get('name', 'Unknown')
        return jsonify({"message": f"Mock enrollment for {name}", "vector_size": 0})

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        speaker_name = request.form.get('name')

        if not speaker_name:
            return jsonify({"error": "Speaker 'name' is required"}), 400
        
        # Determine suffix (browser uploads might default to .webm or .blob)
        suffix = Path(str(file.filename)).suffix or ".webm"
        
        temp_file_path = None
        wav_path = None

        try:
            # 1. Save upload temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                file.save(temp_file)
                temp_file_path = temp_file.name

            # 2. Convert to Standard WAV using ffmpeg
            # This solves the "Unsupported format" issue
            wav_path = convert_to_wav(temp_file_path)

            logger.info(f"Generating embedding for speaker: {speaker_name} from {wav_path}")
            
            # 3. Generate embedding from the clean WAV file
            embedding_tensor = pipeline.get_reference_embedding(wav_path)
            
            # Store in global dictionary
            KNOWN_SPEAKERS[speaker_name] = embedding_tensor
            
            logger.info(f"Enrolled {speaker_name}. Total known speakers: {len(KNOWN_SPEAKERS)}")
            
            shape = embedding_tensor.shape if hasattr(embedding_tensor, 'shape') else "unknown"

            # Save the embedding to disk
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

# Thread because the pipeline takes forever to load and the frontend website was slow to load :)
if PIPELINE_AVAILABLE:
    threading.Thread(target=load_pipeline, daemon=True).start()
    # Load embeddings after a short delay to let pipeline start loading
    def load_embeddings_delayed():
        time.sleep(1)
        load_embeddings()
    threading.Thread(target=load_embeddings_delayed, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)