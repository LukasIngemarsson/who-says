import os
import tempfile
import subprocess
import base64
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
    from utils import load_audio_from_file
    PIPELINE_AVAILABLE = True
except ImportError:
    logger.warning("Could not import WhoSays from main.py. Running in mock mode.")
    PIPELINE_AVAILABLE = False
    load_audio_from_file = None

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

# Session state for tracking consecutive failures (in-memory, per session)
# Format: {session_id: {"consecutive_failures": int, "last_audio_chunks": list}}
SESSION_STATE = {}

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
            # Preserve original format exactly - don't normalize or convert dtype
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
            logger.info(f"Using {len(KNOWN_SPEAKERS)} enrolled speakers: {list(KNOWN_SPEAKERS.keys())}")
            
            # Move known speakers embeddings to pipeline device for consistent operations
            # Preserve original dtype when moving to device
            pipeline_device = pipeline.config.device
            known_speakers_on_device = {
                name: emb.to(pipeline_device) 
                for name, emb in KNOWN_SPEAKERS.items()
            }
            
            result = pipeline(
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
            # Preserve original format exactly as returned by get_reference_embedding()
            embedding_tensor = embedding_tensor.squeeze().cpu()
            if embedding_tensor.dim() != 1:
                embedding_tensor = embedding_tensor.flatten()
            # Don't normalize - preserve original embedding format
            
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

@app.route('/identify_speaker', methods=['POST'])
def identify_speaker():
    """Identify speaker from raw audio data (VAD + Embedding only, no ASR)."""
    if not PIPELINE_AVAILABLE:
        return jsonify({"error": "Pipeline not available"}), 503
    
    try:
        # Check for raw audio data (base64 encoded PCM) - preferred method
        if 'audio_data' in request.form:
            audio_data_b64 = request.form.get('audio_data')
            sample_rate = int(request.form.get('sample_rate', 16000))
            
            try:
                # Decode base64 audio data
                audio_bytes = base64.b64decode(audio_data_b64)
                # Convert bytes to numpy array (float32 PCM)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                
                logger.info(f"Received audio data: {len(audio_array)} samples, sample_rate={sample_rate}")
                
                # Convert to torch tensor
                waveform = torch.from_numpy(audio_array).float()
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0)
                
                logger.info(f"Waveform shape: {waveform.shape}, min: {waveform.min():.4f}, max: {waveform.max():.4f}, mean: {waveform.mean():.4f}")
                
                waveform = waveform.to(pipeline.config.device)
                sr = sample_rate
                
            except Exception as e:
                logger.error(f"Error decoding audio data: {e}")
                return jsonify({"error": f"Invalid audio data: {str(e)}"}), 400
        elif 'audio_chunk' in request.files:
            # Fallback: file upload (for compatibility)
            file = request.files['audio_chunk']
            if file.filename == '':
                return jsonify({"error": "No file provided"}), 400
            
            temp_file_path = None
            wav_path = None
            
            try:
                suffix = Path(str(file.filename)).suffix or ".webm"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    file.save(temp_file)
                    temp_file_path = temp_file.name
                
                wav_path = convert_to_wav(temp_file_path)
                
                # Load audio
                from utils import load_audio_from_file
                waveform, sr = load_audio_from_file(wav_path, sr=16000)
                
                # Ensure mono
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0)
                waveform = waveform.to(pipeline.config.device)
            except Exception as e:
                logger.error(f"Error processing audio file: {e}")
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
                return jsonify({"error": f"Could not process audio: {str(e)}"}), 500
        else:
            return jsonify({"error": "No audio_data or audio_chunk provided"}), 400
        
        # Process audio (common for both paths)
        # waveform and sr are now defined from either path above
        # Run VAD to check if there's speech
        logger.info(f"Running VAD on waveform with shape: {waveform.shape}, duration: {len(waveform)/sr:.2f}s")
        speech_segments = pipeline.vad(waveform)
        logger.info(f"VAD found {len(speech_segments)} speech segments")
        if not speech_segments:
            logger.warning("No speech detected by VAD")
            return jsonify({
                "speaker": None,
                "has_speech": False
            })
        
        # Extract embeddings from speech segments (more accurate than entire chunk)
        try:
            # Extract embeddings for each speech segment
            segment_embeddings = []
            for seg in speech_segments:
                start_sample = int(seg['start'] * sr)
                end_sample = int(seg['end'] * sr)
                segment_waveform = waveform[start_sample:end_sample]
                
                if len(segment_waveform) > 0:
                    seg_emb = pipeline.embedder.embed(segment_waveform, sr)
                    # Squeeze to 1D if needed
                    while seg_emb.dim() > 1:
                        seg_emb = seg_emb.squeeze(0)
                    segment_embeddings.append(seg_emb)
            
            if not segment_embeddings:
                logger.warning("No embeddings extracted from speech segments")
                return jsonify({
                    "speaker": None,
                    "has_speech": True,
                    "error": "Failed to extract embeddings"
                })
            
            # Average embeddings from all segments (like cluster centroid)
            segment_embeddings_tensor = torch.stack(segment_embeddings)
            chunk_embedding = torch.mean(segment_embeddings_tensor, dim=0)
            
            # Preserve original format - don't normalize to match original embedding format
            
            logger.info(f"Extracted {len(segment_embeddings)} segment embeddings, averaged to single embedding")
            
            # Identify speaker by comparing to known speakers
            best_speaker = None
            best_score = -1.0
            similarity_threshold = 0.5  # Same as post-recording
            
            is_generic_speaker = False
            
            logger.info(f"Comparing to {len(KNOWN_SPEAKERS)} known speakers")
            if KNOWN_SPEAKERS:
                # Ensure chunk_embedding device for moving ref_emb
                device = chunk_embedding.device
                for name, ref_emb in KNOWN_SPEAKERS.items():
                    # Move ref_emb to same device as chunk_embedding, preserve dtype
                    ref_emb_device = ref_emb.to(device)
                    score = torch.nn.functional.cosine_similarity(
                        chunk_embedding.unsqueeze(0),
                        ref_emb_device.unsqueeze(0)
                    ).item()
                    
                    logger.debug(f"Speaker {name}: similarity={score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        if score >= similarity_threshold:
                            best_speaker = name
                
                logger.info(f"Best match: {best_speaker} (score: {best_score:.4f}, threshold: {similarity_threshold})")
            else:
                logger.warning("No known speakers enrolled. Please add speakers first.")
            
            # Check if speaker is generic (SPEAKER_X pattern)
            if best_speaker and best_speaker.startswith("SPEAKER_"):
                is_generic_speaker = True
            
            # Track consecutive failures per session
            session_id = request.form.get('session_id', 'default')
            if session_id not in SESSION_STATE:
                SESSION_STATE[session_id] = {
                    "consecutive_failures": 0,
                    "last_audio_chunks": [],
                    "last_successful_speaker": None
                }
            
            session = SESSION_STATE[session_id]
            
            # Track failures for logging, but don't use for any actions
            # Speakers are only added via the "Add Speakers" button, not during recording
            if best_speaker is None or is_generic_speaker:
                session["consecutive_failures"] += 1
            else:
                # Reset on success
                session["consecutive_failures"] = 0
                session["last_successful_speaker"] = best_speaker
            
            # Just show who's talking - no corrections or additions during recording
            # Speakers should be added using the "Add Speakers" button before recording
            
            return jsonify({
                "speaker": best_speaker,
                "has_speech": True,
                "confidence": best_score if best_score > 0 else 0.0,
                "is_generic": is_generic_speaker,
                "consecutive_failures": session["consecutive_failures"] if 'session' in locals() else 0
            })
        
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return jsonify({
                "speaker": None,
                "has_speech": True,
                "error": str(e)
            })
    
    except Exception as e:
        logger.error(f"Error in identify_speaker: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up files if they were created
        if 'temp_file_path' in locals() and temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'wav_path' in locals() and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

@app.route('/correct_speaker', methods=['POST'])
def correct_speaker():
    """This endpoint is disabled - speakers should only be added using the 'Add Speakers' button before recording."""
    return jsonify({
        "error": "Speaker additions and corrections are not allowed during recording. Please use the 'Add Speakers' button before starting a recording."
    }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)