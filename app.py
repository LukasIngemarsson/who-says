import os
import time
import tempfile
import subprocess
import base64
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from loguru import logger
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn.functional as F

import warnings
from utils import load_audio_from_file, match_frequency
warnings.filterwarnings("ignore")

# -------------------------------------------------
# Try importing the WhoSays pipeline
# -------------------------------------------------
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

# -------------------------------------------------
# Initialize pipeline
# -------------------------------------------------
if PIPELINE_AVAILABLE:
    logger.info("Loading WhoSays pipeline... This may take a moment.")
    pipeline = WhoSays()
    logger.info("Pipeline loaded successfully. Server is ready.")
else:
    pipeline = None

# -------------------------------------------------
# Global state
# -------------------------------------------------
ROLLING_BUFFER = None        # 1-second rolling raw audio (torch.Tensor on pipeline device)
SPEECH_BUFFER = None         # rolling speech-only buffer (up to ~2s)
KNOWN_SPEAKERS = {}          # name -> normalized embedding (1D tensor on CPU)

SESSION_STATE = {}           # kept in case you want to use later
# Live ASR per session:
#   session_id -> {
#       "buffer": Tensor,
#       "cursor": int,
#       "last_asr_time": float,
#   }
SESSION_ASR_STATE = {}

EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

CURRENT_SPEAKER = None
CURRENT_CONFIDENCE = 0.0
LAST_SWITCH_TIME = 0.0
SWITCH_COOLDOWN = 0.3  # seconds

# Similarity → confidence scaling (for UI)
SIM_CONF_FLOOR = 0.05  # expected "different speaker" region
SIM_CONF_CEIL = 0.50   # expected "good same speaker" region

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def similarity_to_confidence(score: float,
                             floor: float = SIM_CONF_FLOOR,
                             ceil: float = SIM_CONF_CEIL) -> float:
    """Map cosine similarity into [0,1] for UI purposes."""
    x = (score - floor) / (ceil - floor)
    x = max(0.0, min(1.0, x))
    return float(x)

def save_embedding(name, tensor: torch.Tensor):
    """Save a 1D CPU tensor to disk."""
    try:
        file_path = EMBEDDINGS_DIR / f"{name}.pt"
        torch.save(tensor.cpu(), file_path)
        logger.info(f"Saved embedding for {name} to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save embedding for {name}: {e}")

def load_embeddings():
    """Load normalized 1D embeddings from disk into KNOWN_SPEAKERS (on CPU)."""
    KNOWN_SPEAKERS.clear()
    try:
        for file_path in EMBEDDINGS_DIR.glob("*.pt"):
            name = file_path.stem
            tensor = torch.load(file_path, map_location='cpu').squeeze()
            if tensor.dim() != 1:
                logger.warning(f"Embedding for {name} has unexpected shape {tensor.shape}, flattening")
                tensor = tensor.flatten()
            # Normalize here so all stored embeddings are unit length
            tensor = F.normalize(tensor, p=2, dim=0)
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

def get_live_snippet_for_session(session_id: str, sr: int) -> str:
    """
    Run VAD + ASR only on the NEW part of the session buffer that hasn't
    been transcribed yet. Returns just the new text snippet.
    """
    state = SESSION_ASR_STATE.get(session_id)
    if state is None:
        return ""

    buffer = state.get("buffer")
    cursor = int(state.get("cursor", 0))

    if buffer is None or buffer.numel() == 0 or cursor >= len(buffer):
        return ""

    # Throttle heavy ASR calls – only once every 30s per session
    now = time.time()
    last_asr_time = float(state.get("last_asr_time", 0.0))
    MIN_ASR_INTERVAL = 0.0  # seconds
    if now - last_asr_time < MIN_ASR_INTERVAL:
        return ""

    # Require a minimum amount of new audio before attempting ASR
    # (more context → fewer hallucinations)
    MIN_NEW_SAMPLES = int(1.5 * sr)  # ~1.5s of audio
    if len(buffer) - cursor < MIN_NEW_SAMPLES:
        return ""

    tail = buffer[cursor:]
    tail_start_time = cursor / sr

    # VAD on the tail only
    try:
        speech_segments_tail = pipeline.vad(tail)
    except Exception as e:
        logger.error(f"VAD error in get_live_snippet_for_session({session_id}): {e}")
        return ""

    if not speech_segments_tail:
        return ""

    # Require at least some real speech duration in the tail
    total_speech_tail = sum(max(0.0, seg["end"] - seg["start"]) for seg in speech_segments_tail)
    if total_speech_tail < 0.5:  # < 0.5s of detected speech → too flimsy for ASR
        return ""

    # Offset segments to full-buffer timeline
    new_segments = []
    for seg in speech_segments_tail:
        seg_start = seg["start"] + tail_start_time
        seg_end = seg["end"] + tail_start_time
        new_segments.append({"start": seg_start, "end": seg_end})

    # ASR on the tail only (live captioning does not require per-segment timestamps)
    try:
        asr_result = pipeline.asr.transcribe(
            tail,
            return_timestamps=False,
        )
    except Exception as e:
        logger.error(f"ASR error in get_live_snippet_for_session({session_id}): {e}")
        return ""

    snippet = (asr_result.get("text") or "").strip() if isinstance(asr_result, dict) else str(asr_result or "").strip()

    if not snippet:
        return ""

    # Advance cursor to the end of the processed tail
    state["cursor"] = len(buffer)


    # Heuristic: drop very common closing hallucinations on short/noisy tails
    lowered = snippet.lower()
    banned_phrases = [
        "thank you very much for watching",
        "thank you for watching",
        "thanks for watching",
        "see you in the next video",
        "don't forget to like and subscribe",
    ]
    if any(p in lowered for p in banned_phrases):
        logger.info(f"Dropping likely hallucinated closing snippet for session {session_id!r}: {snippet!r}")
        return ""

    # Only mark ASR time if we actually produced text
    state["last_asr_time"] = now
    return snippet


def finalize_current_turn(session_id: str, end_sample: int, sr: int) -> dict:
    """
    Slice the current speaker's turn from the per-session ASR buffer,
    run ASR on just that slice, and return a small dict:

      {
        "turn_speaker": <name>,
        "turn_transcript": <text>,
        "turn_start": <seconds>,
        "turn_end": <seconds>,
        "turn_duration": <seconds>,
      }

    If there is no active turn, or ASR returns no text, this returns {}.
    """
    state = SESSION_ASR_STATE.get(session_id)
    if not state:
        return {}

    buffer = state.get("buffer")
    start_sample = state.get("current_turn_start_sample")
    turn_speaker = state.get("current_turn_speaker")

    if (
        buffer is None
        or buffer.numel() == 0
        or turn_speaker is None
        or start_sample is None
        or end_sample is None
        or end_sample <= start_sample
    ):
        return {}

    try:
        start_idx = max(int(start_sample), 0)
        end_idx = min(int(end_sample), len(buffer))

        if end_idx <= start_idx:
            return {}

        turn_audio = buffer[start_idx:end_idx]

        # Run ASR on this single turn (no timestamps needed)
        result = pipeline.asr.transcribe(
            turn_audio,
            return_timestamps=False,
        )

        if isinstance(result, dict):
            text = (result.get("text") or "").strip()
        else:
            text = str(result or "").strip()

        if not text:
            return {}

        start_time = start_idx / float(sr)
        end_time = end_idx / float(sr)

        # Reset current turn so the next speech starts a new one
        state["current_turn_speaker"] = None
        state["current_turn_start_sample"] = None

        # Also move the ASR cursor forward so live snippets don't re-transcribe
        state["cursor"] = max(int(end_idx), int(state.get("cursor", 0)))

        return {
            "turn_speaker": turn_speaker,
            "turn_transcript": text,
            "turn_start": start_time,
            "turn_end": end_time,
            "turn_duration": end_time - start_time,
        }

    except Exception as e:
        logger.error(f"Error while finalizing turn for session {session_id!r}: {e}")
        return {}
def convert_to_wav(input_path):
    """Convert any input audio to 16kHz mono WAV using ffmpeg."""
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

        subprocess.run(command, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
        logger.info(f"Converted {input_path} to {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        raise Exception(f"Could not convert audio file: {e.stderr.decode()}")

if PIPELINE_AVAILABLE:
    load_embeddings()

# -------------------------------------------------
# Routes
# -------------------------------------------------
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

# -------------------------------------------------
# /process - full file diarization / ASR
# -------------------------------------------------
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

# -------------------------------------------------
# /upload_embeddings - ENROLL speakers
# -------------------------------------------------
@app.route('/upload_embeddings', methods=['POST'])
def upload_embeddings():
    """Enroll a speaker: use the SAME embedder path as live mode, and normalize."""
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

            # Load audio using the same loader as the pipeline
            waveform, sr = load_audio_from_file(wav_path, sr=16000)

            # Ensure mono
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)

            # Move to pipeline device
            device = pipeline.config.device
            waveform = waveform.to(device)

            # IMPORTANT: use the SAME embedder as live mode
            embedding_tensor = pipeline.embedder.embed(waveform, 16000)

            # ---- FIX: squeeze FIRST ----
            while embedding_tensor.dim() > 1:
                embedding_tensor = embedding_tensor.squeeze(0)

            # ---- NOW it is safe to compare them ----
            with torch.no_grad():
                live_like_emb = pipeline.embedder.embed(waveform, 16000)

                # squeeze this one too
                while live_like_emb.dim() > 1:
                    live_like_emb = live_like_emb.squeeze(0)

                enroll_norm = F.normalize(embedding_tensor, p=2, dim=0)
                live_norm = F.normalize(live_like_emb, p=2, dim=0)

                sim = torch.dot(enroll_norm, live_norm).item()
                logger.warning(f"[DEBUG] Enrollment self-similarity = {sim:.4f}")

            # ---- continue as before ----
            embedding_tensor = F.normalize(embedding_tensor, p=2, dim=0)
            # Store on CPU in KNOWN_SPEAKERS
            KNOWN_SPEAKERS[speaker_name] = embedding_tensor.cpu()
            logger.info(f"Enrolled {speaker_name}. Total known speakers: {len(KNOWN_SPEAKERS)}")

            save_embedding(speaker_name, embedding_tensor)

            shape = embedding_tensor.shape if hasattr(embedding_tensor, 'shape') else "unknown"

            return jsonify({
                "message": f"Successfully enrolled speaker: {speaker_name}",
                "vector_size": str(shape)
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

# -------------------------------------------------
# /identify_speaker - REAL-TIME
# -------------------------------------------------
@app.route('/identify_speaker', methods=['POST'])
def identify_speaker():
    """Real-time speaker identification.

    - 1s rolling audio buffer (for VAD)
    - up to 2s rolling speech-only buffer (for embedding)
    - cosine similarity vs enrolled, normalized embeddings
    - stable speaker indicator with cooldown and margin
    """
    global ROLLING_BUFFER, SPEECH_BUFFER
    global CURRENT_SPEAKER, CURRENT_CONFIDENCE, LAST_SWITCH_TIME

    if not PIPELINE_AVAILABLE:
        return jsonify({"error": "Pipeline not available"}), 503

    try:
        # 1. Read raw PCM base64 audio
        if 'audio_data' not in request.form:
            return jsonify({"error": "No audio_data provided"}), 400

        audio_data_b64 = request.form.get('audio_data')
        sample_rate = int(request.form.get('sample_rate', 16000))
        session_id = request.form.get('session_id', 'default')

        try:
            audio_bytes = base64.b64decode(audio_data_b64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

            # Start on CPU for potential resampling
            waveform = torch.from_numpy(audio_array).float()
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)

            sr = sample_rate

            # Resample to pipeline sample rate if needed
            target_sr = pipeline.config.sr
            if sr != target_sr:
                waveform_2d = waveform.unsqueeze(0)  # (1, n_samples)
                waveform_2d = match_frequency(waveform_2d, sr, sr=target_sr)
                waveform = waveform_2d.squeeze(0)
                sr = target_sr

            device = pipeline.config.device
            waveform = waveform.to(device)

        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return jsonify({"error": "Invalid audio data"}), 400

        
        # Update per-session ASR buffer (kept separate from VAD/speaker buffers)
        if session_id not in SESSION_ASR_STATE:
            SESSION_ASR_STATE[session_id] = {
                "buffer": torch.zeros(0, device=device),
                "cursor": 0,
                "last_asr_time": 0.0,
                # Turn-tracking state for per-speaker transcripts
                "current_turn_speaker": None,
                "current_turn_start_sample": None,
            }
        asr_state = SESSION_ASR_STATE[session_id]
        prev_total_samples = int(asr_state["buffer"].shape[0])
        asr_state["buffer"] = torch.cat([asr_state["buffer"], waveform.detach()])
        total_samples = int(asr_state["buffer"].shape[0])

        # 2. Initialize buffers on correct device
        if ROLLING_BUFFER is None:
            ROLLING_BUFFER = torch.zeros(sr, device=device)
        if SPEECH_BUFFER is None:
            SPEECH_BUFFER = torch.zeros(0, device=device)

        # 3. Update 1-second rolling audio buffer
        chunk = waveform

        if len(chunk) >= sr:
            ROLLING_BUFFER = chunk[-sr:]
        else:
            needed = sr - len(chunk)
            ROLLING_BUFFER = torch.cat([ROLLING_BUFFER[-needed:], chunk])

        vad_input = ROLLING_BUFFER.clone()



        # 4. Run VAD on rolling 1s window (for speaker detection)
        speech_segments = pipeline.vad(vad_input)


        if not speech_segments:
            # No speech in this window; keep previous speaker if any
            ui_conf = similarity_to_confidence(CURRENT_CONFIDENCE)
            return jsonify({
                "speaker": CURRENT_SPEAKER,
                "has_speech": False if CURRENT_SPEAKER is None else True,
                "confidence": ui_conf,
                "transcript": "",
            })

        # 5. Build rolling speech buffer (up to 2s) for speaker embedding
        speech_portions = []
        for seg in speech_segments:
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            seg_audio = vad_input[start_sample:end_sample]
            if len(seg_audio) > 0:
                speech_portions.append(seg_audio)

        new_speech = torch.cat(speech_portions) if speech_portions else torch.zeros(0, device=device)

        SPEECH_BUFFER = torch.cat([SPEECH_BUFFER, new_speech])

        MAX_SPEECH_SAMPLES = int(2.0 * sr)  # up to 2 seconds of speech
        if len(SPEECH_BUFFER) > MAX_SPEECH_SAMPLES:
            SPEECH_BUFFER = SPEECH_BUFFER[-MAX_SPEECH_SAMPLES:]

  
        # 6. Require minimum speech before embedding
        MIN_SPEECH_SAMPLES = int(0.5 * sr)  # at least 1s of speech
        if len(SPEECH_BUFFER) < MIN_SPEECH_SAMPLES:
            ui_conf = similarity_to_confidence(CURRENT_CONFIDENCE)
            live_snippet = get_live_snippet_for_session(session_id, sr)
            return jsonify({
                "speaker": CURRENT_SPEAKER,
                "has_speech": True,
                "confidence": ui_conf,
                "message": "Collecting more speech...",
                "transcript": live_snippet,
            })

        # 7. Compute embedding from speech buffer
        embedding = pipeline.embedder.embed(SPEECH_BUFFER, sr)
        while embedding.dim() > 1:
            embedding = embedding.squeeze(0)

        # Normalize current embedding
        embedding = F.normalize(embedding, p=2, dim=0)


        # 8. Compare with known speakers (track 2 best)
        best_speaker = None
        best_score = -1.0
        second_best_score = -1.0

        if KNOWN_SPEAKERS:
            for name, ref_emb_cpu in KNOWN_SPEAKERS.items():
                # Move ref emb to device
                ref_emb = ref_emb_cpu.to(device)
                # ref_emb is already normalized on load; but re-normalize for safety
                ref_emb = F.normalize(ref_emb, p=2, dim=0)

                # Cosine similarity between two unit vectors
                score = torch.dot(embedding, ref_emb).item()

                logger.debug(f"{name}: similarity={score:.4f}")

                if score > best_score:
                    second_best_score = best_score
                    best_score = score
                    best_speaker = name
                elif score > second_best_score:
                    second_best_score = score

 
        now = time.time()

        # Tuning thresholds
        MIN_CONFIDENCE = 0.25       # minimum similarity to accept/update speaker
        MARGIN_THRESHOLD = 0.08    # best must beat second-best by at least this much

        margin = best_score - second_best_score if second_best_score > -1.0 else best_score

        if (best_speaker is None or
            best_score < MIN_CONFIDENCE or
            margin < MARGIN_THRESHOLD):
            # Not confident enough in this frame; keep previous speaker
            ui_conf = similarity_to_confidence(CURRENT_CONFIDENCE)
            live_snippet = get_live_snippet_for_session(session_id, sr)
            return jsonify({
                "speaker": CURRENT_SPEAKER,
                "has_speech": True if CURRENT_SPEAKER else False,
                "confidence": ui_conf,
                "transcript": live_snippet,
            })

        # First speaker
        previous_speaker = CURRENT_SPEAKER
        if CURRENT_SPEAKER is None:
            CURRENT_SPEAKER = best_speaker
            CURRENT_CONFIDENCE = best_score
            LAST_SWITCH_TIME = now
        else:
            # Different speaker? apply cooldown
            if best_speaker != CURRENT_SPEAKER:
                if now - LAST_SWITCH_TIME >= SWITCH_COOLDOWN:
                    CURRENT_SPEAKER = best_speaker
                    CURRENT_CONFIDENCE = best_score
                    LAST_SWITCH_TIME = now
                # else: ignore this change (too soon)
            else:
                # Same speaker; gently update confidence (simple EMA)
                ALPHA = 0.3
                CURRENT_CONFIDENCE = ALPHA * best_score + (1 - ALPHA) * CURRENT_CONFIDENCE
                LAST_SWITCH_TIME = now

        # --- Turn-tracking for per-speaker transcripts ---
        turn_metadata: dict = {}
        active_turn_speaker = asr_state.get("current_turn_speaker")

        # Helper to start a new turn at the beginning of this chunk
        def _start_new_turn(speaker_name: str):
            asr_state["current_turn_speaker"] = speaker_name
            asr_state["current_turn_start_sample"] = prev_total_samples

        # If there was an active turn and the (possibly updated) CURRENT_SPEAKER
        # is now different or None, close that turn at the start of this chunk.
        if active_turn_speaker is not None and CURRENT_SPEAKER != active_turn_speaker:
            turn_metadata = finalize_current_turn(
                session_id=session_id,
                end_sample=prev_total_samples,
                sr=sr,
            )

        # Start a new turn if we have a non-None current speaker and no active turn
        # (either initial speech or right after closing a previous turn).
        if CURRENT_SPEAKER is not None and asr_state.get("current_turn_speaker") is None:
            _start_new_turn(CURRENT_SPEAKER)

        # Final response: include latest live snippet (tail-only ASR)
        ui_conf = similarity_to_confidence(CURRENT_CONFIDENCE)
        live_snippet = get_live_snippet_for_session(session_id, sr)

        response = {
            "speaker": CURRENT_SPEAKER,
            "has_speech": True,
            "confidence": ui_conf,
            "transcript": live_snippet,
        }
        if turn_metadata:
            response.update(turn_metadata)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in identify_speaker: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# /correct_speaker - Disabled during live recording
# -------------------------------------------------
@app.route('/correct_speaker', methods=['POST'])
def correct_speaker():
    return jsonify({
        "error": "Speaker additions and corrections are not allowed during recording. "
                 "Please use the 'Add Speakers' button before starting a recording."
    }), 400

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
