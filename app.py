import os
import tempfile
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from loguru import logger
from dotenv import load_dotenv

from main import WhoSays

load_dotenv(".env")

static_folder_path = os.environ.get("FLASK_STATIC_FOLDER", "../frontend/dist")

static_folder_path = os.path.abspath(static_folder_path)
 
app = Flask(__name__, static_folder=static_folder_path, static_url_path='')

""" from flask_cors import CORS
CORS(app) """

logger.info("Loading WhoSays pipeline... This may take a moment.")
pipeline = WhoSays()
logger.info("Pipeline loaded successfully. Server is ready.")

KNOWN_SPEAKERS = {}

@app.route('/')
def index():
    logger.info(f"Serving {app.static_folder} or ../frontend/dist")
    return send_from_directory(app.static_folder or '../frontend/dist', 'index.html')

@app.route('/<path:path>')
def catch_all(path):
    # Check if the file exists in the static folder (e.g. css/js files)
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)

    return send_from_directory(app.static_folder, 'index.html')

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "WhoSays server is running."})

@app.route('/process', methods=['POST'])
def process_audio():
    """
    The main endpoint to process an uploaded audio file.
    Expects a multipart-form request with:
    - 'file': The audio file (e.g., .wav, .mp3)
    - 'num_speakers': (Optional) The number of speakers, defaults to 2
    """
    try:
        # 1. Check if the file part is present
        if 'file' not in request.files:
            logger.warning("No 'file' part in request")
            return jsonify({"error": "No 'file' part in the request"}), 400

        file = request.files['file']

        # 2. Check if a file was selected
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({"error": "No file selected"}), 400

        # 3. Get number of speakers from form data
        num_speakers = request.form.get('num_speakers', 2, type=int)

        temp_file_path = None
        try:
            # 4. Save the file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(str(file.filename)).suffix) as temp_file:
                file.save(temp_file)
                temp_file_path = temp_file.name
            
            logger.info(f"Received file '{file.filename}'. Saved to temp path: {temp_file_path}")
            logger.info(f"Processing with num_speakers={num_speakers}")

            # 5. Run the pipeline
            result = pipeline(temp_file_path, 
                              num_speakers=num_speakers, 
                              include_timing=True, 
                              known_speakers=KNOWN_SPEAKERS)
            
            logger.info(f"Successfully processed file: {temp_file_path}")
            
            # 6. Return the JSON result
            return jsonify(result)

        except Exception as e:
            logger.error(f"Error during pipeline processing: {e}")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500
        
        finally:
            # 7. Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")

    except Exception as e:
        logger.error(f"Unhandled error in /process: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    
@app.route('/upload_embeddings', methods=['POST'])
def upload_embeddings():
    """
    Endpoint to enroll a new speaker.
    Expects:
    - 'file': A short audio file containing ONLY the target speaker.
    - 'name': The name of the speaker (e.g., 'John Doe').
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        speaker_name = request.form.get('name')

        if not speaker_name:
            return jsonify({"error": "Speaker 'name' is required"}), 400
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(str(file.filename)).suffix) as temp_file:
            file.save(temp_file)
            temp_file_path = temp_file.name

        try:
            logger.info(f"Generating embedding for speaker: {speaker_name}")
            
            # Use the pipeline to generate the embedding vector
            embedding_tensor = pipeline.get_reference_embedding(temp_file_path)
            
            # Store in memory
            KNOWN_SPEAKERS[speaker_name] = embedding_tensor
            
            logger.info(f"Enrolled {speaker_name}. Total known speakers: {len(KNOWN_SPEAKERS)}")
            
            return jsonify({
                "message": f"Successfully enrolled speaker: {speaker_name}",
                "vector_size": embedding_tensor.shape[0]
            })

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        logger.error(f"Error in upload_embeddings: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)