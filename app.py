import os
import tempfile
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from loguru import logger
from dotenv import load_dotenv
import time

from pipeline.ASR import ASR
from pipeline.speaker_segmentation import SO, SCD
from pipeline.speaker_recognition import SklearnClustering
from pipeline.speaker_recognition import SpeechBrainEmbedding
from pipeline.speaker_recognition import SpeechBrainSpeakerRecognition
from pipeline.speaker_segmentation import SileroVAD
from pipeline.phoene import SpeechBrainPhoneme
from utils import load_audio_from_file
from utils import load_annotation_file, evaluate_pipeline, format_metrics_report
from config import PipelineConfig as Config

load_dotenv(".env")

class WhoSays(object):
    def __init__(self):
        self.config = Config()
        
        #self.sod = SO(self.config.so)
        self.scd = SCD(**self.config.scd.pyannote.to_dict())
        
        self.vad = SileroVAD(**self.config.vad.silero.to_dict())
        self.asr = ASR(**self.config.asr.to_dict())
        self.phoneme = SpeechBrainPhoneme(**self.config.phoneme.speechbrain.to_dict())

        self.embedder = SpeechBrainEmbedding(**self.config.embedding.speechbrain.to_dict())
        self.clustering = SklearnClustering(**self.config.clustering.kmeans.to_dict()) 
        self.recognition = SpeechBrainSpeakerRecognition(**self.config.recognition.speechbrain.to_dict())
    
    def __call__(
        self,
        audio_file: str,
        num_speakers: int = 2,
        include_timing: bool = False
    ):
        """
        Process audio file through the complete speaker diarization pipeline.

        Args:
            audio_file: Path to audio file
            num_speakers: Expected number of speakers
            include_timing: Whether to include timing metrics for each model run

        Returns:
            dict: Structured diarization results containing:
                - transcription: List of transcribed segments with timestamps
                - speakers: List of speaker segments with IDs and timestamps
                - segments: Detailed segment information with speaker assignments
                - timing: (optional) Timing information for each pipeline step
        """
        timing = {} if include_timing else None

        logger.info(f"Loading audio from {audio_file}")
        if include_timing:
            start_time = time.time()
        waveform, sr = load_audio_from_file(
            file_path=audio_file,
            sr=self.config.sr
        )
        if include_timing:
            timing['audio_loading'] = time.time() - start_time

        # Ensure mono audio for pipeline processing
        if waveform.dim() > 1:
            if waveform.shape[0] > 1 and waveform.shape[0] < waveform.shape[1]:
                # Shape is (n_channels, n_samples) - average channels
                waveform = waveform.mean(dim=0)
            elif waveform.shape[1] > 1 and waveform.shape[1] < waveform.shape[0]:
                # Shape is (n_samples, n_channels) - average channels
                waveform = waveform.mean(dim=1)
            else:
                # Ambiguous shape, assume first dim is channels
                waveform = waveform.squeeze()
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0)

        logger.info(f"Audio shape: {waveform.shape}, sample rate: {sr}Hz")

        # Step 1: Voice Activity Detection
        logger.info("Running Voice Activity Detection (VAD)...")
        if include_timing:
            start_time = time.time()
        speech_segments = self.vad(waveform)
        if include_timing:
            timing['vad'] = time.time() - start_time
        logger.info(f"Found {len(speech_segments)} speech segments")

        # Step 2: Automatic Speech Recognition
        logger.info("Transcribing speech segments...")
        if include_timing:
            start_time = time.time()
        transcriptions = self.asr.transcribe_segments(
            waveform,
            speech_segments,
            return_timestamps=True
        )
        if include_timing:
            timing['asr'] = time.time() - start_time
        logger.info(f"Transcribed {len(transcriptions)} segments")

        # Step 3: Speaker Overlap Detection
        logger.info("Detecting speaker overlaps...")
        #overlap_segments = self.sod.sod_pipeline(waveform)
        #logger.info(f"Found {len(overlap_segments)} overlapping speech regions")

        # Step 4: Speaker Change Detection
        logger.info("Detecting speaker change points...")
        if include_timing:
            start_time = time.time()
        change_points = self.scd(waveform)
        if include_timing:
            timing['scd'] = time.time() - start_time
        logger.info(f"Found {len(change_points)} speaker change points")

        # Step 5: Speaker Embedding
        logger.info("Extracting speaker embeddings...")
        if include_timing:
            start_time = time.time()
        segment_embeddings = self.embedder.embed_segments(waveform, sr, change_points)
        if include_timing:
            timing['embedding'] = time.time() - start_time
        logger.info(f"Extracted embeddings for {segment_embeddings.shape[0]} segments")

        # Step 6: Speaker Clustering
        logger.info(f"Clustering speaker segments into {num_speakers} clusters...")
        if include_timing:
            start_time = time.time()
        segment_clusters = self.clustering.cluster_segments(segment_embeddings, n_clusters=num_speakers)
        if include_timing:
            timing['clustering'] = time.time() - start_time
        logger.info(f"Identified {len(set(segment_clusters.tolist()))} speaker clusters")

        # Step 7: Merge results into structured output
        logger.info("Merging results...")
        if include_timing:
            start_time = time.time()
        result = self._format_output(
            change_points=change_points,
            clusters=segment_clusters,
            transcriptions=transcriptions,
            overlap_segments= [],# overlap_segments,
            waveform_duration=waveform.shape[-1] / sr
        )
        if include_timing:
            timing['formatting'] = time.time() - start_time

        # Add timing to result if requested
        if include_timing:
            result['timing'] = timing
            total_time = sum(timing.values())
            result['total_time'] = total_time
            logger.info(f"Total pipeline time: {total_time:.2f}s")

        logger.info("Pipeline complete!")
        return result

    def _format_output(
        self,
        change_points: list[float],
        clusters: list[int],
        transcriptions: list[dict],
        overlap_segments: list[tuple[float, float]],
        waveform_duration: float
    ) -> dict:
        """
        Format pipeline outputs into a structured result.

        Args:
            change_points: Speaker change timestamps
            clusters: Cluster IDs for each segment
            transcriptions: Transcription results with timestamps
            overlap_segments: List of (start, end) tuples where speaker overlap occurs
            waveform_duration: Total audio duration in seconds

        Returns:
            dict: Structured output with speaker segments and transcriptions
        """
        # Create speaker segments from change points and clusters
        speaker_segments = []
        segment_times = [0.0] + change_points + [waveform_duration]

        for i in range(len(segment_times) - 1):
            speaker_segments.append({
                'start': segment_times[i],
                'end': segment_times[i + 1],
                'speaker': f"SPEAKER_{int(clusters[i])}" if i < len(clusters) else "UNKNOWN",
                'duration': segment_times[i + 1] - segment_times[i]
            })

        # Align transcriptions with speaker segments and check for overlaps
        segments_with_text = []
        for segment in speaker_segments:
            # Find overlapping transcriptions
            segment_text = []
            for trans in transcriptions:
                # Check if transcription overlaps with this speaker segment
                overlap_start = max(segment['start'], trans['start'])
                overlap_end = min(segment['end'], trans['end'])

                if overlap_start < overlap_end:
                    # There's overlap
                    segment_text.append({
                        'text': trans['text'],
                        'start': trans['start'],
                        'end': trans['end'],
                        'chunks': trans.get('chunks', [])
                    })

            # Check if this segment has speaker overlap
            has_overlap = False
            for ovlp_start, ovlp_end in overlap_segments:
                # Check if overlap region intersects with this speaker segment
                if max(segment['start'], ovlp_start) < min(segment['end'], ovlp_end):
                    has_overlap = True
                    break

            segments_with_text.append({
                **segment,
                'transcriptions': segment_text,
                'text': ' '.join([t['text'] for t in segment_text]).strip(),
                'has_overlap': has_overlap
            })

        # Format overlap segments for output
        overlap_regions = [
            {'start': start, 'end': end, 'duration': end - start}
            for start, end in overlap_segments
        ]

        # Create final structured output
        return {
            'duration': waveform_duration,
            'num_speakers': len(set(clusters.tolist())),
            'num_overlaps': len(overlap_segments),
            'transcription': transcriptions,
            'speaker_segments': speaker_segments,
            'overlap_regions': overlap_regions,
            'segments': segments_with_text
        }
    
app = Flask(__name__)

logger.info("Loading WhoSays pipeline... This may take a moment.")
pipeline = WhoSays()
logger.info("Pipeline loaded successfully. Server is ready.")

@app.route('/')
def index():
    logger.info("Serving index.html")
    return send_from_directory('.', 'index.html')

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
            result = pipeline(temp_file_path, num_speakers=num_speakers, include_timing=True)
            
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)