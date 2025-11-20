from loguru import logger
import argparse
from pathlib import Path
import time

import torch

from dotenv import load_dotenv

from pipeline.ASR import ASR
from pipeline.speaker_segmentation import SO, SCD, SileroVAD
from pipeline.speaker_recognition import SklearnClustering
from pipeline.speaker_recognition import SpeechBrainEmbedding
from pipeline.speaker_recognition import SpeechBrainSpeakerRecognition
from pipeline.phoene import SpeechBrainPhoneme
from utils import load_audio_from_file
from utils import load_annotation_file, evaluate_pipeline, format_metrics_report, format_timing_report
from config import PipelineConfig as Config

load_dotenv(".env")

class WhoSays(object):
    def __init__(self, config=Config):
        self.config = config()
        
        logger.info(f"Using device: {self.config.device}")
        
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
        with torch.no_grad():
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
            waveform = waveform.to(self.config.device)
            logger.info(f"Audio Waveform is on device: {waveform.device}")

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

        # Step 3: Phoneme Conversion
        logger.info("Converting transcriptions to phonemes...")
        if include_timing:
            start_time = time.time()
        transcriptions = self.phoneme(transcriptions)
        if include_timing:
            timing['phoneme'] = time.time() - start_time
        logger.info(f"Added phonemes to {len(transcriptions)} segments")

        # Step 4: Speaker Overlap Detection
        logger.info("Detecting speaker overlaps...")
        #overlap_segments = self.sod.sod_pipeline(waveform)
        #logger.info(f"Found {len(overlap_segments)} overlapping speech regions")

        # Step 5: Speaker Change Detection
        logger.info("Detecting speaker change points...")
        if include_timing:
            start_time = time.time()
        change_points = self.scd(waveform)
        if include_timing:
            timing['scd'] = time.time() - start_time
        logger.info(f"Found {len(change_points)} speaker change points")

        # Step 6: Speaker Embedding
        logger.info("Extracting speaker embeddings...")
        if include_timing:
            start_time = time.time()
        segment_embeddings = self.embedder.embed_segments(waveform, sr, change_points)
        if include_timing:
            timing['embedding'] = time.time() - start_time
        logger.info(f"Extracted embeddings for {segment_embeddings.shape[0]} segments")

        # Step 7: Speaker Clustering
        logger.info(f"Clustering speaker segments into {num_speakers} clusters...")
        if include_timing:
            start_time = time.time()
        segment_clusters = self.clustering.cluster_segments(segment_embeddings, n_clusters=num_speakers)
        if include_timing:
            timing['clustering'] = time.time() - start_time
        logger.info(f"Identified {len(set(segment_clusters.tolist()))} speaker clusters")

        # Step 8: Merge results into structured output
        logger.info("Merging results...")
        if include_timing:
            start_time = time.time()
        result = self._format_output(
            change_points=change_points,
            clusters=segment_clusters,
            transcriptions=transcriptions,
            overlap_segments= [],# overlap_segments,
            waveform_duration=waveform.shape[-1] / sr,
            vad_segments=speech_segments
        )
        if include_timing:
            timing['formatting'] = time.time() - start_time

            # Step 6: Speaker Clustering
            logger.info(f"Clustering speaker segments into {num_speakers} clusters...")
            if include_timing:
                start_time = time.time()
            segment_clusters = self.clustering.cluster_segments(segment_embeddings, n_clusters=min(segment_embeddings.shape[0], num_speakers))
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
        waveform_duration: float,
        vad_segments: list[dict]
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
            'vad_segments': vad_segments,
            'overlap_regions': overlap_regions,
            'segments': segments_with_text
        }

if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description="WhoSays - Speaker diarization pipeline")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to process")
    parser.add_argument("--num-speakers", type=int, default=2, help="Expected number of speakers (default: 2)")
    parser.add_argument("--annotation", type=Path, help="Path to gold-standard annotation JSON for metrics evaluation (optional)")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file path (optional)")
    parser.add_argument("--pretty", action="store_true", help="Pretty print the output")
    parser.add_argument("--timing", action="store_true", help="Include timing metrics for each model run")

    args = parser.parse_args()

    if not args.audio_file.exists():
        parser.error(f"Audio file not found: {args.audio_file}")

    pipeline = WhoSays()

    logger.info(f"Processing: {args.audio_file}, num speakers: {args.num_speakers}")

    result = pipeline(
        args.audio_file,
        args.num_speakers,
        include_timing=args.timing
    )

    # Compute metrics if annotation file provided
    metrics = None
    if args.annotation:
        if not args.annotation.exists():
            logger.warning(f"Annotation file not found: {args.annotation}. Skipping metrics computation.")
        else:
            try:
                logger.info(f"Loading annotation from {args.annotation}")
                annotation_data = load_annotation_file(args.annotation)

                logger.info("Computing metrics...")
                metrics = evaluate_pipeline(result, annotation_data)

                result['metrics'] = metrics

                logger.info("Metrics computation complete!")
            except Exception as e:
                logger.error(f"Error computing metrics: {e}")
                metrics = None

    # Display summary
    print("\n" + "="*60)
    print("DIARIZATION RESULTS")
    print("="*60)
    print(f"VAD Model: silero")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"VAD segments: {len(result['vad_segments'])}")
    print(f"Detected speakers: {result['num_speakers']}")
    print(f"Speaker segments: {len(result['segments'])}")
    print(f"Overlap regions: {result['num_overlaps']}")

    if result['overlap_regions']:
        print("\n" + "-"*60)
        print("OVERLAPPING SPEECH REGIONS")
        print("-"*60)
        for overlap in result['overlap_regions']:
            print(f"  [{overlap['start']:.2f}s - {overlap['end']:.2f}s] ({overlap['duration']:.2f}s)")

    print("\n" + "-"*60)
    print("SPEAKER TIMELINE")
    print("-"*60)

    for segment in result['segments']:
        speaker = segment['speaker']
        start = segment['start']
        end = segment['end']
        text = segment['text'] if segment['text'] else "[no speech detected]"
        overlap_marker = " [OVERLAP]" if segment.get('has_overlap') else ""
        print(f"\n[{start:.2f}s - {end:.2f}s] {speaker}{overlap_marker}")
        print(f"  {text}")

    print("\n" + "="*60)

    if metrics:
        print(format_metrics_report(metrics))

    # Display timing if available
    if 'timing' in result:
        print(format_timing_report(result['timing'], result['total_time']))

    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2 if args.pretty else None, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")

    # Print JSON if pretty flag is set
    if args.pretty and not args.output:
        print("\nJSON OUTPUT:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    logger.info("Done!")

