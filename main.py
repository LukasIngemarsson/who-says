from loguru import logger
import argparse
from pathlib import Path
import time
import numpy as np

import torch
import torch.nn.functional as F

from dotenv import load_dotenv

from pipeline.ASR import ASR
from pipeline.speaker_segmentation import SO, SCD, SileroVAD
from pipeline.speaker_recognition import SklearnClustering, CosineSimilarityClustering
from pipeline.speaker_recognition import SpeechBrainEmbedding
from pipeline.speaker_recognition import SpeechBrainSpeakerRecognition
from pipeline.phoene import SpeechBrainPhoneme
from utils import load_audio_from_file
from utils import load_annotation_file, evaluate_pipeline, format_metrics_report, format_timing_report
from config import PipelineConfig as Config

load_dotenv(".env")

class WhoSays(object):
    def __init__(self, config=Config()):
        self.config = config
        
        logger.info(f"Using device: {self.config.device}")
        
        self.sod = SO(self.config.so)
        self.scd = SCD(**self.config.scd.get_config().to_dict(), scd_type=self.config.scd.scd_type)
        
        self.vad = SileroVAD(**self.config.vad.silero.to_dict())
        self.asr = ASR(**self.config.asr.to_dict())
        self.phoneme = SpeechBrainPhoneme(**self.config.phoneme.speechbrain.to_dict())

        self.embedder = SpeechBrainEmbedding(**self.config.embedding.speechbrain.to_dict())

        # Initialize clustering based on config type
        from config import TypeClustering
        clustering_config = self.config.clustering.get_config()
        if self.config.clustering.clustering_type == TypeClustering.COSINE_SIMILARITY:
            self.clustering = CosineSimilarityClustering(**clustering_config.to_dict())
        else:
            self.clustering = SklearnClustering(**clustering_config.to_dict())

        self.recognition = SpeechBrainSpeakerRecognition(**self.config.recognition.speechbrain.to_dict())

    def get_reference_embedding(self, audio_file: str):
        """
        Helper to generate an embedding for a specific user enrollment file.
        Assumes the file contains only the target speaker.
        """
        waveform, sr = load_audio_from_file(file_path=audio_file, sr=self.config.sr)
        
        # Handle channels (mono)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
            
        waveform = waveform.to(self.config.device)
        
        # We treat the whole file as one segment for enrollment
        # Pass a fake 'change_point' that covers the whole duration
        duration = waveform.shape[-1] / sr
        fake_change_points = [duration] 
        
        # Get embedding
        emb = self.embedder.embed_segments(waveform, sr, fake_change_points)
        
        # Return the first (and only) embedding vector
        return emb[0]

    def _identify_clusters(self, segment_embeddings, clusters, known_speakers, threshold=0.5):
        """
        Compare cluster centroids to known speaker embeddings.
        Returns a dictionary mapping Cluster ID (int) -> Name (str).
        """
        cluster_mapping = {}
        unique_clusters = set(clusters.tolist())

        # Pre-process known speakers into a tensor for batch comparison
        if not known_speakers:
            return {c_id: f"SPEAKER_{c_id}" for c_id in unique_clusters}

        # known_names = list(known_speakers.keys())
        # known_embs = torch.stack(list(known_speakers.values()))

        for c_id in unique_clusters:
            # 1. Find indices of all segments belonging to this cluster
            indices = [i for i, x in enumerate(clusters) if x == c_id]
            
            # 2. Calculate the average embedding (Centroid) for this cluster
            # segment_embeddings is likely a Tensor or Numpy array. 
            # If Tensor:
            cluster_embs = segment_embeddings[indices]
            centroid = torch.mean(cluster_embs, dim=0)

            # 3. Compare Centroid vs Known Speakers
            best_score = -1.0
            best_name = f"SPEAKER_{c_id}"

            for name, ref_emb in known_speakers.items():
                # Cosine Similarity
                score = F.cosine_similarity(centroid.unsqueeze(0), ref_emb.unsqueeze(0))
                score = score.item()
                
                if score > best_score:
                    best_score = score
                    if score > threshold:
                        best_name = name
            
            cluster_mapping[c_id] = best_name
            logger.info(f"Cluster {c_id} identified as '{best_name}' (Score: {best_score:.4f})")

        return cluster_mapping

    def _process_overlap_regions(
        self,
        separated_regions: dict,
        overlap_embedding_info: list[dict],
        overlap_clusters: torch.Tensor,
        all_embeddings: torch.Tensor,
        all_clusters: torch.Tensor,
        cluster_names: dict,
        sr: int,
        min_confidence: float = 0.7
    ) -> list[dict]:
        """
        Process separated overlap regions: transcribe and assign speakers from clustering.

        Args:
            separated_regions: Dict mapping (start, end) to {speaker_idx: waveform}
            overlap_embedding_info: List of {region, speaker_idx, embedding_idx} for overlap embeddings
            overlap_clusters: Cluster assignments for overlap embeddings (from main clustering)
            all_embeddings: All embeddings (main + overlap) used in clustering
            all_clusters: All cluster assignments
            cluster_names: Mapping from cluster ID to speaker name
            sr: Sample rate
            min_confidence: Minimum cosine similarity to cluster centroid to accept assignment

        Returns:
            List of processed overlap regions with transcriptions and speaker IDs
        """
        processed_overlaps = []

        # Compute cluster centroids for confidence scoring
        cluster_centroids = {}
        unique_clusters = set(all_clusters.tolist())
        for c_id in unique_clusters:
            indices = [i for i, x in enumerate(all_clusters) if x == c_id]
            cluster_embs = all_embeddings[indices]
            cluster_centroids[c_id] = torch.mean(cluster_embs, dim=0)

        # Create lookup from (region, speaker_idx) to (cluster assignment, embedding index)
        overlap_cluster_lookup = {}
        for i, info in enumerate(overlap_embedding_info):
            key = (info['region'], info['speaker_idx'])
            if i < len(overlap_clusters):
                overlap_cluster_lookup[key] = {
                    'cluster_id': int(overlap_clusters[i].item()),
                    'embedding_idx': info['embedding_idx']
                }

        for (start_time, end_time), speaker_waveforms in separated_regions.items():
            overlap_info = {
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'speakers': []
            }

            for spk_idx, spk_waveform in speaker_waveforms.items():
                # Look up cluster assignment from clustering step
                lookup_key = ((start_time, end_time), spk_idx)
                lookup_data = overlap_cluster_lookup.get(lookup_key)

                cluster_id = -1
                confidence = 0.0
                speaker_name = 'UNKNOWN'

                if lookup_data:
                    cluster_id = lookup_data['cluster_id']
                    emb_idx = lookup_data['embedding_idx']

                    # Compute confidence as cosine similarity to cluster centroid
                    if cluster_id in cluster_centroids and emb_idx < all_embeddings.shape[0]:
                        emb = all_embeddings[emb_idx]
                        centroid = cluster_centroids[cluster_id]
                        confidence = F.cosine_similarity(
                            emb.unsqueeze(0), centroid.unsqueeze(0)
                        ).item()

                    # Only accept assignment if confidence meets threshold
                    if confidence >= min_confidence:
                        speaker_name = cluster_names.get(cluster_id, f"SPEAKER_{cluster_id}")
                    else:
                        logger.debug(f"Rejected overlap speaker {spk_idx} assignment (confidence {confidence:.2f} < {min_confidence})")
                        cluster_id = -1

                speaker_data = {
                    'separated_speaker_idx': spk_idx,
                    'speaker': speaker_name,
                    'cluster_id': cluster_id,
                    'confidence': confidence,
                    'transcription': None
                }

                # Ensure waveform is on correct device
                if spk_waveform.device != torch.device(self.config.device):
                    spk_waveform = spk_waveform.to(self.config.device)

                # Transcribe this separated speaker's audio
                try:
                    segments = [{'start': 0.0, 'end': spk_waveform.shape[-1] / sr}]
                    trans = self.asr.transcribe_segments(
                        spk_waveform, segments, return_timestamps=True
                    )
                    if trans:
                        speaker_data['transcription'] = trans[0].get('text', '')
                except Exception as e:
                    logger.warning(f"Failed to transcribe separated speaker {spk_idx}: {e}")

                overlap_info['speakers'].append(speaker_data)

            processed_overlaps.append(overlap_info)

        return processed_overlaps

    def __call__(
        self,
        audio_file: str,
        num_speakers: int = 2,
        include_timing: bool = False,
        known_speakers: dict | None = None
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

            # Speaker Overlap Detection & Separation
            logger.info("Detecting and separating speaker overlaps...")
            if include_timing:
                start_time = time.time()
            overlap_result = self.sod(waveform, sr)
            overlap_segments = overlap_result['overlap_segments']
            separated_regions = overlap_result['separated_regions']
            if include_timing:
                timing['overlap_detection'] = time.time() - start_time
            logger.info(f"Found {len(overlap_segments)} overlapping speech regions")

            # Clear GPU memory after overlap detection before ASR
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

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


            # Step 5: Speaker Change Detection
            logger.info("Detecting speaker change points...")
            if include_timing:
                start_time = time.time()
            change_points = self.scd(waveform)
            if include_timing:
                timing['scd'] = time.time() - start_time
            logger.info(f"Found {len(change_points)} speaker change points")

            # Step 6: Speaker Embedding (main segments + separated overlap audio)
            logger.info("Extracting speaker embeddings...")
            if include_timing:
                start_time = time.time()

            # Embed main segments
            segment_embeddings = self.embedder.embed_segments(waveform, sr, change_points)
            n_main_segments = segment_embeddings.shape[0]

            # Embed separated overlap speakers and track their metadata
            overlap_embedding_info = []  # List of {region: (start, end), speaker_idx: int, embedding_idx: int}
            overlap_embeddings_list = []

            if separated_regions:
                logger.info(f"Embedding {len(separated_regions)} separated overlap regions...")
                for (start_time_ovlp, end_time_ovlp), speaker_waveforms in separated_regions.items():
                    for spk_idx, spk_waveform in speaker_waveforms.items():
                        try:
                            # Ensure waveform is on correct device
                            if spk_waveform.device != torch.device(self.config.device):
                                spk_waveform = spk_waveform.to(self.config.device)

                            # Get embedding for separated speaker
                            duration = spk_waveform.shape[-1] / sr
                            spk_embedding = self.embedder.embed_segments(spk_waveform, sr, [duration])

                            if spk_embedding.shape[0] > 0:
                                overlap_embeddings_list.append(spk_embedding[0])
                                overlap_embedding_info.append({
                                    'region': (start_time_ovlp, end_time_ovlp),
                                    'speaker_idx': spk_idx,
                                    'embedding_idx': n_main_segments + len(overlap_embeddings_list) - 1
                                })
                        except Exception as e:
                            logger.warning(f"Failed to embed separated speaker {spk_idx} in region {start_time_ovlp}-{end_time_ovlp}: {e}")

            # Combine main and overlap embeddings for clustering
            if overlap_embeddings_list:
                overlap_embeddings = torch.stack(overlap_embeddings_list)
                all_embeddings = torch.cat([segment_embeddings, overlap_embeddings], dim=0)
            else:
                all_embeddings = segment_embeddings

            if include_timing:
                timing['embedding'] = time.time() - start_time
            logger.info(f"Extracted embeddings for {n_main_segments} main segments + {len(overlap_embeddings_list)} overlap speakers")

            # Step 7: Speaker Clustering (all embeddings together)
            n_all_segments = all_embeddings.shape[0]
            if num_speakers is not None and n_all_segments < num_speakers:
                logger.warning(f"Only {n_all_segments} segments detected but {num_speakers} speakers requested, using {n_all_segments} clusters")
                num_speakers = n_all_segments

            logger.info(f"Clustering all segments into {num_speakers} clusters...")
            if include_timing:
                start_time = time.time()
            all_clusters = self.clustering.cluster_segments(all_embeddings, n_clusters=num_speakers)
            if include_timing:
                timing['clustering'] = time.time() - start_time

            # Split clusters back into main segments and overlap speakers
            segment_clusters = all_clusters[:n_main_segments]
            overlap_clusters = all_clusters[n_main_segments:] if len(overlap_embeddings_list) > 0 else torch.tensor([])

            logger.info(f"Identified {len(set(all_clusters.tolist()))} speaker clusters")

            logger.info("Identifying speakers against known registry...")
            if known_speakers:
                cluster_names = self._identify_clusters(all_embeddings, all_clusters, known_speakers)
            else:
                cluster_names = {c: f"SPEAKER_{c}" for c in set(all_clusters.tolist())}

            # Step 8: Process overlap regions (transcribe and assign speakers from clustering)
            processed_overlaps = []
            if separated_regions:
                logger.info("Processing separated overlap regions...")
                if include_timing:
                    start_time = time.time()
                processed_overlaps = self._process_overlap_regions(
                    separated_regions=separated_regions,
                    overlap_embedding_info=overlap_embedding_info,
                    overlap_clusters=overlap_clusters,
                    all_embeddings=all_embeddings,
                    all_clusters=all_clusters,
                    cluster_names=cluster_names,
                    sr=sr,
                    min_confidence=self.config.so.min_overlap_confidence
                )
                if include_timing:
                    timing['overlap_processing'] = time.time() - start_time
                logger.info(f"Processed {len(processed_overlaps)} overlap regions")

        # Step 9: Merge results into structured output
        logger.info("Merging results...")
        if include_timing:
            start_time = time.time()
        result = self._format_output(
            change_points=change_points,
            clusters=segment_clusters,
            cluster_names=cluster_names,
            transcriptions=transcriptions,
            overlap_segments=overlap_segments,
            processed_overlaps=processed_overlaps,
            waveform_duration=waveform.shape[-1] / sr,
            vad_segments=speech_segments
        )
        if include_timing:
            timing['formatting'] = time.time() - start_time

        # Add timing to result if requested
        if include_timing:
            result['timing'] = timing
            total_time = sum(timing.values())
            result['total_time'] = total_time
            logger.info(f"Total pipeline time: {total_time:.2f}s")

        result['embeddings'] = segment_embeddings.cpu().numpy()
        result['cluster_labels'] = segment_clusters.cpu().numpy()

        logger.info("Pipeline complete!")
        return result

    def _format_output(
        self,
        change_points: list[float],
        clusters: list[int],
        cluster_names: dict,
        transcriptions: list[dict],
        overlap_segments: list[tuple[float, float]],
        processed_overlaps: list[dict],
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
            processed_overlaps: Processed overlap data with transcriptions and speaker IDs
            waveform_duration: Total audio duration in seconds

        Returns:
            dict: Structured output with speaker segments and transcriptions
        """
        # Create speaker segments from change points and clusters
        speaker_segments = []
        segment_times = [0.0] + change_points + [waveform_duration]

        for i in range(len(segment_times) - 1):
            c_id = int(clusters[i]) if i < len(clusters) else -1
            speaker_name = cluster_names.get(c_id, "UNKNOWN")

            speaker_segments.append({
                'start': segment_times[i],
                'end': segment_times[i + 1],
                'speaker': speaker_name,
                'cluster_id': c_id,
                'duration': segment_times[i + 1] - segment_times[i]
            })

        # Index processed overlaps by time for quick lookup
        overlap_lookup = {}
        for ovlp in processed_overlaps:
            overlap_lookup[(ovlp['start'], ovlp['end'])] = ovlp

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
            overlap_info = None
            for ovlp_start, ovlp_end in overlap_segments:
                # Check if overlap region intersects with this speaker segment
                if max(segment['start'], ovlp_start) < min(segment['end'], ovlp_end):
                    has_overlap = True
                    # Get processed overlap data
                    if (ovlp_start, ovlp_end) in overlap_lookup:
                        processed = overlap_lookup[(ovlp_start, ovlp_end)]
                        overlap_info = {
                            'overlap_start': ovlp_start,
                            'overlap_end': ovlp_end,
                            'speakers': processed.get('speakers', [])
                        }
                    break

            segments_with_text.append({
                **segment,
                'transcriptions': segment_text,
                'text': ' '.join([t['text'] for t in segment_text]).strip(),
                'has_overlap': has_overlap,
                'overlap_info': overlap_info
            })

        # Create final structured output
        return {
            'duration': waveform_duration,
            'num_speakers': len(set(clusters.tolist())),
            'num_overlaps': len(overlap_segments),
            'transcription': transcriptions,
            'speaker_segments': speaker_segments,
            'vad_segments': vad_segments,
            'overlap_regions': processed_overlaps,
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

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(make_serializable(result), f, indent=2 if args.pretty else None, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")

    # Print JSON if pretty flag is set
    if args.pretty and not args.output:
        print("\nJSON OUTPUT:")
        print(json.dumps(make_serializable(result), indent=2, ensure_ascii=False))

    logger.info("Done!")

