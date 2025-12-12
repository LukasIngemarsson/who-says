"""
Comparison utils
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from tqdm import tqdm

from pipeline.speaker_segmentation.VAD.pyannote_vad import PyannoteVAD
from pipeline.speaker_segmentation.SCD import SCD, TypeSCD
from pipeline.speaker_recognition.embedding.speechbrain import SpeechBrainEmbedding
from pipeline.speaker_recognition.embedding._pyannote import PyAnnoteEmbedding
from pipeline.speaker_recognition.embedding.wav2vec2 import Wav2Vec2Embedding
from pipeline.speaker_recognition.clustering.sklearn import SklearnClustering
from pipeline.ASR.faster_whisper import FasterWhisperASR
from utils.audio import load_audio_from_file
from utils.metrics import load_annotation_file, evaluate_segmentation, evaluate_clustering, evaluate_diarization, evaluate_asr
from utils.constants import SR


def get_system_info() -> Dict:
    """Get GPU and system information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device = "cuda"
    else:
        gpu_name = "CPU"
        vram_gb = 0
        device = "cpu"

    return {
        "gpu": gpu_name,
        "vram": f"{vram_gb:.0f} GB" if vram_gb > 0 else "N/A",
        "device": device
    }


def discover_benchmark_files(
    annotation_dir: Path,
    audio_dir: Path,
    limit: int = None
) -> List[Tuple[Path, Path, str]]:
    """
    Find pairs of annotation JSON and audio files using 1-to-1 positional matching.
    First annotation file matches first audio file (alphabetically sorted).

    Returns:
        List of (audio_file, annotation_file, file_id) tuples
    """
    annotation_files = sorted(Path(annotation_dir).glob("*.json"))
    audio_files = sorted(get_audio_files(audio_dir))

    if limit:
        annotation_files = annotation_files[:limit]
        audio_files = audio_files[:limit]

    if len(annotation_files) != len(audio_files):
        logger.warning(
            f"Mismatch: {len(annotation_files)} annotations but {len(audio_files)} audio files"
        )

    file_pairs = []
    for annotation_file, audio_file in zip(annotation_files, audio_files):
        file_id = annotation_file.stem
        file_pairs.append((audio_file, annotation_file, file_id))
        logger.debug(f"Matched: {audio_file.name} <-> {annotation_file.name}")

    logger.info(f"Found {len(file_pairs)} file pairs")
    return file_pairs


def get_audio_files(audio_dir: Path) -> List[Path]:
    """Get all audio files from directory (sorted alphabetically)."""
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(Path(audio_dir).glob(f"*{ext}"))

    return sorted(audio_files)


def change_points_to_segments(
    change_points: List[float],
    labels: np.ndarray,
    duration: float
) -> List[Dict]:
    """
    Convert SCD change points + cluster labels to segment format.

    Parameters
    ----------
    change_points : List[float]
        List of speaker change point timestamps
    labels : np.ndarray
        Cluster labels for each segment
    duration : float
        Total audio duration in seconds

    Returns
    -------
    List[Dict]
        List of segments with start, end, and speaker labels
    """
    boundaries = [0.0] + change_points + [duration]
    segments = []

    for i in range(len(boundaries) - 1):
        if i < len(labels):
            segments.append({
                'start': boundaries[i],
                'end': boundaries[i + 1],
                'speaker': int(labels[i])
            })

    return segments


def handle_dbscan_noise(labels: torch.Tensor) -> torch.Tensor:
    """
    Handle DBSCAN noise labels (-1) by reassigning to new cluster.

    Parameters
    ----------
    labels : torch.Tensor
        Cluster labels potentially containing -1 for noise points

    Returns
    -------
    torch.Tensor
        Labels with noise points reassigned
    """
    if -1 in labels:
        max_label = labels.max().item()
        noise_mask = labels == -1
        labels[noise_mask] = max_label + 1
    return labels


def compare_vad_models(
    file_pairs: List[Tuple[Path, Path, str]]
) -> Dict:
    """
    Compare Silero VAD vs Pyannote VAD.

    Returns:
        Dictionary with results for both models
    """
    from pipeline.speaker_segmentation.VAD.silero import SileroVAD

    silero_vad = SileroVAD()
    pyannote_vad = PyannoteVAD()

    models = {
        'silero': {'model': silero_vad, 'results': []},
        'pyannote': {'model': pyannote_vad, 'results': []}
    }

    for audio_file, annotation_file, file_id in file_pairs:
        logger.info(f"\nProcessing file: {file_id}")

        waveform, sr = load_audio_from_file(audio_file, sr=SR)

        if waveform.dim() > 1:
            if waveform.shape[0] > 1 and waveform.shape[0] < waveform.shape[1]:
                waveform = waveform.mean(dim=0)
            elif waveform.shape[1] > 1 and waveform.shape[1] < waveform.shape[0]:
                waveform = waveform.mean(dim=1)
            else:
                waveform = waveform.squeeze()
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0)

        duration = waveform.shape[-1] / sr

        annotation_data = load_annotation_file(annotation_file)

        for model_name, model_data in models.items():
            logger.info(f"  Testing {model_name}...")

            start_time = time.time()
            segments = model_data['model'](waveform, sample_rate=sr)
            inference_time = time.time() - start_time

            metrics = evaluate_segmentation(
                reference_segments=annotation_data['segments'],
                prediction_segments=segments,
                total_duration=duration
            )

            model_data['results'].append({
                'file_id': file_id,
                'audio_file': str(audio_file),
                'duration': duration,
                'metrics': metrics,
                'timing': inference_time
            })

            logger.info(f"    F1: {metrics['f1']:.2f}%, Time: {inference_time:.2f}s")

    return models


def aggregate_results(models: Dict) -> Dict:
    """
    Compute mean and std for each model's results.
    """
    for model_name, model_data in models.items():
        results = model_data['results']

        precisions = [r['metrics']['precision'] for r in results]
        recalls = [r['metrics']['recall'] for r in results]
        f1s = [r['metrics']['f1'] for r in results]
        timings = [r['timing'] for r in results]

        model_data['aggregated'] = {
            'precision': {
                'mean': float(np.mean(precisions)),
                'std': float(np.std(precisions))
            },
            'recall': {
                'mean': float(np.mean(recalls)),
                'std': float(np.std(recalls))
            },
            'f1': {
                'mean': float(np.mean(f1s)),
                'std': float(np.std(f1s))
            },
            'timing': {
                'mean': float(np.mean(timings)),
                'std': float(np.std(timings)),
                'total': float(np.sum(timings))
            }
        }

    return models


def compare_sc_models(
    file_pairs: List[Tuple[Path, Path, str]]
) -> Dict:
    """
    Compare speaker embedding models and clustering algorithms.

    Tests 3 embedding models (SpeechBrain, PyAnnote, Wav2Vec2) with
    4 clustering algorithms (KMeans, Agglomerative, DBSCAN, Naive/Cosine) = 12 combinations total.

    Returns
    -------
    Dict
        Dictionary with results for all 12 combinations
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.metrics import silhouette_score, adjusted_rand_score, f1_score
    from scipy.optimize import linear_sum_assignment

    scd_model = SCD()

    embedding_models = {}
    try:
        embedding_models['speechbrain'] = SpeechBrainEmbedding()
    except Exception as e:
        logger.error(f"Failed to load SpeechBrain: {e}")

    try:
        embedding_models['pyannote'] = PyAnnoteEmbedding()
    except Exception as e:
        logger.warning(f"PyAnnote requires HF_TOKEN_PYANNOTE_EMBEDDING: {e}")

    try:
        embedding_models['wav2vec2'] = Wav2Vec2Embedding()
    except Exception as e:
        logger.warning(f"Failed to load Wav2Vec2: {e}")

    if not embedding_models:
        logger.error("No embedding models available!")
        return {}

    # Clustering methods: kmeans, agglomerative, dbscan, naive (cosine similarity)
    clustering_methods = ['kmeans', 'agglomerative', 'dbscan', 'naive']

    models = {}
    for emb_name in embedding_models.keys():
        for clus_name in clustering_methods:
            key = f'{emb_name}_{clus_name}'
            models[key] = {'results': []}

    # Helper function to compute F1 with Hungarian matching
    def compute_f1_hungarian(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        mask = pred_labels != -1
        if not np.any(mask):
            return 0.0

        true_filtered = true_labels[mask]
        pred_filtered = pred_labels[mask]

        true_unique = np.unique(true_filtered)
        pred_unique = np.unique(pred_filtered)

        n_true = len(true_unique)
        n_pred = len(pred_unique)

        cost_matrix = np.zeros((n_pred, n_true))
        for i, p in enumerate(pred_unique):
            for j, t in enumerate(true_unique):
                cost_matrix[i, j] = -np.sum((pred_filtered == p) & (true_filtered == t))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        label_map = {pred_unique[r]: true_unique[c] for r, c in zip(row_ind, col_ind)}
        aligned = np.array([label_map.get(p, -1) for p in pred_filtered])

        valid_mask = aligned != -1
        if np.sum(valid_mask) == 0:
            return 0.0

        return f1_score(true_filtered[valid_mask], aligned[valid_mask], average='macro', zero_division=0)

    for audio_file, annotation_file, file_id in file_pairs:
        logger.info(f"\nProcessing file: {file_id}")

        waveform, sr = load_audio_from_file(audio_file, sr=SR)

        if waveform.dim() > 1:
            if waveform.shape[0] > 1 and waveform.shape[0] < waveform.shape[1]:
                waveform = waveform.mean(dim=0)
            elif waveform.shape[1] > 1 and waveform.shape[1] < waveform.shape[0]:
                waveform = waveform.mean(dim=1)
            else:
                waveform = waveform.squeeze()
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0)

        duration = waveform.shape[-1] / sr

        annotation_data = load_annotation_file(annotation_file)

        n_speakers = len(set(seg['speaker'] for seg in annotation_data['segments']))

        change_points = scd_model(waveform, sample_rate=sr)

        if len(change_points) == 0:
            logger.warning(f"  No change points detected for {file_id}, skipping")
            continue

        for emb_name, embedder in embedding_models.items():
            logger.info(f"  Embedding: {emb_name}")

            try:
                start_time = time.time()
                embeddings = embedder.embed_segments(waveform, sr, change_points)
                emb_time = time.time() - start_time
            except Exception as e:
                logger.warning(f"    Failed to extract embeddings for {file_id} with {emb_name}: {e}")
                continue

            if embeddings.shape[0] == 0:
                logger.warning(f"    No embeddings extracted for {file_id}, skipping {emb_name}")
                continue

            if torch.isnan(embeddings).any():
                logger.warning(f"    NaN values in embeddings for {file_id}, skipping {emb_name}")
                continue

            # L2 normalize embeddings for cosine-based methods
            embeddings_np = embeddings.cpu().numpy()
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings_norm = embeddings_np / norms

            # Get ground truth labels for segments (map segments to speakers)
            # This is an approximation - assign each segment to the speaker with most overlap
            boundaries = [0.0] + change_points + [duration]
            true_labels = []
            speaker_map = {}  # Move outside the loop to maintain consistent mapping
            for i in range(len(boundaries) - 1):
                seg_start = boundaries[i]
                seg_end = boundaries[i + 1]
                # Find which ground truth speaker has most overlap with this segment
                best_speaker = 0
                best_overlap = 0
                for gt_seg in annotation_data['segments']:
                    speaker = gt_seg['speaker']
                    if speaker not in speaker_map:
                        speaker_map[speaker] = len(speaker_map)
                    overlap_start = max(seg_start, gt_seg['start'])
                    overlap_end = min(seg_end, gt_seg['end'])
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = speaker_map[speaker]
                true_labels.append(best_speaker)
            true_labels = np.array(true_labels)

            # Trim true_labels to match embeddings if some segments were filtered (e.g., too short)
            n_embeddings = embeddings_norm.shape[0]
            if len(true_labels) > n_embeddings:
                logger.debug(f"      Trimming true_labels from {len(true_labels)} to {n_embeddings} to match embeddings")
                true_labels = true_labels[:n_embeddings]

            for clus_name in clustering_methods:
                logger.info(f"    Clustering: {clus_name}")

                start_time = time.time()

                if clus_name == 'kmeans':
                    n_clus = min(n_speakers, embeddings_norm.shape[0])
                    clusterer = KMeans(n_clusters=n_clus, random_state=42, n_init=10)
                    labels = clusterer.fit_predict(embeddings_norm)
                    # Silhouette requires: 2 <= n_labels <= n_samples - 1
                    n_unique = len(set(labels))
                    if n_unique > 1 and n_unique < embeddings_norm.shape[0]:
                        silhouette = silhouette_score(embeddings_norm, labels, metric='cosine')
                    else:
                        silhouette = 0.0

                elif clus_name == 'agglomerative':
                    n_clus = min(n_speakers, embeddings_norm.shape[0])
                    clusterer = AgglomerativeClustering(n_clusters=n_clus, metric='cosine', linkage='average')
                    labels = clusterer.fit_predict(embeddings_norm)
                    # Silhouette requires: 2 <= n_labels <= n_samples - 1
                    n_unique = len(set(labels))
                    if n_unique > 1 and n_unique < embeddings_norm.shape[0]:
                        silhouette = silhouette_score(embeddings_norm, labels, metric='cosine')
                    else:
                        silhouette = 0.0

                elif clus_name == 'dbscan':
                    # DBSCAN with eps tuning
                    best_eps = 0.3
                    best_ari = -1
                    best_labels = None
                    for eps in np.linspace(0.1, 1.0, 10):
                        dbscan = DBSCAN(eps=eps, min_samples=2, metric='cosine')
                        dbscan_labels = dbscan.fit_predict(embeddings_norm)
                        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                        if n_clusters >= 2:
                            ari = adjusted_rand_score(true_labels[:len(dbscan_labels)], dbscan_labels)
                            if ari > best_ari:
                                best_ari = ari
                                best_eps = eps
                                best_labels = dbscan_labels

                    if best_labels is not None:
                        labels = best_labels
                        mask = labels != -1
                        if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
                            silhouette = silhouette_score(embeddings_norm[mask], labels[mask], metric='cosine')
                        else:
                            silhouette = 0.0
                    else:
                        # Fallback if no good DBSCAN found
                        labels = np.zeros(embeddings_norm.shape[0], dtype=int)
                        silhouette = 0.0

                elif clus_name == 'naive':
                    # Naive Cosine Similarity with Leave-One-Out
                    # For each embedding, compute centroids from OTHER samples only,
                    # then assign to closest centroid. This avoids circular evaluation.
                    unique_true = np.unique(true_labels)
                    labels = []

                    for i, emb in enumerate(embeddings_norm):
                        # Compute centroids excluding the current sample
                        centroids = {}
                        for speaker_idx in unique_true:
                            # Get mask for this speaker, excluding current sample
                            speaker_mask = (true_labels == speaker_idx)
                            speaker_mask[i] = False  # Exclude current sample
                            if np.any(speaker_mask):
                                centroid = embeddings_norm[speaker_mask].mean(axis=0)
                                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                                centroids[speaker_idx] = centroid

                        # Assign to closest centroid
                        best_sim = -1
                        best_speaker = 0
                        for speaker_idx, centroid in centroids.items():
                            sim = np.dot(emb, centroid)
                            if sim > best_sim:
                                best_sim = sim
                                best_speaker = speaker_idx
                        labels.append(best_speaker)

                    labels = np.array(labels)
                    # No silhouette for naive method (uses ground truth structure)
                    silhouette = None

                clus_time = time.time() - start_time

                # Convert labels to tensor for compatibility
                labels_tensor = torch.tensor(labels)
                labels_tensor = handle_dbscan_noise(labels_tensor)
                labels = labels_tensor.numpy()

                if len(labels) != len(change_points) + 1:
                    logger.warning(f"      Label mismatch: {len(labels)} labels, {len(change_points)+1} segments")
                    min_len = min(len(labels), len(change_points) + 1)
                    labels = labels[:min_len]
                    true_labels_trimmed = true_labels[:min_len]
                else:
                    true_labels_trimmed = true_labels

                pred_segments = change_points_to_segments(
                    change_points[:len(labels)-1] if len(labels) < len(change_points) + 1 else change_points,
                    labels,
                    duration
                )

                # Compute ARI and F1
                ari = adjusted_rand_score(true_labels_trimmed, labels)
                f1 = compute_f1_hungarian(true_labels_trimmed, labels)

                der_metrics = evaluate_diarization(
                    annotation_data['segments'],
                    pred_segments,
                    total_duration=duration
                )

                key = f'{emb_name}_{clus_name}'
                models[key]['results'].append({
                    'file_id': file_id,
                    'audio_file': str(audio_file),
                    'duration': duration,
                    'n_speakers': n_speakers,
                    'n_change_points': len(change_points),
                    'metrics': {
                        'silhouette': silhouette if silhouette is not None else 0.0,
                        'ari': ari,
                        'f1': f1,
                        'der': der_metrics['der'],
                        'miss': der_metrics['miss'],
                        'false_alarm': der_metrics['false_alarm'],
                        'confusion': der_metrics['confusion']
                    },
                    'timing': {
                        'embedding': emb_time,
                        'clustering': clus_time
                    }
                })

                sil_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
                logger.info(f"      ARI: {ari:.3f}, F1: {f1:.3f}, Silhouette: {sil_str}, DER: {der_metrics['der']:.2f}%")

    return models


def aggregate_sc_results(models: Dict) -> Dict:
    """
    Compute mean and std for SC model results.
    """
    for model_name, model_data in models.items():
        results = model_data['results']

        if not results:
            logger.warning(f"No results for {model_name}, skipping aggregation")
            continue

        model_data['aggregated'] = {
            'silhouette': {
                'mean': float(np.mean([r['metrics']['silhouette'] for r in results])),
                'std': float(np.std([r['metrics']['silhouette'] for r in results]))
            },
            'ari': {
                'mean': float(np.mean([r['metrics']['ari'] for r in results])),
                'std': float(np.std([r['metrics']['ari'] for r in results]))
            },
            'f1': {
                'mean': float(np.mean([r['metrics']['f1'] for r in results])),
                'std': float(np.std([r['metrics']['f1'] for r in results]))
            },
            'der': {
                'mean': float(np.mean([r['metrics']['der'] for r in results])),
                'std': float(np.std([r['metrics']['der'] for r in results]))
            },
            'miss': {
                'mean': float(np.mean([r['metrics']['miss'] for r in results])),
                'std': float(np.std([r['metrics']['miss'] for r in results]))
            },
            'false_alarm': {
                'mean': float(np.mean([r['metrics']['false_alarm'] for r in results])),
                'std': float(np.std([r['metrics']['false_alarm'] for r in results]))
            },
            'confusion': {
                'mean': float(np.mean([r['metrics']['confusion'] for r in results])),
                'std': float(np.std([r['metrics']['confusion'] for r in results]))
            },
            'embedding_timing': {
                'mean': float(np.mean([r['timing']['embedding'] for r in results])),
                'std': float(np.std([r['timing']['embedding'] for r in results])),
                'total': float(np.sum([r['timing']['embedding'] for r in results]))
            },
            'clustering_timing': {
                'mean': float(np.mean([r['timing']['clustering'] for r in results])),
                'std': float(np.std([r['timing']['clustering'] for r in results])),
                'total': float(np.sum([r['timing']['clustering'] for r in results]))
            }
        }

    return models


def compare_asr_models(
    file_pairs: List[Tuple[Path, Path, str]]
) -> Dict:
    """
    Compare ASR models with different sizes.

    Tests 7 FasterWhisper models from tiny to large-v3.

    Returns
    -------
    Dict
        Dictionary with results for all 7 models
    """
    models_to_test = [
        'tiny',
        'base',
        'small',
        'medium',
        'large-v3',
        'large-v3-turbo',
        'distil-large-v3'
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    models = {}
    for model_name in models_to_test:
        models[model_name] = {'results': []}

    for audio_file, annotation_file, file_id in file_pairs:
        logger.info(f"\nProcessing file: {file_id}")

        waveform, sr = load_audio_from_file(audio_file, sr=SR)

        if waveform.dim() > 1:
            if waveform.shape[0] > 1 and waveform.shape[0] < waveform.shape[1]:
                waveform = waveform.mean(dim=0)
            elif waveform.shape[1] > 1 and waveform.shape[1] < waveform.shape[0]:
                waveform = waveform.mean(dim=1)
            else:
                waveform = waveform.squeeze()
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0)

        annotation_data = load_annotation_file(annotation_file)

        reference_transcriptions = [seg['text'] for seg in annotation_data['segments']]

        for model_name in models_to_test:
            logger.info(f"  Testing {model_name}...")

            asr = FasterWhisperASR(
                model=model_name,
                device=device,
                compute_type=compute_type
            )

            start_time = time.time()
            result = asr.transcribe(waveform, language="en", return_timestamps=False, word_timestamps=False)
            inference_time = time.time() - start_time

            hypothesis_transcriptions = [result['text']]

            metrics = evaluate_asr(reference_transcriptions, hypothesis_transcriptions)

            models[model_name]['results'].append({
                'file_id': file_id,
                'audio_file': str(audio_file),
                'metrics': metrics,
                'timing': inference_time
            })

            logger.info(f"    WER: {metrics['wer']:.2f}%, Time: {inference_time:.2f}s")

    return models


def aggregate_asr_results(models: Dict) -> Dict:
    """
    Compute mean and std for ASR model results.
    """
    for model_name, model_data in models.items():
        results = model_data['results']

        if not results:
            logger.warning(f"No results for {model_name}, skipping aggregation")
            continue

        wers = [r['metrics']['wer'] for r in results]
        timings = [r['timing'] for r in results]

        model_data['aggregated'] = {
            'wer': {
                'mean': float(np.mean(wers)),
                'std': float(np.std(wers))
            },
            'timing': {
                'mean': float(np.mean(timings)),
                'std': float(np.std(timings)),
                'total': float(np.sum(timings))
            }
        }

    return models


def compare_e2e_pipelines(file_pairs: List[Tuple[Path, Path, str]]) -> Dict:
    """
    Compare end-to-end diarization pipelines.

    Evaluates complete pipeline output using DER and timing metrics.

    Args:
        file_pairs: List of (audio_path, annotation_path, file_id) tuples

    Returns:
        Dict mapping pipeline names to results:
        {
            'pipeline_name': {
                'has_transcription': bool,
                'results': [
                    {
                        'file_id': str,
                        'audio_file': str,
                        'duration': float,
                        'n_speakers_pred': int,
                        'n_speakers_ref': int,
                        'der_metrics': {...},
                        'wer_metrics': {...} or None,
                        'timing': float
                    }
                ]
            }
        }
    """
    from pipeline.pyannote_full_pipeline import PyannoteFullPipeline
    from main import WhoSays

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipelines = {
        'who-says': {
            'instance': WhoSays(),
            'has_transcription': True
        },
        'pyannote-3.1': {
            'instance': PyannoteFullPipeline(device=device),
            'has_transcription': False
        }
    }

    for name in pipelines:
        pipelines[name]['results'] = []

    for audio_file, annotation_file, file_id in tqdm(file_pairs, desc="Processing files"):
        logger.info(f"\nProcessing: {file_id}")

        annotation_data = load_annotation_file(annotation_file)
        waveform, sr = load_audio_from_file(audio_file, sr=SR)
        duration = waveform.shape[-1] / sr

        ref_segments = annotation_data['segments']
        ref_transcriptions = annotation_data.get('transcriptions')
        n_speakers_ref = len(set(seg['speaker'] for seg in ref_segments))

        for pipeline_name, pipeline_data in pipelines.items():
            pipeline = pipeline_data['instance']
            has_transcription = pipeline_data['has_transcription']

            logger.info(f"  Running {pipeline_name}...")

            try:
                start_time = time.time()

                if pipeline_name == 'who-says':
                    result = pipeline(
                        str(audio_file),
                        num_speakers=n_speakers_ref,
                        include_timing=True,
                        return_diarization_time=True
                    )
                    inference_time = result.get('diarization_time', time.time() - start_time)
                else:
                    result = pipeline.process(audio_file)
                    inference_time = time.time() - start_time

                pred_segments = result['segments']
                n_speakers_pred = len(set(seg['speaker'] for seg in pred_segments))

                der_metrics = evaluate_diarization(
                    reference_segments=ref_segments,
                    hypothesis_segments=pred_segments,
                    total_duration=duration
                )

                wer_metrics = None
                if has_transcription and ref_transcriptions:
                    pred_transcriptions = [seg['text'] for seg in pred_segments]
                    wer_metrics = evaluate_asr(ref_transcriptions, pred_transcriptions)

                pipeline_data['results'].append({
                    'file_id': file_id,
                    'audio_file': str(audio_file),
                    'duration': duration,
                    'n_speakers_pred': n_speakers_pred,
                    'n_speakers_ref': n_speakers_ref,
                    'der_metrics': der_metrics,
                    'wer_metrics': wer_metrics,
                    'timing': inference_time
                })

                logger.info(f"    DER: {der_metrics['der']:.2f}%")
                if wer_metrics:
                    logger.info(f"    WER: {wer_metrics['wer']:.2f}%")
                logger.info(f"    Time: {inference_time:.2f}s")

            except Exception as e:
                logger.error(f"    Error with {pipeline_name}: {e}")
                continue

    return pipelines


def aggregate_e2e_results(pipelines: Dict) -> Dict:
    """
    Aggregate end-to-end pipeline results.

    Computes mean and std for DER components and timing.

    Args:
        pipelines: Dict from compare_e2e_pipelines()

    Returns:
        Updated dict with 'aggregated' key for each pipeline
    """
    for pipeline_name, pipeline_data in pipelines.items():
        results = pipeline_data['results']

        if not results:
            logger.warning(f"No results for {pipeline_name}, skipping aggregation")
            continue

        ders = [r['der_metrics']['der'] for r in results]
        misses = [r['der_metrics']['miss'] for r in results]
        false_alarms = [r['der_metrics']['false_alarm'] for r in results]
        confusions = [r['der_metrics']['confusion'] for r in results]
        timings = [r['timing'] for r in results]

        aggregated = {
            'der': {'mean': float(np.mean(ders)), 'std': float(np.std(ders))},
            'miss': {'mean': float(np.mean(misses)), 'std': float(np.std(misses))},
            'false_alarm': {'mean': float(np.mean(false_alarms)), 'std': float(np.std(false_alarms))},
            'confusion': {'mean': float(np.mean(confusions)), 'std': float(np.std(confusions))},
            'timing': {
                'mean': float(np.mean(timings)),
                'std': float(np.std(timings)),
                'total': float(np.sum(timings))
            }
        }

        if pipeline_data['has_transcription'] and results[0]['wer_metrics']:
            wers = [r['wer_metrics']['wer'] for r in results if r['wer_metrics']]
            if wers:
                aggregated['wer'] = {'mean': float(np.mean(wers)), 'std': float(np.std(wers))}

        pipeline_data['aggregated'] = aggregated

    return pipelines


# =============================================================================
# SCD (Speaker Change Detection) Comparison Functions
# =============================================================================

def load_scd_ground_truth(benchmark_dir: Path) -> Tuple[int, List[float]]:
    """
    Load ground truth speaker changes from benchmark annotations.

    Parameters
    ----------
    benchmark_dir : Path
        Directory containing benchmark JSON files (30s chunks named 0.json, 1.json, etc.)

    Returns
    -------
    Tuple[int, List[float]]
        Total number of speaker changes and list of change point timestamps
    """
    total_changes = 0
    all_change_times = []

    for f in sorted(benchmark_dir.glob("*.json")):
        chunk_idx = int(f.stem)
        chunk_offset = chunk_idx * 30.0  # 30 second chunks

        with open(f) as fp:
            data = json.load(fp)

        segs = data['segments']
        for i in range(1, len(segs)):
            if segs[i]['speaker'] != segs[i-1]['speaker']:
                total_changes += 1
                all_change_times.append(chunk_offset + segs[i]['start'])

    return total_changes, sorted(all_change_times)


def match_change_points(
    detected: List[float],
    ground_truth: List[float],
    tolerance: float = 2.0
) -> Tuple[float, float, float]:
    """
    Match detected change points to ground truth within tolerance.

    Parameters
    ----------
    detected : List[float]
        Detected change point timestamps
    ground_truth : List[float]
        Ground truth change point timestamps
    tolerance : float
        Maximum time difference for a match (seconds)

    Returns
    -------
    Tuple[float, float, float]
        Precision, recall, F1 score
    """
    if not detected or not ground_truth:
        if not detected and not ground_truth:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    detected = sorted(detected)
    ground_truth = sorted(ground_truth)

    matched_gt = set()
    matched_det = set()

    for i, gt in enumerate(ground_truth):
        for j, det in enumerate(detected):
            if j not in matched_det and abs(det - gt) <= tolerance:
                matched_gt.add(i)
                matched_det.add(j)
                break

    true_positives = len(matched_gt)
    precision = true_positives / len(detected) if detected else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def compare_scd_models(
    audio_path: str,
    benchmark_dir: Path,
    prominence_values: List[float] = None,
    tolerance_values: List[float] = None,
    include_nemo: bool = True,
    chunk_duration: float = 300.0
) -> Dict:
    """
    Compare SCD models: Pyannote (with different prominence values) vs NeMo Sortformer.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    benchmark_dir : Path
        Directory containing ground truth annotations
    prominence_values : List[float]
        List of prominence thresholds to test for Pyannote
    tolerance_values : List[float]
        List of tolerance values for matching evaluation
    include_nemo : bool
        Whether to include NeMo Sortformer benchmark
    chunk_duration : float
        Chunk duration for processing (to avoid OOM)

    Returns
    -------
    Dict
        Results for all models and configurations
    """
    from pipeline.speaker_segmentation.SCD._nemo import NemoSCD
    from config import PipelineConfig

    if prominence_values is None:
        prominence_values = [0.1, 0.15, 0.2, 0.25, 0.3]
    if tolerance_values is None:
        tolerance_values = [0.5, 1.0, 2.0, 3.0, 5.0]

    config = PipelineConfig()

    # Load ground truth
    logger.info("Loading ground truth annotations...")
    gt_changes, gt_times = load_scd_ground_truth(benchmark_dir)
    logger.info(f"Ground truth: {gt_changes} speaker changes")

    # Load audio
    logger.info(f"Loading audio from {audio_path}...")
    waveform, sr = load_audio_from_file(file_path=audio_path, sr=config.sr)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    waveform = waveform.to(config.device)
    total_duration = waveform.shape[-1] / sr
    logger.info(f"Audio loaded: {total_duration:.1f} seconds")

    results = {
        'ground_truth_count': gt_changes,
        'ground_truth_times': gt_times,
        'total_duration': total_duration,
        'pyannote': [],
        'nemo': None
    }

    # Test Pyannote with different prominence values
    for prominence in prominence_values:
        logger.info(f"Testing Pyannote SCD (prominence={prominence})...")

        scd = SCD(
            scd_type=TypeSCD.PYANNOTE,
            device=config.device,
            model=config.scd.pyannote.model,
            min_prominence=prominence,
            min_duration=config.scd.pyannote.min_duration
        )

        start_time = time.time()
        with torch.no_grad():
            change_points = scd(waveform)
        elapsed = time.time() - start_time

        # Evaluate at all tolerance levels
        tolerance_results = []
        for tol in tolerance_values:
            precision, recall, f1 = match_change_points(change_points, gt_times, tolerance=tol)
            tolerance_results.append({
                'tolerance': tol,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        results['pyannote'].append({
            'prominence': prominence,
            'detected_count': len(change_points),
            'change_points': change_points,
            'time': elapsed,
            'results_by_tolerance': tolerance_results
        })

        logger.info(f"  Detected: {len(change_points)}, Time: {elapsed:.2f}s")

    # Test NeMo Sortformer
    if include_nemo:
        logger.info("Testing NeMo Sortformer SCD...")
        try:
            nemo_scd = NemoSCD(device=torch.device(config.device))

            # Process in chunks
            chunk_samples = int(chunk_duration * sr)
            num_chunks = int(np.ceil(waveform.shape[-1] / chunk_samples))
            waveform_cpu = waveform.cpu()

            all_change_points = []
            start_time = time.time()

            for i in range(num_chunks):
                chunk_start = i * chunk_samples
                chunk_end = min((i + 1) * chunk_samples, waveform_cpu.shape[-1])
                chunk_waveform = waveform_cpu[chunk_start:chunk_end]
                chunk_offset = chunk_start / sr

                with torch.no_grad():
                    chunk_changes = nemo_scd(chunk_waveform, sample_rate=sr)

                for cp in chunk_changes:
                    all_change_points.append(cp + chunk_offset)

            elapsed = time.time() - start_time
            change_points = sorted(all_change_points)

            # Evaluate at all tolerance levels
            tolerance_results = []
            for tol in tolerance_values:
                precision, recall, f1 = match_change_points(change_points, gt_times, tolerance=tol)
                tolerance_results.append({
                    'tolerance': tol,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

            results['nemo'] = {
                'detected_count': len(change_points),
                'change_points': change_points,
                'time': elapsed,
                'results_by_tolerance': tolerance_results
            }

            logger.info(f"  Detected: {len(change_points)}, Time: {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"NeMo SCD failed: {e}")
            results['nemo'] = {'error': str(e)}

    return results


def aggregate_scd_results(results: Dict) -> Dict:
    """Aggregate SCD comparison results with best configs."""
    # Find best Pyannote config for each tolerance
    best_pyannote = {}
    for tol_result in results['pyannote'][0]['results_by_tolerance']:
        tol = tol_result['tolerance']
        best_f1 = 0
        best_config = None

        for pya_result in results['pyannote']:
            for tr in pya_result['results_by_tolerance']:
                if tr['tolerance'] == tol and tr['f1'] > best_f1:
                    best_f1 = tr['f1']
                    best_config = {
                        'prominence': pya_result['prominence'],
                        'f1': tr['f1'],
                        'precision': tr['precision'],
                        'recall': tr['recall']
                    }

        best_pyannote[f'tolerance_{tol}s'] = best_config

    results['aggregated'] = {
        'best_pyannote_by_tolerance': best_pyannote
    }

    if results['nemo'] and 'error' not in results['nemo']:
        results['aggregated']['nemo_by_tolerance'] = {
            f"tolerance_{tr['tolerance']}s": {
                'f1': tr['f1'],
                'precision': tr['precision'],
                'recall': tr['recall']
            }
            for tr in results['nemo']['results_by_tolerance']
        }

    return results


# =============================================================================
# SOD (Speech Overlap Detection) Comparison Functions
# =============================================================================

def extract_overlap_ground_truth(
    benchmark_dir: Path,
    min_overlap_duration: float = 0.05
) -> List[Tuple[float, float]]:
    """
    Extract overlap regions from benchmark JSONs by finding
    overlapping timestamps between different speakers.

    Parameters
    ----------
    benchmark_dir : Path
        Directory containing benchmark JSON annotation files
    min_overlap_duration : float
        Minimum overlap duration in seconds

    Returns
    -------
    List[Tuple[float, float]]
        List of (start, end) tuples representing overlap regions
    """
    all_overlaps = []

    for f in sorted(benchmark_dir.glob("*.json")):
        chunk_idx = int(f.stem)
        chunk_offset = chunk_idx * 30.0

        with open(f) as fp:
            data = json.load(fp)

        segments = data['segments']

        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments):
                if i >= j:
                    continue
                if seg1['speaker'] != seg2['speaker']:
                    overlap_start = max(seg1['start'], seg2['start'])
                    overlap_end = min(seg1['end'], seg2['end'])
                    overlap_duration = overlap_end - overlap_start

                    if overlap_duration >= min_overlap_duration:
                        abs_start = chunk_offset + overlap_start
                        abs_end = chunk_offset + overlap_end
                        all_overlaps.append((abs_start, abs_end))

    # Merge overlapping regions
    all_overlaps = merge_overlapping_segments(all_overlaps)
    all_overlaps = [(s, e) for s, e in all_overlaps if (e - s) >= min_overlap_duration]

    return all_overlaps


def merge_overlapping_segments(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping or adjacent segments into contiguous regions."""
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segs[0]]

    for start, end in sorted_segs[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def overlaps_to_frames(
    overlaps: List[Tuple[float, float]],
    total_duration: float,
    frame_size: float = 0.01
) -> np.ndarray:
    """Convert overlap segments to frame-level binary array."""
    num_frames = int(np.ceil(total_duration / frame_size))
    frames = np.zeros(num_frames, dtype=np.int32)

    for start, end in overlaps:
        start_frame = int(start / frame_size)
        end_frame = min(int(end / frame_size), num_frames)
        frames[start_frame:end_frame] = 1

    return frames


def compute_frame_metrics(ref_frames: np.ndarray, pred_frames: np.ndarray) -> Dict[str, float]:
    """Compute frame-level precision, recall, F1."""
    max_len = max(len(ref_frames), len(pred_frames))
    if len(ref_frames) < max_len:
        ref_frames = np.pad(ref_frames, (0, max_len - len(ref_frames)))
    if len(pred_frames) < max_len:
        pred_frames = np.pad(pred_frames, (0, max_len - len(pred_frames)))

    tp = np.sum((ref_frames == 1) & (pred_frames == 1))
    fp = np.sum((ref_frames == 0) & (pred_frames == 1))
    fn = np.sum((ref_frames == 1) & (pred_frames == 0))

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_segment_iou(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
    """Compute Intersection over Union (IoU) between two segments."""
    intersection_start = max(seg1[0], seg2[0])
    intersection_end = min(seg1[1], seg2[1])
    intersection = max(0, intersection_end - intersection_start)
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - intersection
    return intersection / union if union > 0 else 0.0


def compute_segment_metrics(
    ref_segments: List[Tuple[float, float]],
    pred_segments: List[Tuple[float, float]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Compute segment-level precision, recall, F1 using IoU matching."""
    if not ref_segments and not pred_segments:
        return {'precision': 100.0, 'recall': 100.0, 'f1': 100.0}
    if not pred_segments or not ref_segments:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    matched_pred = set()
    matched_ref = set()

    for i, pred in enumerate(pred_segments):
        for j, ref in enumerate(ref_segments):
            if j not in matched_ref:
                iou = compute_segment_iou(pred, ref)
                if iou >= iou_threshold:
                    matched_pred.add(i)
                    matched_ref.add(j)
                    break

    tp = len(matched_pred)
    precision = tp / len(pred_segments) * 100 if pred_segments else 0.0
    recall = tp / len(ref_segments) * 100 if ref_segments else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def compare_sod_models(
    audio_path: str,
    benchmark_dir: Path,
    onset_thresholds: List[float] = None,
    include_nemo: bool = True,
    min_overlap_duration: float = 0.1
) -> Dict:
    """
    Compare SOD models: PyannoteSOD vs NeMo Sortformer (via diarization).

    Parameters
    ----------
    audio_path : str
        Path to audio file
    benchmark_dir : Path
        Directory containing ground truth annotations
    onset_thresholds : List[float]
        Onset thresholds to test for Pyannote
    include_nemo : bool
        Whether to include NeMo benchmark
    min_overlap_duration : float
        Minimum overlap duration for ground truth

    Returns
    -------
    Dict
        Results for all models
    """
    from pipeline.speaker_segmentation.SO.Detection import PyannoteSOD
    from pipeline.speaker_segmentation.SCD._nemo import NemoSCD
    from config import PipelineConfig

    if onset_thresholds is None:
        onset_thresholds = [0.3, 0.5, 0.7]

    config = PipelineConfig()

    # Load ground truth
    gt_overlaps = extract_overlap_ground_truth(benchmark_dir, min_overlap_duration)
    logger.info(f"Ground truth: {len(gt_overlaps)} overlap regions")

    # Load audio
    waveform, sr = load_audio_from_file(file_path=audio_path, sr=config.sr)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    waveform = waveform.to(config.device)
    total_duration = waveform.shape[-1] / sr

    gt_frames = overlaps_to_frames(gt_overlaps, total_duration)
    gt_total_time = sum(end - start for start, end in gt_overlaps)

    results = {
        'ground_truth_count': len(gt_overlaps),
        'ground_truth_duration': gt_total_time,
        'total_audio_duration': total_duration,
        'pyannote_results': [],
        'nemo_results': None
    }

    # Test Pyannote with different thresholds
    for onset in onset_thresholds:
        logger.info(f"Testing PyannoteSOD (onset={onset})...")

        sod = PyannoteSOD(
            model_name="pyannote/segmentation-3.0",
            onset=onset,
            offset=onset,
            min_duration=0.0,
            device=torch.device(config.device)
        )

        start_time = time.time()
        with torch.no_grad():
            detected_overlaps = sod(waveform, sample_rate=sr)
        elapsed = time.time() - start_time

        pred_frames = overlaps_to_frames(detected_overlaps, total_duration)
        frame_metrics = compute_frame_metrics(gt_frames, pred_frames)
        seg_metrics_05 = compute_segment_metrics(gt_overlaps, detected_overlaps, iou_threshold=0.5)
        seg_metrics_03 = compute_segment_metrics(gt_overlaps, detected_overlaps, iou_threshold=0.3)

        detected_total_time = sum(end - start for start, end in detected_overlaps)

        results['pyannote_results'].append({
            'onset': onset,
            'detected_count': len(detected_overlaps),
            'detected_duration': detected_total_time,
            'time': elapsed,
            'frame_metrics': frame_metrics,
            'segment_metrics_iou05': seg_metrics_05,
            'segment_metrics_iou03': seg_metrics_03
        })

        logger.info(f"  Detected: {len(detected_overlaps)}, Frame F1: {frame_metrics['f1']:.1f}%")

    # Test NeMo (via diarization overlaps)
    if include_nemo:
        logger.info("Testing NeMo Sortformer (via diarization)...")
        try:
            nemo_scd = NemoSCD(device=torch.device(config.device))
            waveform_cpu = waveform.cpu()

            start_time = time.time()
            with torch.no_grad():
                diar_segments = nemo_scd.get_diarization_segments(waveform_cpu, sample_rate=sr)
            elapsed = time.time() - start_time

            # Extract overlaps from diarization
            nemo_overlaps = []
            for i, seg1 in enumerate(diar_segments):
                for j, seg2 in enumerate(diar_segments):
                    if i >= j:
                        continue
                    if seg1['speaker'] != seg2['speaker']:
                        ov_start = max(seg1['start'], seg2['start'])
                        ov_end = min(seg1['end'], seg2['end'])
                        if ov_start < ov_end:
                            nemo_overlaps.append((ov_start, ov_end))

            nemo_overlaps = merge_overlapping_segments(nemo_overlaps)
            pred_frames = overlaps_to_frames(nemo_overlaps, total_duration)
            frame_metrics = compute_frame_metrics(gt_frames, pred_frames)
            seg_metrics_05 = compute_segment_metrics(gt_overlaps, nemo_overlaps, iou_threshold=0.5)
            seg_metrics_03 = compute_segment_metrics(gt_overlaps, nemo_overlaps, iou_threshold=0.3)

            results['nemo_results'] = {
                'detected_count': len(nemo_overlaps),
                'detected_duration': sum(e - s for s, e in nemo_overlaps),
                'time': elapsed,
                'frame_metrics': frame_metrics,
                'segment_metrics_iou05': seg_metrics_05,
                'segment_metrics_iou03': seg_metrics_03
            }

            logger.info(f"  Detected: {len(nemo_overlaps)}, Frame F1: {frame_metrics['f1']:.1f}%")

        except Exception as e:
            logger.error(f"NeMo SOD failed: {e}")
            results['nemo_results'] = {'error': str(e)}

    return results


def aggregate_sod_results(results: Dict) -> Dict:
    """Aggregate SOD results with best config."""
    best_pyannote = max(results['pyannote_results'], key=lambda x: x['frame_metrics']['f1'])

    results['aggregated'] = {
        'best_pyannote': {
            'onset': best_pyannote['onset'],
            'frame_f1': best_pyannote['frame_metrics']['f1'],
            'segment_f1_iou05': best_pyannote['segment_metrics_iou05']['f1']
        }
    }

    if results['nemo_results'] and 'error' not in results['nemo_results']:
        results['aggregated']['nemo'] = {
            'frame_f1': results['nemo_results']['frame_metrics']['f1'],
            'segment_f1_iou05': results['nemo_results']['segment_metrics_iou05']['f1']
        }

    return results


# =============================================================================
# SOS (Speech Overlap Separation) Comparison Functions
# =============================================================================

def compare_sos_models(
    audio_path: str,
    benchmark_dir: Path,
    speaker_dir: Path,
    max_regions: int = 15
) -> Dict:
    """
    Compare SOS models: PyannoteSOS vs SpeechBrainSOS (SepFormer).

    Parameters
    ----------
    audio_path : str
        Path to combined audio file
    benchmark_dir : Path
        Directory containing ground truth annotations
    speaker_dir : Path
        Directory containing individual speaker tracks
    max_regions : int
        Maximum overlap regions to process

    Returns
    -------
    Dict
        Results for both models with SI-SDR metrics
    """
    from pipeline.speaker_segmentation.SO.Separation import PyannoteSOS, SpeechBrainSOS
    from utils.metrics import si_sdr
    from config import PipelineConfig
    import torchaudio

    config = PipelineConfig()

    # Load ground truth overlaps
    gt_overlaps = extract_overlap_ground_truth(benchmark_dir)
    logger.info(f"Found {len(gt_overlaps)} overlap regions")

    # Load audio
    waveform, sr = load_audio_from_file(file_path=audio_path, sr=config.sr)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)

    # Load speaker reference tracks
    speaker_tracks = {
        'GOR': 'gor.mp3', 'JOHAN': 'johan.mp3', 'KALLE': 'kalle.mp3',
        'LUKAS': 'lukas.mp3', 'MARTEN': 'marten.mp3', 'OSCAR': 'oscar.mp3'
    }

    speaker_waveforms = {}
    for speaker, filename in speaker_tracks.items():
        track_path = speaker_dir / filename
        if track_path.exists():
            spk_waveform, _ = load_audio_from_file(file_path=str(track_path), sr=config.sr)
            if spk_waveform.dim() > 1:
                spk_waveform = spk_waveform.mean(dim=0)
            speaker_waveforms[speaker] = spk_waveform.numpy()

    results = {
        'ground_truth_overlaps': len(gt_overlaps),
        'pyannote': None,
        'sepformer': None
    }

    regions_to_process = gt_overlaps[:max_regions]

    # Helper function to run separation benchmark
    def run_separation(model, model_name: str, target_sr: int = None) -> Dict:
        region_results = []
        total_time = 0

        for start, end in regions_to_process:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = waveform[start_sample:end_sample]

            if len(segment) < sr * 0.1:
                continue

            start_time = time.time()
            try:
                with torch.no_grad():
                    try:
                        separated = model(segment, sample_rate=sr)
                    except TypeError:
                        separated = model(segment)
                elapsed = time.time() - start_time
                total_time += elapsed
            except Exception as e:
                logger.warning(f"Separation error: {e}")
                continue

            source_si_sdrs = []
            for source_idx, source_waveform in separated.items():
                if isinstance(source_waveform, torch.Tensor):
                    source_np = source_waveform.numpy()
                else:
                    source_np = source_waveform

                if target_sr and target_sr != sr:
                    resampler = torchaudio.transforms.Resample(target_sr, sr)
                    source_tensor = torch.from_numpy(source_np).float()
                    source_np = resampler(source_tensor).numpy()

                best_si_sdr = float('-inf')
                for speaker, ref_waveform in speaker_waveforms.items():
                    ref_segment = ref_waveform[start_sample:end_sample]
                    min_len = min(len(source_np), len(ref_segment))
                    if min_len < 100:
                        continue
                    sdr = si_sdr(ref_segment[:min_len], source_np[:min_len])
                    if sdr > best_si_sdr:
                        best_si_sdr = sdr

                source_si_sdrs.append(best_si_sdr)

            valid_sdrs = [s for s in source_si_sdrs if s > float('-inf')]
            avg_si_sdr = np.mean(valid_sdrs) if valid_sdrs else float('-inf')

            region_results.append({
                'start': start,
                'end': end,
                'avg_si_sdr': avg_si_sdr,
                'time': elapsed
            })

        all_si_sdrs = [r['avg_si_sdr'] for r in region_results if r['avg_si_sdr'] > float('-inf')]
        mean_si_sdr = np.mean(all_si_sdrs) if all_si_sdrs else float('-inf')
        std_si_sdr = np.std(all_si_sdrs) if len(all_si_sdrs) > 1 else 0

        return {
            'model': model_name,
            'num_regions': len(region_results),
            'total_time': total_time,
            'mean_si_sdr': mean_si_sdr,
            'std_si_sdr': std_si_sdr,
            'region_results': region_results
        }

    # Test PyannoteSOS
    logger.info("Testing PyannoteSOS...")
    try:
        pyannote_sos = PyannoteSOS(device=torch.device(config.device))
        results['pyannote'] = run_separation(pyannote_sos, "PyannoteSOS (separation-ami-1.0)")
        logger.info(f"  Mean SI-SDR: {results['pyannote']['mean_si_sdr']:.1f} dB")
    except Exception as e:
        logger.error(f"PyannoteSOS failed: {e}")
        results['pyannote'] = {'error': str(e)}

    # Test SpeechBrainSOS
    logger.info("Testing SpeechBrainSOS (SepFormer)...")
    try:
        sepformer_sos = SpeechBrainSOS(device=torch.device(config.device))
        results['sepformer'] = run_separation(sepformer_sos, "SepFormer (wsj02mix)", target_sr=8000)
        logger.info(f"  Mean SI-SDR: {results['sepformer']['mean_si_sdr']:.1f} dB")
    except Exception as e:
        logger.error(f"SepFormer failed: {e}")
        results['sepformer'] = {'error': str(e)}

    return results


def aggregate_sos_results(results: Dict) -> Dict:
    """Aggregate SOS results."""
    results['aggregated'] = {}

    for model in ['pyannote', 'sepformer']:
        if results[model] and 'error' not in results[model]:
            results['aggregated'][model] = {
                'mean_si_sdr': results[model]['mean_si_sdr'],
                'std_si_sdr': results[model]['std_si_sdr'],
                'total_time': results[model]['total_time']
            }

    return results


# =============================================================================
# Speaker Identification Comparison Functions
# =============================================================================

def load_ground_truth_segments(benchmark_dir: Path) -> List[Dict]:
    """Load all segments with golden speaker labels."""
    all_segments = []

    for f in sorted(benchmark_dir.glob("*.json")):
        chunk_idx = int(f.stem)
        chunk_offset = chunk_idx * 30.0

        with open(f) as fp:
            data = json.load(fp)

        for seg in data['segments']:
            speaker = seg['speaker'].upper()
            if speaker == 'SPEAKER_01':
                continue
            all_segments.append({
                'start': chunk_offset + seg['start'],
                'end': chunk_offset + seg['end'],
                'speaker': speaker
            })

    return all_segments


def compare_speaker_id_models(
    audio_path: str,
    benchmark_dir: Path,
    reference_dir: Path
) -> Dict:
    """
    Compare speaker identification approaches across embedding models and clustering methods.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    benchmark_dir : Path
        Directory containing ground truth annotations
    reference_dir : Path
        Directory containing reference speaker audio files

    Returns
    -------
    Dict
        Results for all embedding + clustering combinations
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.metrics import accuracy_score
    from config import PipelineConfig

    config = PipelineConfig()

    # Load ground truth
    segments = load_ground_truth_segments(benchmark_dir)
    speakers = set(seg['speaker'] for seg in segments)
    n_clusters = len(speakers)
    logger.info(f"Loaded {len(segments)} segments, {n_clusters} speakers")

    # Load audio
    waveform, sr = load_audio_from_file(audio_path, sr=config.sr)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)

    # Initialize embedders
    embedders = {}
    try:
        embedders['SpeechBrain'] = SpeechBrainEmbedding(device=config.device)
    except Exception as e:
        logger.warning(f"SpeechBrain failed: {e}")
    try:
        embedders['PyAnnote'] = PyAnnoteEmbedding()
    except Exception as e:
        logger.warning(f"PyAnnote failed: {e}")
    try:
        embedders['Wav2Vec2'] = Wav2Vec2Embedding(device=config.device)
    except Exception as e:
        logger.warning(f"Wav2Vec2 failed: {e}")

    results = {
        'num_segments': len(segments),
        'num_speakers': n_clusters,
        'speakers': list(speakers),
        'embeddings': {}
    }

    min_duration = 0.5

    for emb_name, embedder in embedders.items():
        logger.info(f"Testing {emb_name}...")

        # Load reference embeddings
        reference_embeddings = {}
        for audio_file in reference_dir.glob("*.mp3"):
            speaker_name = audio_file.stem.upper()
            ref_waveform, _ = load_audio_from_file(str(audio_file), sr=sr)
            if ref_waveform.dim() > 1:
                ref_waveform = ref_waveform.mean(dim=0)
            emb = embedder.embed(ref_waveform, sr)
            while emb.dim() > 1:
                emb = emb.squeeze(0)
            reference_embeddings[speaker_name] = emb

        # Extract segment embeddings
        embeddings = []
        valid_segments = []

        for seg in segments:
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)

            if (end_sample - start_sample) / sr < min_duration:
                continue
            if start_sample >= waveform.shape[-1] or end_sample > waveform.shape[-1]:
                continue

            segment_audio = waveform[start_sample:end_sample]
            emb = embedder.embed(segment_audio, sr)
            while emb.dim() > 1:
                emb = emb.squeeze(0)

            if torch.isnan(emb).any():
                continue

            embeddings.append(emb)
            valid_segments.append(seg)

        if not embeddings:
            logger.warning(f"No valid embeddings for {emb_name}")
            continue

        segment_embeddings = torch.stack(embeddings)
        ground_truth_labels = [seg['speaker'] for seg in valid_segments]

        emb_results = {}

        # Method 1: Cosine Similarity
        ref_names = list(reference_embeddings.keys())
        ref_embs = torch.stack([reference_embeddings[name] for name in ref_names])

        cosine_preds = []
        for seg_emb in segment_embeddings:
            similarities = F.cosine_similarity(
                seg_emb.unsqueeze(0).expand(len(ref_names), -1),
                ref_embs
            )
            best_idx = similarities.argmax().item()
            cosine_preds.append(ref_names[best_idx])

        emb_results['Cosine Similarity'] = {
            'accuracy': accuracy_score(ground_truth_labels, cosine_preds) * 100
        }

        # Method 2: K-Means
        embeddings_np = segment_embeddings.cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings_np)

        cluster_to_speaker = {}
        for cluster_id in range(n_clusters):
            mask = kmeans_labels == cluster_id
            if not mask.any():
                continue
            centroid = torch.tensor(embeddings_np[mask].mean(axis=0))
            similarities = F.cosine_similarity(
                centroid.unsqueeze(0).expand(len(ref_names), -1),
                ref_embs
            )
            cluster_to_speaker[cluster_id] = ref_names[similarities.argmax().item()]

        kmeans_preds = [cluster_to_speaker.get(c, "UNKNOWN") for c in kmeans_labels]
        emb_results['K-Means'] = {
            'accuracy': accuracy_score(ground_truth_labels, kmeans_preds) * 100
        }

        # Method 3: Agglomerative
        agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        agglo_labels = agglo.fit_predict(embeddings_np)

        cluster_to_speaker = {}
        for cluster_id in range(n_clusters):
            mask = agglo_labels == cluster_id
            if not mask.any():
                continue
            centroid = torch.tensor(embeddings_np[mask].mean(axis=0))
            similarities = F.cosine_similarity(
                centroid.unsqueeze(0).expand(len(ref_names), -1),
                ref_embs
            )
            cluster_to_speaker[cluster_id] = ref_names[similarities.argmax().item()]

        agglo_preds = [cluster_to_speaker.get(c, "UNKNOWN") for c in agglo_labels]
        emb_results['Agglomerative'] = {
            'accuracy': accuracy_score(ground_truth_labels, agglo_preds) * 100
        }

        # Method 4: DBSCAN
        dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        dbscan_labels = dbscan.fit_predict(embeddings_np)

        cluster_to_speaker = {}
        unique_clusters = set(dbscan_labels)
        unique_clusters.discard(-1)
        for cluster_id in unique_clusters:
            mask = dbscan_labels == cluster_id
            if not mask.any():
                continue
            centroid = torch.tensor(embeddings_np[mask].mean(axis=0))
            similarities = F.cosine_similarity(
                centroid.unsqueeze(0).expand(len(ref_names), -1),
                ref_embs
            )
            cluster_to_speaker[cluster_id] = ref_names[similarities.argmax().item()]

        dbscan_preds = [cluster_to_speaker.get(c, "UNKNOWN") for c in dbscan_labels]
        emb_results['DBSCAN'] = {
            'accuracy': accuracy_score(ground_truth_labels, dbscan_preds) * 100
        }

        results['embeddings'][emb_name] = emb_results
        logger.info(f"  Cosine: {emb_results['Cosine Similarity']['accuracy']:.1f}%, K-Means: {emb_results['K-Means']['accuracy']:.1f}%")

    return results


def aggregate_speaker_id_results(results: Dict) -> Dict:
    """Aggregate speaker ID results with best combination."""
    best_acc = 0
    best_combo = None

    for emb_name, methods in results['embeddings'].items():
        for method_name, res in methods.items():
            if res['accuracy'] > best_acc:
                best_acc = res['accuracy']
                best_combo = (emb_name, method_name)

    results['aggregated'] = {
        'best_combination': {
            'embedding': best_combo[0] if best_combo else None,
            'method': best_combo[1] if best_combo else None,
            'accuracy': best_acc
        }
    }

    return results


# =============================================================================
# Embedding Comparison Functions (Metrics Only)
# =============================================================================

def compare_embedding_models(
    audio_path: str,
    benchmark_dir: Path,
    reference_dir: Path = None,
    max_segments_per_speaker: int = 20,
    min_segment_duration: float = 1.0
) -> Dict:
    """
    Compare embedding models by evaluating clustering quality metrics.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    benchmark_dir : Path
        Directory containing ground truth annotations
    reference_dir : Path, optional
        Directory containing reference speaker audio files for cosine similarity
    max_segments_per_speaker : int
        Maximum segments per speaker to embed
    min_segment_duration : float
        Minimum segment duration in seconds

    Returns
    -------
    Dict
        Results including clustering metrics (ARI, Silhouette, F1) for each embedding model
        If reference_dir is provided, also includes cosine similarity accuracy
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.metrics import silhouette_score, adjusted_rand_score, f1_score
    from scipy.optimize import linear_sum_assignment
    from config import PipelineConfig

    config = PipelineConfig()

    # Load ground truth segments
    segments = load_ground_truth_segments(benchmark_dir)
    speakers = set(seg['speaker'] for seg in segments)
    n_speakers = len(speakers)
    logger.info(f"Found {len(segments)} segments, {n_speakers} speakers")

    # Load audio
    waveform, sr = load_audio_from_file(audio_path, sr=config.sr)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)

    # Initialize embedders
    embedders = {}
    try:
        embedders['SpeechBrain'] = SpeechBrainEmbedding(device=config.device)
    except Exception as e:
        logger.warning(f"SpeechBrain failed: {e}")
    try:
        embedders['PyAnnote'] = PyAnnoteEmbedding()
    except Exception as e:
        logger.warning(f"PyAnnote failed: {e}")
    try:
        embedders['Wav2Vec2'] = Wav2Vec2Embedding(device=config.device)
    except Exception as e:
        logger.warning(f"Wav2Vec2 failed: {e}")

    results = {
        'num_speakers': n_speakers,
        'speakers': list(speakers),
        'embeddings': {},
        'has_reference': reference_dir is not None
    }

    # Extract embeddings for each model
    all_embeddings = {}
    all_labels = {}
    reference_embeddings_by_model = {}

    for emb_name, embedder in embedders.items():
        logger.info(f"Extracting embeddings with {emb_name}...")

        # Load reference embeddings if reference_dir is provided
        if reference_dir is not None:
            reference_embeddings = {}
            for audio_file in reference_dir.glob("*.mp3"):
                speaker_name = audio_file.stem.upper()
                try:
                    ref_waveform, _ = load_audio_from_file(str(audio_file), sr=sr)
                    if ref_waveform.dim() > 1:
                        ref_waveform = ref_waveform.mean(dim=0)
                    emb = embedder.embed(ref_waveform, sr)
                    while emb.dim() > 1:
                        emb = emb.squeeze(0)
                    reference_embeddings[speaker_name] = emb.cpu()
                except Exception as e:
                    logger.warning(f"Failed to embed reference {speaker_name}: {e}")
            reference_embeddings_by_model[emb_name] = reference_embeddings
            logger.info(f"  Loaded {len(reference_embeddings)} reference embeddings")

        embeddings = []
        labels = []
        speaker_counts = {}

        # Sort segments by duration (longer first)
        sorted_segments = sorted(segments, key=lambda x: x['end'] - x['start'], reverse=True)

        for seg in sorted_segments:
            speaker = seg['speaker']

            # Limit segments per speaker
            if speaker_counts.get(speaker, 0) >= max_segments_per_speaker:
                continue

            duration = seg['end'] - seg['start']
            if duration < min_segment_duration:
                continue

            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)

            if start_sample >= waveform.shape[-1] or end_sample > waveform.shape[-1]:
                continue

            segment_audio = waveform[start_sample:end_sample]

            try:
                emb = embedder.embed(segment_audio, sr)
                while emb.dim() > 1:
                    emb = emb.squeeze(0)

                if torch.isnan(emb).any():
                    continue

                embeddings.append(emb.cpu().numpy())
                labels.append(speaker)
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            except Exception as e:
                continue

        if embeddings:
            all_embeddings[emb_name] = np.array(embeddings)
            all_labels[emb_name] = labels
            logger.info(f"  {emb_name}: {len(embeddings)} embeddings, speakers: {speaker_counts}")

    # Helper function to compute F1 with Hungarian matching
    def compute_f1_hungarian(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """Compute F1 score using Hungarian algorithm for optimal label assignment."""
        # Filter out noise points (label -1 in DBSCAN)
        mask = pred_labels != -1
        if not np.any(mask):
            return 0.0

        true_filtered = true_labels[mask]
        pred_filtered = pred_labels[mask]

        true_unique = np.unique(true_filtered)
        pred_unique = np.unique(pred_filtered)

        n_true = len(true_unique)
        n_pred = len(pred_unique)

        # Build cost matrix (negative overlap for minimization)
        cost_matrix = np.zeros((n_pred, n_true))
        for i, p in enumerate(pred_unique):
            for j, t in enumerate(true_unique):
                cost_matrix[i, j] = -np.sum((pred_filtered == p) & (true_filtered == t))

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create mapping and align labels
        label_map = {pred_unique[r]: true_unique[c] for r, c in zip(row_ind, col_ind)}
        aligned = np.array([label_map.get(p, -1) for p in pred_filtered])

        # Compute F1
        valid_mask = aligned != -1
        if np.sum(valid_mask) == 0:
            return 0.0

        return f1_score(true_filtered[valid_mask], aligned[valid_mask], average='macro', zero_division=0)

    # Compute clustering metrics for each embedding model
    for emb_name, embeddings in all_embeddings.items():
        labels = all_labels[emb_name]
        if len(embeddings) < 2:
            continue

        unique_speakers = sorted(set(labels))
        true_indices = np.array([unique_speakers.index(l) for l in labels])

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = embeddings / norms

        emb_results = {
            'num_embeddings': len(embeddings),
            'embedding_dim': embeddings.shape[1]
        }

        # KMeans
        logger.info(f"  {emb_name}: Running KMeans...")
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings_norm)
        emb_results['kmeans'] = {
            'silhouette': float(silhouette_score(embeddings_norm, kmeans_labels, metric='cosine')),
            'ari': float(adjusted_rand_score(true_indices, kmeans_labels)),
            'f1': float(compute_f1_hungarian(true_indices, kmeans_labels))
        }

        # Agglomerative
        logger.info(f"  {emb_name}: Running Agglomerative...")
        agglo = AgglomerativeClustering(n_clusters=n_speakers, metric='cosine', linkage='average')
        agglo_labels = agglo.fit_predict(embeddings_norm)
        emb_results['agglomerative'] = {
            'silhouette': float(silhouette_score(embeddings_norm, agglo_labels, metric='cosine')),
            'ari': float(adjusted_rand_score(true_indices, agglo_labels)),
            'f1': float(compute_f1_hungarian(true_indices, agglo_labels))
        }

        # DBSCAN (with optimal eps search)
        logger.info(f"  {emb_name}: Running DBSCAN...")
        best_eps = 0.3
        best_ari = -1
        best_dbscan_labels = None
        for eps in np.linspace(0.1, 1.0, 10):
            dbscan = DBSCAN(eps=eps, min_samples=2, metric='cosine')
            dbscan_labels = dbscan.fit_predict(embeddings_norm)
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            if n_clusters >= 2:
                ari = adjusted_rand_score(true_indices, dbscan_labels)
                if ari > best_ari:
                    best_ari = ari
                    best_eps = eps
                    best_dbscan_labels = dbscan_labels

        if best_ari > -1 and best_dbscan_labels is not None:
            mask = best_dbscan_labels != -1
            if np.sum(mask) > 1 and len(set(best_dbscan_labels[mask])) > 1:
                sil = silhouette_score(embeddings_norm[mask], best_dbscan_labels[mask], metric='cosine')
            else:
                sil = 0.0
            emb_results['dbscan'] = {
                'silhouette': float(sil),
                'ari': float(best_ari),
                'f1': float(compute_f1_hungarian(true_indices, best_dbscan_labels)),
                'eps': float(best_eps),
                'n_noise': int(np.sum(best_dbscan_labels == -1))
            }

        # Cosine Similarity (direct assignment using reference embeddings)
        if emb_name in reference_embeddings_by_model and reference_embeddings_by_model[emb_name]:
            logger.info(f"  {emb_name}: Running Cosine Similarity...")
            ref_embs = reference_embeddings_by_model[emb_name]
            ref_names = list(ref_embs.keys())
            ref_stack = torch.stack([ref_embs[name] for name in ref_names])

            # Convert segment embeddings to tensor for cosine similarity
            segment_embs_tensor = torch.from_numpy(embeddings).float()

            # Assign each segment to the closest reference speaker
            cosine_preds = []
            for seg_emb in segment_embs_tensor:
                similarities = F.cosine_similarity(
                    seg_emb.unsqueeze(0).expand(len(ref_names), -1),
                    ref_stack
                )
                best_idx = similarities.argmax().item()
                cosine_preds.append(ref_names[best_idx])

            # Compute metrics
            # Map true labels to indices
            correct = sum(1 for pred, true in zip(cosine_preds, labels) if pred == true)
            accuracy = correct / len(labels) if labels else 0.0

            # Compute ARI and F1 for consistency with other methods
            # Map predictions to indices using unique_speakers list
            pred_indices = []
            for pred in cosine_preds:
                if pred in unique_speakers:
                    pred_indices.append(unique_speakers.index(pred))
                else:
                    pred_indices.append(-1)  # Unknown speaker
            pred_indices = np.array(pred_indices)

            # Filter out unknown predictions for metrics
            valid_mask = pred_indices != -1
            if np.sum(valid_mask) > 0:
                cosine_ari = adjusted_rand_score(true_indices[valid_mask], pred_indices[valid_mask])
                cosine_f1 = f1_score(true_indices[valid_mask], pred_indices[valid_mask], average='macro', zero_division=0)
            else:
                cosine_ari = 0.0
                cosine_f1 = 0.0

            emb_results['cosine'] = {
                'accuracy': float(accuracy),
                'ari': float(cosine_ari),
                'f1': float(cosine_f1)
            }

        results['embeddings'][emb_name] = emb_results

    return results


def aggregate_embedding_results(results: Dict) -> Dict:
    """Aggregate embedding comparison results."""
    best_model = None
    best_ari = -1
    best_f1 = -1
    best_accuracy = -1
    best_cosine_model = None

    for emb_name, emb_data in results['embeddings'].items():
        for method_name in ['kmeans', 'agglomerative', 'dbscan', 'cosine']:
            if method_name in emb_data:
                metrics = emb_data[method_name]
                if metrics.get('ari', -1) > best_ari:
                    best_ari = metrics['ari']
                    best_model = (emb_name, method_name)
                if metrics.get('f1', -1) > best_f1:
                    best_f1 = metrics['f1']
                # Track best cosine similarity accuracy separately
                if method_name == 'cosine' and metrics.get('accuracy', -1) > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_cosine_model = emb_name

    results['aggregated'] = {
        'best_by_ari': {
            'embedding': best_model[0] if best_model else None,
            'clustering': best_model[1] if best_model else None,
            'ari': best_ari
        },
        'best_f1': best_f1
    }

    # Add best cosine similarity if available
    if best_cosine_model is not None:
        results['aggregated']['best_cosine'] = {
            'embedding': best_cosine_model,
            'accuracy': best_accuracy
        }

    return results


# Keep old function name as alias for backwards compatibility
compare_embedding_visualization = compare_embedding_models
aggregate_embedding_viz_results = aggregate_embedding_results


def compare_cluster_viz(
    audio_folder: Path,
    alignment_file: Path = None,
    embedding_type: str = 'speechbrain'
) -> Dict:
    """
    Compare clustering methods on audio embeddings with UMAP visualization.

    Parameters
    ----------
    audio_folder : Path
        Directory containing either:
        - Speaker subdirectories with audio_chunks/ subfolders (e.g., marten/audio_chunks/*.mp3)
        - Or flat audio files with alignment_file mapping
    alignment_file : Path, optional
        JSON file mapping audio filenames to speaker names.
        If None, uses directory structure where folder name = speaker name
    embedding_type : str
        One of 'pyannote', 'wav2vec2', 'speechbrain'

    Returns
    -------
    Dict
        Results including embeddings, UMAP coordinates, and clustering results
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from scipy.optimize import linear_sum_assignment
    from config import PipelineConfig

    # Try to import UMAP
    try:
        from umap import UMAP
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
        logger.warning("UMAP not installed. Install with: pip install umap-learn")

    config = PipelineConfig()

    # Build alignment from directory structure or load from file
    alignment = {}
    reference_files = set()  # Track which files are reference/speaking_alignment samples

    if alignment_file is not None and alignment_file.exists():
        # Load alignment file
        with open(alignment_file, 'r') as f:
            alignment = json.load(f)
        logger.info(f"Loaded alignment with {len(alignment)} entries")
    else:
        # Auto-discover from directory structure
        # Look for speaker subdirectories with audio_chunks/ subfolders
        logger.info(f"Auto-discovering speakers from directory structure: {audio_folder}")
        for speaker_dir in sorted(audio_folder.iterdir()):
            if not speaker_dir.is_dir():
                continue

            # Only use directories that have audio_chunks subfolder
            audio_chunks_dir = speaker_dir / "audio_chunks"
            if not audio_chunks_dir.exists():
                continue

            speaker_name = speaker_dir.name.upper()

            # Find audio files in audio_chunks
            for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
                for audio_file in audio_chunks_dir.glob(ext):
                    # Store relative path from audio_folder
                    rel_path = audio_file.relative_to(audio_folder)
                    alignment[str(rel_path)] = speaker_name

        logger.info(f"Discovered {len(alignment)} audio files from directory structure")

    # Also check for speaking_alignment folder with reference audio
    speaking_alignment_dir = audio_folder / "speaking_alignment"
    if speaking_alignment_dir.exists():
        logger.info(f"Found speaking_alignment folder, loading reference audio...")
        for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
            for audio_file in speaking_alignment_dir.glob(ext):
                # Speaker name from filename (e.g., marten.mp3 -> MARTEN)
                speaker_name = audio_file.stem.upper()
                rel_path = audio_file.relative_to(audio_folder)
                alignment[str(rel_path)] = speaker_name
                reference_files.add(str(rel_path))
        logger.info(f"Loaded {len(reference_files)} reference audio files from speaking_alignment")

    if not alignment:
        raise ValueError("No audio files found. Provide alignment file or use speaker directory structure.")

    speakers = sorted(set(alignment.values()))
    n_speakers = len(speakers)
    logger.info(f"Found {n_speakers} unique speakers: {speakers}")

    # Initialize embedder based on type
    if embedding_type == 'speechbrain':
        embedder = SpeechBrainEmbedding(device=config.device)
        emb_name = 'SpeechBrain'
    elif embedding_type == 'pyannote':
        embedder = PyAnnoteEmbedding()
        emb_name = 'PyAnnote'
    elif embedding_type == 'wav2vec2':
        embedder = Wav2Vec2Embedding(device=config.device)
        emb_name = 'Wav2Vec2'
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    logger.info(f"Using {emb_name} embedder")

    # Extract embeddings for each audio file
    embeddings = []
    labels = []
    filenames = []
    is_reference = []  # Track which samples are from speaking_alignment

    # Suppress MP3 decoding warnings
    import warnings
    import os
    old_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)

    for filename, speaker in alignment.items():
        audio_path = audio_folder / filename
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            continue

        try:
            # Temporarily redirect stderr to suppress libmpg123 warnings
            os.dup2(devnull, 2)
            waveform, sr = load_audio_from_file(str(audio_path), sr=config.sr)
            os.dup2(old_stderr, 2)
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)

            emb = embedder.embed(waveform, sr)
            while emb.dim() > 1:
                emb = emb.squeeze(0)

            if torch.isnan(emb).any():
                logger.warning(f"NaN embedding for {filename}, skipping")
                continue

            embeddings.append(emb.cpu().numpy())
            labels.append(speaker)
            filenames.append(filename)
            is_reference.append(filename in reference_files)
        except Exception as e:
            os.dup2(old_stderr, 2)  # Restore stderr on error
            logger.warning(f"Failed to embed {filename}: {e}")
            continue

    # Clean up stderr redirection
    os.close(devnull)

    if len(embeddings) < 2:
        raise ValueError(f"Not enough valid embeddings: {len(embeddings)}")

    embeddings = np.array(embeddings)
    logger.info(f"Extracted {len(embeddings)} embeddings with shape {embeddings.shape}")

    # L2 normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_norm = embeddings / norms

    # Convert labels to indices
    unique_speakers = sorted(set(labels))
    true_indices = np.array([unique_speakers.index(l) for l in labels])

    # Initialize results
    results = {
        'embedding_type': emb_name,
        'embedding_dim': embeddings.shape[1],
        'num_embeddings': len(embeddings),
        'num_speakers': n_speakers,
        'speakers': unique_speakers,
        'ground_truth_labels': labels,
        'filenames': filenames,
        'is_reference': is_reference,  # Boolean list indicating reference/speaking_alignment samples
        'clustering_results': {}
    }
    logger.info(f"Reference samples: {sum(is_reference)} out of {len(is_reference)} total")

    # Helper function to compute F1 with Hungarian matching
    def compute_f1_hungarian(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        from sklearn.metrics import f1_score
        mask = pred_labels != -1
        if not np.any(mask):
            return 0.0

        true_filtered = true_labels[mask]
        pred_filtered = pred_labels[mask]

        true_unique = np.unique(true_filtered)
        pred_unique = np.unique(pred_filtered)

        n_true = len(true_unique)
        n_pred = len(pred_unique)

        cost_matrix = np.zeros((n_pred, n_true))
        for i, p in enumerate(pred_unique):
            for j, t in enumerate(true_unique):
                cost_matrix[i, j] = -np.sum((pred_filtered == p) & (true_filtered == t))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        label_map = {pred_unique[r]: true_unique[c] for r, c in zip(row_ind, col_ind)}
        aligned = np.array([label_map.get(p, -1) for p in pred_filtered])

        valid_mask = aligned != -1
        if np.sum(valid_mask) == 0:
            return 0.0

        return f1_score(true_filtered[valid_mask], aligned[valid_mask], average='macro', zero_division=0)

    # Separate reference samples from samples to be clustered
    # Reference samples (from speaking_alignment) are static and should not be clustered
    is_ref_arr = np.array(is_reference)
    non_ref_mask = ~is_ref_arr
    n_non_ref = np.sum(non_ref_mask)

    if n_non_ref < 2:
        logger.warning(f"Not enough non-reference samples to cluster: {n_non_ref}")
        # Still include reference info for visualization
        results['clustering_results'] = {}
    else:
        # Get non-reference embeddings and labels for clustering
        embeddings_to_cluster = embeddings_norm[non_ref_mask]
        labels_to_cluster = [l for l, is_ref in zip(labels, is_reference) if not is_ref]
        true_indices_to_cluster = np.array([unique_speakers.index(l) for l in labels_to_cluster])

        logger.info(f"Clustering {n_non_ref} non-reference samples (excluding {np.sum(is_ref_arr)} reference samples)")

        # Helper to expand cluster labels back to full array (reference samples get -2 = "reference")
        def expand_labels(cluster_labels):
            full_labels = np.full(len(is_reference), -2, dtype=int)  # -2 = reference
            full_labels[non_ref_mask] = cluster_labels
            return full_labels

        # KMeans clustering (using normalized embeddings = cosine distance)
        # Since embeddings are L2-normalized, Euclidean distance is equivalent to cosine distance
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b = 2 - 2*cos(a,b) when ||a||=||b||=1
        logger.info("Running KMeans clustering (on L2-normalized embeddings)...")
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        kmeans_labels_clustered = kmeans.fit_predict(embeddings_to_cluster)
        kmeans_labels = expand_labels(kmeans_labels_clustered)
        kmeans_sil = silhouette_score(embeddings_to_cluster, kmeans_labels_clustered, metric='cosine')
        kmeans_ari = adjusted_rand_score(true_indices_to_cluster, kmeans_labels_clustered)
        kmeans_f1 = compute_f1_hungarian(true_indices_to_cluster, kmeans_labels_clustered)
        results['clustering_results']['kmeans'] = {
            'labels': kmeans_labels.tolist(),
            'silhouette': float(kmeans_sil),
            'ari': float(kmeans_ari),
            'f1': float(kmeans_f1)
        }

        # Agglomerative clustering
        logger.info("Running Agglomerative clustering...")
        agglo = AgglomerativeClustering(n_clusters=n_speakers, metric='cosine', linkage='average')
        agglo_labels_clustered = agglo.fit_predict(embeddings_to_cluster)
        agglo_labels = expand_labels(agglo_labels_clustered)
        agglo_sil = silhouette_score(embeddings_to_cluster, agglo_labels_clustered, metric='cosine')
        agglo_ari = adjusted_rand_score(true_indices_to_cluster, agglo_labels_clustered)
        agglo_f1 = compute_f1_hungarian(true_indices_to_cluster, agglo_labels_clustered)
        results['clustering_results']['agglomerative'] = {
            'labels': agglo_labels.tolist(),
            'silhouette': float(agglo_sil),
            'ari': float(agglo_ari),
            'f1': float(agglo_f1)
        }

        # DBSCAN with eps tuning
        logger.info("Running DBSCAN clustering...")
        best_eps = 0.3
        best_ari = -1
        best_dbscan_labels_clustered = None
        for eps in np.linspace(0.1, 1.0, 10):
            dbscan = DBSCAN(eps=eps, min_samples=2, metric='cosine')
            dbscan_labels_clustered = dbscan.fit_predict(embeddings_to_cluster)
            n_clusters = len(set(dbscan_labels_clustered)) - (1 if -1 in dbscan_labels_clustered else 0)
            if n_clusters >= 2:
                ari = adjusted_rand_score(true_indices_to_cluster, dbscan_labels_clustered)
                if ari > best_ari:
                    best_ari = ari
                    best_eps = eps
                    best_dbscan_labels_clustered = dbscan_labels_clustered

        if best_dbscan_labels_clustered is not None:
            best_dbscan_labels = expand_labels(best_dbscan_labels_clustered)
            mask = best_dbscan_labels_clustered != -1
            if np.sum(mask) > 1 and len(set(best_dbscan_labels_clustered[mask])) > 1:
                dbscan_sil = silhouette_score(embeddings_to_cluster[mask], best_dbscan_labels_clustered[mask], metric='cosine')
            else:
                dbscan_sil = 0.0
            dbscan_f1 = compute_f1_hungarian(true_indices_to_cluster, best_dbscan_labels_clustered)
            results['clustering_results']['dbscan'] = {
                'labels': best_dbscan_labels.tolist(),
                'silhouette': float(dbscan_sil),
                'ari': float(best_ari),
                'f1': float(dbscan_f1),
                'eps': float(best_eps),
                'n_noise': int(np.sum(best_dbscan_labels_clustered == -1))
            }

        # Cosine Similarity - compute centroids from ALL ground truth samples
        # but only predict non-reference samples
        logger.info("Running Cosine Similarity assignment...")
        # Compute centroid for each speaker from ALL samples (not just reference)
        speaker_centroids = {}
        for speaker in unique_speakers:
            speaker_mask = np.array([l == speaker for l in labels])
            speaker_embs = embeddings_norm[speaker_mask]
            centroid = speaker_embs.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # L2 normalize centroid
            speaker_centroids[speaker] = centroid

        # Assign each NON-REFERENCE embedding to closest centroid
        # Reference samples stay static (not predicted)
        cosine_preds = []
        cosine_pred_indices_list = []
        for i, (emb, is_ref) in enumerate(zip(embeddings_norm, is_reference)):
            if is_ref:
                # Reference samples keep their ground truth label (not predicted)
                cosine_preds.append(labels[i])
                cosine_pred_indices_list.append(-2)  # -2 = reference (not predicted)
            else:
                best_sim = -1
                best_speaker = None
                for speaker, centroid in speaker_centroids.items():
                    sim = np.dot(emb, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_speaker = speaker
                cosine_preds.append(best_speaker)
                cosine_pred_indices_list.append(unique_speakers.index(best_speaker))

        cosine_pred_indices = np.array(cosine_pred_indices_list)

        # Compute metrics only on non-reference samples
        non_ref_preds = [p for p, is_ref in zip(cosine_preds, is_reference) if not is_ref]
        non_ref_true = [l for l, is_ref in zip(labels, is_reference) if not is_ref]
        correct = sum(1 for pred, true in zip(non_ref_preds, non_ref_true) if pred == true)
        cosine_accuracy = correct / len(non_ref_true) if non_ref_true else 0.0

        # For ARI/F1, only use non-reference samples
        non_ref_pred_indices = np.array([unique_speakers.index(p) for p in non_ref_preds])
        cosine_ari = adjusted_rand_score(true_indices_to_cluster, non_ref_pred_indices)
        cosine_f1 = compute_f1_hungarian(true_indices_to_cluster, non_ref_pred_indices)

        results['clustering_results']['cosine'] = {
            'labels': cosine_pred_indices.tolist(),
            'predicted_speakers': cosine_preds,
            'accuracy': float(cosine_accuracy),
            'ari': float(cosine_ari),
            'f1': float(cosine_f1)
        }

    # Compute UMAP projection
    if HAS_UMAP:
        logger.info("Computing UMAP projection...")
        n_neighbors = min(15, max(2, len(embeddings) - 1))
        umap = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        umap_coords = umap.fit_transform(embeddings_norm)
        results['umap_coords'] = umap_coords.tolist()
    else:
        # Fallback to PCA if UMAP not available
        from sklearn.decomposition import PCA
        logger.info("UMAP not available, using PCA...")
        pca = PCA(n_components=2, random_state=42)
        umap_coords = pca.fit_transform(embeddings_norm)
        results['umap_coords'] = umap_coords.tolist()

    return results
