"""
Comparison utils
"""

import time
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import numpy as np
from loguru import logger
from tqdm import tqdm

from pipeline.speaker_segmentation.VAD.pyannote_vad import PyannoteVAD
from pipeline.speaker_segmentation.SCD import SCD
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
    2 clustering algorithms (KMeans, DBSCAN) = 6 combinations total.

    Returns
    -------
    Dict
        Dictionary with results for all 6 combinations
    """
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

    clustering_configs = {
        'kmeans': {'algorithm': 'kmeans'},
        'dbscan': {'algorithm': 'dbscan', 'eps': 0.5, 'min_samples': 2}
    }

    models = {}
    for emb_name in embedding_models.keys():
        for clus_name in clustering_configs.keys():
            key = f'{emb_name}_{clus_name}'
            models[key] = {'results': []}

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

            for clus_name, clus_config in clustering_configs.items():
                logger.info(f"    Clustering: {clus_name}")

                clusterer = SklearnClustering(**clus_config)

                start_time = time.time()
                n_clus = n_speakers if clus_name == 'kmeans' else None

                if n_clus is not None and embeddings.shape[0] < n_clus:
                    logger.warning(f"      Only {embeddings.shape[0]} segments but {n_clus} speakers in ground truth, using {embeddings.shape[0]} clusters")
                    n_clus = embeddings.shape[0]

                labels = clusterer.cluster_segments(embeddings, n_clusters=n_clus)
                clus_time = time.time() - start_time

                labels = handle_dbscan_noise(labels)

                if len(labels) != len(change_points) + 1:
                    logger.warning(f"      Label mismatch: {len(labels)} labels, {len(change_points)+1} segments")
                    min_len = min(len(labels), len(change_points) + 1)
                    labels = labels[:min_len]

                pred_segments = change_points_to_segments(
                    change_points[:min_len-1] if len(labels) < len(change_points) + 1 else change_points,
                    labels.numpy(),
                    duration
                )

                n_unique_labels = len(set(labels.tolist()))
                if n_unique_labels < 2 or n_unique_labels >= embeddings.shape[0]:
                    logger.warning(f"      Cannot compute silhouette: {n_unique_labels} unique labels for {embeddings.shape[0]} samples")
                    silhouette = 0.0
                else:
                    silhouette = evaluate_clustering(embeddings.cpu(), labels)['silhouette']

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
                        'silhouette': silhouette,
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

                logger.info(f"      Silhouette: {silhouette:.2f}, DER: {der_metrics['der']:.2f}%")

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
                    result = pipeline(str(audio_file), num_speakers=n_speakers_ref, include_timing=False)
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
