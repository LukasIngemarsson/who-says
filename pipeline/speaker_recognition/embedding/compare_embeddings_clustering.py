"""Compare speaker embedding models using SCD + clustering evaluation.

Compares SpeechBrain ECAPA vs Wav2Vec2 embeddings by:
1. Running SCD (Speaker Change Detection) to get speaker boundaries
2. Extracting embeddings for each speaker segment
3. Clustering embeddings with K-means
4. Computing silhouette score to measure cluster separation

Usage:
    python -m pipeline.speaker_recognition.embedding.compare_embeddings_clustering \
        samples/multi_speaker_sample.mp3 --num-speakers 2
"""
import argparse
import time
from pathlib import Path
import torch
import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv

from utils import load_audio_from_file
from utils.constants import SR
from pipeline.speaker_segmentation import SCD

load_dotenv(".env")


def compute_cluster_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2 or len(embeddings) < 2:
        return {"silhouette": 0.0}

    # Silhouette score (-1 to 1, higher = better separation)
    sil_score = silhouette_score(embeddings, labels)

    return {"silhouette": sil_score}


def compare_embeddings(audio_file: Path, num_speakers: int = 2):
    logger.info(f"Loading audio: {audio_file}")
    waveform, sr = load_audio_from_file(audio_file, sr=SR)

    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0) if waveform.shape[0] < waveform.shape[1] else waveform.mean(dim=1)

    total_duration = waveform.shape[-1] / sr
    logger.info(f"Audio duration: {total_duration:.2f}s")

    logger.info("Running Speaker Change Detection (SCD)...")
    scd = SCD()
    change_points = scd(waveform, sample_rate=sr)
    logger.info(f"Found {len(change_points)} speaker change points")

    if len(change_points) < 1:
        logger.error("Need at least 1 change point to compare embeddings")
        return

    # Define embedding models to compare
    embedding_models = []

    # SpeechBrain ECAPA
    try:
        from pipeline.speaker_recognition.embedding.speechbrain import SpeechBrainEmbedding
        embedding_models.append(("SpeechBrain ECAPA", SpeechBrainEmbedding()))
        logger.info("Loaded SpeechBrain ECAPA")
    except Exception as e:
        logger.warning(f"Failed to load SpeechBrain: {e}")

    # Wav2Vec2
    try:
        from pipeline.speaker_recognition.embedding.wav2vec2 import Wav2Vec2Embedding
        embedding_models.append(("Wav2Vec2-base", Wav2Vec2Embedding()))
        logger.info("Loaded Wav2Vec2")
    except Exception as e:
        logger.warning(f"Failed to load Wav2Vec2: {e}")

    if not embedding_models:
        logger.error("No embedding models available")
        return

    results = []

    for model_name, embedder in embedding_models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_name}")
        logger.info(f"{'='*60}")

        try:
            start_time = time.time()
            embeddings = embedder.embed_segments(waveform, sr, change_points)
            inference_time = time.time() - start_time

            embeddings_np = embeddings.cpu().numpy()
            logger.info(f"Extracted {embeddings_np.shape[0]} embeddings, dim={embeddings_np.shape[1]}")
            logger.info(f"Inference time: {inference_time:.2f}s")

            n_clusters = min(num_speakers, embeddings_np.shape[0])
            if n_clusters < 2:
                logger.warning(f"Not enough segments for clustering")
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_np)

            metrics = compute_cluster_metrics(embeddings_np, labels)
            metrics["inference_time"] = inference_time
            metrics["embedding_dim"] = embeddings_np.shape[1]
            metrics["num_segments"] = embeddings_np.shape[0]

            results.append({"model": model_name, **metrics})

        except Exception as e:
            logger.error(f"Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 90)
    print("SPEAKER EMBEDDING COMPARISON")
    print("=" * 90)
    print(f"Audio: {audio_file.name}")
    print(f"Duration: {total_duration:.2f}s")
    print(f"Change points: {len(change_points)}")
    print(f"Target clusters: {num_speakers}")
    print("-" * 90)

    header = f"{'Model':<20} {'Dim':>6} {'Silhouette':>12} {'Time':>8}"
    print(header)
    print("-" * 90)

    for r in results:
        print(
            f"{r['model']:<20} "
            f"{r['embedding_dim']:>6} "
            f"{r['silhouette']:>12.4f} "
            f"{r['inference_time']:>7.2f}s"
        )

    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare speaker embedding models")
    parser.add_argument("audio_file", type=Path, help="Path to audio file")
    parser.add_argument("--num-speakers", type=int, default=2, help="Expected number of speakers")

    args = parser.parse_args()

    if not args.audio_file.exists():
        parser.error(f"Audio file not found: {args.audio_file}")

    compare_embeddings(args.audio_file, args.num_speakers)
    logger.info("Comparison complete!")
