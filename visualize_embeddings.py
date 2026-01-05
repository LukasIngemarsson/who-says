"""
Speaker Embedding Visualization Script

This script:
1. Loads reference speaker audio files from speaking_alignment/
2. Extracts segments from benchmark JSON files using combined.mp3
3. Generates embeddings using wav2vec2, speechbrain, and pyannote
4. Visualizes embeddings using T-SNE, PCA, and UMAP
5. Compares DBSCAN vs KMeans clustering
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Try to import UMAP (optional)
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")

from utils import load_audio_from_file, to_mono
from utils.constants import SR


# Color palette for speakers
SPEAKER_COLORS = {
    'GOR': '#e41a1c',
    'JOHAN': '#377eb8',
    'MARTEN': '#4daf4a',
    'KALLE': '#984ea3',
    'LUKAS': '#ff7f00',
    'OSCAR': '#ffff33',
}

# Marker styles for embedding models
MODEL_MARKERS = {
    'wav2vec2': 'o',
    'speechbrain': 's',
    'pyannote': '^',
}


def load_reference_embeddings(
    reference_dir: str,
    embedder,
    model_name: str
) -> Tuple[np.ndarray, List[str]]:
    """Load and embed reference speaker audio files."""
    embeddings = []
    labels = []

    reference_path = Path(reference_dir)
    for audio_file in sorted(reference_path.glob("*.mp3")):
        speaker_name = audio_file.stem.upper()
        print(f"  [{model_name}] Embedding reference: {speaker_name}")

        audio, freq = load_audio_from_file(str(audio_file))
        audio = to_mono(audio)

        emb = embedder.embed(audio, freq)
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().detach().numpy()
        emb = emb.squeeze()

        embeddings.append(emb)
        labels.append(speaker_name)

    return np.array(embeddings), labels


def load_benchmark_segments(
    benchmark_dir: str,
    audio_file: str,
    embedder,
    model_name: str,
    max_segments: int = 100
) -> Tuple[np.ndarray, List[str]]:
    """Load benchmark JSON files and extract segment embeddings from audio."""
    embeddings = []
    labels = []

    # Load the combined audio
    print(f"  [{model_name}] Loading audio: {audio_file}")
    audio, freq = load_audio_from_file(audio_file)
    audio = to_mono(audio)

    # Ensure audio is 1D
    if audio.dim() > 1:
        audio = audio.squeeze()

    benchmark_path = Path(benchmark_dir)
    segment_count = 0

    for json_file in sorted(benchmark_path.glob("*.json")):
        if segment_count >= max_segments:
            break

        with open(json_file, 'r') as f:
            data = json.load(f)

        segments = data.get('segments', [])

        for seg in segments:
            if segment_count >= max_segments:
                break

            start = seg.get('start', 0)
            end = seg.get('end', 0)
            speaker = seg.get('speaker', 'UNKNOWN').upper()

            # Skip very short segments
            duration = end - start
            if duration < 0.5:
                continue

            # Extract segment audio
            start_sample = int(start * freq)
            end_sample = int(end * freq)

            if end_sample > len(audio):
                continue

            segment_audio = audio[start_sample:end_sample]

            try:
                emb = embedder.embed(segment_audio, freq)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().detach().numpy()
                emb = emb.squeeze()

                embeddings.append(emb)
                labels.append(speaker)
                segment_count += 1
            except Exception as e:
                print(f"    Warning: Failed to embed segment: {e}")
                continue

    print(f"  [{model_name}] Embedded {segment_count} benchmark segments")
    return np.array(embeddings) if embeddings else np.array([]), labels


def apply_dimensionality_reduction(
    embeddings: np.ndarray,
    method: str = 'tsne',
    n_components: int = 2,
    perplexity: int = 30,
    use_cosine: bool = True
) -> np.ndarray:
    """Apply dimensionality reduction to embeddings.

    Args:
        embeddings: Input embeddings array
        method: 'tsne', 'pca', or 'umap'
        n_components: Output dimensions (default 2)
        perplexity: T-SNE perplexity (will be auto-adjusted)
        use_cosine: If True, use L2 normalization (cosine similarity space)
                   which often works better for speaker embeddings
    """
    if len(embeddings) < 2:
        return embeddings

    # L2 normalize for cosine similarity space (better for speaker embeddings)
    # vs StandardScaler which centers and scales by variance
    if use_cosine:
        # L2 normalization: each embedding becomes unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        embeddings_normalized = embeddings / norms
    else:
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)

    if method == 'tsne':
        # Better perplexity heuristic: ~n_samples/3, clamped to [5, 50]
        auto_perp = max(5, min(50, len(embeddings) // 3))
        perp = min(auto_perp, len(embeddings) - 1)

        reducer = TSNE(
            n_components=n_components,
            perplexity=perp,
            max_iter=2000,        # More iterations for convergence (sklearn 1.7+)
            learning_rate='auto', # Auto-adjust learning rate
            init='pca',           # PCA init is more stable than random
            metric='cosine' if use_cosine else 'euclidean',
            random_state=42
        )
        return reducer.fit_transform(embeddings_normalized)

    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        return reducer.fit_transform(embeddings_normalized)

    elif method == 'umap' and HAS_UMAP:
        # UMAP parameters tuned for speaker clustering
        n_neighbors = min(15, max(2, len(embeddings) - 1))
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,         # Tighter clusters (default 0.1, try 0.05-0.2)
            metric='cosine' if use_cosine else 'euclidean',
            random_state=42
        )
        return reducer.fit_transform(embeddings_normalized)

    else:
        raise ValueError(f"Unknown method: {method}")


def find_optimal_dbscan_eps(
    embeddings: np.ndarray,
    n_clusters_target: int,
    eps_range: Tuple[float, float] = (0.1, 2.0),
    n_steps: int = 20,
    min_samples: int = 2
) -> Tuple[float, np.ndarray]:
    """Find optimal eps for DBSCAN to match target number of clusters."""
    # L2 normalize for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_norm = embeddings / norms

    best_eps = eps_range[0]
    best_labels = None
    best_diff = float('inf')

    for eps in np.linspace(eps_range[0], eps_range[1], n_steps):
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clusterer.fit_predict(embeddings_norm)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        diff = abs(n_clusters - n_clusters_target)
        if diff < best_diff or (diff == best_diff and eps > best_eps):
            best_diff = diff
            best_eps = eps
            best_labels = labels

    return best_eps, best_labels


def apply_clustering(
    embeddings: np.ndarray,
    method: str = 'kmeans',
    n_clusters: int = 6,
    eps: float = 0.5,
    min_samples: int = 2,
    use_cosine: bool = True
) -> Tuple[np.ndarray, dict]:
    """Apply clustering to embeddings.

    Returns:
        Tuple of (cluster_labels, info_dict)
    """
    if len(embeddings) < 2:
        return np.array([0]), {'method': method}

    # L2 normalize for cosine similarity
    if use_cosine:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_proc = embeddings / norms
    else:
        scaler = StandardScaler()
        embeddings_proc = scaler.fit_transform(embeddings)

    info = {'method': method}

    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings_proc)
        info['n_clusters'] = n_clusters

    elif method == 'dbscan':
        # Find optimal eps
        best_eps, labels = find_optimal_dbscan_eps(
            embeddings, n_clusters, min_samples=min_samples
        )
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        info['eps'] = best_eps
        info['n_clusters'] = n_found
        info['n_noise'] = np.sum(labels == -1)

    elif method == 'hierarchical':
        # Agglomerative clustering with cosine affinity
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clusterer.fit_predict(embeddings_proc)
        info['n_clusters'] = n_clusters
        info['linkage'] = 'average'

    elif method == 'cosine_threshold':
        # Cluster by cosine similarity threshold
        labels = cluster_by_cosine_similarity(embeddings_proc, n_clusters)
        info['n_clusters'] = len(set(labels))

    else:
        raise ValueError(f"Unknown method: {method}")

    return labels, info


def cluster_by_cosine_similarity(
    embeddings: np.ndarray,
    n_clusters: int
) -> np.ndarray:
    """Cluster embeddings using cosine similarity with hierarchical clustering."""
    # Compute cosine distance matrix (1 - similarity)
    cos_sim = cosine_similarity(embeddings)
    cos_dist = 1 - cos_sim

    # Use scipy hierarchical clustering with precomputed distances
    # Convert to condensed distance matrix
    condensed_dist = pdist(embeddings, metric='cosine')

    # Hierarchical clustering
    Z = linkage(condensed_dist, method='average')

    # Cut tree to get desired number of clusters
    labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-indexed

    return labels


def align_cluster_labels(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """Align predicted cluster labels to ground truth using Hungarian algorithm.

    Since clustering doesn't know the "correct" label names, we need to find
    the optimal mapping between predicted clusters and true labels.
    """
    # Filter out noise points (label -1 in DBSCAN)
    mask = pred_labels != -1
    if not np.any(mask):
        return pred_labels

    true_filtered = true_labels[mask]
    pred_filtered = pred_labels[mask]

    true_unique = np.unique(true_filtered)
    pred_unique = np.unique(pred_filtered)

    # Build confusion matrix
    n_true = len(true_unique)
    n_pred = len(pred_unique)

    # Create cost matrix (negative overlap for minimization)
    cost_matrix = np.zeros((n_pred, n_true))
    for i, p in enumerate(pred_unique):
        for j, t in enumerate(true_unique):
            cost_matrix[i, j] = -np.sum((pred_filtered == p) & (true_filtered == t))

    # Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create mapping
    label_map = {}
    for r, c in zip(row_ind, col_ind):
        label_map[pred_unique[r]] = true_unique[c]

    # Apply mapping
    aligned = np.full_like(pred_labels, -1)
    for old_label, new_label in label_map.items():
        aligned[pred_labels == old_label] = new_label

    return aligned


def compute_clustering_metrics(
    embeddings: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: List[str]
) -> dict:
    """Compute clustering quality metrics including F1, precision, recall."""
    unique_speakers = sorted(set(true_labels))
    true_label_indices = np.array([unique_speakers.index(l) for l in true_labels])

    metrics = {}

    # Adjusted Rand Index (agreement with ground truth)
    try:
        metrics['ARI'] = adjusted_rand_score(true_label_indices, predicted_labels)
    except:
        metrics['ARI'] = 0.0

    # Silhouette score (cluster cohesion/separation)
    n_unique = len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)
    if n_unique > 1 and n_unique < len(embeddings):
        try:
            mask = predicted_labels != -1
            if np.sum(mask) > 1:
                metrics['Silhouette'] = silhouette_score(
                    embeddings[mask], predicted_labels[mask], metric='cosine'
                )
            else:
                metrics['Silhouette'] = 0.0
        except:
            metrics['Silhouette'] = 0.0
    else:
        metrics['Silhouette'] = 0.0

    # Align cluster labels to ground truth for F1/precision/recall
    aligned_labels = align_cluster_labels(true_label_indices, predicted_labels)

    # Filter out noise points for classification metrics
    mask = aligned_labels != -1
    if np.sum(mask) > 0:
        true_filtered = true_label_indices[mask]
        pred_filtered = aligned_labels[mask]

        try:
            # Macro average: compute metric for each class, then average
            metrics['Precision'] = precision_score(
                true_filtered, pred_filtered, average='macro', zero_division=0
            )
            metrics['Recall'] = recall_score(
                true_filtered, pred_filtered, average='macro', zero_division=0
            )
            metrics['F1'] = f1_score(
                true_filtered, pred_filtered, average='macro', zero_division=0
            )

            # Also compute weighted versions (accounts for class imbalance)
            metrics['F1_weighted'] = f1_score(
                true_filtered, pred_filtered, average='weighted', zero_division=0
            )
        except Exception as e:
            metrics['Precision'] = 0.0
            metrics['Recall'] = 0.0
            metrics['F1'] = 0.0
            metrics['F1_weighted'] = 0.0
    else:
        metrics['Precision'] = 0.0
        metrics['Recall'] = 0.0
        metrics['F1'] = 0.0
        metrics['F1_weighted'] = 0.0

    return metrics


def plot_embeddings_comparison(
    all_embeddings: Dict[str, np.ndarray],
    all_labels: Dict[str, List[str]],
    reduction_method: str,
    output_file: str
):
    """Create a comparison plot for all embedding models."""
    n_models = len(all_embeddings)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    fig.suptitle(f'Speaker Embeddings - {reduction_method.upper()} Projection', fontsize=14)

    # Handle single model case
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, embeddings) in enumerate(all_embeddings.items()):
        ax = axes[idx]
        labels = all_labels[model_name]

        if len(embeddings) == 0:
            ax.set_title(f'{model_name}\n(No data)')
            continue

        # Apply dimensionality reduction
        reduced = apply_dimensionality_reduction(embeddings, method=reduction_method)

        # Plot each speaker with different color
        unique_speakers = list(set(labels))
        for speaker in unique_speakers:
            mask = np.array([l == speaker for l in labels])
            color = SPEAKER_COLORS.get(speaker, '#999999')
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=color,
                label=speaker,
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidth=0.5
            )

        ax.set_title(f'{model_name}')
        ax.set_xlabel(f'{reduction_method.upper()} 1')
        ax.set_ylabel(f'{reduction_method.upper()} 2')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_clustering_comparison(
    all_embeddings: Dict[str, np.ndarray],
    all_labels: Dict[str, List[str]],
    output_file: str
):
    """Create comprehensive clustering comparison plot with all methods."""
    n_models = len(all_embeddings)
    # 5 methods: Ground Truth, KMeans, DBSCAN (optimal), Hierarchical, Cosine Similarity
    methods = ['ground_truth', 'kmeans', 'dbscan', 'hierarchical', 'cosine_threshold']
    method_names = ['Ground Truth', 'KMeans', 'DBSCAN\n(optimal eps)', 'Hierarchical\n(avg linkage)', 'Cosine\nSimilarity']

    fig, axes = plt.subplots(n_models, 5, figsize=(25, 6 * n_models))
    fig.suptitle('Clustering Method Comparison (T-SNE Visualization)', fontsize=16, y=1.02)

    # Handle single model case
    if n_models == 1:
        axes = axes.reshape(1, -1)

    # Store results for summary table
    all_results = {}

    for row_idx, (model_name, embeddings) in enumerate(all_embeddings.items()):
        labels = all_labels[model_name]
        all_results[model_name] = {}

        if len(embeddings) == 0:
            continue

        # Apply T-SNE for visualization (compute once per model)
        reduced = apply_dimensionality_reduction(embeddings, method='tsne')

        unique_speakers = sorted(set(labels))
        n_speakers = len(unique_speakers)
        ground_truth_indices = np.array([unique_speakers.index(l) for l in labels])

        for col_idx, (method, method_name) in enumerate(zip(methods, method_names)):
            ax = axes[row_idx, col_idx]

            if method == 'ground_truth':
                cluster_labels = ground_truth_indices
                info = {'n_clusters': n_speakers}
                title = f'{model_name}\n{method_name}'
            else:
                cluster_labels, info = apply_clustering(
                    embeddings, method=method, n_clusters=n_speakers
                )
                title = f'{model_name}\n{method_name}'

            # Plot
            scatter = ax.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=cluster_labels,
                cmap='tab10',
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidth=0.5
            )

            ax.set_title(title, fontsize=10)
            ax.set_xlabel('T-SNE 1', fontsize=8)
            ax.set_ylabel('T-SNE 2', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Calculate and display metrics
            if method != 'ground_truth':
                metrics = compute_clustering_metrics(embeddings, cluster_labels, labels)
                all_results[model_name][method] = {**metrics, **info}

                # Build info text
                info_lines = [f"ARI: {metrics['ARI']:.3f}", f"Sil: {metrics['Silhouette']:.3f}"]
                if 'eps' in info:
                    info_lines.append(f"eps: {info['eps']:.2f}")
                if 'n_noise' in info and info['n_noise'] > 0:
                    info_lines.append(f"noise: {info['n_noise']}")
                info_lines.append(f"k: {info.get('n_clusters', n_speakers)}")

                ax.text(0.02, 0.98, '\n'.join(info_lines),
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    return all_results


def print_clustering_results_table(results: dict):
    """Print a formatted table of clustering results."""
    print("\n" + "=" * 100)
    print("CLUSTERING RESULTS SUMMARY")
    print("=" * 100)

    methods = ['kmeans', 'dbscan', 'hierarchical', 'cosine_threshold']
    method_headers = ['KMeans', 'DBSCAN', 'Hierarchical', 'Cosine Sim']

    # Header
    print(f"\n{'Model':<15} | ", end="")
    for h in method_headers:
        print(f"{h:^18} | ", end="")
    print()
    print("-" * 15 + "-+-" + "-+-".join(["-" * 18] * len(methods)) + "-+")

    # Define metrics to print
    metric_info = [
        ('F1', 'F1 Score (macro avg) - higher = better:'),
        ('Precision', 'Precision (macro avg) - higher = better:'),
        ('Recall', 'Recall (macro avg) - higher = better:'),
        ('ARI', 'Adjusted Rand Index - higher = better:'),
        ('Silhouette', 'Silhouette Score - higher = better:'),
    ]

    for metric_key, metric_title in metric_info:
        print(f"\n{metric_title}")
        for model_name, model_results in results.items():
            print(f"{model_name:<15} | ", end="")
            for method in methods:
                if method in model_results:
                    val = model_results[method].get(metric_key, 0)
                    print(f"{val:^18.3f} | ", end="")
                else:
                    print(f"{'N/A':^18} | ", end="")
            print()

    # Find best method per model by F1
    print("\n" + "-" * 100)
    print("BEST METHOD BY F1 SCORE:")
    for model_name, model_results in results.items():
        if model_results:
            best_method = max(model_results.keys(), key=lambda m: model_results[m].get('F1', 0))
            best_f1 = model_results[best_method].get('F1', 0)
            best_prec = model_results[best_method].get('Precision', 0)
            best_rec = model_results[best_method].get('Recall', 0)
            print(f"  {model_name}: {best_method} (F1={best_f1:.3f}, P={best_prec:.3f}, R={best_rec:.3f})")

    print("=" * 100)

    # Return JSON-serializable results
    return results


def save_clustering_results_json(results: dict, output_file: str):
    """Save clustering results to JSON file."""
    import json

    # Convert to serializable format
    output = {
        'embedding_models': {},
        'summary': {}
    }

    methods = ['kmeans', 'dbscan', 'hierarchical', 'cosine_threshold']

    for model_name, model_results in results.items():
        output['embedding_models'][model_name] = {}
        for method in methods:
            if method in model_results:
                output['embedding_models'][model_name][method] = {
                    'F1': round(model_results[method].get('F1', 0), 4),
                    'Precision': round(model_results[method].get('Precision', 0), 4),
                    'Recall': round(model_results[method].get('Recall', 0), 4),
                    'ARI': round(model_results[method].get('ARI', 0), 4),
                    'Silhouette': round(model_results[method].get('Silhouette', 0), 4),
                    'n_clusters': model_results[method].get('n_clusters', 0),
                }
                if 'eps' in model_results[method]:
                    output['embedding_models'][model_name][method]['eps'] = round(
                        model_results[method]['eps'], 4
                    )

        # Find best method
        if model_results:
            best_method = max(model_results.keys(), key=lambda m: model_results[m].get('F1', 0))
            output['summary'][model_name] = {
                'best_method': best_method,
                'best_F1': round(model_results[best_method].get('F1', 0), 4)
            }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def plot_combined_view(
    all_embeddings: Dict[str, np.ndarray],
    all_labels: Dict[str, List[str]],
    output_file: str
):
    """Create a combined view with all reduction methods."""
    reduction_methods = ['pca', 'tsne']
    if HAS_UMAP:
        reduction_methods.append('umap')

    n_methods = len(reduction_methods)
    n_models = len(all_embeddings)

    fig, axes = plt.subplots(n_methods, n_models, figsize=(6 * n_models, 5 * n_methods))
    fig.suptitle('Speaker Embeddings: Model × Reduction Method Comparison', fontsize=16, y=1.02)

    # Handle edge cases for axes indexing
    if n_methods == 1 and n_models == 1:
        axes = np.array([[axes]])
    elif n_methods == 1:
        axes = axes.reshape(1, -1)
    elif n_models == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, method in enumerate(reduction_methods):
        for col_idx, (model_name, embeddings) in enumerate(all_embeddings.items()):
            ax = axes[row_idx, col_idx]
            labels = all_labels[model_name]

            if len(embeddings) == 0:
                ax.set_title(f'{model_name} - {method.upper()}\n(No data)')
                continue

            # Apply reduction
            try:
                reduced = apply_dimensionality_reduction(embeddings, method=method)
            except Exception as e:
                ax.set_title(f'{model_name} - {method.upper()}\n(Error: {str(e)[:30]})')
                continue

            # Plot each speaker
            unique_speakers = sorted(set(labels))
            for speaker in unique_speakers:
                mask = np.array([l == speaker for l in labels])
                color = SPEAKER_COLORS.get(speaker, '#999999')
                ax.scatter(
                    reduced[mask, 0],
                    reduced[mask, 1],
                    c=color,
                    label=speaker,
                    alpha=0.7,
                    s=50,
                    edgecolors='white',
                    linewidth=0.5
                )

            ax.set_title(f'{model_name} - {method.upper()}')
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
            if col_idx == n_models - 1:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def load_segments_by_speaker(
    benchmark_dir: str,
    audio_file: str,
    embedder,
    model_name: str,
    min_duration: float = 1.5,     # Increased: longer segments = better embeddings
    max_segments_per_speaker: int = 20
) -> Tuple[np.ndarray, List[str]]:
    """Load benchmark segments, ensuring balanced representation per speaker."""
    embeddings = []
    labels = []
    speaker_counts = {}

    # Load the combined audio
    print(f"  [{model_name}] Loading audio: {audio_file}")
    audio, freq = load_audio_from_file(audio_file)
    audio = to_mono(audio)

    if audio.dim() > 1:
        audio = audio.squeeze()

    benchmark_path = Path(benchmark_dir)
    all_segments = []

    # Collect all segments first
    for json_file in sorted(benchmark_path.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
        segments = data.get('segments', [])
        for seg in segments:
            duration = seg.get('end', 0) - seg.get('start', 0)
            if duration >= min_duration:
                all_segments.append(seg)

    # Sort by duration (longer segments first for better embeddings)
    all_segments.sort(key=lambda x: x.get('end', 0) - x.get('start', 0), reverse=True)

    for seg in all_segments:
        speaker = seg.get('speaker', 'UNKNOWN').upper()

        # Limit segments per speaker
        if speaker_counts.get(speaker, 0) >= max_segments_per_speaker:
            continue

        start = seg.get('start', 0)
        end = seg.get('end', 0)
        start_sample = int(start * freq)
        end_sample = int(end * freq)

        if end_sample > len(audio):
            continue

        segment_audio = audio[start_sample:end_sample]

        try:
            emb = embedder.embed(segment_audio, freq)
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().detach().numpy()
            emb = emb.squeeze()

            embeddings.append(emb)
            labels.append(speaker)
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        except Exception as e:
            continue

    print(f"  [{model_name}] Segments per speaker: {speaker_counts}")
    return np.array(embeddings) if embeddings else np.array([]), labels


def plot_reference_only(
    all_embeddings: Dict[str, np.ndarray],
    all_labels: Dict[str, List[str]],
    output_file: str
):
    """Plot only reference speaker embeddings (cleaner view)."""
    n_models = len(all_embeddings)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    fig.suptitle('Reference Speaker Embeddings (T-SNE)', fontsize=14)

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, embeddings) in enumerate(all_embeddings.items()):
        ax = axes[idx]
        labels = all_labels[model_name]

        if len(embeddings) == 0:
            ax.set_title(f'{model_name}\n(No data)')
            continue

        reduced = apply_dimensionality_reduction(embeddings, method='tsne')

        unique_speakers = sorted(set(labels))
        for speaker in unique_speakers:
            mask = np.array([l == speaker for l in labels])
            color = SPEAKER_COLORS.get(speaker, '#999999')
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=color,
                label=speaker,
                alpha=0.8,
                s=120,
                edgecolors='black',
                linewidth=1
            )
            # Add speaker name annotation
            if np.sum(mask) > 0:
                centroid = reduced[mask].mean(axis=0)
                ax.annotate(speaker, centroid, fontsize=8, ha='center', va='bottom',
                           fontweight='bold')

        ax.set_title(f'{model_name}')
        ax.set_xlabel('T-SNE 1')
        ax.set_ylabel('T-SNE 2')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    # Paths
    reference_dir = "samples/meetings/meeting3-en/speaking_alignment"
    benchmark_dir = "samples/benchmarks/english"
    audio_file = "samples/meetings/meeting3-en/combined.mp3"

    # Check paths exist
    if not os.path.exists(reference_dir):
        print(f"Error: Reference directory not found: {reference_dir}")
        return
    if not os.path.exists(benchmark_dir):
        print(f"Error: Benchmark directory not found: {benchmark_dir}")
        return
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return

    print("=" * 60)
    print("Speaker Embedding Visualization")
    print("=" * 60)

    # Store reference-only and segment embeddings separately
    ref_embeddings = {}
    ref_labels = {}
    seg_embeddings = {}
    seg_labels = {}
    all_embeddings = {}
    all_labels = {}

    # 1. Wav2Vec2 Embeddings
    print("\n[1/3] Loading Wav2Vec2 embedder...")
    try:
        from pipeline.speaker_recognition.embedding.wav2vec2 import Wav2Vec2Embedding
        wav2vec2_embedder = Wav2Vec2Embedding()

        ref_emb, ref_lab = load_reference_embeddings(reference_dir, wav2vec2_embedder, "wav2vec2")
        seg_emb, seg_lab = load_segments_by_speaker(benchmark_dir, audio_file, wav2vec2_embedder, "wav2vec2")

        ref_embeddings['wav2vec2'] = ref_emb
        ref_labels['wav2vec2'] = ref_lab
        seg_embeddings['wav2vec2'] = seg_emb
        seg_labels['wav2vec2'] = seg_lab

        if len(ref_emb) > 0 and len(seg_emb) > 0:
            all_embeddings['wav2vec2'] = np.vstack([ref_emb, seg_emb])
            all_labels['wav2vec2'] = ref_lab + seg_lab
        elif len(seg_emb) > 0:
            all_embeddings['wav2vec2'] = seg_emb
            all_labels['wav2vec2'] = seg_lab
        print(f"  Wav2Vec2: {len(all_embeddings.get('wav2vec2', []))} total embeddings")
    except Exception as e:
        print(f"  Error loading Wav2Vec2: {e}")

    # 2. SpeechBrain Embeddings
    print("\n[2/3] Loading SpeechBrain embedder...")
    try:
        from pipeline.speaker_recognition.embedding.speechbrain import SpeechBrainEmbedding
        speechbrain_embedder = SpeechBrainEmbedding()

        ref_emb, ref_lab = load_reference_embeddings(reference_dir, speechbrain_embedder, "speechbrain")
        seg_emb, seg_lab = load_segments_by_speaker(benchmark_dir, audio_file, speechbrain_embedder, "speechbrain")

        ref_embeddings['speechbrain'] = ref_emb
        ref_labels['speechbrain'] = ref_lab
        seg_embeddings['speechbrain'] = seg_emb
        seg_labels['speechbrain'] = seg_lab

        if len(ref_emb) > 0 and len(seg_emb) > 0:
            all_embeddings['speechbrain'] = np.vstack([ref_emb, seg_emb])
            all_labels['speechbrain'] = ref_lab + seg_lab
        elif len(seg_emb) > 0:
            all_embeddings['speechbrain'] = seg_emb
            all_labels['speechbrain'] = seg_lab
        print(f"  SpeechBrain: {len(all_embeddings.get('speechbrain', []))} total embeddings")
    except Exception as e:
        print(f"  Error loading SpeechBrain: {e}")

    # 3. PyAnnote Embeddings
    print("\n[3/3] Loading PyAnnote embedder...")
    try:
        from pipeline.speaker_recognition.embedding._pyannote import PyAnnoteEmbedding
        pyannote_embedder = PyAnnoteEmbedding()

        ref_emb, ref_lab = load_reference_embeddings(reference_dir, pyannote_embedder, "pyannote")
        seg_emb, seg_lab = load_segments_by_speaker(benchmark_dir, audio_file, pyannote_embedder, "pyannote")

        ref_embeddings['pyannote'] = ref_emb
        ref_labels['pyannote'] = ref_lab
        seg_embeddings['pyannote'] = seg_emb
        seg_labels['pyannote'] = seg_lab

        if len(ref_emb) > 0 and len(seg_emb) > 0:
            all_embeddings['pyannote'] = np.vstack([ref_emb, seg_emb])
            all_labels['pyannote'] = ref_lab + seg_lab
        elif len(seg_emb) > 0:
            all_embeddings['pyannote'] = seg_emb
            all_labels['pyannote'] = seg_lab
        print(f"  PyAnnote: {len(all_embeddings.get('pyannote', []))} total embeddings")
    except Exception as e:
        print(f"  Error loading PyAnnote: {e}")

    if not seg_embeddings:
        print("\nNo embeddings could be generated. Exiting.")
        return

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    # Segment-only plots (cleaner, no mixing with reference)
    print("\n[1/6] Creating segment-only T-SNE plot...")
    plot_embeddings_comparison(seg_embeddings, seg_labels, 'tsne', 'embedding_segments_tsne.png')

    print("\n[2/6] Creating segment-only PCA plot...")
    plot_embeddings_comparison(seg_embeddings, seg_labels, 'pca', 'embedding_segments_pca.png')

    if HAS_UMAP:
        print("\n[3/6] Creating segment-only UMAP plot...")
        plot_embeddings_comparison(seg_embeddings, seg_labels, 'umap', 'embedding_segments_umap.png')

    # Clustering comparison on segments only (all 4 methods + ground truth)
    print("\n[4/6] Creating clustering comparison plot (all methods)...")
    clustering_results = plot_clustering_comparison(seg_embeddings, seg_labels, 'embedding_clustering_comparison.png')

    # Print results table and save to JSON
    if clustering_results:
        print_clustering_results_table(clustering_results)
        save_clustering_results_json(clustering_results, 'embedding_clustering_results.json')

    # Combined view with segments
    print("\n[5/6] Creating combined view plot...")
    plot_combined_view(seg_embeddings, seg_labels, 'embedding_combined_view.png')

    # Reference-only plot if we have references
    if ref_embeddings and any(len(v) > 0 for v in ref_embeddings.values()):
        print("\n[6/6] Creating reference-only plot...")
        plot_reference_only(ref_embeddings, ref_labels, 'embedding_references_only.png')

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - embedding_segments_tsne.png (segments only)")
    print("  - embedding_segments_pca.png (segments only)")
    if HAS_UMAP:
        print("  - embedding_segments_umap.png (segments only)")
    print("  - embedding_clustering_comparison.png (5 methods: GT, KMeans, DBSCAN, Hierarchical, Cosine)")
    print("  - embedding_clustering_results.json (F1, Precision, Recall, ARI, Silhouette)")
    print("  - embedding_combined_view.png")
    print("  - embedding_references_only.png (reference speakers)")


if __name__ == "__main__":
    main()
