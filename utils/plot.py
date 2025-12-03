from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger


def plot_metrics(
    models: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """Generate metrics comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.35

    silero_means = [models['silero']['aggregated'][m]['mean'] for m in metrics]
    pyannote_means = [models['pyannote']['aggregated'][m]['mean'] for m in metrics]

    ax.bar(x - width/2, silero_means, width,
           label='Silero VAD', color='#2E86AB')
    ax.bar(x + width/2, pyannote_means, width,
           label='Pyannote VAD', color='#A23B72')

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(['Precision', 'Recall', 'F1'], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    total_duration_min = dataset_info['total_duration_seconds'] / 60
    title = f"VAD Model Comparison - Metrics\n"
    title += f"Language: {dataset_info['language'].title()} | "
    title += f"Files: {dataset_info['num_files']} | "
    title += f"Duration: {total_duration_min:.1f} min\n"
    title += f"Hardware: {system_info['gpu']} ({system_info['vram']})"
    ax.set_title(title, fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved metrics plot: {output_path}")


def plot_timing(
    models: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """Generate timing comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    model_names = ['Silero VAD', 'Pyannote VAD']
    means = [
        models['silero']['aggregated']['timing']['mean'],
        models['pyannote']['aggregated']['timing']['mean']
    ]
    totals = [
        models['silero']['aggregated']['timing']['total'],
        models['pyannote']['aggregated']['timing']['total']
    ]

    x = np.arange(len(model_names))
    bars = ax.bar(x, means, color=['#2E86AB', '#A23B72'])

    max_height = max(means)
    text_height_estimate = max_height * 0.08
    y_max = max_height + text_height_estimate + (max_height * 0.15)
    ax.set_ylim(0, y_max)

    for i, (bar, total) in enumerate(zip(bars, totals)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max_height * 0.03),
                f'Total: {total:.1f}s',
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Mean Time per File (seconds)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    total_duration_min = dataset_info['total_duration_seconds'] / 60
    title = f"VAD Model Comparison - Inference Time\n"
    title += f"Language: {dataset_info['language'].title()} | "
    title += f"Files: {dataset_info['num_files']} | "
    title += f"Duration: {total_duration_min:.1f} min\n"
    title += f"Hardware: {system_info['gpu']} ({system_info['vram']})"
    ax.set_title(title, fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved timing plot: {output_path}")


def plot_sc_timing(
    models: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """Generate SC timing comparison plot (stacked: embedding + clustering)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = list(models.keys())
    emb_times = [models[name]['aggregated']['embedding_timing']['mean'] for name in model_names]
    clus_times = [models[name]['aggregated']['clustering_timing']['mean'] for name in model_names]
    total_times = [emb + clus for emb, clus in zip(emb_times, clus_times)]
    totals = [models[name]['aggregated']['embedding_timing']['total'] + models[name]['aggregated']['clustering_timing']['total'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.6

    bars1 = ax.bar(x, emb_times, width, label='Embedding', color='#2E86AB')
    bars2 = ax.bar(x, clus_times, width, bottom=emb_times, label='Clustering', color='#A23B72', alpha=0.7)

    max_height = max(total_times)
    text_height_estimate = max_height * 0.08
    y_max = max_height + text_height_estimate + (max_height * 0.15)
    ax.set_ylim(0, y_max)

    for i, (total_time, total_sum) in enumerate(zip(total_times, totals)):
        ax.text(x[i], total_time + (max_height * 0.03),
                f'Total: {total_sum:.1f}s',
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Mean Time per File (seconds)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', '\n') for name in model_names], fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    total_duration_min = dataset_info['total_duration_seconds'] / 60
    title = f"Speaker Clustering Comparison - Timing\n"
    title += f"Language: {dataset_info['language'].title()} | "
    title += f"Files: {dataset_info['num_files']} | "
    title += f"Duration: {total_duration_min:.1f} min\n"
    title += f"Hardware: {system_info['gpu']} ({system_info['vram']})"
    ax.set_title(title, fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved SC timing plot: {output_path}")


def plot_sc_silhouette(
    models: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """Generate SC silhouette score comparison plot (only KMeans, excludes DBSCAN)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    kmeans_models = {name: data for name, data in models.items() if 'kmeans' in name}
    model_names = list(kmeans_models.keys())
    silhouette_scores = [kmeans_models[name]['aggregated']['silhouette']['mean'] for name in model_names]

    x = np.arange(len(model_names))

    embedding_colors = {
        'speechbrain': '#2E86AB',
        'pyannote': '#A23B72',
        'wav2vec2': '#F18F01'
    }

    bar_colors = []
    for name in model_names:
        for emb_name, color in embedding_colors.items():
            if emb_name in name:
                bar_colors.append(color)
                break

    bars = ax.bar(x, silhouette_scores, color=bar_colors, alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_ylabel('Silhouette Score', fontsize=12)
    max_score = max(silhouette_scores) if silhouette_scores else 0.5
    min_score = min(silhouette_scores) if silhouette_scores else -0.5
    y_margin = max(0.1, (max_score - min_score) * 0.2)
    ax.set_ylim(min_score - y_margin, max_score + y_margin)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', '\n').replace('kmeans', 'KMeans') for name in model_names], fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    total_duration_min = dataset_info['total_duration_seconds'] / 60
    title = f"Speaker Clustering Comparison - Silhouette Score\n"
    title += f"Language: {dataset_info['language'].title()} | "
    title += f"Files: {dataset_info['num_files']} | "
    title += f"Duration: {total_duration_min:.1f} min\n"
    title += f"Hardware: {system_info['gpu']} ({system_info['vram']})"
    ax.set_title(title, fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved SC silhouette plot: {output_path}")


def plot_sc_der(
    models: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """Generate SC DER metrics comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    metrics = ['der', 'miss', 'confusion']
    metric_labels = ['DER', 'Miss Rate', 'Confusion']
    model_names = list(models.keys())

    x = np.arange(len(metrics))
    width = 0.12
    num_models = len(model_names)

    embedding_colors = {
        'speechbrain_kmeans': '#2E86AB',
        'speechbrain_dbscan': '#4A9CC1',
        'pyannote_kmeans': '#A23B72',
        'pyannote_dbscan': '#C65996',
        'wav2vec2_kmeans': '#F18F01',
        'wav2vec2_dbscan': '#F5A742'
    }

    for i, model_name in enumerate(model_names):
        means = [models[model_name]['aggregated'][m]['mean'] for m in metrics]
        offset = (i - num_models/2) * width + width/2
        color = embedding_colors.get(model_name, '#808080')
        ax.bar(x + offset, means, width, label=model_name.replace('_', ' + '), color=color)

    ax.set_ylabel('Score (%)', fontsize=12)
    max_val = max([models[name]['aggregated'][m]['mean'] for name in model_names for m in metrics])
    ax.set_ylim(0, max_val * 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    total_duration_min = dataset_info['total_duration_seconds'] / 60
    title = f"Speaker Clustering Comparison - DER Metrics\n"
    title += f"Language: {dataset_info['language'].title()} | "
    title += f"Files: {dataset_info['num_files']} | "
    title += f"Duration: {total_duration_min:.1f} min\n"
    title += f"Hardware: {system_info['gpu']} ({system_info['vram']})"
    ax.set_title(title, fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved SC DER plot: {output_path}")
