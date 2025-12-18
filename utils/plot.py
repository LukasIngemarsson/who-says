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
    fig, ax = plt.subplots(figsize=(14, 6))

    # Filter to models with aggregated data
    model_names = [name for name in models.keys() if 'aggregated' in models[name]]
    emb_times = [models[name]['aggregated']['embedding_timing']['mean'] for name in model_names]
    clus_times = [models[name]['aggregated']['clustering_timing']['mean'] for name in model_names]
    total_times = [emb + clus for emb, clus in zip(emb_times, clus_times)]
    totals = [models[name]['aggregated']['embedding_timing']['total'] + models[name]['aggregated']['clustering_timing']['total'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.6

    bars1 = ax.bar(x, emb_times, width, label='Embedding', color='#2E86AB')
    bars2 = ax.bar(x, clus_times, width, bottom=emb_times, label='Clustering', color='#A23B72', alpha=0.7)

    max_height = max(total_times) if total_times else 1
    text_height_estimate = max_height * 0.08
    y_max = max_height + text_height_estimate + (max_height * 0.15)
    ax.set_ylim(0, y_max)

    for i, (total_time, total_sum) in enumerate(zip(total_times, totals)):
        ax.text(x[i], total_time + (max_height * 0.03),
                f'Total: {total_sum:.1f}s',
                ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Mean Time per File (seconds)', fontsize=12)
    ax.set_xticks(x)
    labels = [name.replace('_', '\n').replace('agglomerative', 'Agglom.').replace('naive', 'Naive') for name in model_names]
    ax.set_xticklabels(labels, fontsize=8)
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
    """Generate SC silhouette score comparison plot (excludes naive which has no silhouette)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter to models with silhouette (exclude naive)
    silhouette_models = {name: data for name, data in models.items()
                         if 'naive' not in name and 'aggregated' in data}
    model_names = list(silhouette_models.keys())
    silhouette_scores = [silhouette_models[name]['aggregated']['silhouette']['mean'] for name in model_names]

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
        else:
            bar_colors.append('#808080')

    bars = ax.bar(x, silhouette_scores, color=bar_colors, alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_ylabel('Silhouette Score', fontsize=12)
    max_score = max(silhouette_scores) if silhouette_scores else 0.5
    min_score = min(silhouette_scores) if silhouette_scores else -0.5
    y_margin = max(0.1, (max_score - min_score) * 0.2)
    ax.set_ylim(min_score - y_margin, max_score + y_margin)
    ax.set_xticks(x)
    labels = [name.replace('_', '\n').replace('kmeans', 'KMeans').replace('agglomerative', 'Agglom.').replace('dbscan', 'DBSCAN') for name in model_names]
    ax.set_xticklabels(labels, fontsize=9)
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
    fig, ax = plt.subplots(figsize=(12, 7))

    metrics = ['der', 'miss', 'false_alarm', 'confusion']
    metric_labels = ['DER', 'Miss Rate', 'False Alarm', 'Confusion']
    model_names = list(models.keys())

    x = np.arange(len(metrics))
    width = 0.12
    num_models = len(model_names)

    # Color palette for different embedding + clustering combinations
    embedding_colors = {
        'speechbrain_kmeans': '#2E86AB',
        'speechbrain_agglomerative': '#1E5F7A',
        'speechbrain_dbscan': '#4A9CC1',
        'speechbrain_naive': '#6BB8D9',
        'pyannote_kmeans': '#A23B72',
        'pyannote_agglomerative': '#7A2D56',
        'pyannote_dbscan': '#C65996',
        'pyannote_naive': '#E07AB8',
        'wav2vec2_kmeans': '#F18F01',
        'wav2vec2_agglomerative': '#C47200',
        'wav2vec2_dbscan': '#F5A742',
        'wav2vec2_naive': '#F9C07A'
    }

    for i, model_name in enumerate(model_names):
        if 'aggregated' not in models[model_name]:
            continue
        means = [models[model_name]['aggregated'][m]['mean'] for m in metrics]
        offset = (i - num_models/2) * width + width/2
        color = embedding_colors.get(model_name, '#808080')
        label = model_name.replace('_', ' + ').replace('naive', 'Naive').replace('agglomerative', 'Agglom.')
        ax.bar(x + offset, means, width, label=label, color=color)

    ax.set_ylabel('Score (%)', fontsize=12)
    max_val = max([models[name]['aggregated'][m]['mean'] for name in model_names for m in metrics if 'aggregated' in models[name]], default=100)
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


def plot_sc_clustering_metrics(
    models: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """
    Generate SC clustering metrics comparison plot (ARI, F1, Silhouette).
    Similar to cluster-viz style - grouped by embedding type, comparing clustering methods.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Group models by embedding type
    embedding_types = ['speechbrain', 'pyannote', 'wav2vec2']
    clustering_methods = ['kmeans', 'agglomerative', 'dbscan', 'naive']
    metrics = ['ari', 'f1', 'silhouette']
    metric_labels = ['ARI', 'F1', 'Silhouette']

    # Filter to only available embeddings
    available_embeddings = []
    for emb in embedding_types:
        if any(emb in name for name in models.keys()):
            available_embeddings.append(emb)

    x = np.arange(len(clustering_methods))
    width = 0.25
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red

    embedding_labels = {
        'speechbrain': 'SpeechBrain',
        'pyannote': 'PyAnnote',
        'wav2vec2': 'Wav2Vec2'
    }

    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # For silhouette, exclude naive method (no silhouette for naive)
        if metric == 'silhouette':
            methods_to_plot = ['kmeans', 'agglomerative', 'dbscan']
            x_positions = np.arange(len(methods_to_plot))
        else:
            methods_to_plot = clustering_methods
            x_positions = x

        for i, emb_type in enumerate(available_embeddings):
            values = []
            for method in methods_to_plot:
                key = f'{emb_type}_{method}'
                if key in models and 'aggregated' in models[key]:
                    values.append(models[key]['aggregated'][metric]['mean'])
                else:
                    values.append(0)

            bars = ax.bar(x_positions + i * width, values, width,
                         label=embedding_labels.get(emb_type, emb_type),
                         color=colors[i % len(colors)])

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Clustering Method')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} Comparison')
        ax.set_xticks(x_positions + width)

        method_labels = ['K-Means', 'Agglom.', 'DBSCAN', 'Naive']
        if metric == 'silhouette':
            ax.set_xticklabels(method_labels[:3])
        else:
            ax.set_xticklabels(method_labels)

        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

    total_duration_min = dataset_info['total_duration_seconds'] / 60
    title = f"Speaker Clustering Performance Comparison\n"
    title += f"Language: {dataset_info['language'].title()} | "
    title += f"Files: {dataset_info['num_files']} | "
    title += f"Duration: {total_duration_min:.1f} min | "
    title += f"Hardware: {system_info['gpu']}"
    plt.suptitle(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved SC clustering metrics plot: {output_path}")


def plot_asr_wer(
    models: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """Generate ASR WER comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = list(models.keys())
    wer_means = [models[name]['aggregated']['wer']['mean'] for name in model_names]

    x = np.arange(len(model_names))
    bars = ax.bar(x, wer_means, color='#2E86AB')

    ax.set_ylabel('WER (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    total_duration_min = dataset_info['total_duration_seconds'] / 60
    title = f"ASR Model Comparison - Word Error Rate\n"
    title += f"Language: {dataset_info['language'].title()} | "
    title += f"Files: {dataset_info['num_files']} | "
    title += f"Duration: {total_duration_min:.1f} min\n"
    title += f"Hardware: {system_info['gpu']} ({system_info['vram']})"
    ax.set_title(title, fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved ASR WER plot: {output_path}")


def plot_asr_timing(
    models: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """Generate ASR timing comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = list(models.keys())
    means = [models[name]['aggregated']['timing']['mean'] for name in model_names]
    totals = [models[name]['aggregated']['timing']['total'] for name in model_names]

    x = np.arange(len(model_names))
    bars = ax.bar(x, means, color='#A23B72')

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
    ax.set_xlabel('Model', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    total_duration_min = dataset_info['total_duration_seconds'] / 60
    title = f"ASR Model Comparison - Inference Time\n"
    title += f"Language: {dataset_info['language'].title()} | "
    title += f"Files: {dataset_info['num_files']} | "
    title += f"Duration: {total_duration_min:.1f} min\n"
    title += f"Hardware: {system_info['gpu']} ({system_info['vram']})"
    ax.set_title(title, fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved ASR timing plot: {output_path}")


def plot_e2e_der(
    pipelines: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """Generate end-to-end DER comparison plot."""
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics = ['der', 'miss', 'false_alarm', 'confusion']
    metric_labels = ['DER', 'Miss Rate', 'False Alarm', 'Confusion']
    pipeline_names = [name for name in pipelines.keys() if 'aggregated' in pipelines[name]]

    x = np.arange(len(metrics))
    num_pipelines = len(pipeline_names)
    width = 0.8 / num_pipelines

    import matplotlib.cm as cm
    color_groups = {
        'who-says': '#2E86AB',
        'pyannote-3.1': '#F18F01',
    }

    whisperx_cmap = cm.get_cmap('Purples')
    whisperx_models = ['tiny', 'medium', 'large-v3', 'distil-large-v3']
    for i, model in enumerate(whisperx_models):
        color_groups[f'whisperx-{model}'] = whisperx_cmap(0.3 + i * 0.15)

    for i, pipeline_name in enumerate(pipeline_names):
        agg = pipelines[pipeline_name]['aggregated']
        means = [agg[m]['mean'] for m in metrics]
        offset = (i - num_pipelines/2) * width + width/2
        color = color_groups.get(pipeline_name, '#808080')

        model_info = pipelines[pipeline_name].get('model_info', '')
        if model_info:
            label = f"{pipeline_name}\n({model_info})"
        else:
            label = pipeline_name

        ax.bar(x + offset, means, width, label=label, color=color)

    ax.set_ylabel('Score (%)', fontsize=12)
    max_val = max([pipelines[name]['aggregated'][m]['mean']
                   for name in pipeline_names
                   for m in metrics])
    ax.set_ylim(0, max_val * 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)

    if num_pipelines > 4:
        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    else:
        ax.legend(fontsize=10, loc='upper right')

    ax.grid(axis='y', alpha=0.3)

    total_duration_min = dataset_info['total_duration_seconds'] / 60
    title = f"End-to-End Pipeline Comparison - DER Metrics\n"
    title += f"Language: {dataset_info['language'].title()} | "
    title += f"Files: {dataset_info['num_files']} | "
    title += f"Duration: {total_duration_min:.1f} min\n"
    title += f"Hardware: {system_info['gpu']} ({system_info['vram']})"
    ax.set_title(title, fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved E2E DER plot: {output_path}")


def plot_e2e_wer(
    pipelines: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """Generate end-to-end WER comparison plot (only pipelines with ASR)."""
    transcription_pipelines = {
        name: data for name, data in pipelines.items()
        if data['has_transcription'] and 'wer' in data.get('aggregated', {})
    }

    if not transcription_pipelines:
        logger.warning("No pipelines with WER metrics to plot")
        return

    pipeline_names = list(transcription_pipelines.keys())
    wer_means = [transcription_pipelines[name]['aggregated']['wer']['mean']
                 for name in pipeline_names]

    fig_width = max(10, len(pipeline_names) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    x = np.arange(len(pipeline_names))

    import matplotlib.cm as cm
    colors = []
    for name in pipeline_names:
        if name == 'who-says':
            colors.append('#2E86AB')
        elif name.startswith('whisperx'):
            whisperx_models = ['tiny', 'medium', 'large-v3', 'distil-large-v3']
            for i, model in enumerate(whisperx_models):
                if name == f'whisperx-{model}':
                    cmap = cm.get_cmap('Purples')
                    colors.append(cmap(0.3 + i * 0.15))
                    break
        else:
            colors.append('#808080')

    bars = ax.bar(x, wer_means, color=colors)

    ax.set_ylabel('WER (%)', fontsize=12)
    ax.set_xticks(x)

    labels_with_info = []
    for name in pipeline_names:
        model_info = transcription_pipelines[name].get('model_info', '')
        if model_info and len(pipeline_names) <= 4:
            labels_with_info.append(f"{name}\n({model_info})")
        else:
            labels_with_info.append(name)

    ax.set_xticklabels(labels_with_info, fontsize=10, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    total_duration_min = dataset_info['total_duration_seconds'] / 60
    title = f"End-to-End Pipeline Comparison - Word Error Rate\n"
    title += f"Language: {dataset_info['language'].title()} | "
    title += f"Files: {dataset_info['num_files']} | "
    title += f"Duration: {total_duration_min:.1f} min\n"
    title += f"Hardware: {system_info['gpu']} ({system_info['vram']})"
    ax.set_title(title, fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved E2E WER plot: {output_path}")


def plot_e2e_timing(
    pipelines: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """Generate end-to-end timing comparison plot with stacked ASR time for WhoSays."""
    pipeline_names = [name for name in pipelines.keys() if 'aggregated' in pipelines[name]]
    diarization_means = []
    asr_means = []
    totals = []

    for name in pipeline_names:
        agg = pipelines[name]['aggregated']
        if name == 'who-says' and 'component_timing' in agg and 'asr' in agg['component_timing']:
            diar_time = agg['timing']['mean']
            asr_time = agg['component_timing']['asr']['mean']
            diarization_means.append(diar_time)
            asr_means.append(asr_time)
            totals.append(agg['timing']['total'] + agg['component_timing']['asr']['total'])
        else:
            diarization_means.append(agg['timing']['mean'])
            asr_means.append(0)
            totals.append(agg['timing']['total'])

    fig_width = max(10, len(pipeline_names) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    x = np.arange(len(pipeline_names))

    import matplotlib.cm as cm
    color_groups = {
        'who-says': '#2E86AB',
        'pyannote-3.1': '#F18F01',
    }

    whisperx_cmap = cm.get_cmap('Purples')
    whisperx_models = ['tiny', 'medium', 'large-v3', 'distil-large-v3']
    for i, model in enumerate(whisperx_models):
        color_groups[f'whisperx-{model}'] = whisperx_cmap(0.3 + i * 0.15)

    bar_colors = [color_groups.get(name, '#808080') for name in pipeline_names]

    bars_diar = ax.bar(x, diarization_means, color=bar_colors, label='Diarization')
    bars_asr = ax.bar(x, asr_means, bottom=diarization_means, color='#90C978', label='ASR (WhoSays only)', alpha=0.8)

    max_height = max([d + a for d, a in zip(diarization_means, asr_means)])
    text_height_estimate = max_height * 0.08
    y_max = max_height + text_height_estimate + (max_height * 0.15)
    ax.set_ylim(0, y_max)

    for i, total in enumerate(totals):
        height = diarization_means[i] + asr_means[i]
        ax.text(x[i], height + (max_height * 0.03),
                f'{total:.1f}s',
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Mean Time per File (seconds)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pipeline_names, fontsize=10, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    total_duration_min = dataset_info['total_duration_seconds'] / 60
    title = f"End-to-End Pipeline Comparison - Inference Timing\n"
    title += f"WhoSays = diarization + ASR | Pyannote = diarization only | WhisperX = diarization + ASR \n"
    title += f"Language: {dataset_info['language'].title()} | "
    title += f"Files: {dataset_info['num_files']} | "
    title += f"Duration: {total_duration_min:.1f} min\n"
    title += f"Hardware: {system_info['gpu']} ({system_info['vram']})"
    ax.set_title(title, fontsize=10, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved E2E timing plot: {output_path}")


def plot_embedding_comparison(
    results: Dict,
    system_info: Dict,
    output_path: Path
):
    """
    Generate embedding model comparison plot showing ARI, F1, and Silhouette/Accuracy scores.

    Parameters
    ----------
    results : Dict
        Results from compare_embedding_models containing embeddings metrics
    system_info : Dict
        System information (GPU, etc.)
    output_path : Path
        Path to save the plot
    """
    embeddings_data = results.get('embeddings', {})
    if not embeddings_data:
        logger.warning("No embedding data to plot")
        return

    # Check if cosine similarity is available
    has_cosine = any('cosine' in emb_data for emb_data in embeddings_data.values())

    # Prepare data
    embedding_names = list(embeddings_data.keys())
    if has_cosine:
        clustering_methods = ['cosine', 'kmeans', 'agglomerative', 'dbscan']
        method_labels = ['Cosine', 'KMeans', 'Agglomerative', 'DBSCAN']
    else:
        clustering_methods = ['kmeans', 'agglomerative', 'dbscan']
        method_labels = ['KMeans', 'Agglomerative', 'DBSCAN']

    metrics = ['ari', 'f1']
    metric_labels = ['ARI', 'F1 Score']

    # Add accuracy or silhouette as 3rd metric depending on cosine availability
    if has_cosine:
        # Create figure with 3 subplots: ARI, F1, and Accuracy (for cosine)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        metrics.append('accuracy')
        metric_labels.append('Accuracy (Cosine Only)')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics.append('silhouette')
        metric_labels.append('Silhouette')

    fig.suptitle(f'Embedding Model Comparison\n{results["num_speakers"]} speakers | Hardware: {system_info.get("gpu", "N/A")}',
                 fontsize=12, y=1.02)

    # Colors for embedding models
    colors = {
        'SpeechBrain': '#2E86AB',
        'PyAnnote': '#A23B72',
        'Wav2Vec2': '#F18F01'
    }

    x = np.arange(len(clustering_methods))
    width = 0.25
    n_embeddings = len(embedding_names)

    for metric_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[metric_idx]

        for emb_idx, emb_name in enumerate(embedding_names):
            emb_data = embeddings_data[emb_name]
            values = []
            for method in clustering_methods:
                if method in emb_data and metric in emb_data[method]:
                    values.append(emb_data[method][metric])
                elif metric == 'accuracy' and method != 'cosine':
                    # Accuracy only applies to cosine similarity
                    values.append(0)
                else:
                    values.append(0)

            offset = (emb_idx - n_embeddings/2) * width + width/2
            color = colors.get(emb_name, '#808080')
            dim = emb_data.get('embedding_dim', '?')
            ax.bar(x + offset, values, width, label=f'{emb_name} ({dim}d)', color=color)

        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=7, padding=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved embedding comparison plot: {output_path}")


def plot_cluster_umap(
    results: Dict,
    system_info: Dict,
    output_path: Path
):
    """
    Generate UMAP visualization of embeddings colored by clustering results.

    Parameters
    ----------
    results : Dict
        Results from compare_cluster_viz containing:
        - umap_coords: 2D UMAP coordinates
        - ground_truth_labels: speaker labels
        - clustering_results: {method: {labels, silhouette, ari, f1}}
        - embedding_type: model name
        - num_speakers: count
        - is_reference: optional list of booleans indicating reference/alignment samples
    system_info : Dict
        System information (GPU, etc.)
    output_path : Path
        Path to save the plot
    """
    umap_coords = np.array(results.get('umap_coords', []))
    if len(umap_coords) == 0:
        logger.warning("No UMAP coordinates to plot")
        return

    ground_truth_labels = results.get('ground_truth_labels', [])
    clustering_results = results.get('clustering_results', {})
    embedding_type = results.get('embedding_type', 'Unknown')
    num_speakers = results.get('num_speakers', 0)
    speakers = results.get('speakers', [])
    is_reference = np.array(results.get('is_reference', [False] * len(umap_coords)))

    # Color palette for speakers
    speaker_colors = {
        'GOR': '#e41a1c',
        'JOHAN': '#377eb8',
        'MARTEN': '#4daf4a',
        'KALLE': '#984ea3',
        'LUKAS': '#ff7f00',
        'OSCAR': '#ffff33',
    }

    # Cluster colors (for non-speaker assignments)
    cluster_cmap = plt.cm.get_cmap('tab10')

    # Methods to plot: ground truth + clustering methods
    methods = ['ground_truth'] + list(clustering_results.keys())
    method_names = {
        'ground_truth': 'Ground Truth',
        'kmeans': 'KMeans',
        'agglomerative': 'Agglomerative',
        'dbscan': 'DBSCAN',
        'cosine': 'Cosine Similarity'
    }

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))

    if n_methods == 1:
        axes = [axes]

    fig.suptitle(
        f'Clustering Comparison - {embedding_type} Embeddings (UMAP)\n'
        f'{num_speakers} speakers, {len(umap_coords)} samples | Hardware: {system_info.get("gpu", "N/A")}',
        fontsize=12, y=1.02
    )

    for idx, method in enumerate(methods):
        ax = axes[idx]

        if method == 'ground_truth':
            # Color by ground truth speaker labels
            # First plot non-reference points (regular)
            for speaker in speakers:
                mask = np.array([l == speaker for l in ground_truth_labels])
                non_ref_mask = mask & ~is_reference
                color = speaker_colors.get(speaker, '#808080')
                if np.any(non_ref_mask):
                    ax.scatter(
                        umap_coords[non_ref_mask, 0],
                        umap_coords[non_ref_mask, 1],
                        c=color,
                        label=speaker,
                        alpha=0.7,
                        s=50,
                        edgecolors='white',
                        linewidth=0.5
                    )

            # Then plot reference points with highlight (star markers, black edge)
            ref_count = 0
            for speaker in speakers:
                mask = np.array([l == speaker for l in ground_truth_labels])
                ref_mask = mask & is_reference
                color = speaker_colors.get(speaker, '#808080')
                if np.any(ref_mask):
                    label = f'{speaker} (ref)' if ref_count == 0 else None
                    ax.scatter(
                        umap_coords[ref_mask, 0],
                        umap_coords[ref_mask, 1],
                        c=color,
                        label=label if ref_count == 0 else f'{speaker} (ref)',
                        alpha=1.0,
                        s=200,
                        marker='*',
                        edgecolors='black',
                        linewidth=1.5
                    )
                    ref_count += 1

            ax.set_title(method_names.get(method, method), fontsize=11)
            ax.legend(loc='best', fontsize=8)

            # Add annotation if references exist
            if np.any(is_reference):
                ax.text(
                    0.02, 0.02, f'★ = Reference ({np.sum(is_reference)} samples)',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='bottom', fontfamily='sans-serif',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8)
                )
        else:
            # Color by cluster assignment
            cluster_data = clustering_results.get(method, {})
            labels = np.array(cluster_data.get('labels', []))

            if len(labels) == 0:
                ax.set_title(f'{method_names.get(method, method)}\n(No data)', fontsize=11)
                continue

            # Reference samples have label -2 (they were not clustered)
            reference_mask = (labels == -2)

            # For cosine similarity, labels are already speaker indices - plot directly
            if method == 'cosine':
                # Labels are speaker indices (0, 1, 2...) or -2 for reference
                for speaker_idx, speaker in enumerate(speakers):
                    mask = (labels == speaker_idx)
                    color = speaker_colors.get(speaker, cluster_cmap(speaker_idx % 10))
                    if np.any(mask):
                        ax.scatter(
                            umap_coords[mask, 0],
                            umap_coords[mask, 1],
                            c=[color],
                            label=speaker,
                            alpha=0.7,
                            s=50,
                            edgecolors='white',
                            linewidth=0.5
                        )
                # No unknown for cosine - every sample gets assigned
                unknown_mask = np.zeros(len(labels), dtype=bool)
            else:
                # For other methods, use Hungarian algorithm to match clusters to speakers
                from scipy.optimize import linear_sum_assignment

                # Build ground truth indices
                gt_indices = np.array([speakers.index(l) for l in ground_truth_labels])
                unique_clusters = sorted(set(labels) - {-1, -2})  # Exclude noise and reference

                # Build cost matrix for Hungarian matching (only match up to n_speakers clusters)
                cluster_to_speaker = {}
                matched_clusters = set()
                if len(unique_clusters) > 0:
                    # Only use first n_speakers clusters for matching
                    n_to_match = min(len(unique_clusters), len(speakers))
                    cost_matrix = np.zeros((n_to_match, len(speakers)))
                    for i, cluster in enumerate(unique_clusters[:n_to_match]):
                        for j, speaker in enumerate(speakers):
                            # Count how many points in this cluster belong to this speaker
                            cluster_mask = labels == cluster
                            speaker_mask = gt_indices == j
                            overlap = np.sum(cluster_mask & speaker_mask)
                            cost_matrix[i, j] = -overlap  # Negative for minimization

                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for r, c in zip(row_ind, col_ind):
                        cluster_to_speaker[unique_clusters[r]] = speakers[c]
                        matched_clusters.add(unique_clusters[r])

                # Identify unmatched clusters (extra clusters beyond n_speakers) and noise
                unmatched_clusters = set(unique_clusters) - matched_clusters
                unknown_mask = np.zeros(len(labels), dtype=bool)
                for label in unmatched_clusters:
                    unknown_mask |= (labels == label)
                # Also include noise points (-1) in UNKNOWN
                unknown_mask |= (labels == -1)

                # Plot matched clusters with speaker names
                for label in sorted(matched_clusters):
                    mask = labels == label
                    speaker_name = cluster_to_speaker[label]
                    color = speaker_colors.get(speaker_name, cluster_cmap(label % 10))
                    if np.any(mask):
                        ax.scatter(
                            umap_coords[mask, 0],
                            umap_coords[mask, 1],
                            c=[color],
                            label=speaker_name,
                            alpha=0.7,
                            s=50,
                            edgecolors='white',
                            linewidth=0.5
                        )

                # Plot all unmatched/noise as single UNKNOWN group
                if np.any(unknown_mask):
                    ax.scatter(
                        umap_coords[unknown_mask, 0],
                        umap_coords[unknown_mask, 1],
                        c='#808080',
                        label='UNKNOWN',
                        alpha=0.7,
                        s=50,
                        edgecolors='white',
                        linewidth=0.5
                    )

            # Plot reference samples with star markers (they were not clustered)
            # Color them by their ground truth speaker label
            n_ref = np.sum(reference_mask)
            if n_ref > 0:
                for speaker in speakers:
                    speaker_gt_mask = np.array([l == speaker for l in ground_truth_labels])
                    ref_speaker_mask = reference_mask & speaker_gt_mask
                    color = speaker_colors.get(speaker, '#808080')
                    if np.any(ref_speaker_mask):
                        ax.scatter(
                            umap_coords[ref_speaker_mask, 0],
                            umap_coords[ref_speaker_mask, 1],
                            c=[color],
                            alpha=1.0,
                            s=200,
                            marker='*',
                            edgecolors='black',
                            linewidth=1.5
                        )

                # Add reference annotation
                ax.text(
                    0.02, 0.02, f'★ = Reference ({n_ref})',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='bottom', fontfamily='sans-serif',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8)
                )

            # Build metrics text
            metrics_text = []
            if 'ari' in cluster_data:
                metrics_text.append(f"ARI: {cluster_data['ari']:.3f}")
            if 'silhouette' in cluster_data:
                metrics_text.append(f"Sil: {cluster_data['silhouette']:.3f}")
            if 'f1' in cluster_data:
                metrics_text.append(f"F1: {cluster_data['f1']:.3f}")
            if 'accuracy' in cluster_data:
                metrics_text.append(f"Acc: {cluster_data['accuracy']:.3f}")
            if 'eps' in cluster_data:
                metrics_text.append(f"eps: {cluster_data['eps']:.2f}")
            if 'n_noise' in cluster_data and cluster_data['n_noise'] > 0:
                metrics_text.append(f"noise: {cluster_data['n_noise']}")

            ax.set_title(method_names.get(method, method), fontsize=11)

            if metrics_text:
                ax.text(
                    0.02, 0.98, '\n'.join(metrics_text),
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

            ax.legend(loc='best', fontsize=7)

        ax.set_xlabel('UMAP 1', fontsize=9)
        ax.set_ylabel('UMAP 2', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved cluster UMAP plot: {output_path}")


def plot_sos_comparison(
    results: Dict,
    system_info: Dict,
    output_path: Path
):
    """
    Generate Speech Overlap Separation (SOS) comparison plot.

    Shows SI-SDR metrics (if available) or energy ratio, and timing comparison
    between separation models (PyannoteSOS vs SpeechBrain SepFormer).

    Parameters
    ----------
    results : Dict
        Results from compare_sos_models containing:
        - ground_truth_overlaps: number of overlap regions
        - has_references: bool indicating if SI-SDR is available
        - pyannote: dict with metrics and region_results
        - sepformer: dict with metrics and region_results
    system_info : Dict
        System information (GPU, etc.)
    output_path : Path
        Path to save the plot
    """
    # Collect model data
    models = []
    model_names = []
    colors = []

    if results.get('pyannote') and 'error' not in results['pyannote']:
        models.append(results['pyannote'])
        model_names.append('PyannoteSOS\n(separation-ami-1.0)')
        colors.append('#A23B72')

    if results.get('sepformer') and 'error' not in results['sepformer']:
        models.append(results['sepformer'])
        model_names.append('SpeechBrain\n(SepFormer)')
        colors.append('#2E86AB')

    if not models:
        logger.warning("No SOS model results to plot")
        return

    has_references = results.get('has_references', False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(models))

    # Plot 1: Quality metric (SI-SDR if available, else Energy Ratio)
    ax1 = axes[0]
    if has_references:
        # SI-SDR plot
        si_sdrs = [m.get('mean_si_sdr', 0) for m in models]
        std_sdrs = [m.get('std_si_sdr', 0) for m in models]

        bars1 = ax1.bar(x, si_sdrs, color=colors, yerr=std_sdrs, capsize=5, alpha=0.8)
        ax1.set_ylabel('SI-SDR (dB)', fontsize=12)
        ax1.set_title('Source Separation Quality (SI-SDR)', fontsize=11)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        for bar, val, std in zip(bars1, si_sdrs, std_sdrs):
            if val != float('-inf') and val is not None:
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.5,
                        f'{val:.1f} dB', ha='center', va='bottom', fontsize=9)
    else:
        # Energy ratio plot
        energy_ratios = [m.get('mean_energy_ratio', 0) for m in models]
        std_ratios = [m.get('std_energy_ratio', 0) for m in models]

        bars1 = ax1.bar(x, energy_ratios, color=colors, yerr=std_ratios, capsize=5, alpha=0.8)
        ax1.set_ylabel('Energy Ratio', fontsize=12)
        ax1.set_title('Source Separation Quality (Energy Ratio)', fontsize=11)
        ax1.set_ylim(0, 1.1)

        for bar, val in zip(bars1, energy_ratios):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Success rate (percentage of positive SI-SDR)
    ax2 = axes[1]
    if has_references:
        success_rates = []
        for m in models:
            region_results = m.get('region_results', [])
            if region_results:
                positive_count = sum(1 for r in region_results
                                    if r.get('avg_si_sdr') is not None and r.get('avg_si_sdr') > 0)
                success_rate = 100 * positive_count / len(region_results)
            else:
                success_rate = 0
            success_rates.append(success_rate)

        bars2 = ax2.bar(x, success_rates, color=colors, alpha=0.8)
        ax2.set_ylabel('Success Rate (%)', fontsize=12)
        ax2.set_title('Success Rate (SI-SDR > 0 dB)', fontsize=11)
        ax2.set_ylim(0, 105)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

        for bar, val in zip(bars2, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    else:
        # If no SI-SDR, show number of sources
        num_sources = [m.get('mean_num_sources', 0) for m in models]
        bars2 = ax2.bar(x, num_sources, color=colors, alpha=0.8)
        ax2.set_ylabel('Avg Sources', fontsize=12)
        ax2.set_title('Average Number of Sources', fontsize=11)

        for bar, val in zip(bars2, num_sources):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Total processing time
    ax3 = axes[2]
    times = [m['total_time'] for m in models]

    bars3 = ax3.bar(x, times, color=colors, alpha=0.8)
    ax3.set_ylabel('Total Time (seconds)', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, fontsize=10)
    ax3.set_title('Processing Time', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars3, times):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(times) * 0.02,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)

    # Overall title
    n_overlaps = results.get('ground_truth_overlaps', 0)
    metric_type = "SI-SDR" if has_references else "Energy Ratio"
    title = f"Speech Overlap Separation (SOS) Comparison\n"
    title += f"Overlap Regions: {n_overlaps} | Metric: {metric_type} | Hardware: {system_info.get('gpu', 'N/A')}"
    fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved SOS comparison plot: {output_path}")


def plot_sod_comparison(
    results: Dict,
    system_info: Dict,
    output_path: Path
):
    """
    Generate Speech Overlap Detection (SOD) comparison plot.

    Shows grouped bar graph comparing F1, Precision, and Recall for each model.

    Parameters
    ----------
    results : Dict
        Results from compare_sod_models containing:
        - ground_truth_count: number of overlap regions
        - ground_truth_duration: total duration of overlaps
        - aggregated: dict with best results for each model
    system_info : Dict
        System information (GPU, etc.)
    output_path : Path
        Path to save the plot
    """
    aggregated = results.get('aggregated', {})
    if not aggregated:
        logger.warning("No aggregated SOD results to plot")
        return

    optimize_metric = aggregated.get('optimize_metric', 'frame_f1')

    # Collect best results for each model
    model_data = []

    # Pyannote best
    best_pyannote = aggregated.get('best_pyannote', {})
    if best_pyannote:
        time_str = f"{best_pyannote.get('time', 0):.2f}s"
        model_data.append({
            'name': 'Pyannote',
            'label': f"Pyannote\n({time_str})",
            'frame_f1': best_pyannote.get('frame_f1', 0),
            'frame_precision': best_pyannote.get('frame_precision', 0),
            'frame_recall': best_pyannote.get('frame_recall', 0),
            'segment_f1': best_pyannote.get('segment_f1_iou05', 0),
            'color': '#3498DB'
        })

    # WavLM best
    best_wavlm = aggregated.get('best_wavlm', {})
    if best_wavlm:
        time_str = f"{best_wavlm.get('time', 0):.2f}s"
        model_data.append({
            'name': 'WavLM',
            'label': f"WavLM\n({time_str})",
            'frame_f1': best_wavlm.get('frame_f1', 0),
            'frame_precision': best_wavlm.get('frame_precision', 0),
            'frame_recall': best_wavlm.get('frame_recall', 0),
            'segment_f1': best_wavlm.get('segment_f1_iou05', 0),
            'color': '#27AE60'
        })

    # NeMo
    nemo = aggregated.get('nemo', {})
    if nemo:
        time_str = f"{nemo.get('time', 0):.2f}s"
        model_data.append({
            'name': 'NeMo',
            'label': f"NeMo\n({time_str})",
            'frame_f1': nemo.get('frame_f1', 0),
            'frame_precision': nemo.get('frame_precision', 0),
            'frame_recall': nemo.get('frame_recall', 0),
            'segment_f1': nemo.get('segment_f1_iou05', 0),
            'color': '#E74C3C'
        })

    if not model_data:
        logger.warning("No valid SOD model results to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(model_data))
    bar_width = 0.25

    # Frame-level metrics (F1, Precision, Recall)
    f1_vals = [m['frame_f1'] for m in model_data]
    prec_vals = [m['frame_precision'] for m in model_data]
    rec_vals = [m['frame_recall'] for m in model_data]

    bars_f1 = ax.bar(x - bar_width, f1_vals, bar_width, label='F1', color='#3498DB', alpha=0.85, edgecolor='black', linewidth=1)
    bars_prec = ax.bar(x, prec_vals, bar_width, label='Precision', color='#27AE60', alpha=0.85, edgecolor='black', linewidth=1)
    bars_rec = ax.bar(x + bar_width, rec_vals, bar_width, label='Recall', color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bars, vals in [(bars_f1, f1_vals), (bars_prec, prec_vals), (bars_rec, rec_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Frame-level Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m['label'] for m in model_data], fontsize=11)
    all_vals = f1_vals + prec_vals + rec_vals
    ax.set_ylim(0, max(all_vals) * 1.2 if all_vals else 100)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # Overall title
    gt_count = results.get('ground_truth_count', 0)
    gt_duration = results.get('ground_truth_duration', 0)

    title = f"Speech Overlap Detection (SOD) - Best Model Comparison\n"
    title += f"Ground Truth: {gt_count} overlaps ({gt_duration:.1f}s) | {system_info.get('gpu', 'N/A')}"
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved SOD comparison plot: {output_path}")


def plot_scd_comparison(
    results: Dict,
    system_info: Dict,
    output_path: Path
):
    """
    Generate Speaker Change Detection (SCD) comparison plot.

    Shows grouped bar graph comparing F1, Precision, and Recall for each model at tolerance 3.0s.

    Parameters
    ----------
    results : Dict
        Results from compare_scd_models containing model results and aggregated data
    system_info : Dict
        System information (GPU, etc.)
    output_path : Path
        Path to save the plot
    """
    aggregated = results.get('aggregated', {})

    if not aggregated:
        logger.warning("No SCD results to plot")
        return

    # Use tolerance 3.0s
    tol_key = 'tolerance_3.0s'
    optimize_metric = aggregated.get('optimize_metric', 'f1')

    # Define colors for each model
    model_colors = {
        'pyannote': '#3498DB',
        'nemo': '#E74C3C',
        'naive_pyannote': '#27AE60',
        'naive_speechbrain': '#9B59B6',
        'naive_wav2vec2': '#F39C12'
    }

    # Collect results for each model
    model_data = []

    # Pyannote
    pyannote_by_tol = aggregated.get('best_pyannote_by_tolerance', {})
    if pyannote_by_tol.get(tol_key):
        config = pyannote_by_tol[tol_key]
        model_data.append({
            'name': 'pyannote',
            'label': f"Pyannote\n(prom={config['prominence']})",
            'f1': config['f1'] * 100,
            'precision': config['precision'] * 100,
            'recall': config['recall'] * 100,
            'color': model_colors['pyannote']
        })

    # NeMo
    nemo_by_tol = aggregated.get('nemo_by_tolerance', {})
    if nemo_by_tol and nemo_by_tol.get(tol_key):
        metrics = nemo_by_tol[tol_key]
        model_data.append({
            'name': 'nemo',
            'label': 'NeMo\nSortformer',
            'f1': metrics['f1'] * 100,
            'precision': metrics['precision'] * 100,
            'recall': metrics['recall'] * 100,
            'color': model_colors['nemo']
        })

    # Naive models
    for naive_model in ['naive_pyannote', 'naive_speechbrain', 'naive_wav2vec2']:
        naive_by_tol = aggregated.get(f'best_{naive_model}_by_tolerance', {})
        if naive_by_tol and naive_by_tol.get(tol_key):
            config = naive_by_tol[tol_key]
            emb_name = naive_model.replace('naive_', '')
            model_data.append({
                'name': naive_model,
                'label': f"Naive\n({emb_name})",
                'f1': config['f1'] * 100,
                'precision': config['precision'] * 100,
                'recall': config['recall'] * 100,
                'color': model_colors[naive_model]
            })

    if not model_data:
        logger.warning("No valid SCD model results to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(model_data))
    bar_width = 0.25

    # Create grouped bars for F1, Precision, Recall
    f1_vals = [m['f1'] for m in model_data]
    prec_vals = [m['precision'] for m in model_data]
    rec_vals = [m['recall'] for m in model_data]

    bars_f1 = ax.bar(x - bar_width, f1_vals, bar_width, label='F1', color='#3498DB', alpha=0.85, edgecolor='black', linewidth=1)
    bars_prec = ax.bar(x, prec_vals, bar_width, label='Precision', color='#27AE60', alpha=0.85, edgecolor='black', linewidth=1)
    bars_rec = ax.bar(x + bar_width, rec_vals, bar_width, label='Recall', color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bars, vals in [(bars_f1, f1_vals), (bars_prec, prec_vals), (bars_rec, rec_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Highlight best model (by optimized metric) with gold border
    metric_map = {'f1': (bars_f1, f1_vals), 'precision': (bars_prec, prec_vals), 'recall': (bars_rec, rec_vals)}
    if optimize_metric in metric_map:
        opt_bars, opt_vals = metric_map[optimize_metric]
        best_idx = opt_vals.index(max(opt_vals))
        opt_bars[best_idx].set_edgecolor('#FFD700')
        opt_bars[best_idx].set_linewidth(3)

    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Speaker Change Detection - Model Comparison (optimized: {optimize_metric})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m['label'] for m in model_data], fontsize=11)
    all_vals = f1_vals + prec_vals + rec_vals
    ax.set_ylim(0, max(all_vals) * 1.2 if all_vals else 100)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    # Overall title
    gt_count = results.get('ground_truth_count', 0)
    total_duration = results.get('total_duration', 0)

    title = f"Speaker Change Detection (SCD) Comparison\n"
    title += f"Ground Truth: {gt_count} changes | Audio: {total_duration:.1f}s | {system_info.get('gpu', 'N/A')}"
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved SCD comparison plot: {output_path}")


def plot_full_e2e_comparison(
    results: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """
    Generate comprehensive comparison plots for full-e2e pipeline combinations.

    Creates a multi-panel figure showing:
    1. Top 10 configurations by DER
    2. DER component breakdown (miss, false alarm, confusion)
    3. Timing comparison
    4. WER comparison (if available)

    Args:
        results: Dict from aggregate_full_e2e_results()
        dataset_info: Dataset information dict
        system_info: System/hardware information dict
        output_path: Path to save the plot
    """
    summary = results.get('_summary', {})
    rankings = summary.get('rankings', [])

    if not rankings:
        logger.warning("No rankings available for plotting")
        return

    # Take top 15 configurations by DER for visualization
    top_n = min(15, len(rankings))
    top_configs = rankings[:top_n]

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # Create GridSpec for custom layout
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # Color palette for components
    colors = {
        'der': '#E74C3C',
        'miss': '#3498DB',
        'false_alarm': '#2ECC71',
        'confusion': '#9B59B6',
        'wer': '#F39C12',
        'timing': '#1ABC9C'
    }

    # Short labels for configurations
    def get_short_label(config):
        """Create a short label from config dict."""
        sod = config.get('sod', 'unk')[:3]
        sos = config.get('sos', 'unk')[:3]
        scd = config.get('scd', 'unk')[:3]
        emb = config.get('embedding', 'unk')[:3]
        clust = config.get('clustering', 'unk')[:4]
        return f"{sod}/{sos}/{scd}/{emb}/{clust}"

    labels = [get_short_label(c['config']) for c in top_configs]

    # =========================================================================
    # Panel 1: Top configurations by DER (horizontal bar chart)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])

    ders = [c['der'] for c in top_configs]
    y_pos = np.arange(len(labels))

    bars = ax1.barh(y_pos, ders, color=colors['der'], alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, ders)):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('DER (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Pipeline Configurations by DER', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()  # Best at top
    ax1.grid(axis='x', alpha=0.3)

    # Highlight best config
    bars[0].set_color('#27AE60')
    bars[0].set_edgecolor('#1E8449')
    bars[0].set_linewidth(2)

    # =========================================================================
    # Panel 2: DER Component Breakdown (stacked bar)
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    misses = [c['miss'] for c in top_configs]
    false_alarms = [c['false_alarm'] for c in top_configs]
    confusions = [c['confusion'] for c in top_configs]

    x = np.arange(len(labels))
    width = 0.6

    ax2.bar(x, misses, width, label='Miss', color=colors['miss'], alpha=0.8)
    ax2.bar(x, false_alarms, width, bottom=misses, label='False Alarm', color=colors['false_alarm'], alpha=0.8)
    ax2.bar(x, confusions, width, bottom=np.array(misses) + np.array(false_alarms),
            label='Confusion', color=colors['confusion'], alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('DER Component Breakdown', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # =========================================================================
    # Panel 3: Timing Comparison
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    timings = [c['timing'] for c in top_configs]
    bars3 = ax3.bar(x, timings, width, color=colors['timing'], alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars3, timings):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Time (s/file)', fontsize=12, fontweight='bold')
    ax3.set_title('Average Processing Time per File', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # =========================================================================
    # Panel 4: WER Comparison (if available)
    # =========================================================================
    ax4 = fig.add_subplot(gs[2, 0])

    wers = [c.get('wer') for c in top_configs]
    has_wer = any(w is not None for w in wers)

    if has_wer:
        wers_clean = [w if w is not None else 0 for w in wers]
        bars4 = ax4.bar(x, wers_clean, width, color=colors['wer'], alpha=0.8, edgecolor='black', linewidth=0.5)

        for bar, val, orig in zip(bars4, wers_clean, wers):
            if orig is not None:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('WER (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Word Error Rate Comparison', fontsize=13, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'WER data not available', ha='center', va='center',
                fontsize=14, transform=ax4.transAxes)
        ax4.set_title('Word Error Rate Comparison', fontsize=13, fontweight='bold')

    # =========================================================================
    # Panel 5: Summary Table
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Create summary text
    best_der = summary.get('best_by_der', {})
    best_wer = summary.get('best_by_wer', {})
    best_time = summary.get('best_by_time', {})

    summary_text = "BEST CONFIGURATIONS\n" + "="*50 + "\n\n"

    if best_der:
        summary_text += "🏆 Best by DER:\n"
        summary_text += f"   Config: {best_der.get('name', 'N/A')}\n"
        summary_text += f"   DER: {best_der.get('der', 0):.2f}%\n"
        summary_text += f"   Miss: {best_der.get('miss', 0):.2f}% | FA: {best_der.get('false_alarm', 0):.2f}% | Conf: {best_der.get('confusion', 0):.2f}%\n\n"

    if best_wer:
        summary_text += "📝 Best by WER:\n"
        summary_text += f"   Config: {best_wer.get('name', 'N/A')}\n"
        summary_text += f"   WER: {best_wer.get('wer', 0):.2f}%\n\n"

    if best_time:
        summary_text += "⚡ Best by Speed:\n"
        summary_text += f"   Config: {best_time.get('name', 'N/A')}\n"
        summary_text += f"   Time: {best_time.get('timing', 0):.2f}s/file\n"

    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    # =========================================================================
    # Overall title
    # =========================================================================
    if dataset_info:
        total_duration_min = dataset_info.get('total_duration_seconds', 0) / 60
        title = f"Full E2E Pipeline Comparison - {len(rankings)} Configurations Tested\n"
        title += f"Language: {dataset_info.get('language', 'unknown').title()} | "
        title += f"Files: {dataset_info.get('num_files', 0)} | "
        title += f"Duration: {total_duration_min:.1f} min | "
        title += f"Hardware: {system_info.get('gpu', 'N/A')}"
    else:
        title = f"Full E2E Pipeline Comparison - {len(rankings)} Configurations Tested\n"
        title += f"Hardware: {system_info.get('gpu', 'N/A')}"

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved full E2E comparison plot: {output_path}")


def plot_full_e2e_heatmap(
    results: Dict,
    dataset_info: Dict,
    system_info: Dict,
    output_path: Path
):
    """
    Generate a heatmap showing DER for different component combinations.

    Creates separate heatmaps for embedding vs clustering at each SCD type.

    Args:
        results: Dict from aggregate_full_e2e_results()
        dataset_info: Dataset information dict
        system_info: System/hardware information dict
        output_path: Path to save the plot
    """
    import pandas as pd

    summary = results.get('_summary', {})
    rankings = summary.get('rankings', [])

    if not rankings:
        logger.warning("No rankings available for heatmap plotting")
        return

    # Create dataframe from rankings
    df = pd.DataFrame(rankings)

    # Extract component values from config
    df['sod'] = df['config'].apply(lambda x: x.get('sod', 'unknown'))
    df['sos'] = df['config'].apply(lambda x: x.get('sos', 'unknown'))
    df['scd'] = df['config'].apply(lambda x: x.get('scd', 'unknown'))
    df['embedding'] = df['config'].apply(lambda x: x.get('embedding', 'unknown'))
    df['clustering'] = df['config'].apply(lambda x: x.get('clustering', 'unknown'))

    # Get unique values
    scd_types = df['scd'].unique()
    embeddings = df['embedding'].unique()
    clusterings = df['clustering'].unique()

    # Create figure with subplots for each SCD type
    n_scd = len(scd_types)
    fig, axes = plt.subplots(1, n_scd, figsize=(8 * n_scd, 6))

    if n_scd == 1:
        axes = [axes]

    for idx, scd_type in enumerate(scd_types):
        ax = axes[idx]

        # Filter data for this SCD type (averaging over SOD/SOS combinations)
        scd_data = df[df['scd'] == scd_type]

        # Create pivot table: embedding vs clustering, values = mean DER
        pivot = scd_data.pivot_table(
            values='der',
            index='embedding',
            columns='clustering',
            aggfunc='mean'
        )

        # Plot heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('DER (%)', fontsize=10)

        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(pivot.index, fontsize=10)

        # Add value annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val > pivot.values.mean() else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                           fontsize=9, fontweight='bold', color=text_color)

        ax.set_xlabel('Clustering', fontsize=12, fontweight='bold')
        ax.set_ylabel('Embedding', fontsize=12, fontweight='bold')
        ax.set_title(f'SCD: {scd_type.upper()}', fontsize=13, fontweight='bold')

    # Overall title
    if dataset_info:
        title = f"DER Heatmap: Embedding × Clustering (averaged over SOD/SOS)\n"
        title += f"Language: {dataset_info.get('language', 'unknown').title()} | "
        title += f"Hardware: {system_info.get('gpu', 'N/A')}"
    else:
        title = f"DER Heatmap: Embedding × Clustering (averaged over SOD/SOS)\n"
        title += f"Hardware: {system_info.get('gpu', 'N/A')}"

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved full E2E heatmap plot: {output_path}")
