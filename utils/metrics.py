import json
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from loguru import logger


def load_annotation_file(annotation_path):
    file_path = Path(annotation_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {annotation_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'segments' not in data:
        raise ValueError("Missing 'segments' in annotation file")

    return data


def segments_to_frames(segments, total_duration, frame_size=0.01):
    num_frames = int(np.ceil(total_duration / frame_size))
    frames = np.zeros(num_frames, dtype=np.int32)

    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']

        start_frame = int(start_time / frame_size)
        end_frame = int(end_time / frame_size)

        end_frame = min(end_frame, num_frames)

        frames[start_frame:end_frame] = 1

    return frames


def compute_precision(reference_frames, prediction_frames):
    precision = precision_score(reference_frames, prediction_frames, zero_division=0)
    precision_percentage = precision * 100.0

    return precision_percentage


def compute_recall(reference_frames, prediction_frames):
    recall = recall_score(reference_frames, prediction_frames, zero_division=0)
    recall_percentage = recall * 100.0

    return recall_percentage


def compute_f1(reference_frames, prediction_frames):
    f1 = f1_score(reference_frames, prediction_frames, zero_division=0)
    f1_percentage = f1 * 100.0

    return f1_percentage


def evaluate_segmentation(reference_segments, prediction_segments, total_duration):
    """
    Evaluate segmentation quality by comparing prediction against reference.
    Works for VAD, SCD, or any segment based evaluation.
    """
    frame_size = 0.01

    ref_frames = segments_to_frames(reference_segments, total_duration, frame_size)
    pred_frames = segments_to_frames(prediction_segments, total_duration, frame_size)

    max_length = max(len(ref_frames), len(pred_frames))

    if len(ref_frames) < max_length:
        padding = max_length - len(ref_frames)
        ref_frames = np.pad(ref_frames, (0, padding), mode='constant')

    if len(pred_frames) < max_length:
        padding = max_length - len(pred_frames)
        pred_frames = np.pad(pred_frames, (0, padding), mode='constant')

    precision = compute_precision(ref_frames, pred_frames)
    recall = compute_recall(ref_frames, pred_frames)
    f1 = compute_f1(ref_frames, pred_frames)

    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return results


def evaluate_pipeline(pipeline_output, annotation_data):
    # TODO: Add other component metrics
    logger.info("Computing metrics...")

    total_duration = pipeline_output['duration']
    reference_segments = annotation_data['segments']

    vad_metrics = evaluate_segmentation(
        reference_segments,
        pipeline_output['vad_segments'],
        total_duration
    )

    scd_metrics = evaluate_segmentation(
        reference_segments,
        pipeline_output['speaker_segments'],
        total_duration
    )

    return {
        'vad': vad_metrics,
        'scd': scd_metrics
    }


def format_metrics_report(metrics):
    lines = []

    lines.append("")
    lines.append("=" * 60)
    lines.append("EVALUATION METRICS")
    lines.append("=" * 60)
    lines.append(f"{'Component':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    lines.append("-" * 60)

    vad = metrics['vad']
    lines.append(f"{'Voice Activity (VAD)':<25} {vad['precision']:>9.2f}% {vad['recall']:>9.2f}% {vad['f1']:>9.2f}%")

    scd = metrics['scd']
    lines.append(f"{'Speaker Change (SCD)':<25} {scd['precision']:>9.2f}% {scd['recall']:>9.2f}% {scd['f1']:>9.2f}%")

    lines.append("=" * 60)

    report = "\n".join(lines)

    return report


def format_timing_report(timing, total_time):
    """
    Format timing metrics with bar graph visualization.

    Args:
        timing: Dictionary with timing for each component
        total_time: Total pipeline execution time

    Returns:
        str: Formatted timing report with bar graphs
    """
    lines = []

    lines.append("")
    lines.append("=" * 60)
    lines.append("TIMING METRICS")
    lines.append("=" * 60)

    # Component display names
    component_names = {
        'audio_loading': 'Audio Loading',
        'vad': 'Voice Activity Detection',
        'asr': 'Automatic Speech Recognition',
        'scd': 'Speaker Change Detection',
        'embedding': 'Speaker Embedding',
        'clustering': 'Speaker Clustering',
        'formatting': 'Result Formatting'
    }

    # Find the maximum time for scaling the bar graph
    max_time = max(timing.values()) if timing else 1.0
    bar_width = 40  # Width of the bar in characters

    # Sort components by time (descending)
    sorted_timing = sorted(timing.items(), key=lambda x: x[1], reverse=True)

    for component, time_val in sorted_timing:
        # Get display name
        display_name = component_names.get(component, component.replace('_', ' ').title())

        # Calculate percentage
        percentage = (time_val / total_time) * 100 if total_time > 0 else 0

        # Create bar graph
        bar_length = int((time_val / max_time) * bar_width)
        bar = '█' * bar_length

        # Format the line with component name, bar, time, and percentage
        lines.append(f"{display_name:30} {bar:40} {time_val:6.2f}s ({percentage:5.1f}%)")

    lines.append("-" * 60)
    lines.append(f"{'Total Time':30} {'':<40} {total_time:6.2f}s (100.0%)")
    lines.append("=" * 60)

    report = "\n".join(lines)

    return report
