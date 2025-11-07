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


def evaluate_vad(reference_segments, prediction_segments, total_duration):
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
    logger.info("Computing VAD metrics...")

    total_duration = pipeline_output['duration']
    reference_segments = annotation_data['segments']
    prediction_segments = pipeline_output['speaker_segments']

    vad_metrics = evaluate_vad(reference_segments, prediction_segments, total_duration)

    results = {
        'vad': vad_metrics
    }

    return results


def format_metrics_report(metrics):
    lines = []

    lines.append("")
    lines.append("=" * 60)
    lines.append("VAD METRICS")
    lines.append("=" * 60)

    vad = metrics['vad']
    lines.append(f"Precision: {vad['precision']:.2f}%")
    lines.append(f"Recall:    {vad['recall']:.2f}%")
    lines.append(f"F1 Score:  {vad['f1']:.2f}%")

    lines.append("=" * 60)

    report = "\n".join(lines)

    return report
