import json
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score
from loguru import logger
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


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


def segments_to_annotation(segments, uri="audio"):
    """
    Convert segment list to pyannote.core.Annotation object.

    Parameters
    ----------
    segments : list of dict
        List of segments with 'start', 'end', and 'speaker' keys.
    uri : str, optional
        Uniform Resource Identifier for the annotation.

    Returns
    -------
    pyannote.core.Annotation
        Annotation object with segments mapped to speaker labels.
    """
    annotation = Annotation(uri=uri)

    for segment in segments:
        # Create a Segment object with start and end times
        seg = Segment(segment['start'], segment['end'])

        # Add the segment to the annotation with the speaker label
        annotation[seg] = segment['speaker']

    return annotation


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
 
# ASR
def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) between reference and hypothesis strings.
    WER = (S + D + I) / N
    S = substitutions, D = deletions, I = insertions, N = number of words in reference
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    N = len(ref_words)

    d = np.zeros((len(ref_words)+1, len(hyp_words)+1), dtype=np.uint32)
    for i in range(len(ref_words)+1):
        d[i][0] = i
    for j in range(len(hyp_words)+1):
        d[0][j] = j

    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            if ref_words[i-1] == hyp_words[j-1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )
    wer = 100 * d[len(ref_words)][len(hyp_words)] / N if N > 0 else 0.0
    return wer

#Phoneme
def phoneme_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Phoneme Error Rate (PER) between reference and hypothesis
    phoneme strings.

    Both inputs are expected to be space-separated phoneme sequences, e.g.:

        reference  = "HH AH L OW W ER L D"
        hypothesis = "HH AH L OW ER L D"

    PER = 100 * (S + D + I) / N

    where
      S = substitutions
      D = deletions
      I = insertions
      N = number of phoneme tokens in the reference.
    """
    # Tokenize by whitespace – phonemes are treated just like "words"
    ref_phonemes = reference.strip().split()
    hyp_phonemes = hypothesis.strip().split()

    N = len(ref_phonemes)
    M = len(hyp_phonemes)

    # If there are no reference phonemes, define PER as 0.0 to avoid div-by-zero
    if N == 0:
        return 0.0

    # (N+1) x (M+1) dynamic programming matrix
    d = np.zeros((N + 1, M + 1), dtype=np.uint32)

    # Initialization: distance from empty sequence
    for i in range(1, N + 1):
        d[i, 0] = i
    for j in range(1, M + 1):
        d[0, j] = j

    # Fill DP table
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = 0 if ref_phonemes[i - 1] == hyp_phonemes[j - 1] else 1

            d[i, j] = min(
                d[i - 1, j] + 1,        # deletion
                d[i, j - 1] + 1,        # insertion
                d[i - 1, j - 1] + cost  # substitution
            )

    # Edit distance is at bottom-right of matrix
    distance = d[N, M]

    # Convert to percentage
    per = 100.0 * distance / N
    return per

def extract_change_points(segments):
    """
    Extract change point timestamps from segment boundaries.

    Parameters
    ----------
    segments : list of dict
        Segments with 'start' and 'end' times.

    Returns
    -------
    list of float
        Change point timestamps (end of each segment except the last).
    """
    if len(segments) <= 1:
        return []

    change_points = []
    for i in range(len(segments) - 1):
        change_points.append(segments[i]['end'])

    return change_points


def evaluate_change_point_detection(reference_segments, predicted_segments, tolerance=0.5):
    """
    Evaluate speaker change point detection with tolerance window.

    Params:
    reference_segments : list of dict
        Ground truth segments (to extract change points from).
    predicted_segments : list of dict
        Predicted segments (to extract change points from).
    tolerance :
        Time tolerance in seconds
        A predicted change point matches a reference if within +-tolerance.

    Returns
    -------
    dict
        Precision, Recall, F1 scores for change point detection.
    """
    ref_change_points = extract_change_points(reference_segments)
    pred_change_points = extract_change_points(predicted_segments)

    if len(ref_change_points) == 0 and len(pred_change_points) == 0:
        return {'precision': 100.0, 'recall': 100.0, 'f1': 100.0}

    if len(ref_change_points) == 0:
        return {'precision': 0.0, 'recall': 100.0, 'f1': 0.0}

    if len(pred_change_points) == 0:
        return {'precision': 100.0, 'recall': 0.0, 'f1': 0.0}

    matched_predictions = set()
    matched_references = set()

    for i, pred_cp in enumerate(pred_change_points):
        for j, ref_cp in enumerate(ref_change_points):
            if abs(pred_cp - ref_cp) <= tolerance:
                matched_predictions.add(i)
                matched_references.add(j)
                break 

    true_positives = len(matched_predictions)

    precision = (true_positives / len(pred_change_points) * 100.0) if len(pred_change_points) > 0 else 0.0
    recall = (true_positives / len(ref_change_points) * 100.0) if len(ref_change_points) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_asr(reference_transcriptions, hypothesis_transcriptions):
    reference = " ".join(reference_transcriptions)
    hypothesis = " ".join(hypothesis_transcriptions)
    wer = word_error_rate(reference, hypothesis)

    metrics = {"wer": wer}

    return metrics

def evaluate_phonemes(reference_phoneme_sequences, hypothesis_phoneme_sequences):
    if not reference_phoneme_sequences or not hypothesis_phoneme_sequences:
        return {"per": None}
    # Evaluate PER over a list of phoneme sequences.
    reference = " ".join(reference_phoneme_sequences)
    hypothesis = " ".join(hypothesis_phoneme_sequences)
    per = phoneme_error_rate(reference, hypothesis)
    return {"per": per}


def evaluate_diarization(reference_segments, hypothesis_segments):
    """
    Compute Diarization Error Rate (DER) between reference and hypothesis.

    Params:
        reference_segments : list of dict
            Gold-standard segments with start, end, and speaker keys.
        hypothesis_segments : list of dict
            Predicted segments with start, end, and speaker keys.

    Returns:
        Dictionary containing:
        - der: Overall diarization error rate (percentage)
        - miss: Miss rate (percentage)
        - false_alarm: False alarm rate (percentage)
        - confusion: Speaker confusion rate (percentage)
    """
    from pyannote.core import Timeline

    reference = segments_to_annotation(reference_segments)
    hypothesis = segments_to_annotation(hypothesis_segments)

    uem = Timeline(uri="audio")
    for seg in reference_segments:
        uem.add(Segment(seg['start'], seg['end']))
    uem = uem.support()  # Merge overlapping segments

    metric = DiarizationErrorRate()
    # Compute overall DER with UEM (returns a single value between 0 and 1)
    der_value = metric(reference, hypothesis, uem=uem)
    components = metric.compute_components(reference, hypothesis, uem=uem)

    total = components['total']
    confusion = components.get('confusion', 0.0)
    miss = components.get('missed detection', 0.0)
    false_alarm = components.get('false alarm', 0.0)

    der_percentage = der_value * 100.0
    miss_rate = (miss / total * 100.0) if total > 0 else 0.0
    fa_rate = (false_alarm / total * 100.0) if total > 0 else 0.0
    confusion_rate = (confusion / total * 100.0) if total > 0 else 0.0

    return {
        'der': der_percentage,
        'miss': miss_rate,
        'false_alarm': fa_rate,
        'confusion': confusion_rate
    }


def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute clustering quality using silhouette score (unsupervised).

    Parameters:
        embeddings : np.ndarray
            Speaker embeddings array of shape (n_segments, embedding_dim)
        labels : np.ndarray
            Cluster labels for each segment

    Returns:
        dict containing:
        - silhouette: Silhouette coefficient (-1 to 1, higher is better)
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(embeddings)

    if n_clusters < 2 or n_samples < 2:
        logger.warning(f"Cannot compute clustering metrics: {n_clusters} clusters, {n_samples} samples")
        return {'silhouette': 0.0}

    return {'silhouette': silhouette_score(embeddings, labels)}


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

    scd_metrics = evaluate_change_point_detection(
        reference_segments,
        pipeline_output['speaker_segments'],
        tolerance=0.5
    )

    reference_transcriptions = [seg["text"] for seg in reference_segments]
    output_transcriptions = [seg["text"] for seg in pipeline_output['transcription']]
    asr_metrics = evaluate_asr(reference_transcriptions, output_transcriptions)

    # Phoneme-level metrics (optional, only if phonemes exist)
    ref_phoneme_sequences = []
    hyp_phoneme_sequences = []

    # Align reference and hypothesis by index, and collect only if both have phonemes
    for ref_seg, hyp_seg in zip(reference_segments, pipeline_output['transcription']):
        ref_ph = ref_seg.get("phonemes")
        hyp_ph = hyp_seg.get("phonemes")

        if ref_ph and hyp_ph:
            ref_phoneme_sequences.append(ref_ph)
            hyp_phoneme_sequences.append(hyp_ph)

    if ref_phoneme_sequences and hyp_phoneme_sequences:
        phoneme_metrics = evaluate_phonemes(ref_phoneme_sequences, hyp_phoneme_sequences)
    else:
        logger.warning("No phoneme ground truth available – PER not computed.")
        phoneme_metrics = {"per": None}

    diarization_metrics = evaluate_diarization(
        reference_segments,
        pipeline_output['speaker_segments']
    )

    clustering_metrics = {'silhouette': 0.0}
    if 'embeddings' in pipeline_output and 'cluster_labels' in pipeline_output:
        clustering_metrics = evaluate_clustering(
            pipeline_output['embeddings'],
            pipeline_output['cluster_labels']
        )

    return {
        'vad': vad_metrics,
        'scd': scd_metrics,
        'asr': asr_metrics,
        'phoneme': phoneme_metrics,
        'diarization': diarization_metrics,
        'clustering': clustering_metrics,
    }

def format_metrics_report(metrics):
    lines = []

    lines.append("")
    lines.append("=" * 60)
    lines.append("EVALUATION METRICS")
    lines.append("=" * 60)
    lines.append(f"{'Component':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'WER':>10} {'PER':>10}")
    lines.append("-" * 60)

    vad = metrics['vad']
    lines.append(f"{'Voice Activity (VAD)':<25} {vad['precision']:>9.2f}% {vad['recall']:>9.2f}% {vad['f1']:>9.2f}% {0:>9.2f}% {0:>9.2f}%")

    scd = metrics['scd']
    lines.append(f"{'Speaker Change (SCD)':<25} {scd['precision']:>9.2f}% {scd['recall']:>9.2f}% {scd['f1']:>9.2f}% {0:>9.2f}% {0:>9.2f}%")

    asr = metrics['asr']
    lines.append(f"{'ASR':<25} {0:>9.2f}% {0:>9.2f}% {0:>9.2f}% {asr['wer']:>9.2f}% {0:>9.2f}%")

    phoneme = metrics['phoneme']
    per_val = phoneme.get('per') if phoneme is not None else None
    if per_val is None:
        per_val = 0.0  # or you could choose to display this as 0.0% meaning "not computed"
    
    lines.append(
        f"{'Phoneme':<25} "
        f"{0:>9.2f}% {0:>9.2f}% {0:>9.2f}% "
        f"{0:>9.2f}% {per_val:>9.2f}%"
    )

    lines.append("")
    lines.append("Diarization Error Rate (DER)")
    lines.append("-" * 60)

    der = metrics['diarization']
    lines.append(f"  Overall DER:            {der['der']:>9.2f}%")
    lines.append(f"  Miss Rate:              {der['miss']:>9.2f}%")
    lines.append(f"  False Alarm:            {der['false_alarm']:>9.2f}%")
    lines.append(f"  Speaker Confusion:      {der['confusion']:>9.2f}%")

    if 'clustering' in metrics:
        lines.append("")
        lines.append("Clustering Quality")
        lines.append("-" * 60)
        clustering = metrics['clustering']
        lines.append(f"  Silhouette Score:       {clustering['silhouette']:>9.4f}  (higher is better, -1 to 1)")

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
        'phoneme': 'Phoneme Conversion',
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
