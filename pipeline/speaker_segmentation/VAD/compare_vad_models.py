import argparse
import time
from pathlib import Path
from loguru import logger
from utils import load_audio_from_file, load_annotation_file, evaluate_segmentation
from utils.constants import SR
from pipeline.speaker_segmentation.VAD.silero import SileroVAD
from pipeline.speaker_segmentation.VAD.pyannote_vad import PyannoteVAD


def compare_vad_models(audio_file: Path, annotation_file: Path = None):
    """
    Compare multiple VAD models on the same audio file.

    Parameters:
        audio_file : Path
            Path to the audio file to process
        annotation_file : Path
            Optional path to gold-standard annotation JSON for metrics evaluation
    """
    logger.info(f"Loading audio: {audio_file}")
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

    total_duration = waveform.shape[-1] / sr
    logger.info(f"Audio duration: {total_duration:.2f}s")

    annotation_data = None
    if annotation_file and annotation_file.exists():
        logger.info(f"Loading annotation from {annotation_file}")
        annotation_data = load_annotation_file(annotation_file)

    vad_models = [
        ('silero', SileroVAD()),
        ('pyannote', PyannoteVAD()),
    ]

    results = []

    for model_name, vad in vad_models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_name.upper()}")
        logger.info(f"{'='*60}")

        start_time = time.time()
        segments = vad(waveform, sample_rate=sr)
        inference_time = time.time() - start_time

        logger.info(f"Found {len(segments)} speech segments")
        logger.info(f"Inference time: {inference_time:.2f}s")

        result = {
            'model': model_name,
            'num_segments': len(segments),
            'inference_time': inference_time,
            'segments': segments
        }

        if annotation_data:
            metrics = evaluate_segmentation(
                reference_segments=annotation_data['segments'],
                prediction_segments=segments,
                total_duration=total_duration
            )
            result['metrics'] = metrics

        results.append(result)

    if annotation_data:
        print("\n" + "="*80)
        print("REFERENCE ANNOTATION")
        print("="*80)
        print(f"\nANNOTATION ({len(annotation_data['segments'])} segments)")
        print("-"*80)
        for i, seg in enumerate(annotation_data['segments'][:10]): 
            duration = seg['end'] - seg['start']
            print(f"  Segment {i+1}: [{seg['start']:6.2f}s - {seg['end']:6.2f}s] ({duration:5.2f}s)")
        if len(annotation_data['segments']) > 10:
            print(f"  ... and {len(annotation_data['segments']) - 10} more segments")
        print("="*80)

    print("\n" + "="*80)
    print("DETECTED SEGMENTS BY MODEL")
    print("="*80)
    for result in results:
        print(f"\n{result['model'].upper()} ({result['num_segments']} segments)")
        print("-"*80)
        for i, seg in enumerate(result['segments'][:10]): 
            duration = seg['end'] - seg['start']
            print(f"  Segment {i+1}: [{seg['start']:6.2f}s - {seg['end']:6.2f}s] ({duration:5.2f}s)")
        if result['num_segments'] > 10:
            print(f"  ... and {result['num_segments'] - 10} more segments")
    print("="*80)

    print("\n" + "="*80)
    print("VAD MODEL COMPARISON")
    print("="*80)
    print(f"Audio: {audio_file.name}")
    print(f"Duration: {total_duration:.2f}s")
    if annotation_data:
        print(f"Reference segments: {len(annotation_data['segments'])}")
    print("-"*80)

    print(f"{'Model':<15} {'Segments':<10} {'Time (s)':<10}")
    print("-"*80)

    for result in results:
        model = result['model']
        num_seg = result['num_segments']
        inf_time = result['inference_time']
        print(f"{model:<15} {num_seg:<10} {inf_time:<10.2f}")

    print("="*80)

    if annotation_data:
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        print(f"{'Model':<15} {'Precision':>12} {'Recall':>12} {'F1':>12}")
        print("-"*80)
        for result in results:
            if 'metrics' in result:
                model = result['model']
                metrics = result['metrics']
                print(f"{model:<15} {metrics['precision']:>11.2f}% {metrics['recall']:>11.2f}% {metrics['f1']:>11.2f}%")
        print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple VAD models")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to process")
    parser.add_argument("--annotation", type=Path, help="Path to gold-standard annotation JSON for metrics evaluation")

    args = parser.parse_args()

    if not args.audio_file.exists():
        parser.error(f"Audio file not found: {args.audio_file}")

    if args.annotation and not args.annotation.exists():
        parser.error(f"Annotation file not found: {args.annotation}")

    compare_vad_models(args.audio_file, args.annotation)

    logger.info("Comparison complete!")
