#!/usr/bin/env python3
"""
Helper script to run WhisperX comparison in isolated venv subprocess.
Outputs JSON to stdout for parent process to consume.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import discover_benchmark_files, aggregate_e2e_results, load_audio_from_file
from utils.constants import SR
from utils.metrics import evaluate_diarization, evaluate_asr
from pipeline.whisperx_pipeline import WhisperXPipeline
import torch


def main():
    parser = argparse.ArgumentParser(description="Run WhisperX E2E comparison in venv")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--limit", type=int, help="Limit number of files")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisperx_models = [
        'tiny', 'medium',
        'large-v3', 'distil-large-v3'
    ]

    file_pairs = discover_benchmark_files(
        args.annotation_dir,
        args.audio_dir,
        args.limit
    )

    pipelines = {}
    for model_size in whisperx_models:
        pipeline_name = f'whisperx-{model_size}'
        pipelines[pipeline_name] = {
            'instance': WhisperXPipeline(device=device, model=model_size),
            'has_transcription': True,
            'model_info': f'whisper-X({model_size})',
            'results': []
        }

    for audio_file, annotation_file, file_id in file_pairs:
        with open(annotation_file) as f:
            annotation = json.load(f)

        n_speakers_ref = len(set(
            seg['speaker'] for seg in annotation['segments']
        ))

        for pipeline_name, pipeline_data in pipelines.items():
            start_time = time.time()
            result = pipeline_data['instance'].process(str(audio_file))
            inference_time = time.time() - start_time

            der_metrics = evaluate_diarization(
                result,
                annotation,
                collar=0.25
            )

            wer_metrics = evaluate_asr(result, annotation)

            pipeline_data['results'].append({
                'file_id': file_id,
                'audio_file': str(audio_file),
                'duration': result.get('duration', 0),
                'n_speakers_pred': len(set(
                    seg.get('speaker', 'UNKNOWN')
                    for seg in result['segments']
                )),
                'n_speakers_ref': n_speakers_ref,
                'der_metrics': der_metrics,
                'wer_metrics': wer_metrics,
                'timing': inference_time
            })

    pipelines = aggregate_e2e_results(pipelines)

    output = {
        'pipelines': {
            name: {
                'has_transcription': data['has_transcription'],
                'model_info': data.get('model_info', 'N/A'),
                'per_file_results': data['results'],
                'aggregated': data['aggregated']
            }
            for name, data in pipelines.items()
        }
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
