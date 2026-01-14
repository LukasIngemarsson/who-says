#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from contextlib import contextmanager

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
import numpy as np


@contextmanager
def redirect_stdout_to_stderr():
    old_stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
        yield
    finally:
        sys.stdout = old_stdout


def discover_benchmark_files(annotation_dir: Path, audio_dir: Path, limit: int = None) -> List[Tuple[Path, Path, str]]:
    annotation_files = sorted(annotation_dir.glob("*.json"))
    if limit:
        annotation_files = annotation_files[:limit]

    file_pairs = []
    for annotation_file in annotation_files:
        file_id = annotation_file.stem

        for ext in ['.mp3', '.wav', '.flac', '.m4a']:

            audio_file = audio_dir / f"{file_id}{ext}"
            if audio_file.exists():
                file_pairs.append((audio_file, annotation_file, file_id))
                break
            audio_file = audio_dir / f"combined_part{file_id}{ext}"
            if audio_file.exists():
                file_pairs.append((audio_file, annotation_file, file_id))
                break

    return file_pairs


def evaluate_diarization(hypothesis_segments: List[Dict], reference: Dict, collar: float = 0.25) -> Dict:
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref_segments = reference['segments']
    duration = max(seg['end'] for seg in ref_segments)

    uem = Timeline([Segment(0, duration)], uri="audio")

    ref_annotation = Annotation()
    for seg in ref_segments:
        ref_annotation[Segment(seg['start'], seg['end'])] = seg['speaker']

    hyp_annotation = Annotation()
    for seg in hypothesis_segments:
        hyp_annotation[Segment(seg['start'], seg['end'])] = seg.get('speaker', 'UNKNOWN')

    metric = DiarizationErrorRate(collar=collar)
    der_value = metric(ref_annotation, hyp_annotation, uem=uem)
    components = metric.compute_components(ref_annotation, hyp_annotation, uem=uem)

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


def aggregate_e2e_results(pipelines: Dict) -> Dict:
    for pipeline_name, pipeline_data in pipelines.items():
        results = pipeline_data.get('results', [])

        if not results:
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

        pipeline_data['aggregated'] = aggregated

    return pipelines


class WhisperXPipeline:

    def __init__(self, device: str = "cuda", model: str = "large-v2"):
        self.model_size = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.language = "en" 
        self.asr_model = None
        self.diarize_model = None

    def load(self):
        if self.asr_model is not None:
            return

        self.asr_model = whisperx.load_model(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
        )

    def _get_diarize_model(self):
        if self.diarize_model is None:
            hf_token = os.environ.get('HF_TOKEN')
            if not hf_token:
                raise RuntimeError("HF_TOKEN not set for diarization")

            self.diarize_model = DiarizationPipeline(
                use_auth_token=hf_token,
                device=self.device,
            )
        return self.diarize_model

    def process(self, audio_path: str, num_speakers: int = None) -> Dict:
        if self.asr_model is None:
            self.load()

        result = self.asr_model.transcribe(audio_path, batch_size=16, language=self.language)
        detected_language = self.language

        audio_data = whisperx.load_audio(audio_path)

        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=self.device,
        )
        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_data,
            device=self.device,
        )

        del model_a
        torch.cuda.empty_cache() if self.device == "cuda" else None

        diarizer = self._get_diarize_model()
        if num_speakers is not None:
            diarize_segments = diarizer(
                audio_data,
                min_speakers=num_speakers,
                max_speakers=num_speakers,
            )
        else:
            diarize_segments = diarizer(audio_data)

        result_with_speakers = whisperx.assign_word_speakers(
            diarize_segments,
            result_aligned,
        )

        torch.cuda.empty_cache() if self.device == "cuda" else None

        segments = []
        for seg in result_with_speakers.get("segments", []):
            segments.append({
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'speaker': seg.get('speaker', 'UNKNOWN'),
                'text': seg.get('text', '').strip()
            })

        return {'segments': segments}


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

    checkpoint_file = Path("/tmp/whisperx_checkpoint.json")

    if checkpoint_file.exists():
        print(f"Found checkpoint, loading...", file=sys.stderr)
        with open(checkpoint_file) as f:
            pipelines = json.load(f)
        for pipeline_name in pipelines:
            model_size = pipeline_name.replace('whisperx-', '')
            pipelines[pipeline_name]['instance'] = WhisperXPipeline(device=device, model=model_size)
    else:
        pipelines = {}
        for model_size in whisperx_models:
            pipeline_name = f'whisperx-{model_size}'
            pipelines[pipeline_name] = {
                'instance': WhisperXPipeline(device=device, model=model_size),
                'has_transcription': True,
                'model_info': f'whisper-X({model_size})',
                'results': []
            }

    for file_idx, (audio_file, annotation_file, file_id) in enumerate(file_pairs):
        with open(annotation_file) as f:
            annotation = json.load(f)

        n_speakers_ref = len(set(
            seg['speaker'] for seg in annotation['segments']
        ))

        # Skip files already processed
        processed_files = set()
        for pipeline_data in pipelines.values():
            for r in pipeline_data['results']:
                processed_files.add(r['file_id'])

        if file_id in processed_files:
            print(f"[{file_idx+1}/{len(file_pairs)}] Skipping {file_id} (already processed)", file=sys.stderr)
            continue

        print(f"[{file_idx+1}/{len(file_pairs)}] Processing {file_id}...", file=sys.stderr)

        for pipeline_name, pipeline_data in pipelines.items():
            print(f"\n  {pipeline_name}: Processing {audio_file.name}...", file=sys.stderr)
            start_time = time.time()

            try:
                # Redirect WhisperX stdout to stderr to keep stdout clean for JSON
                with redirect_stdout_to_stderr():
                    result = pipeline_data['instance'].process(str(audio_file), num_speakers=n_speakers_ref)
                inference_time = time.time() - start_time

                der_metrics = evaluate_diarization(
                    result['segments'],
                    annotation,
                    collar=0.25
                )

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
                    'timing': inference_time
                })

                print(f"Time: {inference_time:.1f}s", file=sys.stderr)
                print(f"DER: {der_metrics['der']:.2f}% (miss: {der_metrics['miss']:.2f}%, FA: {der_metrics['false_alarm']:.2f}%, conf: {der_metrics['confusion']:.2f}%)", file=sys.stderr)
            except Exception as e:
                print(f"    ✗ ERROR: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)

        checkpoint_data = {
            name: {
                'has_transcription': data['has_transcription'],
                'model_info': data['model_info'],
                'results': data['results']
            }
            for name, data in pipelines.items()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        print(f"  Checkpoint saved", file=sys.stderr)

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
            if name != 'instance'  # Don't serialize the instance
        }
    }

    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Checkpoint cleaned up", file=sys.stderr)

    print(json.dumps(output))


if __name__ == "__main__":
    main()
