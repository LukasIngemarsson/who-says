#!/usr/bin/env python3
"""
Comparison tool for who-says pipeline components.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv(".env")

from utils import (
    get_system_info,
    discover_benchmark_files,
    compare_vad_models,
    compare_sc_models,
    compare_asr_models,
    compare_e2e_pipelines,
    aggregate_results,
    aggregate_sc_results,
    aggregate_asr_results,
    aggregate_e2e_results,
    compare_scd_models,
    aggregate_scd_results,
    compare_sod_models,
    aggregate_sod_results,
    compare_sos_models,
    aggregate_sos_results,
    compare_speaker_id_models,
    aggregate_speaker_id_results,
    compare_embedding_models,
    aggregate_embedding_results,
    compare_cluster_viz,
    compare_full_e2e_pipelines,
    aggregate_full_e2e_results,
    plot_metrics,
    plot_timing,
    plot_sc_timing,
    plot_sc_silhouette,
    plot_sc_der,
    plot_sc_clustering_metrics,
    plot_asr_wer,
    plot_asr_timing,
    plot_e2e_der,
    plot_e2e_wer,
    plot_e2e_timing,
    plot_embedding_comparison,
    plot_cluster_umap,
    plot_sos_comparison,
    plot_sod_comparison,
    plot_scd_comparison,
    plot_full_e2e_comparison,
    plot_full_e2e_heatmap,
    load_audio_from_file
)
from utils.constants import SR


def run_whisperx_in_venv(audio_dir, annotation_dir, language, limit):
    """Run WhisperX comparison in isolated venv, return pipelines dict."""
    import subprocess
    import sys

    PROJECT_ROOT = Path(__file__).parent
    VENV_DIR = PROJECT_ROOT / ".venv_whisperx"
    VENV_PYTHON = VENV_DIR / "bin/python"

    if not VENV_DIR.exists():
        logger.info("Creating WhisperX venv ")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

        logger.info("Installing WhisperX dependencies...")
        subprocess.run([
            str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)

        subprocess.run([
            str(VENV_PYTHON), "-m", "pip", "install",
            "whisperx>=3.1.1",
            "torch==2.5.1",
            "torchaudio==2.5.1",
            "pyannote.audio>=3.3.0",
            "faster-whisper>=1.0.0",
            "transformers==4.49.0",
            "loguru==0.7.0",
            "numpy>=2.0.2,<2.1.0",
            "soundfile>=0.12.0",
            "python-dotenv>=1.0.0",
            "scikit-learn>=1.3.0",
            "jiwer>=3.0.0" 
        ], check=True)

    helper_script = Path(__file__).parent / "_whisperx_helper.py"

    cmd = [
        str(VENV_PYTHON),
        str(helper_script),
        "--audio-dir", str(audio_dir),
        "--annotation-dir", str(annotation_dir),
        "--language", language
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    logger.info("Running WhisperX subprocess...")

    result = subprocess.run(cmd, capture_output=False, stdout=subprocess.PIPE, text=True)

    if result.returncode != 0:
        logger.error("WhisperX subprocess failed!")
        logger.error(f"STDOUT:\n{result.stdout}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout)

    if not result.stdout or not result.stdout.strip():
        logger.error("WhisperX subprocess produced no output!")
        logger.error(f"STDOUT:\n{result.stdout}")
        raise ValueError("WhisperX subprocess produced no output")

    try:
        whisperx_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse WhisperX output as JSON: {e}")
        logger.error(f"STDOUT (first 2000 chars):\n{result.stdout[:2000]}")
        logger.error(f"Check if WhisperX printed errors above (stderr was not captured)")
        raise

    return whisperx_data['pipelines']


def main():
    parser = argparse.ArgumentParser(
        description="Compare models for who-says pipeline components"
    )
    parser.add_argument(
        "--component",
        required=True,
        choices=["vad", "sc", "asr", "e2e", "full-e2e", "scd", "sod", "sos", "speaker-id", "embedding-viz", "cluster-viz"],
        help="Component to compare: vad, sc (speaker clustering), asr, e2e (end-to-end), "
             "full-e2e (all component combinations), scd (speaker change detection), "
             "sod (speech overlap detection), sos (speech overlap separation), "
             "speaker-id (speaker identification), embedding-viz (embedding visualization), "
             "cluster-viz (clustering with UMAP)"
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        help="Path to directory containing audio files (for vad, sc, asr, e2e)"
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="Path to directory containing annotation JSON files (benchmark dir)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparison/english"),
        help="Output directory"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="unknown",
        help="Language of the audio dataset (default: unknown)"
    )
    # Arguments for new benchmark components (scd, sod, sos, speaker-id)
    parser.add_argument(
        "--audio",
        type=str,
        help="Single audio file path (for scd, sod, sos, speaker-id components)"
    )
    parser.add_argument(
        "--speaker-dir",
        type=Path,
        help="Directory containing individual speaker tracks (for sos component)"
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Directory containing reference speaker audio files (for speaker-id and scd components)"
    )
    parser.add_argument(
        "--skip-nemo",
        action="store_true",
        help="Skip NeMo benchmark (for scd, sod components)"
    )
    parser.add_argument(
        "--skip-naive",
        action="store_true",
        help="Skip Naive (cosine similarity) benchmark (for scd component)"
    )
    parser.add_argument(
        "--skip-wavlm",
        action="store_true",
        help="Skip WavLM benchmark (for sod component)"
    )
    parser.add_argument(
        "--skip-overlap",
        action="store_true",
        help="Skip overlap detection and processing (for e2e component)"
    )
    parser.add_argument(
        "--include-whisperx",
        action="store_true",
        help="Include WhisperX models (7 sizes, runs in separate venv) for e2e component"
    )
    parser.add_argument(
        "--skip-sod-sos",
        action="store_true",
        help="Skip SOD/SOS variations to reduce combinations (for full-e2e component)"
    )
    parser.add_argument(
        "--scd-metric",
        type=str,
        choices=["f1", "precision", "recall"],
        default="f1",
        help="Metric to optimize for when selecting best SCD config (default: f1)"
    )
    parser.add_argument(
        "--sod-metric",
        type=str,
        choices=["frame_f1", "frame_precision", "frame_recall"],
        default="frame_f1",
        help="Metric to optimize for when selecting best SOD config (default: frame_f1)"
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=15,
        help="Maximum overlap regions to process (for sos component)"
    )
    # Arguments for cluster-viz component
    parser.add_argument(
        "--embedding-type",
        type=str,
        choices=["pyannote", "wav2vec2", "speechbrain"],
        help="Embedding model type for cluster-viz component"
    )
    parser.add_argument(
        "--audio-folder",
        type=Path,
        help="Folder containing audio segments for cluster-viz component"
    )
    parser.add_argument(
        "--alignment-file",
        type=Path,
        help="JSON file mapping audio filenames to speaker names for cluster-viz"
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Component: {args.component.upper()}")
    logger.info(f"Annotation dir: {args.annotation_dir}")

    system_info = get_system_info()
    logger.info(f"Device: {system_info['device']} ({system_info['gpu']})")

    # New components (scd, sod, sos, speaker-id, embedding-viz, cluster-viz) use single audio file, not file pairs
    if args.component in ["scd", "sod", "sos", "speaker-id", "embedding-viz", "cluster-viz"]:
        if args.audio:
            logger.info(f"Audio file: {args.audio}")
        file_pairs = None
        dataset_info = None
    else:
        # Original components use file pair discovery
        if not args.audio_dir:
            logger.error(f"--audio-dir is required for {args.component} component")
            return
        logger.info(f"Audio dir: {args.audio_dir}")

        file_pairs = discover_benchmark_files(
            args.annotation_dir,
            args.audio_dir,
            args.limit
        )

        if not file_pairs:
            logger.error("No file pairs found!")
            return

        logger.info("Calculating total dataset duration...")
        total_duration = sum(
            load_audio_from_file(audio_file, sr=SR)[0].shape[-1] / SR
            for audio_file, _, _ in file_pairs
        )

        dataset_info = {
            "language": args.language,
            "num_files": len(file_pairs),
            "total_duration_seconds": float(total_duration)
        }

        logger.info(f"Total duration: {total_duration/60:.1f} minutes")

    if args.component == "vad":
        logger.info("\n" + "="*60)
        logger.info("Running VAD comparison...")
        logger.info("="*60)

        models = compare_vad_models(file_pairs)
        models = aggregate_results(models)

        output_data = {
            "comparison_type": "vad",
            "timestamp": timestamp,
            "system_info": system_info,
            "dataset": dataset_info,
            "models": {
                name: {
                    "per_file_results": data['results'],
                    "aggregated": data['aggregated']
                }
                for name, data in models.items()
            }
        }

        json_path = args.output_dir / f"vad_comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nSaved results: {json_path}")

        metrics_plot = args.output_dir / f"vad_metrics_{timestamp}.png"
        timing_plot = args.output_dir / f"vad_timing_{timestamp}.png"

        plot_metrics(models, dataset_info, system_info, metrics_plot)
        plot_timing(models, dataset_info, system_info, timing_plot)

        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"\nSilero VAD:")
        print(f"  Precision: {models['silero']['aggregated']['precision']['mean']:.2f}% "
              f"(±{models['silero']['aggregated']['precision']['std']:.2f})")
        print(f"  Recall:    {models['silero']['aggregated']['recall']['mean']:.2f}% "
              f"(±{models['silero']['aggregated']['recall']['std']:.2f})")
        print(f"  F1:        {models['silero']['aggregated']['f1']['mean']:.2f}% "
              f"(±{models['silero']['aggregated']['f1']['std']:.2f})")
        print(f"  Avg Time:  {models['silero']['aggregated']['timing']['mean']:.2f}s/file")

        print(f"\nPyannote VAD:")
        print(f"  Precision: {models['pyannote']['aggregated']['precision']['mean']:.2f}% "
              f"(±{models['pyannote']['aggregated']['precision']['std']:.2f})")
        print(f"  Recall:    {models['pyannote']['aggregated']['recall']['mean']:.2f}% "
              f"(±{models['pyannote']['aggregated']['recall']['std']:.2f})")
        print(f"  F1:        {models['pyannote']['aggregated']['f1']['mean']:.2f}% "
              f"(±{models['pyannote']['aggregated']['f1']['std']:.2f})")
        print(f"  Avg Time:  {models['pyannote']['aggregated']['timing']['mean']:.2f}s/file")
        print("="*60)

    elif args.component == "sc":
        logger.info("\n" + "="*60)
        logger.info("Running SC comparison...")
        logger.info("="*60)

        models = compare_sc_models(file_pairs)
        models = aggregate_sc_results(models)

        output_data = {
            "comparison_type": "sc",
            "timestamp": timestamp,
            "system_info": system_info,
            "dataset": dataset_info,
            "models": {
                name: {
                    "per_file_results": data['results'],
                    "aggregated": data['aggregated']
                }
                for name, data in models.items()
            }
        }

        json_path = args.output_dir / f"sc_comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nSaved results: {json_path}")

        timing_plot = args.output_dir / f"sc_timing_{timestamp}.png"
        silhouette_plot = args.output_dir / f"sc_silhouette_{timestamp}.png"
        der_plot = args.output_dir / f"sc_der_{timestamp}.png"
        clustering_metrics_plot = args.output_dir / f"sc_clustering_metrics_{timestamp}.png"

        plot_sc_timing(models, dataset_info, system_info, timing_plot)
        plot_sc_silhouette(models, dataset_info, system_info, silhouette_plot)
        plot_sc_der(models, dataset_info, system_info, der_plot)
        plot_sc_clustering_metrics(models, dataset_info, system_info, clustering_metrics_plot)

        print("\n" + "="*60)
        print("SPEAKER CLUSTERING COMPARISON SUMMARY")
        print("="*60)
        for name, data in models.items():
            if 'aggregated' not in data:
                continue
            agg = data['aggregated']
            print(f"\n{name.upper().replace('_', ' + ')}:")
            print(f"  ARI:        {agg['ari']['mean']:6.3f} (±{agg['ari']['std']:.3f})")
            print(f"  F1:         {agg['f1']['mean']:6.3f} (±{agg['f1']['std']:.3f})")
            print(f"  Silhouette: {agg['silhouette']['mean']:6.3f} (±{agg['silhouette']['std']:.3f})")
            print(f"  DER:        {agg['der']['mean']:6.2f}% (±{agg['der']['std']:.2f}%)")
            print(f"  Miss:       {agg['miss']['mean']:6.2f}% (±{agg['miss']['std']:.2f}%)")
            print(f"  False Alarm:{agg['false_alarm']['mean']:6.2f}% (±{agg['false_alarm']['std']:.2f}%)")
            print(f"  Confusion:  {agg['confusion']['mean']:6.2f}% (±{agg['confusion']['std']:.2f}%)")
            print(f"  Embedding:  {agg['embedding_timing']['mean']:6.2f}s/file")
            print(f"  Clustering: {agg['clustering_timing']['mean']:6.2f}s/file")
        print("="*60)

    elif args.component == "asr":
        logger.info("\n" + "="*60)
        logger.info("Running ASR comparison...")
        logger.info("="*60)

        models = compare_asr_models(file_pairs)
        models = aggregate_asr_results(models)

        output_data = {
            "comparison_type": "asr",
            "timestamp": timestamp,
            "system_info": system_info,
            "dataset": dataset_info,
            "models": {
                name: {
                    "per_file_results": data['results'],
                    "aggregated": data['aggregated']
                }
                for name, data in models.items()
            }
        }

        json_path = args.output_dir / f"asr_comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nSaved results: {json_path}")

        wer_plot = args.output_dir / f"asr_wer_{timestamp}.png"
        timing_plot = args.output_dir / f"asr_timing_{timestamp}.png"

        plot_asr_wer(models, dataset_info, system_info, wer_plot)
        plot_asr_timing(models, dataset_info, system_info, timing_plot)

        print("\n" + "="*60)
        print("ASR COMPARISON SUMMARY")
        print("="*60)
        for name, data in models.items():
            if 'aggregated' not in data:
                continue
            agg = data['aggregated']
            print(f"\n{name.upper()}:")
            print(f"  WER:      {agg['wer']['mean']:6.2f}% (±{agg['wer']['std']:.2f}%)")
            print(f"  Avg Time: {agg['timing']['mean']:6.2f}s/file")
        print("="*60)

    elif args.component == "e2e":
        logger.info("\n" + "="*60)
        logger.info("Running End-to-End Pipeline comparison...")
        logger.info("="*60)

        whisperx_pipelines = {}
        if args.include_whisperx:
            logger.info("\n" + "="*60)
            logger.info("Running full WhisperX setup")
            logger.info("="*60)

            whisperx_pipelines = run_whisperx_in_venv(
                args.audio_dir,
                args.annotation_dir,
                args.language,
                args.limit
            )
            logger.info(f"WhisperX setup successful! Tested {len(whisperx_pipelines)} models")

        logger.info("\n" + "="*60)
        logger.info("Running WhoSays and Pyannote comparison...")
        logger.info("="*60)

        pipelines = compare_e2e_pipelines(file_pairs, skip_overlap=args.skip_overlap)
        pipelines = aggregate_e2e_results(pipelines)

        if whisperx_pipelines:
            pipelines.update(whisperx_pipelines)
            logger.info(f"Merged {len(whisperx_pipelines)} WhisperX pipelines")

        output_data = {
            "comparison_type": "e2e",
            "timestamp": timestamp,
            "system_info": system_info,
            "dataset": dataset_info,
            "evaluation_settings": {
                "collar": 0.25,
                "skip_overlap": False,
                "timing_note": "WhoSays: diarization + ASR (separate). Pyannote: diarization only. WhisperX: diarization + ASR (combined)."
            },
            "pipelines": {
                name: {
                    "has_transcription": data['has_transcription'],
                    "model_info": data.get('model_info', 'N/A'),
                    "per_file_results": data.get('per_file_results', data.get('results', [])),
                    "aggregated": data['aggregated']
                }
                for name, data in pipelines.items()
            }
        }

        json_path = args.output_dir / f"e2e_comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nSaved results: {json_path}")

        der_plot = args.output_dir / f"e2e_der_{timestamp}.png"
        wer_plot = args.output_dir / f"e2e_wer_{timestamp}.png"
        timing_plot = args.output_dir / f"e2e_timing_{timestamp}.png"

        plot_e2e_der(pipelines, dataset_info, system_info, der_plot)
        plot_e2e_wer(pipelines, dataset_info, system_info, wer_plot)
        plot_e2e_timing(pipelines, dataset_info, system_info, timing_plot)

        print("\n" + "="*60)
        print("END-TO-END PIPELINE COMPARISON SUMMARY")
        print("="*60)
        print("\nNOTE: Timing measurements (diarization-only for fair comparison):")
        print("      - WhoSays: diarization reported below, ASR shown separately in plot")
        print("      - Pyannote: diarization only (no ASR component)")
        print("      - WhisperX: full pipeline (cannot separate ASR from diarization)\n")
        print("-"*60)
        for name, data in pipelines.items():
            if 'aggregated' not in data:
                continue
            agg = data['aggregated']
            model_info = data.get('model_info', 'N/A')
            print(f"\n{name.upper()}:")
            if model_info != 'N/A':
                print(f"  Model:       {model_info}")
            print(f"  DER:         {agg['der']['mean']:6.2f}% (±{agg['der']['std']:.2f}%)")
            print(f"  Miss:        {agg['miss']['mean']:6.2f}% (±{agg['miss']['std']:.2f}%)")
            print(f"  False Alarm: {agg['false_alarm']['mean']:6.2f}% (±{agg['false_alarm']['std']:.2f}%)")
            print(f"  Confusion:   {agg['confusion']['mean']:6.2f}% (±{agg['confusion']['std']:.2f}%)")
            if 'wer' in agg:
                print(f"  WER:         {agg['wer']['mean']:6.2f}% (±{agg['wer']['std']:.2f}%)")
            print(f"  Avg Time:    {agg['timing']['mean']:6.2f}s/file")

            # Print component timing breakdown if available
            if 'component_timing' in agg:
                print(f"\n  Component Timing (avg per file):")
                ct = agg['component_timing']
                component_labels = [
                    ('audio_loading', 'Audio Loading'),
                    ('vad', 'VAD'),
                    ('overlap_detection', 'Overlap Detection'),
                    ('scd', 'Speaker Change Detection'),
                    ('embedding', 'Embedding'),
                    ('clustering', 'Clustering'),
                    ('asr', 'ASR'),
                    ('phoneme', 'Phoneme Alignment'),
                    ('overlap_processing', 'Overlap Processing'),
                    ('formatting', 'Formatting')
                ]
                for key, label in component_labels:
                    if key in ct:
                        print(f"    {label:<22} {ct[key]['mean']:6.2f}s (±{ct[key]['std']:.2f}s)")
        print("="*60)

    elif args.component == "full-e2e":
        # Full E2E comparison - all component combinations
        # ASR model is constant - always use large-v3
        ASR_MODEL = "openai/whisper-tiny"

        logger.info("\n" + "="*60)
        logger.info("Running Full E2E Pipeline comparison (all component combinations)...")
        logger.info("="*60)
        logger.info(f"ASR Model: {ASR_MODEL} (constant)")

        if args.skip_sod_sos:
            logger.info("Skipping SOD/SOS variations (using defaults)")
            num_combos = 2 * 3 * 4  # SCD × Embedding × Clustering = 24
        else:
            num_combos = 2 * 2 * 2 * 3 * 4  # SOD × SOS × SCD × Embedding × Clustering = 96
        logger.info(f"Total combinations to test: {num_combos}")
        logger.info(f"Files to process per combination: {len(file_pairs)}")

        # Checkpoint directory for resumable runs
        checkpoint_dir = args.output_dir / "checkpoints"

        pipelines = compare_full_e2e_pipelines(
            file_pairs,
            skip_sod_sos=args.skip_sod_sos,
            checkpoint_dir=checkpoint_dir
        )
        pipelines = aggregate_full_e2e_results(pipelines)

        # Get summary for output
        summary = pipelines.get('_summary', {})

        output_data = {
            "comparison_type": "full-e2e",
            "timestamp": timestamp,
            "system_info": system_info,
            "dataset": dataset_info,
            "evaluation_settings": {
                "collar": 0.25,
                "skip_sod_sos": args.skip_sod_sos,
                "num_combinations": num_combos,
                "asr_model": ASR_MODEL,
                "timing_note": "Diarization time only (excludes ASR for comparability)"
            },
            "summary": summary,
            "pipelines": {
                name: {
                    "config": data.get('config'),
                    "has_transcription": data.get('has_transcription', True),
                    "per_file_results": data.get('results', []),
                    "aggregated": data.get('aggregated')
                }
                for name, data in pipelines.items()
                if name != '_summary' and 'error' not in data
            },
            "failed_pipelines": {
                name: data
                for name, data in pipelines.items()
                if name != '_summary' and 'error' in data
            }
        }

        json_path = args.output_dir / f"full_e2e_comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nSaved results: {json_path}")

        # Generate plots
        comparison_plot = args.output_dir / f"full_e2e_comparison_{timestamp}.png"
        heatmap_plot = args.output_dir / f"full_e2e_heatmap_{timestamp}.png"

        plot_full_e2e_comparison(pipelines, dataset_info, system_info, comparison_plot)
        plot_full_e2e_heatmap(pipelines, dataset_info, system_info, heatmap_plot)

        print("\n" + "="*60)
        print("FULL E2E PIPELINE COMPARISON SUMMARY")
        print("="*60)
        print(f"ASR Model: {ASR_MODEL} (constant for all combinations)")
        print(f"Tested {len([p for p in pipelines if p != '_summary' and 'error' not in pipelines[p]])} configurations")
        print(f"Failed: {len([p for p in pipelines if p != '_summary' and 'error' in pipelines.get(p, {})])} configurations")

        if summary.get('best_by_der'):
            best = summary['best_by_der']
            print(f"\n🏆 BEST BY DER:")
            print(f"   Config: {best['name']}")
            print(f"   SOD: {best['config'].get('sod')} | SOS: {best['config'].get('sos')} | SCD: {best['config'].get('scd')}")
            print(f"   Embedding: {best['config'].get('embedding')} | Clustering: {best['config'].get('clustering')}")
            print(f"   DER:       {best['der']:.2f}%")
            print(f"   Miss:      {best['miss']:.2f}%")
            print(f"   False Alarm: {best['false_alarm']:.2f}%")
            print(f"   Confusion: {best['confusion']:.2f}%")
            print(f"   Time:      {best['timing']:.2f}s/file")

        if summary.get('best_by_wer'):
            best = summary['best_by_wer']
            print(f"\n📝 BEST BY WER:")
            print(f"   Config: {best['name']}")
            print(f"   WER:   {best['wer']:.2f}%")
            print(f"   DER:   {best['der']:.2f}%")

        if summary.get('best_by_time'):
            best = summary['best_by_time']
            print(f"\n⚡ FASTEST:")
            print(f"   Config: {best['name']}")
            print(f"   Time:  {best['timing']:.2f}s/file")
            print(f"   DER:   {best['der']:.2f}%")

        # Print top 5 rankings
        rankings = summary.get('rankings', [])[:5]
        if rankings:
            print(f"\n📊 TOP 5 CONFIGURATIONS BY DER:")
            print("-"*60)
            for i, r in enumerate(rankings, 1):
                print(f"  {i}. {r['name']}")
                print(f"     DER: {r['der']:.2f}% | Miss: {r['miss']:.2f}% | FA: {r['false_alarm']:.2f}% | Conf: {r['confusion']:.2f}%")
                if r.get('wer'):
                    print(f"     WER: {r['wer']:.2f}% | Time: {r['timing']:.2f}s")
                else:
                    print(f"     Time: {r['timing']:.2f}s")

        print("="*60)
        print(f"\nResults saved to: {json_path}")
        print(f"Comparison plot: {comparison_plot}")
        print(f"Heatmap plot: {heatmap_plot}")
        print(f"Checkpoint: {checkpoint_dir / 'full_e2e_checkpoint.json'}")
        print(f"\nTip: Re-run the same command to resume from checkpoint if interrupted.")

    elif args.component == "scd":
        # Speaker Change Detection comparison (Pyannote vs NeMo vs Naive)
        logger.info("\n" + "="*60)
        logger.info("Running SCD (Speaker Change Detection) comparison...")
        logger.info("="*60)

        if not args.audio:
            logger.error("--audio argument required for scd component")
            return

        if args.reference_dir:
            logger.info(f"Reference dir: {args.reference_dir} (naive SCD enabled)")

        results = compare_scd_models(
            audio_path=args.audio,
            benchmark_dir=args.annotation_dir,
            include_nemo=not args.skip_nemo,
            include_naive=not args.skip_naive,
            reference_dir=args.reference_dir
        )
        results = aggregate_scd_results(results, optimize_metric=args.scd_metric)

        output_data = {
            "comparison_type": "scd",
            "timestamp": timestamp,
            "system_info": system_info,
            "results": results
        }

        json_path = args.output_dir / f"scd_comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info(f"\nSaved results: {json_path}")

        print("\n" + "="*60)
        print(f"SCD COMPARISON SUMMARY (tolerance=3.0s, optimize={args.scd_metric})")
        print("="*60)
        print(f"Ground Truth: {results['ground_truth_count']} speaker changes")

        # Use tolerance_3.0s for all results
        tol_key = 'tolerance_3.0s'

        # Pyannote SCD - find best config's detected count
        pyannote_config = results['aggregated']['best_pyannote_by_tolerance'].get(tol_key)
        if pyannote_config:
            # Find the result with this prominence to get detected_count
            detected = None
            for r in results.get('pyannote', []):
                if r['prominence'] == pyannote_config['prominence']:
                    detected = r['detected_count']
                    break
            print(f"\nPyannote SCD (prominence={pyannote_config['prominence']}):")
            if detected:
                print(f"  Detected: {detected} changes")
            print(f"  F1: {pyannote_config['f1']:.3f}, Precision: {pyannote_config['precision']:.3f}, Recall: {pyannote_config['recall']:.3f}")

        # NeMo Sortformer
        if results.get('nemo') and 'error' not in results['nemo']:
            nemo_metrics = results['aggregated']['nemo_by_tolerance'].get(tol_key)
            if nemo_metrics:
                detected = results['nemo'].get('detected_count')
                print(f"\nNeMo Sortformer:")
                if detected:
                    print(f"  Detected: {detected} changes")
                print(f"  F1: {nemo_metrics['f1']:.3f}, Precision: {nemo_metrics['precision']:.3f}, Recall: {nemo_metrics['recall']:.3f}")

        # Naive SCD models
        for emb_model in ['pyannote', 'speechbrain', 'wav2vec2']:
            agg_key = f'best_naive_{emb_model}_by_tolerance'
            naive_key = f'naive_{emb_model}'
            if results['aggregated'].get(agg_key):
                config = results['aggregated'][agg_key].get(tol_key)
                if config:
                    # Find detected count
                    detected = None
                    for r in results.get(naive_key, []) or []:
                        if r.get('window_duration') == config['window_duration'] and r.get('similarity_threshold') == config['similarity_threshold']:
                            detected = r.get('detected_count')
                            break
                    print(f"\nNaive SCD ({emb_model}) (window={config['window_duration']}s, thresh={config['similarity_threshold']}):")
                    if detected:
                        print(f"  Detected: {detected} changes")
                    print(f"  F1: {config['f1']:.3f}, Precision: {config['precision']:.3f}, Recall: {config['recall']:.3f}")
        print("="*60)

        # Generate plot
        plot_path = args.output_dir / f"scd_comparison_{timestamp}.png"
        plot_scd_comparison(results, system_info, plot_path)
        print(f"\nPlot saved: {plot_path}")

    elif args.component == "sod":
        # Speech Overlap Detection comparison (Pyannote vs WavLM vs NeMo)
        logger.info("\n" + "="*60)
        logger.info("Running SOD (Speech Overlap Detection) comparison...")
        logger.info("="*60)

        if not args.audio:
            logger.error("--audio argument required for sod component")
            return

        # speaker_dir is optional - if provided, creates synthetic mix from speaker files
        if args.speaker_dir:
            logger.info(f"Speaker dir: {args.speaker_dir} (synthetic mix mode)")
        else:
            logger.info(f"Using audio file directly: {args.audio}")

        results = compare_sod_models(
            audio_path=args.audio,
            benchmark_dir=args.annotation_dir,
            speaker_dir=args.speaker_dir,
            include_nemo=not args.skip_nemo,
            include_wavlm=not args.skip_wavlm
        )
        results = aggregate_sod_results(results, optimize_metric=args.sod_metric)

        output_data = {
            "comparison_type": "sod",
            "timestamp": timestamp,
            "system_info": system_info,
            "results": results
        }

        json_path = args.output_dir / f"sod_comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nSaved results: {json_path}")

        print("\n" + "="*60)
        print(f"SOD COMPARISON SUMMARY")
        print("="*60)
        print(f"Ground Truth: {results['ground_truth_count']} overlap regions ({results['ground_truth_duration']:.2f}s)")
        print(f"\nBest Pyannote (onset={results['aggregated']['best_pyannote']['onset']}):")
        print(f"  Frame Precision: {results['aggregated']['best_pyannote']['frame_precision']:.1f}%")
        print(f"  Frame Recall: {results['aggregated']['best_pyannote']['frame_recall']:.1f}%")
        print(f"  Frame F1: {results['aggregated']['best_pyannote']['frame_f1']:.1f}%")

        if results['aggregated'].get('best_wavlm'):
            print(f"\nBest WavLM (onset={results['aggregated']['best_wavlm']['onset']}):")
            print(f"  Frame Precision: {results['aggregated']['best_wavlm']['frame_precision']:.1f}%")
            print(f"  Frame Recall: {results['aggregated']['best_wavlm']['frame_recall']:.1f}%")
            print(f"  Frame F1: {results['aggregated']['best_wavlm']['frame_f1']:.1f}%")

        if results.get('nemo_results') and 'error' not in results['nemo_results']:
            print(f"\nNeMo Sortformer (via diarization):")
            print(f"  Frame Precision: {results['aggregated']['nemo']['frame_precision']:.1f}%")
            print(f"  Frame Recall: {results['aggregated']['nemo']['frame_recall']:.1f}%")
            print(f"  Frame F1: {results['aggregated']['nemo']['frame_f1']:.1f}%")
        print("="*60)

        # Generate plot
        plot_path = args.output_dir / f"sod_comparison_{timestamp}.png"
        plot_sod_comparison(results, system_info, plot_path)
        print(f"\nPlot saved: {plot_path}")

    elif args.component == "sos":
        # Speech Overlap Separation comparison (Pyannote vs SpeechBrain)
        logger.info("\n" + "="*60)
        logger.info("Running SOS (Speech Overlap Separation) comparison...")
        logger.info("="*60)

        if not args.audio:
            logger.error("--audio argument required for sos component")
            return

        # speaker_dir is now optional
        if args.speaker_dir:
            logger.info(f"Speaker dir: {args.speaker_dir} (SI-SDR evaluation enabled)")
        else:
            logger.info("No speaker-dir provided, using alternative metrics")

        results = compare_sos_models(
            audio_path=args.audio,
            benchmark_dir=args.annotation_dir,
            speaker_dir=args.speaker_dir,  # Can be None
            max_regions=args.max_regions
        )
        results = aggregate_sos_results(results)

        output_data = {
            "comparison_type": "sos",
            "timestamp": timestamp,
            "system_info": system_info,
            "results": results
        }

        json_path = args.output_dir / f"sos_comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            def json_serializer(x):
                import numpy as np
                if isinstance(x, (np.floating, np.integer)):
                    return float(x) if not np.isinf(x) else None
                if x == float('-inf') or x == float('inf'):
                    return None
                raise TypeError(f"Object of type {type(x)} is not JSON serializable")
            json.dump(output_data, f, indent=2, default=json_serializer)
        logger.info(f"\nSaved results: {json_path}")

        # Generate plot
        plot_path = args.output_dir / f"sos_comparison_{timestamp}.png"
        plot_sos_comparison(results, system_info, plot_path)

        print("\n" + "="*60)
        print("SOS (SPEECH OVERLAP SEPARATION) COMPARISON SUMMARY")
        print("="*60)
        print(f"Processed {results['ground_truth_overlaps']} overlap regions")

        has_references = results.get('has_references', False)
        if has_references:
            print(f"\n{'Model':<40} {'Mean SI-SDR':<15} {'Std':<10} {'Time':<10}")
            print("-"*75)
            for model in ['pyannote', 'sepformer']:
                if results[model] and 'error' not in results[model]:
                    name = results[model]['model']
                    mean_sdr = results[model].get('mean_si_sdr', float('-inf'))
                    std_sdr = results[model].get('std_si_sdr', 0)
                    total_time = results[model]['total_time']
                    print(f"{name:<40} {mean_sdr:>8.1f} dB    ±{std_sdr:<6.1f}   {total_time:.2f}s")
                elif results[model]:
                    print(f"{model:<40} ERROR: {results[model].get('error', 'Unknown')}")
        else:
            print(f"\n{'Model':<40} {'Energy Ratio':<15} {'Sources':<10} {'Time':<10}")
            print("-"*75)
            for model in ['pyannote', 'sepformer']:
                if results[model] and 'error' not in results[model]:
                    name = results[model]['model']
                    energy_ratio = results[model].get('mean_energy_ratio', 0)
                    num_sources = results[model].get('mean_num_sources', 0)
                    total_time = results[model]['total_time']
                    print(f"{name:<40} {energy_ratio:>10.3f}     {num_sources:>6.1f}     {total_time:.2f}s")
                elif results[model]:
                    print(f"{model:<40} ERROR: {results[model].get('error', 'Unknown')}")
        print("="*60)

    elif args.component == "speaker-id":
        # Speaker Identification comparison (embedding models vs clustering methods)
        logger.info("\n" + "="*60)
        logger.info("Running Speaker Identification comparison...")
        logger.info("="*60)

        if not args.audio:
            logger.error("--audio argument required for speaker-id component")
            return
        if not args.reference_dir:
            logger.error("--reference-dir argument required for speaker-id component")
            return

        results = compare_speaker_id_models(
            audio_path=args.audio,
            benchmark_dir=args.annotation_dir,
            reference_dir=args.reference_dir
        )
        results = aggregate_speaker_id_results(results)

        output_data = {
            "comparison_type": "speaker-id",
            "timestamp": timestamp,
            "system_info": system_info,
            "results": results
        }

        json_path = args.output_dir / f"speaker_id_comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nSaved results: {json_path}")

        print("\n" + "="*60)
        print("SPEAKER IDENTIFICATION COMPARISON SUMMARY")
        print("="*60)
        print(f"Segments: {results['num_segments']}, Speakers: {results['num_speakers']}")
        print(f"\nBest: {results['aggregated']['best_combination']['embedding']} + "
              f"{results['aggregated']['best_combination']['method']} = "
              f"{results['aggregated']['best_combination']['accuracy']:.2f}%")

        print(f"\n{'Embedding':<15} {'Cosine':<12} {'K-Means':<12} {'Agglom.':<12} {'DBSCAN':<12}")
        print("-"*63)
        for emb_name, methods in results['embeddings'].items():
            row = f"{emb_name:<15}"
            for method in ['Cosine Similarity', 'K-Means', 'Agglomerative', 'DBSCAN']:
                if method in methods:
                    row += f" {methods[method]['accuracy']:>10.2f}%"
                else:
                    row += f" {'N/A':>10}"
            print(row)
        print("="*60)
    
    elif args.component == "cluster-viz":
        # Cluster visualization with UMAP
        logger.info("\n" + "="*60)
        logger.info("Running Cluster Visualization...")
        logger.info("="*60)

        if not args.embedding_type:
            logger.error("--embedding-type argument required for cluster-viz component")
            return
        if not args.audio_folder:
            logger.error("--audio-folder argument required for cluster-viz component")
            return
        # alignment_file is optional - if not provided, uses directory structure

        results = compare_cluster_viz(
            audio_folder=args.audio_folder,
            alignment_file=args.alignment_file,  # Can be None
            embedding_type=args.embedding_type
        )

        output_data = {
            "comparison_type": "cluster-viz",
            "timestamp": timestamp,
            "system_info": system_info,
            "results": {
                'embedding_type': results['embedding_type'],
                'embedding_dim': results['embedding_dim'],
                'num_embeddings': results['num_embeddings'],
                'num_speakers': results['num_speakers'],
                'speakers': results['speakers'],
                'clustering_results': results['clustering_results']
            }
        }

        json_path = args.output_dir / f"cluster_viz_{args.embedding_type}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nSaved results: {json_path}")

        # Generate plot
        plot_path = args.output_dir / f"cluster_viz_{args.embedding_type}_umap_{timestamp}.png"
        plot_cluster_umap(results, system_info, plot_path)

        print("\n" + "="*60)
        print("CLUSTER VISUALIZATION SUMMARY")
        print("="*60)
        print(f"Embedding: {results['embedding_type']} ({results['embedding_dim']}d)")
        print(f"Samples: {results['num_embeddings']}, Speakers: {results['num_speakers']}")
        print(f"\n{'Method':<18} {'ARI':<10} {'Silhouette':<12} {'F1':<10}")
        print("-"*50)

        for method, data in results['clustering_results'].items():
            ari = data.get('ari', 0)
            sil = data.get('silhouette', 'N/A')
            f1 = data.get('f1', 0)
            if isinstance(sil, float):
                print(f"{method:<18} {ari:<10.3f} {sil:<12.3f} {f1:<10.3f}")
            else:
                print(f"{method:<18} {ari:<10.3f} {'N/A':<12} {f1:<10.3f}")

        print(f"\nPlot saved: {plot_path}")
        print("="*60)

    elif args.component == "embedding-viz":
        # Embedding comparison (metrics only)
        logger.info("\n" + "="*60)
        logger.info("Running Embedding Model comparison...")
        logger.info("="*60)

        if not args.audio:
            logger.error("--audio argument required for embedding-viz component")
            return

        results = compare_embedding_models(
            audio_path=args.audio,
            benchmark_dir=args.annotation_dir,
            reference_dir=args.reference_dir
        )
        results = aggregate_embedding_results(results)

        output_data = {
            "comparison_type": "embedding",
            "timestamp": timestamp,
            "system_info": system_info,
            "results": {
                'num_speakers': results['num_speakers'],
                'speakers': results['speakers'],
                'embeddings': results['embeddings'],
                'aggregated': results['aggregated']
            }
        }

        json_path = args.output_dir / f"embedding_comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nSaved results: {json_path}")

        # Generate plot
        plot_path = args.output_dir / f"embedding_comparison_{timestamp}.png"
        plot_embedding_comparison(results, system_info, plot_path)

        print("\n" + "="*60)
        print("EMBEDDING MODEL COMPARISON SUMMARY")
        print("="*60)
        print(f"Speakers: {results['num_speakers']}")
        print(f"\nBest by ARI: {results['aggregated']['best_by_ari']['embedding']} + "
              f"{results['aggregated']['best_by_ari']['clustering']} "
              f"(ARI={results['aggregated']['best_by_ari']['ari']:.3f})")
        print(f"Best F1: {results['aggregated']['best_f1']:.3f}")
        if 'best_cosine' in results['aggregated']:
            print(f"Best Cosine: {results['aggregated']['best_cosine']['embedding']} "
                  f"(Accuracy={results['aggregated']['best_cosine']['accuracy']:.3f})")

        # Print detailed metrics table
        has_cosine = any('cosine' in emb_data for emb_data in results['embeddings'].values())
        if has_cosine:
            print(f"\n{'Embedding':<12} {'Dim':<6} {'Method':<12} {'ARI':<8} {'F1':<8} {'Accuracy':<10} {'Silhouette':<10}")
            print("-"*76)
        else:
            print(f"\n{'Embedding':<12} {'Dim':<6} {'Method':<12} {'ARI':<8} {'F1':<8} {'Silhouette':<10}")
            print("-"*66)
        for emb_name, emb_data in results['embeddings'].items():
            dim = emb_data.get('embedding_dim', 'N/A')
            for method in ['cosine', 'kmeans', 'agglomerative', 'dbscan']:
                if method in emb_data:
                    m = emb_data[method]
                    if method == 'cosine':
                        print(f"{emb_name:<12} {dim:<6} {method:<12} {m['ari']:<8.3f} {m['f1']:<8.3f} {m['accuracy']:<10.3f} {'N/A':<10}")
                    else:
                        if has_cosine:
                            print(f"{emb_name:<12} {dim:<6} {method:<12} {m['ari']:<8.3f} {m['f1']:<8.3f} {'N/A':<10} {m['silhouette']:<10.3f}")
                        else:
                            print(f"{emb_name:<12} {dim:<6} {method:<12} {m['ari']:<8.3f} {m['f1']:<8.3f} {m['silhouette']:<10.3f}")
                    emb_name = ""  # Don't repeat embedding name
                    dim = ""
        print(f"\nPlot saved: {plot_path}")
        print("="*60)


if __name__ == "__main__":
    main()
