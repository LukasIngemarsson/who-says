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
    plot_metrics,
    plot_timing,
    plot_sc_timing,
    plot_sc_silhouette,
    plot_sc_der,
    plot_asr_wer,
    plot_asr_timing,
    plot_e2e_der,
    plot_e2e_wer,
    plot_e2e_timing,
    load_audio_from_file
)
from utils.constants import SR


def main():
    parser = argparse.ArgumentParser(
        description="Compare models for who-says pipeline components"
    )
    parser.add_argument(
        "--component",
        required=True,
        choices=["vad", "sc", "asr", "e2e"],
        help="Component to compare"
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Path to directory containing audio files"
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="Path to directory containing annotation JSON files"
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

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Component: {args.component.upper()}")
    logger.info(f"Annotation dir: {args.annotation_dir}")
    logger.info(f"Audio dir: {args.audio_dir}")

    system_info = get_system_info()
    logger.info(f"Device: {system_info['device']} ({system_info['gpu']})")

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

        plot_sc_timing(models, dataset_info, system_info, timing_plot)
        plot_sc_silhouette(models, dataset_info, system_info, silhouette_plot)
        plot_sc_der(models, dataset_info, system_info, der_plot)

        print("\n" + "="*60)
        print("SPEAKER CLUSTERING COMPARISON SUMMARY")
        print("="*60)
        for name, data in models.items():
            if 'aggregated' not in data:
                continue
            agg = data['aggregated']
            print(f"\n{name.upper().replace('_', ' + ')}:")
            print(f"  Silhouette: {agg['silhouette']['mean']:6.2f} (±{agg['silhouette']['std']:.2f})")
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

        pipelines = compare_e2e_pipelines(file_pairs)
        pipelines = aggregate_e2e_results(pipelines)

        output_data = {
            "comparison_type": "e2e",
            "timestamp": timestamp,
            "system_info": system_info,
            "dataset": dataset_info,
            "evaluation_settings": {
                "collar": 0.25,
                "skip_overlap": False
            },
            "pipelines": {
                name: {
                    "has_transcription": data['has_transcription'],
                    "per_file_results": data['results'],
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
        for name, data in pipelines.items():
            if 'aggregated' not in data:
                continue
            agg = data['aggregated']
            print(f"\n{name.upper()}:")
            print(f"  DER:         {agg['der']['mean']:6.2f}% (±{agg['der']['std']:.2f}%)")
            print(f"  Miss:        {agg['miss']['mean']:6.2f}% (±{agg['miss']['std']:.2f}%)")
            print(f"  False Alarm: {agg['false_alarm']['mean']:6.2f}% (±{agg['false_alarm']['std']:.2f}%)")
            print(f"  Confusion:   {agg['confusion']['mean']:6.2f}% (±{agg['confusion']['std']:.2f}%)")
            if 'wer' in agg:
                print(f"  WER:         {agg['wer']['mean']:6.2f}% (±{agg['wer']['std']:.2f}%)")
            print(f"  Avg Time:    {agg['timing']['mean']:6.2f}s/file")
        print("="*60)


if __name__ == "__main__":
    main()
