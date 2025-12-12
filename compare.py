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
        choices=["vad", "sc", "asr", "e2e", "scd", "sod", "sos", "speaker-id", "embedding-viz", "cluster-viz"],
        help="Component to compare: vad, sc (speaker clustering), asr, e2e (end-to-end), "
             "scd (speaker change detection), sod (speech overlap detection), "
             "sos (speech overlap separation), speaker-id (speaker identification), "
             "embedding-viz (embedding visualization), cluster-viz (clustering with UMAP)"
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
        help="Directory containing reference speaker audio files (for speaker-id component)"
    )
    parser.add_argument(
        "--skip-nemo",
        action="store_true",
        help="Skip NeMo benchmark (for scd, sod components)"
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

        pipelines = compare_e2e_pipelines(file_pairs)
        pipelines = aggregate_e2e_results(pipelines)

        output_data = {
            "comparison_type": "e2e",
            "timestamp": timestamp,
            "system_info": system_info,
            "dataset": dataset_info,
            "evaluation_settings": {
                "collar": 0.25,
                "skip_overlap": False,
                "timing_note": "WhoSays reports diarization-only time (excluding ASR) for fair comparison diarization-only pipelines"
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
        print("\nNOTE: Timing measures diarization only (VAD, SCD, embedding, clustering).")
        print("      ASR transcription excluded from WhoSays timing for fair comparison.\n")
        print("-"*60)
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

    elif args.component == "scd":
        # Speaker Change Detection comparison (Pyannote vs NeMo)
        logger.info("\n" + "="*60)
        logger.info("Running SCD (Speaker Change Detection) comparison...")
        logger.info("="*60)

        if not args.audio:
            logger.error("--audio argument required for scd component")
            return

        results = compare_scd_models(
            audio_path=args.audio,
            benchmark_dir=args.annotation_dir,
            include_nemo=not args.skip_nemo
        )
        results = aggregate_scd_results(results)

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
        print("SCD COMPARISON SUMMARY")
        print("="*60)
        print(f"Ground Truth: {results['ground_truth_count']} speaker changes")
        print(f"\nPyannote SCD (best by tolerance):")
        for tol_key, config in results['aggregated']['best_pyannote_by_tolerance'].items():
            if config:
                print(f"  {tol_key}: prominence={config['prominence']}, F1={config['f1']:.3f}")

        if results.get('nemo') and 'error' not in results['nemo']:
            print(f"\nNeMo Sortformer:")
            for tol_key, metrics in results['aggregated']['nemo_by_tolerance'].items():
                print(f"  {tol_key}: F1={metrics['f1']:.3f}")
        print("="*60)

    elif args.component == "sod":
        # Speech Overlap Detection comparison (Pyannote vs NeMo)
        logger.info("\n" + "="*60)
        logger.info("Running SOD (Speech Overlap Detection) comparison...")
        logger.info("="*60)

        if not args.audio:
            logger.error("--audio argument required for sod component")
            return

        results = compare_sod_models(
            audio_path=args.audio,
            benchmark_dir=args.annotation_dir,
            include_nemo=not args.skip_nemo
        )
        results = aggregate_sod_results(results)

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
        print("SOD COMPARISON SUMMARY")
        print("="*60)
        print(f"Ground Truth: {results['ground_truth_count']} overlap regions ({results['ground_truth_duration']:.2f}s)")
        print(f"\nBest Pyannote (onset={results['aggregated']['best_pyannote']['onset']}):")
        print(f"  Frame F1: {results['aggregated']['best_pyannote']['frame_f1']:.1f}%")
        print(f"  Segment F1@0.5: {results['aggregated']['best_pyannote']['segment_f1_iou05']:.1f}%")

        if results.get('nemo_results') and 'error' not in results['nemo_results']:
            print(f"\nNeMo Sortformer (via diarization):")
            print(f"  Frame F1: {results['aggregated']['nemo']['frame_f1']:.1f}%")
            print(f"  Segment F1@0.5: {results['aggregated']['nemo']['segment_f1_iou05']:.1f}%")
        print("="*60)

    elif args.component == "sos":
        # Speech Overlap Separation comparison (Pyannote vs SpeechBrain)
        logger.info("\n" + "="*60)
        logger.info("Running SOS (Speech Overlap Separation) comparison...")
        logger.info("="*60)

        if not args.audio:
            logger.error("--audio argument required for sos component")
            return
        if not args.speaker_dir:
            logger.error("--speaker-dir argument required for sos component")
            return

        results = compare_sos_models(
            audio_path=args.audio,
            benchmark_dir=args.annotation_dir,
            speaker_dir=args.speaker_dir,
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
            json.dump(output_data, f, indent=2, default=lambda x: None if x == float('-inf') else x)
        logger.info(f"\nSaved results: {json_path}")

        print("\n" + "="*60)
        print("SOS (SPEECH OVERLAP SEPARATION) COMPARISON SUMMARY")
        print("="*60)
        print(f"Processed {results['ground_truth_overlaps']} overlap regions")
        print(f"\n{'Model':<40} {'Mean SI-SDR':<15} {'Std':<10} {'Time':<10}")
        print("-"*75)

        for model in ['pyannote', 'sepformer']:
            if results[model] and 'error' not in results[model]:
                name = results[model]['model']
                mean_sdr = results[model]['mean_si_sdr']
                std_sdr = results[model]['std_si_sdr']
                total_time = results[model]['total_time']
                print(f"{name:<40} {mean_sdr:>8.1f} dB    ±{std_sdr:<6.1f}   {total_time:.2f}s")
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
