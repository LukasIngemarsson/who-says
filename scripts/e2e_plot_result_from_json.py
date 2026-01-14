#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from loguru import logger

from utils import (
    plot_e2e_der,
    plot_e2e_wer,
    plot_e2e_timing
)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate E2E comparison plots from existing JSON results"
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        required=True,
        help="Path to e2e_comparison JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for plots (defaults to JSON file's directory)"
    )
    args = parser.parse_args()

    if not args.json_file.exists():
        logger.error(f"JSON file not found: {args.json_file}")
        sys.exit(1)

    logger.info(f"Loading e2e comparison results from: {args.json_file}")

    try:
        with open(args.json_file, 'r') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {e}")
        sys.exit(1)

    required_keys = ['timestamp', 'pipelines', 'dataset', 'system_info']
    missing_keys = [key for key in required_keys if key not in json_data]
    if missing_keys:
        logger.error(f"JSON file missing required keys: {', '.join(missing_keys)}")
        sys.exit(1)

    if json_data.get('comparison_type') != 'e2e':
        logger.warning(f"JSON file has comparison_type='{json_data.get('comparison_type')}', expected 'e2e'")

    timestamp = json_data['timestamp']
    pipelines = json_data['pipelines']
    dataset_info = json_data['dataset']
    system_info = json_data['system_info']

    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = args.json_file.parent

    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Pipelines: {', '.join(pipelines.keys())}")
    logger.info(f"Dataset: {dataset_info['num_files']} files, {dataset_info['total_duration_seconds']/60:.1f} min, language={dataset_info['language']}")

    logger.info("\nGenerating plots...")

    der_plot = output_dir / f"e2e_der_{timestamp}.png"
    wer_plot = output_dir / f"e2e_wer_{timestamp}.png"
    timing_plot = output_dir / f"e2e_timing_{timestamp}.png"

    try:
        plot_e2e_der(pipelines, dataset_info, system_info, der_plot)
        logger.info(f"DER plot: {der_plot}")
    except Exception as e:
        logger.error(f"Failed to generate DER plot: {e}")

    try:
        plot_e2e_wer(pipelines, dataset_info, system_info, wer_plot)
        logger.info(f"WER plot: {wer_plot}")
    except Exception as e:
        logger.error(f"Failed to generate WER plot: {e}")

    try:
        plot_e2e_timing(pipelines, dataset_info, system_info, timing_plot)
        logger.info(f"Timing plot: {timing_plot}")
    except Exception as e:
        logger.error(f"Failed to generate timing plot: {e}")

    logger.info("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
