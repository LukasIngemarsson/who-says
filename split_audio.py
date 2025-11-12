#!/usr/bin/env python3
"""
Split audio files into chunks of specified duration.
"""

import argparse
import subprocess
from pathlib import Path


def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(file_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    return float(result.stdout.strip())


def split_audio(input_file, output_dir, chunk_duration_minutes=5):
    """
    Split audio file into chunks of specified duration.

    Args:
        input_file: Path to input audio file
        output_dir: Directory to save chunks
        chunk_duration_minutes: Duration of each chunk in minutes
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get total duration
    total_duration = get_audio_duration(input_path)
    chunk_duration_seconds = chunk_duration_minutes * 60

    print(f"Input file: {input_path}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print(f"Chunk duration: {chunk_duration_seconds}s ({chunk_duration_minutes} minutes)")

    # Calculate number of chunks
    num_chunks = int(total_duration / chunk_duration_seconds) + (1 if total_duration % chunk_duration_seconds > 0 else 0)
    print(f"Will create {num_chunks} chunks\n")

    # Get file extension
    file_extension = input_path.suffix
    base_name = input_path.stem

    # Split audio into chunks
    for i in range(num_chunks):
        start_time = i * chunk_duration_seconds
        output_file = output_path / f"{base_name}_part{i:03d}{file_extension}"

        print(f"Creating chunk {i+1}/{num_chunks}: {output_file.name}")

        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ss', str(start_time),
            '-t', str(chunk_duration_seconds),
            '-c', 'copy',  # Copy codec without re-encoding for speed
            '-y',  # Overwrite output file if exists
            str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: ffmpeg failed for chunk {i}: {result.stderr}")
        else:
            print(f"  ✓ Created: {output_file}")

    print(f"\nDone! Created {num_chunks} chunks in {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Split audio files into chunks")
    parser.add_argument("input_file", type=Path, help="Path to input audio file")
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Output directory for chunks (default: same directory as input file)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=5,
        help="Duration of each chunk in minutes (default: 5)"
    )

    args = parser.parse_args()

    # Use input file directory if output directory not specified
    output_dir = args.output_dir if args.output_dir else args.input_file.parent

    try:
        split_audio(args.input_file, output_dir, args.duration)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
