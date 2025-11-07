#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


DOCKER_IMAGE = "who-says-pipeline"


def log(text: str):
    print(f"\n-----\n{text}\n-----\n")


def build_image():
    log(f"Building Docker image: {DOCKER_IMAGE}")
    subprocess.run(["docker", "build", "-t", DOCKER_IMAGE, "."], check=True)


def run_pipeline(audio_file: str, annotation_file: str = None):
    if not Path(audio_file).is_file():
        log(f"File not found: {audio_file}")
        sys.exit(1)

    if annotation_file and not Path(annotation_file).is_file():
        log(f"Annotation file not found: {annotation_file}")
        sys.exit(1)

    samples_dir = Path.cwd() / "samples"

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{samples_dir.absolute()}:/app/samples:ro",
        "--env-file", ".env",
        DOCKER_IMAGE,
        "python", "main.py", audio_file
    ]

    if annotation_file:
        cmd.extend(["--annotation", annotation_file])

    log(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_component(module_path: str):
    cmd = [
        "docker", "run", "--rm",
        "--env-file", ".env",
        DOCKER_IMAGE,
        "python", "-m", module_path
    ]
    log(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("\t./docker_run.py pipeline <audio_file> [--annotation <annotation_file>]")
        print("\t./docker_run.py component <module_path>")
        sys.exit(1)

    # always build the Docker image first
    build_image()

    mode = sys.argv[1]
    if mode == "pipeline":
        audio_file = sys.argv[2]
        annotation_file = None
        
        if len(sys.argv) > 3 and sys.argv[3] == "--annotation":
            if len(sys.argv) < 5:
                print("Error: --annotation requires a file path")
                sys.exit(1)
            annotation_file = sys.argv[4]

        run_pipeline(audio_file, annotation_file)
    elif mode == "component":
        run_component(sys.argv[2])
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
