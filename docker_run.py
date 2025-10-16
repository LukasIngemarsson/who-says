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


def run_pipeline(audio_file: str):
    if not Path(audio_file).is_file():
        log(f"File not found: {audio_file}")
        sys.exit(1)

    cmd = [
        "docker", "run", "--rm",
        DOCKER_IMAGE,
        "python", "main.py", audio_file
    ]
    log(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_component(module_path: str):
    cmd = [
        "docker", "run", "--rm",
        DOCKER_IMAGE,
        "python", "-m", module_path
    ]
    log(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("\t./docker_run.py pipeline <audio_file> (runs main)")
        print("\t./docker_run.py component <module_path> (runs specified module)")
        sys.exit(1)

    # always build the Docker image first
    build_image()

    mode = sys.argv[1]
    if mode == "pipeline":
        run_pipeline(sys.argv[2])
    elif mode == "component":
        run_component(sys.argv[2])
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
