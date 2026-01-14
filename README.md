# WhoSays

WhoSays is a real-time pipeline for multi-speaker diarization and transcription that answers three practical questions: who spoke, when, and what did they say?

It combines voice activity detection (VAD), speaker change/overlap handling, speaker embedding, clustering and recognition, and automatic speech recognition (ASR) into a streaming workflow. This project is primarily optimized for speaker recognition quality; transcription accuracy was a secondary goal.

The following diagram illustrates the complete pipeline flow, showing how audio input is processed through speaker segmentation, diarization, and ASR components to produce the final transcribed output with speaker labels:

![Pipeline Architecture](docs/Pipeline.png)

For further insight, see the [demo](docs/demo.mp4) or the [final presentation](docs/who-says.pptx).

## Setup, Build, and Run

Create a `.env` file at the root of the repo containing:

```
HF_TOKEN=yourToken
```

You can create an HF token with read access on [HuggingFace](https://huggingface.co) and must also accept the model terms for:

- [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1),
- [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0),
- [`pyannote/embedding`](https://huggingface.co/pyannote/embedding),
- [`pyannote/separation-ami-1.0`](https://huggingface.co/pyannote/separation-ami-1.0).

Once the token is set in `.env`, add execution rights to `run_docker.sh` and run it:

```bash
chmod +x run_docker.sh
./run_docker.sh
```

This builds a Docker image containing the pipeline, backend, and frontend.

Once it’s up, you can open the app at [localhost:8000](http://localhost:8000).

> **Note:**  
> To get Docker to run you need to have Docker Desktop or simply the Docker process running in the background.
> Furthermore, since everything runs locally, low-end hardware will cause the build process to take a long time, and will cause the performance of the app to be highly unreliable. Therefore, it is recommended to use a modern graphics card if available.

To stop the container:

```bash
docker stop whosays-container
```

To follow logs:

```bash
docker logs -f whosays-container
```

## Compare Pipeline Models

Benchmark/analysis CLIs live under `scripts/`.

- Host: `python -m scripts.compare --help`
- Container: `docker exec -it whosays-container python -m scripts.compare --help`

### Compare Models with Benchmark Datasets

See `python -m scripts.compare --help` for all options and components. A typical end-to-end run looks like:

```bash
# Base comparison (WhoSays + Pyannote 3.1)
python -m scripts.compare --component e2e \
    --audio-dir data/benchmark/chunks \
    --annotation-dir data/benchmark/annotations \
    --language english

# Include WhisperX - runs in separate environment component because of conflicting dependencies with the main pipeline (only english version of whisperX is used for now)
python -m scripts.compare --component e2e \
    --audio-dir data/benchmark/chunks \
    --annotation-dir data/benchmark/annotations \
    --language english \
    --include-whisperx
```

By default, comparison JSON files and plots are written under `results/`. This folder is intentionally gitignored and won’t exist in a fresh clone until you run a comparison. You can alternatively choose your own directory with `--output-dir`.

**Regenerate E2E Plots From Existing JSON Results:**

If you've made changes to plot styling or want to regenerate plots without re-running the entire comparison (which can take time), you can use the plot regeneration script:

```bash
python -m scripts.e2e_plot_result_from_json \
    --json-file "results/comparison/english/e2e_comparison_*.json"

```
