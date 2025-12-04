# Who says pipeline

In many real-world speech systems—like smart home assistants, meeting transcribers, or secure voice interfaces—it's crucial not only to transcribe multi-person conversations accurately but also to distinguish who spoke when and to recognize specific users for access control or personalization. This project will build a pipeline that integrates voice activity detection (VAD), speaker diarization, and speaker recognition, enabling systems to selectively trust and process commands from authorized individuals.

The following diagram illustrates the complete pipeline flow, showing how audio input is processed through VAD, speaker diarization, speaker recognition, and ASR components to produce the final transcribed output with speaker labels:

![Pipeline Architecture](Pipeline.png)

## Run with docker

You can use `docker_run.py` to conveniently build the image, and run the full pipeline or test a single component w/ Docker.
Use `chmod +x docker_run.py` to make the script runnable as `./docker_run.py` (only needed once).

> **Note:**  
> To get Docker to run you need to have Docker Desktop or simply the Docker process running in the background.

Usage:

```bash
./docker_run.py pipeline <audio_file> # (runs main / full pipeline)
./docker_run.py component <module_path> # (runs specified module / isolated component)
```

### Running with evaluation metrics

To evaluate pipeline performance against gold standard annotations, use the `--annotation` flag:

For example:

```bash
./docker_run.py pipeline samples/multi_speaker_sample.mp3 --annotation samples/annotations/multi_speaker_sample.json
```

```bash
docker build -t my-diarization-api .
docker run -p 8000:8000 -v .env:/app/.env my-diarization-api
```

 - "audio_file", type=Path, help="Path to the audio file to process"
 - "--num-speakers", type=int, default=2, help="Expected number of speakers (default: 2)"
 - "--annotation", type=Path, help="Path to gold-standard annotation JSON for metrics evaluation (optional)"
 - "--output", "-o", type=Path, help="Output JSON file path (optional)"
 - "--pretty", action="store_true", help="Pretty print the output"
 - "--timing", action="store_true", help="Include timing metrics for each model run"

### Compare pipeline models
Run from inside docker (after running `./run.sh start`)

#### Compare models with benchmark datasets

**Options:**
- `--component`: Component to compare (`vad` or `sc`)
- `--audio-dir`: Directory containing audio files (required)
- `--annotation-dir`: Directory containing annotation JSON files (required)
- `--language`: Language of dataset (default: `unknown`)
- `--limit`: Limit number of files for quick testing
- `--output-dir`: Output directory (default: `results/comparison/english`)


**VAD Comparison** (Silero vs Pyannote):
```bash
python compare.py --component vad \
    --audio-dir samples/meetings/meeting3-en/chunks \
    --annotation-dir samples/benchmarks/english \
    --language english
```

**Speaker Clustering Comparison**:
```bash
python compare.py --component sc \
    --audio-dir samples/meetings/meeting3-en/chunks \
    --annotation-dir samples/benchmarks/english \
    --language english
```

**ASR Comparison** (7 Whisper models from tiny to large):
```bash
python compare.py --component asr \
    --audio-dir samples/meetings/meeting3-en/chunks \
    --annotation-dir samples/benchmarks/english \
    --language english
```

#### Single file comparison
**VAD:**
```bash
python -m pipeline.speaker_segmentation.VAD.compare_vad_models <audioFile> --annotation <annotationFile>
```

**Speaker embedding and clustering:**
```bash
python -m pipeline.speaker_recognition.embedding.compare_embeddings_clustering <audioFile> --num-speakers 2
```
 
<!-- Running w/o the script: -->
<!-- Build image -->
<!---->
<!-- ```bash -->
<!-- docker build -t who-says-pipeline . -->
<!-- ``` -->
<!-- ```bash -->
<!-- docker run --rm who-says-pipeline python main.py multi_speaker_sample.mp3 -->
<!-- ``` -->
<!-- To run any module that isn’t at the project root, use the -m flag so Python treats the project as a package. -->
<!-- ```bash -->
<!-- docker run --rm who-says-pipeline python -m pipeline.<component_name>.<library_name>.<etc.> -->
<!-- ``` -->
<!-- This ensures imports (e.g. from `utils/`) work correctly. -->

> **Note:**  
> Every code-containing subfolder (e.g. `pipeline/`, `speaker_recognition/`, `embedding/`, `utils/`, etc.)  
> must include an empty `__init__.py` file so Python recognizes it as a package.

## Adding new pipeline components

1. Create your class in `pipeline/[component]/your_file.py`
2. Export it in `pipeline/[component]/__init__.py`:
   ```python
   from .your_file import YourClass
   __all__ = ["YourClass"]
   ```
3. Import in `main.py`: `from pipeline.[component] import YourClass`
4. Add dependencies to `requirements.txt`
5. Rebuild Docker: `docker build -t who-says-pipeline .`


## Update `requirements.txt`

For now, manually add necessary packages that are not yet installed in the Docker container, i.e.,
add the library (and if needed, the specific version) as a new line in `requirements.txt`.

<!-- We can try this more automated alternative as well, but we need to ensure that it properly includes -->
<!-- the necessary packages (and their version requirements). -->
<!-- Install `pipreqs` -->
<!---->
<!-- ```bash -->
<!-- pip install pipreqs -->
<!-- ``` -->
<!---->
<!-- Use this command to create a new version of `requirements.txt` while in the root directory -->
<!---->
<!-- ```bash -->
<!-- python -m pipreqs.pipreqs . --force -->
<!-- ``` -->
<!---->
<!-- This will override the current `requirements.txt` file with a new one.  -->
