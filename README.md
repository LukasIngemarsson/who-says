# Who says pipeline

## Run with docker

Build image

```bash
docker build -t who-says-pipeline .
```

Run the container
```bash
docker run --rm who-says-pipeline python main.py multi_speaker_sample.mp3
```

To get docker to run you need to have Docker Desktop or similar running in the background.

## Adding new pipeline components

1. Create module in `pipeline/[component_name]/`
2. Add dependencies to `requirements.txt`
3. Import and use in `main.py`
4. Rebuild Docker: `docker build -t who-says-pipeline .`
5. Run the container `docker run --rm who-says-pipeline python main.py <audiofile>`
