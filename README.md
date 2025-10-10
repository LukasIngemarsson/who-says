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
