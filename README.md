# Who says pipeline

## Run with docker

Build image

```bash
docker build -t who-says-pipeline .
```

Run the container
> To get docker to run you need to have Docker Desktop or similar running in the background.
```bash
docker run --rm who-says-pipeline python main.py multi_speaker_sample.mp3
```
To run any module that isn’t at the project root, use the -m flag so Python treats the project as a package.
```bash
docker run --rm who-says-pipeline python -m pipeline.<component_name>.<library_name>.<etc.>
```
This ensures imports (e.g. from `utils/`) work correctly.

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

Install `pipreqs`

```bash
pip install pipreqs
```

Use this command to create a new version of `requirements.txt` while in the root directory

```bash
python -m pipreqs.pipreqs . --force
```

This will override the current `requirements.txt` file with a new one. 
