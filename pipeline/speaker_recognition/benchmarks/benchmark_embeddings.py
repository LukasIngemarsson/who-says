import time
import numpy as np

from utils import load_audio_from_file
from ..abc import PyAnnoteEmbedding, SpeechBrainEmbedding

def time_function(func, *args, repeat=3, **kwargs):
    """Utility to measure average runtime of a function."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)


def benchmark_models(audio_files, repeat=3):
    models = {
        "SpeechBrain": SpeechBrainEmbedding(),
        "PyAnnote": PyAnnoteEmbedding(),
    }

    results = []
    for name, model in models.items():
        print(f"\n=== {name} ===")
        for file_path in audio_files:
            audio, sr = load_audio_from_file(file_path)
            mean_t, std_t = time_function(model.embed, audio, sr, repeat=repeat)
            results.append((name, file_path, mean_t, std_t))
            print(f"{file_path}: {mean_t:.3f}s ± {std_t:.3f}s")

    return results
