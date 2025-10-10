from config.constants import DESIRED_FREQUENCY, TENSOR_DTYPE

import soundfile as sf
import torchaudio
import torch


def load_audio_from_file(file_path: str) -> tuple[torch.Tensor, int]:
    if file_path.endswith(".wav"):
        audio, frequency = sf.read(file_path)
        audio = torch.tensor(audio, dtype=TENSOR_DTYPE)
    elif file_path.endswith(".mp3") or file_path.endswith(".flac"):
        audio, frequency = torchaudio.load(file_path)
    else:
        raise ValueError(f"Unsupported audio format for file: {file_path}")

    return audio, frequency


def match_frequency(audio: torch.Tensor, frequency: int) -> torch.Tensor:
    if frequency != DESIRED_FREQUENCY:
        audio = torchaudio.functional.resample(audio, orig_freq=frequency, new_freq=DESIRED_FREQUENCY)
    return audio

