from config.constants import DESIRED_FREQUENCY

import soundfile as sf
import torchaudio
import torch


def load_audio_from_file(file_path):
    if file_path.endswith(".wav"):
        audio, frequency = sf.read(file_path)
        audio = torch.tensor(audio, dtype=torch.float32)
    elif file_path.endswith(".mp3") or file_path.endswith(".flac"):
        audio, frequency = torchaudio.load(file_path)
    else:
        raise ValueError(f"Unsupported audio format for file: {file_path}")

    return audio, frequency


def match_frequency(audio, frequency):
    if frequency != DESIRED_FREQUENCY:
        audio = torchaudio.functional.resample(audio, orig_freq=frequency, new_freq=DESIRED_FREQUENCY)
    return audio

