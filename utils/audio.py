from utils.constants import TENSOR_DTYPE, SR
from pathlib import Path

import soundfile as sf
import torchaudio
import torch


def load_audio_from_file(file_path: str | Path, sr: int = SR, convert_to_mono:bool=False) -> tuple[torch.Tensor, int]:
    file_path = str(file_path)
    if file_path.endswith(".wav"):
        audio, sr = sf.read(file_path)
        audio = torch.tensor(audio, dtype=TENSOR_DTYPE)
    elif file_path.endswith(".mp3") or file_path.endswith(".flac"):
        audio, org_sr = torchaudio.load(file_path)
        audio = match_frequency(audio, org_sr, sr)
    else:
        raise ValueError(f"Unsupported audio format for file: {file_path}")

    if convert_to_mono:
        audio = to_mono(audio)

    return audio, sr


def match_frequency(audio: torch.Tensor, frequency: int, sr: int = SR) -> torch.Tensor:
    if frequency != sr:
        audio = torchaudio.functional.resample(audio, orig_freq=frequency, new_freq=sr)
    return audio


def to_mono(audio):
    """
    Converts multi-channel audio to mono by averaging across channels.
    Args:
        audio (torch.Tensor): Audio tensor of shape (channels, time) or (time,)
    Returns:
        torch.Tensor: Mono audio tensor of shape (time,)
    """
    if audio.ndim == 2:
        # Average across channels
        return audio.mean(dim=0)
    return audio

