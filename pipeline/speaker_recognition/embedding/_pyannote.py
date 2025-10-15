from utils.constants import TENSOR_DTYPE
from utils import match_frequency

import os
from dotenv import load_dotenv
from pyannote.audio import Model, Inference
import torch

from utils import load_audio_from_file


# TODO: Add types to class/function args
# TODO: Set up general class for different models later (if wanted)

class PyAnnoteEmbedding:
    def __init__(self, model: str = "pyannote/embedding", batch_size: int = 1) -> None:
        load_dotenv("./.env")
        # .env is in root and not ignored
        token = os.getenv("HF_TOKEN_PYANNOTE_EMBEDDING")
        if token is None:
            raise ValueError("Missing HF_TOKEN_PYANNOTE_EMBEDDING in environment variables.")
        
        self.model = Model.from_pretrained(model, use_auth_token=token)
        self.inference = Inference(self.model, window="whole", batch_size=batch_size)

    def embed(self, audio: torch.Tensor, frequency: int) -> torch.Tensor:
        """Embed directly from audio tensor + sample rate"""
        audio = match_frequency(audio, frequency)

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # (1, num_samples)
        elif audio.ndim > 2:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")
        
        embedding = self.inference({"waveform": audio, "sample_rate": frequency})
        return torch.tensor(embedding, dtype=TENSOR_DTYPE)

    def embed_from_file(self, file_path: str) -> torch.Tensor:
        """Convenience wrapper for file-based input"""
        audio, frequency = load_audio_from_file(file_path)
        return self.embed(audio, frequency)

if __name__ == "__main__":
    file_path = "samples/single_speaker_sample.wav"

    audio, frequency = load_audio_from_file(file_path)

    print(f"loaded file: {file_path}\naudio length: {len(audio)}\nfrequency: {frequency}")

    model = PyAnnoteEmbedding()

    embed_audio_frequency = model.embed(audio, frequency)

    print(f"embeddings shape: {embed_audio_frequency.shape}")
