from utils.constants import TENSOR_DTYPE
from utils import match_frequency

import os
from dotenv import load_dotenv
from pyannote.audio import Model, Inference
import torch

from utils import load_audio_from_file


class PyAnnoteEmbedding:
    """
    Wrapper for pyannote audio speaker embedding model.

    Methods
    -------
    embed(audio: torch.Tensor, frequency: int) -> torch.Tensor
        Embed audio tensor and return embedding.
    embed_from_file(file_path: str) -> torch.Tensor
        Embed audio loaded from file and return embedding.
    embed_segments(audio: torch.Tensor, frequency: int, change_points: list[float]) -> torch.Tensor
        Embed each segment between speaker change points and return stacked embeddings.
    """

    def __init__(self, model: str = "pyannote/embedding", batch_size: int = 1) -> None:
        """
        Initialize PyAnnoteEmbedding with a pretrained pyannote model.

        Parameters
        ----------
        model : str, optional
            HuggingFace model identifier for pyannote embedding.
        batch_size : int, optional
            Batch size for inference.
        Raises
        ------
        ValueError
            If HF_TOKEN_PYANNOTE_EMBEDDING is missing in environment variables.
        """
        load_dotenv("./.env")
        # .env is in root and not ignored
        token = os.getenv("HF_TOKEN_PYANNOTE_EMBEDDING") or os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError("Missing HF_TOKEN_PYANNOTE_EMBEDDING or HF_TOKEN in environment variables.")
        
        self.model = Model.from_pretrained(model, use_auth_token=token)
        self.inference = Inference(self.model, window="whole", batch_size=batch_size)

    def embed(self, audio: torch.Tensor, frequency: int) -> torch.Tensor:
        """
        Embed audio tensor using pyannote embedding model.

        Parameters
        ----------
        audio : torch.Tensor
            Audio tensor to embed.
        frequency : int
            Sample rate of the audio.

        Returns
        -------
        torch.Tensor
            Embedding tensor for the input audio.
        Raises
        ------
        ValueError
            If audio tensor has unexpected shape.
        """
        audio = match_frequency(audio, frequency)

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # (1, num_samples)
        elif audio.ndim > 2:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")
        
        embedding = self.inference({"waveform": audio, "sample_rate": frequency})
        return torch.tensor(embedding, dtype=TENSOR_DTYPE)

    def embed_from_file(self, file_path: str) -> torch.Tensor:
        """
        Embed audio from file using pyannote embedding model.

        Parameters
        ----------
        file_path : str
            Path to the audio file.

        Returns
        -------
        torch.Tensor
            Embedding tensor for the audio file.
        """
        audio, frequency = load_audio_from_file(file_path)
        return self.embed(audio, frequency)


    def embed_segments(self, audio: torch.Tensor, frequency: int, change_points: list[float]) -> torch.Tensor:
        """
        embed each segment between speaker change points from an audio tensor.

        parameters
        ----------
        audio : torch.Tensor
            loaded audio Tensor.
        frequency : int
            sample rate of the audio.
        change_points : list of float
            timestamps (in seconds) where speaker changes occur.

        returns
        -------
        torch.Tensor
            tensor of shape (num_segments, embedding_dim)
        """
        points = [0.0] + change_points + [audio.shape[-1] / frequency]
        embeddings = []
        for i in range(len(points) - 1):
            start = int(points[i] * frequency)
            end = int(points[i+1] * frequency)
            segment = audio[..., start:end]
            emb = self.embed(segment, frequency)
            embeddings.append(emb.squeeze(0))
        return torch.stack(embeddings)


if __name__ == "__main__":
    file_path = "samples/single_speaker_sample.wav"

    audio, frequency = load_audio_from_file(file_path)

    print(f"loaded file: {file_path}\naudio length: {len(audio)}\nfrequency: {frequency}")

    model = PyAnnoteEmbedding()

    embed_audio_frequency = model.embed(audio, frequency)

    print(f"embeddings shape: {embed_audio_frequency.shape}")
