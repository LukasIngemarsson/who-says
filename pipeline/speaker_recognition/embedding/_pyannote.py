from utils.constants import TENSOR_DTYPE, SR
from utils import match_frequency

import os
from dotenv import load_dotenv
import torch

# Fix for PyTorch 2.6+ weights_only default change
import torch.serialization
import typing

try:
    import pytorch_lightning.callbacks.early_stopping
    import pytorch_lightning.callbacks.model_checkpoint
    from omegaconf import ListConfig, DictConfig
    from omegaconf.base import ContainerMetadata

    safe_globals = [
        pytorch_lightning.callbacks.early_stopping.EarlyStopping,
        pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint,
        ListConfig,
        DictConfig,
        ContainerMetadata,
        typing.Any,
    ]

    # Add additional omegaconf types if available
    try:
        from omegaconf._utils import ValueKind
        # Only add ValueKind (an enum class), not MISSING/SI/II which are string constants
        safe_globals.append(ValueKind)
    except ImportError:
        pass

    # Add collections types
    try:
        import collections
        safe_globals.append(collections.OrderedDict)
    except ImportError:
        pass

    torch.serialization.add_safe_globals(safe_globals)
except (ImportError, AttributeError):
    pass

# Alternative: Monkey-patch torch.load to use weights_only=False for pyannote models
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Always force weights_only=False for pyannote model loading
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from pyannote.audio import Model, Inference

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

    def __init__(self, model: str = "pyannote/embedding", batch_size: int = 1, device: str = None, **kwargs) -> None:
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
            If HF_TOKEN is missing in environment variables.
        """
        load_dotenv("./.env")
        # .env is in root and not ignored
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError("Missing HF_TOKEN in environment variables.")
        
        self.model = Model.from_pretrained(model, use_auth_token=token)
        # pyannote 3.x: duration in seconds, step as ratio of duration
        self.inference = Inference(self.model, window="sliding", duration=1.5, step=0.25/1.5, batch_size=batch_size)

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
        audio = match_frequency(audio, frequency)  # Resamples to SR (16000)

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # (1, num_samples)
        elif audio.ndim > 2:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")

        embedding = self.inference({"waveform": audio, "sample_rate": SR})  # Use target SR after resampling
        # SlidingWindowFeature has shape (n_frames, embedding_dim), aggregate by mean
        embedding_data = embedding.data.mean(axis=0)  # shape: (512,)
        return torch.tensor(embedding_data, dtype=TENSOR_DTYPE, device=audio.device)

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


    def embed_segments(self, audio: torch.Tensor, frequency: int, change_points: list[float],
                       return_indices: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[int]]:
        """
        Embed each segment between speaker change points from an audio tensor.

        Parameters
        ----------
        audio : torch.Tensor
            Loaded audio Tensor.
        frequency : int
            Sample rate of the audio.
        change_points : list of float
            Timestamps (in seconds) where speaker changes occur.
        return_indices : bool
            If True, also return list of segment indices that were embedded.

        Returns
        -------
        torch.Tensor or tuple
            If return_indices=False: Embeddings tensor (shape: [n_embedded, embedding_dim])
            If return_indices=True: (embeddings, segment_indices) where segment_indices
                lists which original segments (0-indexed) were embedded.
        """
        points = [0.0] + change_points + [audio.shape[-1] / frequency]
        embeddings = []
        embedded_indices = []

        # Minimum segment duration in seconds
        min_duration = 0.5  # 500ms minimum
        min_samples = int(min_duration * frequency)

        for i in range(len(points) - 1):
            start = int(points[i] * frequency)
            end = int(points[i+1] * frequency)

            # Skip segments that are too short
            if end - start < min_samples:
                continue

            segment = audio[..., start:end]
            emb = self.embed(segment, frequency)
            embeddings.append(emb.squeeze(0))
            embedded_indices.append(i)

        result_embeddings = torch.stack(embeddings) if embeddings else torch.empty(0, 512)  # pyannote embedding dim

        if return_indices:
            return result_embeddings, embedded_indices
        return result_embeddings


if __name__ == "__main__":
    file_path = "data/single_speaker_sample.wav"

    audio, frequency = load_audio_from_file(file_path)

    print(f"loaded file: {file_path}\naudio length: {len(audio)}\nfrequency: {frequency}")

    model = PyAnnoteEmbedding()

    embed_audio_frequency = model.embed(audio, frequency)

    print(f"embeddings shape: {embed_audio_frequency.shape}")
