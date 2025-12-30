"""Wav2Vec2-based speaker embedding using mean pooling over hidden states."""
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from utils import load_audio_from_file, match_frequency


class Wav2Vec2Embedding:
    """
    Speaker embedding using Wav2Vec2 hidden states with mean pooling.
    """

    def __init__(
        self,
        model: str = "facebook/wav2vec2-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        self.device = device
        self.model_name = model
        self.processor = Wav2Vec2Processor.from_pretrained(model)
        self.model = Wav2Vec2Model.from_pretrained(model).to(device)
        self.model.eval()

    def embed(self, audio: torch.Tensor, frequency: int) -> torch.Tensor:
        """
        Embed audio tensor using Wav2Vec2 with mean pooling.

        Parameters
        ----------
        audio : torch.Tensor
            Audio tensor (1D or 2D with batch).
        frequency : int
            Sample rate of the audio.

        Returns
        -------
        torch.Tensor
            Embedding tensor (mean pooled hidden states).
        """
        audio = match_frequency(audio, frequency)

        if audio.dim() == 2:
            audio = audio.squeeze(0)

        # Wav2Vec2 expects 16kHz
        audio_np = audio.cpu().numpy()

        inputs = self.processor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Mean pooling over time dimension
        embedding = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        return embedding

    def embed_from_file(self, file_path: str) -> torch.Tensor:
        audio, frequency = load_audio_from_file(file_path)
        return self.embed(audio, frequency)

    def embed_segments(
        self,
        audio: torch.Tensor,
        frequency: int,
        change_points: list[float],
        return_indices: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[int]]:
        """
        Embed each segment between speaker change points.

        Parameters
        ----------
        return_indices : bool
            If True, also return list of segment indices that were embedded.
        """
        if audio.dim() > 1:
            audio = audio.squeeze()

        audio_duration = audio.shape[0] / frequency
        points = [0.0] + change_points + [audio_duration]
        embeddings = []
        embedded_indices = []

        min_duration = 0.5  # 500ms minimum
        min_samples = int(min_duration * frequency)

        for i in range(len(points) - 1):
            start = int(points[i] * frequency)
            end = int(points[i + 1] * frequency)

            if end - start < min_samples:
                continue

            segment = audio[start:end]
            emb = self.embed(segment, frequency)
            emb = emb.squeeze(0)
            embeddings.append(emb)
            embedded_indices.append(i)

        result_embeddings = torch.stack(embeddings) if embeddings else torch.empty(0, 768)

        if return_indices:
            return result_embeddings, embedded_indices
        return result_embeddings


if __name__ == "__main__":
    embedder = Wav2Vec2Embedding()
    file_path = "samples/single_speaker_sample.wav"
    audio, freq = load_audio_from_file(file_path)
    print(f"Audio shape: {audio.shape}, freq: {freq}")
    emb = embedder.embed(audio, freq)
    print(f"Embedding shape: {emb.shape}")
