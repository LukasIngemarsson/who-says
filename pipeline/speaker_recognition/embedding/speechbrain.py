from utils import load_audio_from_file, match_frequency

from speechbrain.inference.speaker import EncoderClassifier
import torch


class SpeechBrainEmbedding:
    """
    Wrapper for SpeechBrain speaker embedding model.

    Methods
    -------
    embed(audio: torch.Tensor, frequency: int) -> torch.Tensor
        Embed audio tensor and return embedding.
    embed_from_file(file_path: str) -> torch.Tensor
        Embed audio loaded from file and return embedding.
    embed_segments(audio: torch.Tensor, frequency: int, change_points: list[float]) -> torch.Tensor
        Embed each segment between speaker change points and return stacked embeddings.
    """

    def __init__(self, 
                 model: str = "speechbrain/spkrec-ecapa-voxceleb",
                 device: str = "cuda") -> None:
        """
        Initialize SpeechBrainEmbedding with a pretrained model.

        Parameters
        ----------
        model : str, optional
            HuggingFace model identifier for SpeechBrain speaker embedding.
        """

        self.model = EncoderClassifier.from_hparams(source=model)
        self.device = device
        if model is not None:
            self.model.to(self.device)        


    def embed(self, audio: torch.Tensor, frequency: int) -> torch.Tensor:
        """
        Embed audio tensor using SpeechBrain model.

        Parameters
        ----------
        audio : torch.Tensor
            Audio tensor to embed (1D: n_samples or 2D: batch x n_samples).
        frequency : int
            Sample rate of the audio.

        Returns
        -------
        torch.Tensor
            Embedding tensor for the input audio.

        Raises
        ------
        ValueError
            If model is None.
        """
        if not self.model:
            raise ValueError("Model is None.")

        audio = match_frequency(audio, frequency)

        # Ensure audio has batch dimension for encode_batch
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension: (n_samples,) -> (1, n_samples)

        embedding = self.model.encode_batch(audio)
        return embedding

    def embed_from_file(self, file_path: str) -> torch.Tensor:
        """
        Embed audio from file using SpeechBrain model.

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
        Embed each segment between speaker change points from an audio tensor.

        Parameters
        ----------
        audio : torch.Tensor
            Loaded audio tensor (1D: n_samples).
        frequency : int
            Sample rate of the audio.
        change_points : list of float
            Timestamps (in seconds) where speaker changes occur.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_segments, embedding_dim)
        """
        # Ensure audio is 1D
        if audio.dim() > 1:
            audio = audio.squeeze()

        # Calculate audio duration
        audio_duration = audio.shape[0] / frequency
        points = [0.0] + change_points + [audio_duration]

        embeddings = []
        for i in range(len(points) - 1):
            start = int(points[i] * frequency)
            end = int(points[i+1] * frequency)

            # Extract segment
            segment = audio[start:end]

            # Embed the segment
            emb = self.embed(segment, frequency)

            # Squeeze all batch dimensions to get (embedding_dim,)
            while emb.dim() > 1:
                emb = emb.squeeze(0)

            embeddings.append(emb)

        return torch.stack(embeddings)


if __name__ == "__main__":
    embedder = SpeechBrainEmbedding()
    file_path = "samples/single_speaker_sample.wav"
    audio, freq = load_audio_from_file(file_path)
    print(f"\naudio dim: {audio.shape}, freq: {freq}")
    embd = embedder.embed(audio, freq)
    print(f"embedding dim: {embd.shape}")
