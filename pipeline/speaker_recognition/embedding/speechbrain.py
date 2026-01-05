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
                device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=model,
            run_opts={"device": self.device}
        )   
        
        if self.model is not None:
            # Use .to() which recursively moves all submodules
            self.model = self.model.to(self.device)
            # Also set eval mode to ensure all layers behave correctly
            self.model.eval()

        # ECAPA-TDNN embedding dimension
        self.embedding_dim = 192   


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
            audio = audio.unsqueeze(0) 

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

    def embed_segments(self, audio: torch.Tensor, frequency: int, change_points: list[float],
                       return_indices: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[int]]:
        """
        Embed each segment between speaker change points.

        Parameters
        ----------
        audio : torch.Tensor
            Audio tensor (1D or 2D).
        frequency : int
            Sample rate of the audio.
        change_points : list[float]
            List of speaker change timestamps in seconds.
        return_indices : bool
            If True, also return list of segment indices that were embedded.

        Returns
        -------
        torch.Tensor or tuple
            If return_indices=False: Embeddings tensor (shape: [n_embedded, embedding_dim])
            If return_indices=True: (embeddings, segment_indices) where segment_indices
                lists which original segments (0-indexed) were embedded.
        """
        # Ensure audio is 1D
        if audio.dim() > 1:
            audio = audio.squeeze()

        # Calculate audio duration
        audio_duration = audio.shape[0] / frequency
        points = [0.0] + change_points + [audio_duration]
        embeddings = []
        embedded_indices = []  # Track which segments were actually embedded

        # Minimum segment duration in seconds (adjust as needed)
        min_duration = 0.5  # 500ms minimum
        min_samples = int(min_duration * frequency)

        for i in range(len(points) - 1):
            start = int(points[i] * frequency)
            end = int(points[i+1] * frequency)

            # Skip segments that are too short
            if end - start < min_samples:
                continue

            # Extract segment
            segment = audio[start:end]

            # Embed the segment
            emb = self.embed(segment, frequency)

            # Squeeze all batch dimensions to get (embedding_dim,)
            while emb.dim() > 1:
                emb = emb.squeeze(0)

            embeddings.append(emb)
            embedded_indices.append(i)

        result_embeddings = torch.stack(embeddings) if embeddings else torch.empty(0, self.embedding_dim)

        if return_indices:
            return result_embeddings, embedded_indices
        return result_embeddings


if __name__ == "__main__":
    embedder = SpeechBrainEmbedding()
    file_path = "samples/single_speaker_sample.wav"
    audio, freq = load_audio_from_file(file_path)
    print(f"\naudio dim: {audio.shape}, freq: {freq}")
    embd = embedder.embed(audio, freq)
    print(f"embedding dim: {embd.shape}")
