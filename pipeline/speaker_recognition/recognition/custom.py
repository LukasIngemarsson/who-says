import torch
import torch.nn.functional as F

from pipeline.speaker_recognition.embedding.speechbrain import SpeechBrainEmbedding


class CustomSpeakerRecognition:
    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def verify(self, emb1: torch.Tensor, emb2: torch.Tensor): #  -> tuple[float, bool]:
        """
        Compare two speaker embeddings and predict if they are from the same speaker.

        Args:
            emb1 (torch.Tensor): First speaker embedding.
            emb2 (torch.Tensor): Second speaker embedding.

        Returns:
            tuple[float, bool]: Similarity score and prediction (True if same speaker).
        """

        score = F.cosine_similarity(emb1, emb2)
        prediction = score > self.threshold

        return score, prediction


if __name__ == "__main__":
    print("Started...")
    embedder = SpeechBrainEmbedding()

    print("Embed 1...")
    path1 = "data/benchmark/speaker_references/lukas.mp3"
    emb1 = embedder.embed_from_file(path1).squeeze(0)
    print(emb1.shape)

    print("Embed 2...")
    path2 = "data/benchmark/speaker_references/marten.mp3"
    emb2 = embedder.embed_from_file(path2).squeeze(0)
    print(emb2.shape)

    recognizer = CustomSpeakerRecognition()
    score, prediction = recognizer.verify(emb1, emb2)
    print(f"score: {score}, prediction: {prediction}") 

