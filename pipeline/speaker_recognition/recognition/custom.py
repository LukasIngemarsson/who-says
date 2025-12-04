from utils import load_audio_from_file

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
    embedder = SpeechBrainEmbedding()

    path1 = "samples/meetings/meeting3-en/lukas/audio_chunks/"
    audio1, sr1 = load_audio_from_file(path1)
    emb1 = embedder.embed(audio1, sr1)

    path2 = "samples/multi_speaker_sample.mp3"
    audio2, sr2 = load_audio_from_file(path2)
    emb2 = embedder.embed(audio2, sr2)

    recognizer = CustomSpeakerRecognition()
    score_1, prediction_1 = recognizer.verify(emb, emb_sep_1)
    print(f"score: {score_1}, prediction: {prediction_1}") 
