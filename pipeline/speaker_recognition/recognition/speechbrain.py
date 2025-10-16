from utils import load_audio_from_file

from speechbrain.inference.speaker import SpeakerRecognition
from pipeline.speaker_recognition.embedding.speechbrain import SpeechBrainEmbedding
import torch


class SpeechBrainSpeakerRecognition:
    """
    Speaker recognition using SpeechBrain's ECAPA-TDNN model.
    """

    def __init__(self, model: str = "speechbrain/spkrec-ecapa-voxceleb", threshold: float = 0.6) -> None:
        """
        Initialize the speaker recognition model.

        Args:
            model (str): Model source string for SpeechBrain ECAPA-TDNN.
        """
        self.model = SpeakerRecognition.from_hparams(source=model)
        self.threshold = threshold

    def verify(self, emb1: torch.Tensor, emb2: torch.Tensor) -> tuple[float, bool]:
        """
        Compare two speaker embeddings and predict if they are from the same speaker.

        Args:
            emb1 (torch.Tensor): First speaker embedding.
            emb2 (torch.Tensor): Second speaker embedding.

        Returns:
            tuple[float, bool]: Similarity score and prediction (True if same speaker).
        """
        if not self.model:
            raise ValueError("Model is None.")

        score = self.model.similarity(emb1, emb2)
        prediction = score > self.threshold
        return score, prediction

    # TODO: Decide how recognition should work, specifcally in the real-time case


if __name__ == "__main__":
    embedder = SpeechBrainEmbedding()

    path_multi = "samples/multi_speaker_sample.mp3"
    audio_multi, sr_multi = load_audio_from_file(path_multi)
    emb_multi = embedder.embed(audio_multi, sr_multi)

    # should be true
    path_sep_1 = "samples/sep_1.wav"
    audio_sep_1, sr_sep_1 = load_audio_from_file(path_sep_1)
    emb_sep_1 = embedder.embed(audio_sep_1, sr_sep_1)

    # should be true
    path_sep_2 = "samples/sep_2.wav"
    audio_sep_2, sr_sep_2 = load_audio_from_file(path_sep_2)
    emb_sep_2 = embedder.embed(audio_sep_2, sr_sep_2)

    # should be false
    path_single = "samples/single_speaker_sample.wav"
    audio_single, sr_single = load_audio_from_file(path_single)
    emb_single = embedder.embed(audio_single, sr_single)

    recognizer = SpeechBrainSpeakerRecognition()
    score_1, prediction_1 = recognizer.verify(emb_multi, emb_sep_1)
    score_2, prediction_2 = recognizer.verify(emb_multi, emb_sep_2)
    score_false, prediction_false = recognizer.verify(emb_multi, emb_single)
    print(f"score 1: {score_1}, prediction 1: {prediction_1}") 
    print(f"score 2: {score_2}, prediction 2: {prediction_2}") 
    print(f"score false {score_false}, prediction false {prediction_false}") 
