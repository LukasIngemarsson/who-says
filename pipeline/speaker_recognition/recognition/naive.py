import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import json

from pipeline.speaker_recognition.embedding.speechbrain import SpeechBrainEmbedding
from utils.audio import load_audio_from_file


class NaiveSpeakerRecognition:
    def __init__(self, embedder = SpeechBrainEmbedding(), n_speakers: int = 3) -> None:
        self.embedder = embedder
        self.reference_embeddings = {}

        self.n_speakers = n_speakers
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_speakers,
            random_state=42,
            batch_size=10,
            n_init='auto'
        )
        self.cluster_to_speaker = {}


    def create_reference_embeddings(
        self,
        speaker_to_audio: dict[str, list[tuple[torch.Tensor, int]]]
    ) -> None:
        """
        Given a dict mapping speaker names to reference audio (in-memory),
        compute and store mean embeddings.
        Args:
            speaker_to_audio (dict): {speaker_name: (audio, sr) or list of (audio, sr)}
        """
        for speaker, refs in speaker_to_audio.items():
            embs = []
            for audio, sr in refs:
                emb = self.embedder.embed(audio, sr).squeeze(0)
                embs.append(emb)
            mean_emb = torch.stack(embs).mean(dim=0)
            self.reference_embeddings[speaker] = mean_emb


    def verify(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        sr1: int | None = None,
        sr2: int | None = None,
        thres: float = 0.5
    ) -> tuple[float, bool]:
        """
        Computes similarity score between two audio chunks or embeddings.
        If sample rates are provided, treats x1 and x2 as audio and embeds them.
        Otherwise, treats them as embeddings.
        """
        if sr1 is not None and sr2 is not None:
            emb1 = self.embedder.embed(x1, sr1).squeeze(0)
            emb2 = self.embedder.embed(x2, sr2).squeeze(0)
        else:
            emb1 = x1
            emb2 = x2
        score = F.cosine_similarity(emb1, emb2).item()
        prediction = score > thres
        return score, prediction


    def predict_speaker(
        self,
        x: torch.Tensor,
        sr: int | None = None
    ) -> tuple[str, dict[str, float]]:
        """
        Given audio or embedding, returns the speaker with the highest similarity and the similarity scores.
        If sample rate is provided, treats x as audio and embeds it.
        Otherwise, treats x as embedding.
        """
        emb = self.embedder.embed(x, sr).squeeze(0) if sr is not None else x
        similarities = {}
        for speaker, ref_emb in self.reference_embeddings.items():
            score, _ = self.verify(emb, ref_emb)
            similarities[speaker] = score
        best_speaker = max(similarities, key=lambda k: similarities[k])
        return best_speaker, similarities


if __name__ == "__main__":
    #### OFFLINE RECOG TEST
    print("1. Running offline test...")
    embedder = SpeechBrainEmbedding()

    path1 = "data/benchmark/speaker_references/lukas.mp3"
    audio1, sr1 = load_audio_from_file(path1, convert_to_mono=True)

    path2 = "data/benchmark/speaker_references/marten.mp3"
    audio2, sr2 = load_audio_from_file(path2, convert_to_mono=True)

    recognizer = NaiveSpeakerRecognition()
    score, prediction = recognizer.verify(audio1, audio2, sr1, sr2)
    print(f"Mårten vs. Lukas - score: {score:.4f}, prediction: {prediction}") 


    #### ONLINE RECOG TEST W/ SINGLE REFERENCE (NO CLUSTERING)
    print("2. Running online test...")

    path_annot = "data/benchmark/annotations/001.json"
    with open(path_annot, "r") as f:
        annotations = json.load(f)
    segments = annotations["segments"]

    def get_speaker_for_second(segments, t):
        speakers = []
        for seg in segments:
            if seg["start"] <= t < seg["end"]:
                speakers.append(seg["speaker"])
        return speakers 

    path_comb = "data/benchmark/chunks/combined_part001.mp3"
    audio_comb, sr_comb = load_audio_from_file(path_comb, convert_to_mono=True)

    reference_audio = {
        "speaker1": [
            load_audio_from_file("data/benchmark/speaker_references/lukas.mp3", convert_to_mono=True),
            # Add more (audio, sr) tuples as needed
        ],
        "speaker2": [
            load_audio_from_file("data/benchmark/speaker_references/marten.mp3", convert_to_mono=True),
            # Add more (audio, sr) tuples as needed
        ],
    }

    recognizer = NaiveSpeakerRecognition()
    recognizer.create_reference_embeddings(reference_audio)

    # New audio to identify
    new_audio_path = "data/benchmark/chunks/combined_part001.mp3"
    new_audio, new_sr = load_audio_from_file(new_audio_path, convert_to_mono=True)
    duration = int(new_audio.shape[-1] / new_sr)

    STEP_SIZE = 2

    for t in range(0, duration, STEP_SIZE):
        start_sample = t * new_sr
        end_sample = (t + STEP_SIZE) * new_sr
        chunk = new_audio[start_sample:end_sample]

        actual_speaker = get_speaker_for_second(segments, t)

        if len(actual_speaker) == 0:
            print(f"Second {t}-{t+STEP_SIZE}: SILENCE.")
            continue
        if len(actual_speaker) > 1:
            print(f"Second {t}-{t+STEP_SIZE}: OVERLAP. | GT: {actual_speaker}")
            continue

        best_speaker, similarities = recognizer.predict_speaker(chunk, new_sr)
        similarities = {k: f"{v:.3f}" for k, v in similarities.items()}
        print(f"Second {t}-{t+STEP_SIZE}: Assigned speaker: {best_speaker} | GT speaker: {actual_speaker} | Similarities: {similarities}")

