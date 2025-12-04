import torch
import torch.nn.functional as F
import json
from scipy.optimize import linear_sum_assignment

from pipeline.speaker_recognition.clustering.sklearn import SklearnClustering
from pipeline.speaker_recognition.embedding.speechbrain import SpeechBrainEmbedding
from utils.audio import load_audio_from_file


class CustomSpeakerRecognition:
    def __init__(self, threshold: float = 0.6) -> None:
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

        score = F.cosine_similarity(emb1, emb2)
        assert len(score) == 1

        score = score.item()
        prediction = score > self.threshold

        return score, prediction


if __name__ == "__main__":
    print("1. Running cosine distance test...")
    embedder = SpeechBrainEmbedding()

    path1 = "samples/meetings/meeting3-en/lukas/audio_chunks/lukas_part000.mp3"
    audio1, sr1 = load_audio_from_file(path1)
    emb1 = embedder.embed(audio1, sr1).squeeze(0)

    path2 = "samples/meetings/meeting3-en/marten/audio_chunks/marten_chunk_010.mp3"
    audio2, sr2 = load_audio_from_file(path2)
    emb2 = embedder.embed(audio2, sr2).squeeze(0)

    recognizer = CustomSpeakerRecognition()
    score, prediction = recognizer.verify(emb1, emb2)
    print(f"Mårten vs. Lukas - score: {score:.4f}, prediction: {prediction}") 


    print("2. Running progressive clustering test...")
    clustering = SklearnClustering(algorithm="minibatchkmeans", n_clusters=2)

    path_annot = "samples/benchmarks/english/001.json"
    with open(path_annot, "r") as f:
        annotations = json.load(f)
    segments = annotations["segments"]

    def get_speaker_for_second(segments, t):
        for seg in segments:
            if seg["start"] <= t < seg["end"]:
                return seg["speaker"]
        return "unknown"

    path_comb = "samples/meetings/meeting3-en/chunks/combined_part001.mp3"
    audio_comb, sr_comb = load_audio_from_file(path_comb)

    duration = int(audio_comb.shape[-1] / sr_comb)
    embeddings = []
    prev_centroids = None

    for t in range(0, duration):
        start_sample = t * sr_comb
        end_sample = (t + 1) * sr_comb
        chunk = audio_comb[:, start_sample:end_sample]

        if chunk.shape[0] == 0:
            continue

        emb = embedder.embed(chunk, sr_comb).squeeze()
        embeddings.append(emb.cpu())

        if len(embeddings) < 2:
            continue

        # Recluster all embeddings seen so far
        clustering = SklearnClustering(algorithm="kmeans", n_clusters=2)
        X = torch.stack(embeddings)
        clustering.model.fit(X)
        centroids = torch.tensor(clustering.model.cluster_centers_)
        labels = torch.from_numpy(clustering.model.labels_)

        # Align cluster IDs to previous centroids
        if prev_centroids is not None:
            # Compute cost matrix using torch
            cost = torch.cdist(prev_centroids, centroids)
            row_ind, col_ind = linear_sum_assignment(cost.numpy())
            label_map = {new: old for old, new in zip(row_ind, col_ind)}
            aligned_labels = torch.tensor([label_map[int(label)] for label in labels])
        else:
            aligned_labels = labels

        prev_centroids = centroids

        ref_label = get_speaker_for_second(segments, t)
        print(f"Second {t}-{t+1}: Cluster {aligned_labels[-1].item()} | Reference: {ref_label}")
