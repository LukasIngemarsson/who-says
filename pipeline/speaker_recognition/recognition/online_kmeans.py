import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import json
import numpy as np

from pipeline.speaker_recognition.embedding.speechbrain import SpeechBrainEmbedding
from utils.audio import load_audio_from_file


class OnlineKMeansSpeakerRecognition:
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
                emb = embedder.embed(audio, sr).squeeze(0)
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


    def init_online_kmeans_with_references(self) -> None:
        """
        Initialize MiniBatchKMeans using reference embeddings as initial cluster centers.
        Each cluster is associated with a speaker.
        """
        speakers = list(self.reference_embeddings.keys())
        centers = torch.stack([self.reference_embeddings[s].squeeze(0) for s in speakers]).cpu().numpy()
        self.kmeans = MiniBatchKMeans(
            n_clusters=len(speakers),
            random_state=42,
            batch_size=10,
            init='k-means++',
            n_init='auto'
        )
        # Fit once with reference embeddings to initialize internal attributes
        self.kmeans.fit(centers)
        # Overwrite cluster centers with reference embeddings
        self.kmeans.cluster_centers_ = centers
        self.cluster_to_speaker = {i: speakers[i] for i in range(len(speakers))}

    
    def predict_speaker_online(
        self,
        x: torch.Tensor,
        sr: int | None = None
    ) -> tuple[str, int, np.ndarray]:
        """
        Predict speaker using online K-Means clustering.
        Returns:
            speaker_name: The mapped speaker name
            cluster_id: The assigned cluster ID
            distances: Distances to each cluster center
        """
        if self.kmeans is None:
            self.init_online_kmeans_with_references()

        emb = self.embedder.embed(x, sr).squeeze(0) if sr is not None else x
        emb_np = emb.cpu().numpy().reshape(1, -1)

        self.kmeans.partial_fit(emb_np)
        cluster_id = int(self.kmeans.predict(emb_np)[0])
        distances = self.kmeans.transform(emb_np)[0]
        speaker_name = self.get_speaker_from_cluster(cluster_id)
        return speaker_name, cluster_id, distances

    def get_speaker_from_cluster(self, cluster_id: int) -> str:
        """
        Returns the speaker name associated with a cluster ID.
        """
        return self.cluster_to_speaker.get(cluster_id, f"unknown_{cluster_id}")


if __name__ == "__main__":
    #### OFFLINE RECOG TEST
    print("1. Running offline test...")
    embedder = SpeechBrainEmbedding()

    path1 = "samples/meetings/meeting3-en/lukas/audio_chunks/lukas_part000.mp3"
    audio1, sr1 = load_audio_from_file(path1, convert_to_mono=True)

    path2 = "samples/meetings/meeting3-en/marten/audio_chunks/marten_chunk_010.mp3"
    audio2, sr2 = load_audio_from_file(path2, convert_to_mono=True)

    recognizer = OnlineKMeansSpeakerRecognition()
    score, prediction = recognizer.verify(audio1, audio2, sr1, sr2)
    print(f"Mårten vs. Lukas - score: {score:.4f}, prediction: {prediction}") 


    #### ONLINE KMEANS TEST WITH REFERENCE EMBEDDINGS
    path_annot = "samples/benchmarks/english/001.json"
    with open(path_annot, "r") as f:
        annotations = json.load(f)
    segments = annotations["segments"]

    def get_speaker_for_second(segments, t):
        speakers = []
        for seg in segments:
            if seg["start"] <= t < seg["end"]:
                speakers.append(seg["speaker"])
        return speakers 

    path_comb = "samples/meetings/meeting3-en/chunks/combined_part001.mp3"
    audio_comb, sr_comb = load_audio_from_file(path_comb, convert_to_mono=True)

    reference_audio = {
        "lukas": [
            load_audio_from_file("samples/meetings/meeting3-en/lukas/audio_chunks/lukas_part000.mp3", convert_to_mono=True),
            # Add more (audio, sr) tuples as needed
        ],
        "marten": [
            load_audio_from_file("samples/meetings/meeting3-en/marten/audio_chunks/marten_chunk_010.mp3", convert_to_mono=True),
            # Add more (audio, sr) tuples as needed
        ],
        "gor": [
            load_audio_from_file("samples/meetings/meeting3-en/gor/audio_chunks/meeting3_gor_002.mp3", convert_to_mono=True),
            # Add more (audio, sr) tuples as needed
        ]
    }

    # New audio to identify
    new_audio_path = "samples/meetings/meeting3-en/chunks/combined_part001.mp3"
    new_audio, new_sr = load_audio_from_file(new_audio_path, convert_to_mono=True)
    duration = int(new_audio.shape[-1] / new_sr)

    STEP_SIZE = 2

    print("\n2. Running online K-Means test (reference-based)...")
    recognizer_online = OnlineKMeansSpeakerRecognition(n_speakers=3)
    recognizer_online.create_reference_embeddings(reference_audio)
    recognizer_online.init_online_kmeans_with_references()

    print("Cluster ID to Speaker mapping:")
    for cluster_id, speaker in recognizer_online.cluster_to_speaker.items():
        print(f"  Cluster {cluster_id}: {speaker}")

    # Collect all embeddings and their cluster assignments
    all_embeddings = []
    all_cluster_ids = []

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

        speaker_name, cluster_id, distances = recognizer_online.predict_speaker_online(chunk, new_sr)
        print(f"Second {t}-{t+STEP_SIZE}: Cluster: {cluster_id} | Assigned speaker: {speaker_name} | GT: {actual_speaker} | Distances: {distances}")

        emb = recognizer_online.embedder.embed(chunk, new_sr).squeeze(0).cpu().numpy()
        emb = np.squeeze(emb)
        all_embeddings.append(emb)
        all_cluster_ids.append(cluster_id)

    all_embeddings = np.array(all_embeddings)
    centroids = recognizer_online.kmeans.cluster_centers_

    # Reduce to 2D for plotting
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    all_embeddings_2d = pca.fit_transform(all_embeddings)
    centroids_2d = pca.transform(centroids)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('tab10')
    num_clusters = len(recognizer_online.cluster_to_speaker)
    colors = [cmap(i) for i in range(num_clusters)]

    # Plot each cluster's points and its centroid with matching color
    for cluster_id, speaker in recognizer_online.cluster_to_speaker.items():
        idx = [i for i, cid in enumerate(all_cluster_ids) if cid == cluster_id]
        # Cluster points
        plt.scatter(
            all_embeddings_2d[idx, 0],
            all_embeddings_2d[idx, 1],
            color=colors[cluster_id],
            alpha=0.7,
            label=f'{speaker} (cluster {cluster_id})'
        )
        # Centroid for this cluster
        plt.scatter(
            centroids_2d[cluster_id, 0],
            centroids_2d[cluster_id, 1],
            color=colors[cluster_id],
            marker='X',
            s=160,
            edgecolor='black',
            linewidths=2,
            label=f'Centroid {cluster_id}'
        )

    plt.title("Speaker Embeddings and Cluster Centroids (PCA 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.legend()
    plt.savefig("cluster_plot.png")
    plt.close()

