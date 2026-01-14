import torch
from sklearn.cluster import MiniBatchKMeans
import json
import numpy as np

from pipeline.speaker_recognition.embedding.speechbrain import SpeechBrainEmbedding
from utils.audio import load_audio_from_file


class OnlineKMeansSpeakerRecognition:
    """
    Online speaker recognition using MiniBatchKMeans clustering on speaker embeddings.

    Attributes:
        embedder (SpeechBrainEmbedding): Embedding model for audio.
        reference_embeddings (dict): Mean embeddings for each reference speaker.
        n_speakers (int): Number of speakers/clusters.
        kmeans (MiniBatchKMeans): Online KMeans clustering model.
        cluster_to_speaker (dict): Mapping from cluster index to speaker name.
    """

    def __init__(
        self, 
        speaker_to_audio: dict[str, list[tuple[torch.Tensor, int]]],
        n_speakers: int,
        embedder: SpeechBrainEmbedding = SpeechBrainEmbedding(), 
    ) -> None:
        """
        Initializes the online speaker recognition system.

        Args:
            speaker_to_audio (dict): Mapping from speaker names to lists of (audio, sample_rate) tuples.
            n_speakers (int): Number of speakers/clusters.
            embedder (SpeechBrainEmbedding, optional): Embedding model instance.
        """
        self.embedder = embedder
        self.n_speakers = n_speakers

        self.reference_embeddings = self._create_reference_embeddings(speaker_to_audio)
        self.kmeans, self.cluster_to_speaker = self._init_online_kmeans_with_references() 

    def _create_reference_embeddings(
        self,
        speaker_to_audio: dict[str, list[tuple[torch.Tensor, int]]] 
    ) -> dict[str, torch.Tensor]:
        """
        Computes mean embeddings for each speaker from provided reference audio.

        Args:
            speaker_to_audio (dict): 
                A dictionary mapping speaker names to a list of (audio, sample_rate) tuples.

        Returns:
            dict: A dictionary mapping each speaker name to their mean embedding tensor.
        """
        reference_embeddings = {}
        for speaker, refs in speaker_to_audio.items():
            embs = []
            for audio, sr in refs:
                emb = self.embedder.embed(audio, sr).squeeze(0)
                embs.append(emb)
            mean_emb = torch.stack(embs).mean(dim=0)
            reference_embeddings[speaker] = mean_emb
        return reference_embeddings

    def _init_online_kmeans_with_references(self) -> tuple[MiniBatchKMeans, dict[int, str]]:
        """
        Initializes MiniBatchKMeans using reference embeddings as initial cluster centers.
        Each cluster is associated with a speaker.

        Returns:
            tuple: (MiniBatchKMeans instance, cluster-to-speaker mapping dictionary)
        """
        speakers = list(self.reference_embeddings.keys())
        centers = torch.stack([self.reference_embeddings[s].squeeze(0) for s in speakers]).cpu().numpy()
        kmeans = MiniBatchKMeans(
            n_clusters=len(speakers),
            random_state=42,
            batch_size=10,
            init='k-means++',
            n_init='auto'
        )
        # Fit once with reference embeddings to initialize internal attributes
        kmeans.fit(centers)
        # Overwrite cluster centers with reference embeddings
        kmeans.cluster_centers_ = centers
        cluster_to_speaker = {i: speakers[i] for i in range(len(speakers))}
        return kmeans, cluster_to_speaker

    def predict_speaker_online(
        self,
        x: torch.Tensor,
        sr: int | None = None
    ) -> tuple[str, int, np.ndarray]:
        """
        Predicts the speaker for a given audio segment using online K-Means clustering.

        Args:
            x (torch.Tensor): Audio segment or embedding tensor.
            sr (int, optional): Sample rate of the audio. If None, x is assumed to be an embedding.

        Returns:
            tuple:
                - speaker_name (str): The mapped speaker name.
                - cluster_id (int): The assigned cluster ID.
                - distances (np.ndarray): Distances to each cluster center.
        """
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

        Args:
            cluster_id (int): Cluster index.

        Returns:
            str: Speaker name or 'unknown_{cluster_id}' if not mapped.
        """
        return self.cluster_to_speaker.get(cluster_id, f"unknown_{cluster_id}")


if __name__ == "__main__":
    #### ONLINE KMEANS TEST WITH REFERENCE EMBEDDINGS
    print("\nRunning online K-Means test...")
    path_annot = "data/benchmark/annotations/001.json"
    with open(path_annot, "r") as f:
        annotations = json.load(f)
    segments = annotations["segments"]

    def get_speaker_for_second(segments, t):
        """
        Returns the list of speakers active at time t.

        Args:
            segments (list): List of segment dicts with 'start', 'end', and 'speaker'.
            t (int): Time in seconds.

        Returns:
            list: List of speaker names active at time t.
        """
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
        ],
        "speaker2": [
            load_audio_from_file("data/benchmark/speaker_references/marten.mp3", convert_to_mono=True),
        ],
    }

    # New audio to identify
    new_audio_path = "data/benchmark/chunks/combined_part001.mp3"
    new_audio, new_sr = load_audio_from_file(new_audio_path, convert_to_mono=True)
    duration = int(new_audio.shape[-1] / new_sr)

    STEP_SIZE = 2

    recognizer_online = OnlineKMeansSpeakerRecognition(reference_audio, n_speakers=2)
    # reference embeddings and kmeans are initialized in __init__

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
