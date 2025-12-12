from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
import torch
import torch.nn.functional as F


class SklearnClustering:
    """
    Wrapper for sklearn clustering algorithms.

    Parameters
    ----------
    algorithm : str
        Name of the clustering algorithm ('agglomerative', 'kmeans', 'dbscan').
    kwargs : dict
        Additional keyword arguments for the clustering algorithm.
    """

    def __init__(self, algorithm: str = "agglomerative", **kwargs):
        if algorithm == "agglomerative":
            self.model = AgglomerativeClustering(**kwargs)
        elif algorithm == "kmeans":
            self.model = KMeans(**kwargs)
        elif algorithm == "dbscan":
            self.model = DBSCAN(**kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def cluster_segments(self, embeddings: torch.Tensor, n_clusters: int = None) -> torch.Tensor:
        """
        Assign clusters to segment embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            Embeddings for segments (shape: [num_segments, embedding_dim]).
        n_clusters : int, optional
            Number of clusters to create. If provided, overrides the model's n_clusters.

        Returns
        -------
        torch.Tensor
            Cluster labels for each segment.
        """
        # If n_clusters is provided and the model supports it, update it
        if n_clusters is not None and hasattr(self.model, 'n_clusters'):
            self.model.n_clusters = n_clusters

        labels = self.model.fit_predict(embeddings.cpu().numpy())
        return torch.tensor(labels)


class CosineSimilarityClustering:
    """
    Speaker assignment using cosine similarity to reference embeddings.

    Use this when you have known speaker reference embeddings and want to
    match segments to those speakers without clustering.

    Parameters
    ----------
    threshold : float
        Minimum cosine similarity to assign a segment to a speaker.
        Below this threshold, segment is assigned to a new/unknown cluster.
    normalize : bool
        Whether to L2-normalize embeddings before computing cosine similarity.
        Default True - critical for proper cosine similarity.
    use_spectral_init : bool
        Whether to use spectral clustering for initial cluster assignment.
        Default True - provides better global optimization than greedy.
    refinement_passes : int
        Number of refinement passes after initial clustering.
        Default 2 - helps fix incorrect initial assignments.
    """

    def __init__(self, algorithm: str = "cosine_similarity", threshold: float = 0.7,
                 normalize: bool = True, use_spectral_init: bool = True,
                 refinement_passes: int = 2, **kwargs):
        self.threshold = threshold
        self.normalize = normalize
        self.use_spectral_init = use_spectral_init
        self.refinement_passes = refinement_passes
        self.reference_embeddings = None
        self.reference_labels = None

    def _normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """L2-normalize embeddings along the feature dimension."""
        if self.normalize:
            return F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def set_reference_speakers(self, embeddings: torch.Tensor, labels: torch.Tensor = None):
        """
        Set reference speaker embeddings for matching.

        Parameters
        ----------
        embeddings : torch.Tensor
            Reference embeddings (shape: [num_speakers, embedding_dim])
        labels : torch.Tensor, optional
            Labels for reference speakers. If None, uses indices 0, 1, 2, ...
        """
        self.reference_embeddings = self._normalize(embeddings)
        if labels is None:
            self.reference_labels = torch.arange(embeddings.shape[0])
        else:
            self.reference_labels = labels

    def cluster_segments(
        self,
        embeddings: torch.Tensor,
        n_clusters: int = None,
        reference_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Assign segments to speakers using cosine similarity.

        If reference embeddings are provided (either via set_reference_speakers or
        as parameter), matches to those. Otherwise, uses spectral clustering for
        initial assignment followed by refinement passes.

        Parameters
        ----------
        embeddings : torch.Tensor
            Embeddings for segments (shape: [num_segments, embedding_dim]).
        n_clusters : int, optional
            Number of speakers/clusters expected.
        reference_embeddings : torch.Tensor, optional
            Reference embeddings to match against.

        Returns
        -------
        torch.Tensor
            Cluster labels for each segment.
        """
        # L2 normalize embeddings - critical for cosine similarity
        embeddings = self._normalize(embeddings)

        refs = reference_embeddings if reference_embeddings is not None else self.reference_embeddings
        if refs is not None:
            refs = self._normalize(refs)

        if refs is not None:
            # Match to reference embeddings
            return self._match_to_references(embeddings, refs)
        elif n_clusters is not None:
            # Use improved clustering with spectral init and refinement
            if self.use_spectral_init and embeddings.shape[0] >= n_clusters:
                return self._spectral_cluster_with_refinement(embeddings, n_clusters)
            else:
                # Fall back to greedy for very small inputs
                return self._greedy_cluster(embeddings, n_clusters)
        else:
            raise ValueError("Either reference_embeddings or n_clusters must be provided")

    def _match_to_references(self, embeddings: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        """Match each embedding to the closest reference using cosine similarity."""
        labels = []
        next_unknown_label = references.shape[0]

        for emb in embeddings:
            # Compute cosine similarity with all references
            similarities = F.cosine_similarity(
                emb.unsqueeze(0),
                references,
                dim=1
            )
            best_score, best_idx = similarities.max(dim=0)

            if best_score.item() >= self.threshold:
                if self.reference_labels is not None:
                    labels.append(self.reference_labels[best_idx].item())
                else:
                    labels.append(best_idx.item())
            else:
                # Below threshold - assign to unknown
                labels.append(next_unknown_label)
                next_unknown_label += 1

        return torch.tensor(labels)

    def _greedy_cluster(self, embeddings: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """
        Greedy clustering using cosine similarity.
        Assigns each embedding to the most similar existing centroid,
        or creates a new cluster if below threshold (up to n_clusters).
        """
        if embeddings.shape[0] == 0:
            return torch.tensor([])

        labels = [-1] * embeddings.shape[0]
        centroids = []

        for i, emb in enumerate(embeddings):
            if len(centroids) == 0:
                # First embedding becomes first centroid
                centroids.append(emb)
                labels[i] = 0
            else:
                # Find most similar centroid
                centroid_stack = torch.stack(centroids)
                similarities = F.cosine_similarity(
                    emb.unsqueeze(0),
                    centroid_stack,
                    dim=1
                )
                best_score, best_idx = similarities.max(dim=0)

                if best_score.item() >= self.threshold:
                    # Assign to existing cluster
                    labels[i] = best_idx.item()
                    # Update centroid (running average)
                    cluster_members = [j for j, l in enumerate(labels) if l == best_idx.item()]
                    centroids[best_idx] = torch.stack([embeddings[j] for j in cluster_members]).mean(dim=0)
                elif len(centroids) < n_clusters:
                    # Create new cluster
                    labels[i] = len(centroids)
                    centroids.append(emb)
                else:
                    # Max clusters reached, assign to closest anyway
                    labels[i] = best_idx.item()

        return torch.tensor(labels)

    def _spectral_cluster_with_refinement(self, embeddings: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """
        Use agglomerative hierarchical clustering with cosine distance.

        Agglomerative clustering with average linkage and cosine distance is the
        standard approach used in speaker diarization (including pyannote).
        """
        n_samples = embeddings.shape[0]

        if n_samples <= n_clusters:
            # Each sample gets its own cluster
            return torch.arange(n_samples)

        # Convert to numpy for sklearn
        emb_np = embeddings.cpu().numpy()

        # Use agglomerative clustering with cosine distance and average linkage
        # This is what pyannote and other diarization systems use
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = agg.fit_predict(emb_np)
        labels = torch.tensor(labels)

        # Refinement passes: reassign samples to nearest centroid
        for _ in range(self.refinement_passes):
            labels = self._refine_clusters(embeddings, labels, n_clusters)

        return labels

    def _refine_clusters(self, embeddings: torch.Tensor, labels: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """
        Refine cluster assignments by reassigning each sample to the nearest centroid.

        This helps fix incorrect initial assignments, especially for borderline cases.
        """
        # Compute centroids for each cluster
        centroids = []
        for c_id in range(n_clusters):
            mask = labels == c_id
            if mask.sum() > 0:
                centroid = embeddings[mask].mean(dim=0)
                # Re-normalize centroid
                centroid = F.normalize(centroid, p=2, dim=0)
                centroids.append(centroid)
            else:
                # Empty cluster - use zero vector (will get lowest similarity)
                centroids.append(torch.zeros(embeddings.shape[1], device=embeddings.device))

        centroids = torch.stack(centroids)

        # Compute similarity to all centroids for each embedding
        # For normalized vectors: similarity = dot product
        similarities = torch.mm(embeddings, centroids.T)

        # Assign each embedding to the most similar centroid
        new_labels = similarities.argmax(dim=1)

        return new_labels

    def _compute_cluster_quality(self, embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Compute quality metrics for the clustering.

        Returns:
            dict with:
                - silhouette: Overall silhouette score (-1 to 1, higher is better)
                - avg_intra_sim: Average intra-cluster similarity
                - avg_inter_sim: Average inter-cluster similarity
        """
        n_samples = embeddings.shape[0]
        unique_labels = labels.unique()
        n_clusters = len(unique_labels)

        if n_clusters <= 1 or n_samples <= n_clusters:
            return {'silhouette': 0.0, 'avg_intra_sim': 1.0, 'avg_inter_sim': 0.0}

        # Compute centroids
        centroids = []
        for c_id in unique_labels:
            mask = labels == c_id
            centroid = embeddings[mask].mean(dim=0)
            centroid = F.normalize(centroid, p=2, dim=0)
            centroids.append(centroid)
        centroids = torch.stack(centroids)

        # Compute intra-cluster similarities (each sample to its centroid)
        intra_sims = []
        for i, emb in enumerate(embeddings):
            c_idx = (unique_labels == labels[i]).nonzero().item()
            sim = F.cosine_similarity(emb.unsqueeze(0), centroids[c_idx].unsqueeze(0)).item()
            intra_sims.append(sim)

        # Compute inter-cluster similarities (between centroids)
        inter_sims = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                sim = F.cosine_similarity(centroids[i].unsqueeze(0), centroids[j].unsqueeze(0)).item()
                inter_sims.append(sim)

        # Compute silhouette score
        try:
            emb_np = embeddings.cpu().numpy()
            labels_np = labels.cpu().numpy()
            sil_score = silhouette_score(emb_np, labels_np, metric='cosine')
        except Exception:
            sil_score = 0.0

        return {
            'silhouette': sil_score,
            'avg_intra_sim': np.mean(intra_sims) if intra_sims else 0.0,
            'avg_inter_sim': np.mean(inter_sims) if inter_sims else 0.0
        }
