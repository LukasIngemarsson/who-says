from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
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
    """

    def __init__(self, algorithm: str = "cosine_similarity", threshold: float = 0.7, **kwargs):
        self.threshold = threshold
        self.reference_embeddings = None
        self.reference_labels = None

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
        self.reference_embeddings = embeddings
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
        as parameter), matches to those. Otherwise, uses the first n_clusters
        embeddings as initial centroids and iteratively assigns.

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
        refs = reference_embeddings if reference_embeddings is not None else self.reference_embeddings

        if refs is not None:
            # Match to reference embeddings
            return self._match_to_references(embeddings, refs)
        elif n_clusters is not None:
            # No references - use greedy clustering with cosine similarity
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
