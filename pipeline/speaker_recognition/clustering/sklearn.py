from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN 
import torch

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
