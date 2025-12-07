"""Comment clustering using LaBSE embeddings and HDBSCAN (robustified)."""
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.decomposition import PCA
import umap

# ---------- Extract topics from comments ----------
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords if not already
nltk.download('stopwords')

# Combine English and Russian stopwords
EN_STOPWORDS = set(stopwords.words('english'))
RU_STOPWORDS = set(stopwords.words('russian'))
MULTI_STOPWORDS = EN_STOPWORDS.union(RU_STOPWORDS)

# LaBSE model for multilingual embeddings
LABSE_MODEL = "sentence-transformers/LaBSE"

# ---------- Init models (lazy) ----------
_labse_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def init_labse_model(model_name: str = LABSE_MODEL):
    """Initialize LaBSE model for embeddings (lazy)."""
    global _labse_model
    if _labse_model is None:
        _labse_model = SentenceTransformer(model_name, device=_device)
        # ensure eval mode
        _labse_model.eval()


def get_embeddings(
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """
    Get LaBSE embeddings for a list of texts.

    Returns:
        embeddings: np.ndarray shape (n_texts, dim), dtype=float32
    """
    init_labse_model()
    if not texts:
        return np.empty((0, 768), dtype=np.float32)  # default LaBSE dim fallback

    # SentenceTransformer.encode is already batched and uses torch.no_grad internally,
    # but calling within a no_grad context is safe.
    with torch.no_grad():
        embeddings = _labse_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we will control normalization explicitly
        )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if normalize and embeddings.size:
        # L2 normalize rows to unit length (helps with cosine-based clustering)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    return embeddings


def cluster_comments(
    texts: List[str],
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Cluster comments using LaBSE embeddings and HDBSCAN.

    Args:
        texts: list of comment texts
        min_cluster_size: HDBSCAN min cluster size
        min_samples: HDBSCAN min_samples (if None defaults to min_cluster_size)
        metric: distance metric; if embeddings are normalized, 'euclidean' approximates cosine
        cluster_selection_method: 'eom' or 'leaf'
        random_state: random seed (affects UMAP only if used downstream)

    Returns:
        dict with:
            - labels: list[int] cluster labels (-1 = noise)
            - embeddings: np.ndarray (n_samples, dim)
            - n_clusters: int
            - cluster_sizes: dict[int, int]
            - clusterer: the fitted HDBSCAN object (for inspection)  # optional
    """

    if min_samples is None:
        min_samples = min_cluster_size

    n_texts = len(texts)
    if n_texts == 0:
        return {
            "labels": [],
            "embeddings": np.empty((0, 0), dtype=np.float32),
            "n_clusters": 0,
            "cluster_sizes": {},
            "clusterer": None,
        }

    # If too few texts to form even one cluster, label all as noise
    if n_texts < min_cluster_size:
        return {
            "labels": [-1] * n_texts,
            "embeddings": get_embeddings(texts),
            "n_clusters": 0,
            "cluster_sizes": {-1: n_texts},
            "clusterer": None,
        }

    # Get embeddings (normalized)
    embeddings = get_embeddings(texts, normalize=True)

    # HDBSCAN: if embeddings are normalized, Euclidean distance is monotonic with cosine,
    # and tends to be numerically more stable / faster.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,  # optional: enables soft clustering / probabilities
    )

    labels = clusterer.fit_predict(embeddings)

    # Compute cluster sizes (including -1 for noise)
    cluster_sizes: Dict[int, int] = {}
    for lb in labels:
        cluster_sizes[lb] = cluster_sizes.get(lb, 0) + 1

    n_clusters = len([l for l in set(labels) if l >= 0])

    return {
        "labels": labels.tolist(),
        "embeddings": embeddings,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "clusterer": clusterer,
    }


def reduce_dimensions_for_plot(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: Optional[int] = None,
    min_dist: float = 0.1,
    random_state: int = 42,
    use_pca: bool = True,
    pca_dim: int = 64,
) -> np.ndarray:
    """
    Reduce embedding dimensions for visualization using optional PCA + UMAP.

    Args:
        embeddings: np.ndarray shape (n_samples, dim)
        n_components: 2 or 3
        n_neighbors: UMAP n_neighbors; if None defaults to min(15, n_samples-1) but at least 2
        use_pca: whether to run PCA before UMAP to speed things up for high-dim data
        pca_dim: intermediate PCA dimension to reduce to before UMAP

    Returns:
        reduced: np.ndarray shape (n_samples, n_components)
    """
    if embeddings is None or embeddings.size == 0:
        return np.empty((0, n_components), dtype=np.float32)

    n_samples = embeddings.shape[0]
    if n_neighbors is None:
        n_neighbors = max(2, min(15, n_samples - 1))

    # Optional PCA pre-reduction for speed (useful when embeddings dim is large and dataset is big)
    if use_pca and embeddings.shape[1] > pca_dim and n_samples > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        emb_reduced = pca.fit_transform(embeddings)
    else:
        emb_reduced = embeddings
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=int(n_neighbors),
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    reduced = reducer.fit_transform(emb_reduced)
    return reduced
