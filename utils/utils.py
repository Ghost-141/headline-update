import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import os
import json
from typing import Dict, Any, Tuple


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LOCAL_DIR = "./models/all-mpnet-base-v2"
DATA_DIR = "database"


def get_model():
    """
    Ensures the model is available locally, then loads it.
    """
    if not os.path.exists(LOCAL_DIR) or not os.listdir(LOCAL_DIR):
        print("Downloading full model from HF...")
        model = SentenceTransformer(MODEL_NAME)
        model.save(LOCAL_DIR)
    else:
        print("Loading model from local directory...")
        model = SentenceTransformer(LOCAL_DIR)
        print("✔ Model loaded successfully!")
    return model


def get_config():
    """Initialize and return configuration parameters for the headline processing system.

    Returns:
        tuple: (model, DIM, faiss_index, DATA_DIR, INDEX_PATH, METADATA_PATH, SIM_THRESHOLD)
    """

    # model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    model = get_model()

    DIM = 768

    faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))

    INDEX_PATH = os.path.join(DATA_DIR, "vector_store.index")
    METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

    SIM_THRESHOLD = 0.95

    return (
        model,
        DIM,
        faiss_index,
        DATA_DIR,
        INDEX_PATH,
        METADATA_PATH,
        SIM_THRESHOLD,
    )


def load_faiss_and_metadata(
    data_dir: str, index_path: str, metadata_path: str, dim: int
) -> Tuple[faiss.IndexIDMap, Dict[int, Dict[str, Any]], int]:
    """
    Load or create FAISS index and metadata, and determine next available ID.

    Returns:
        faiss_index: FAISS IndexIDMap
        metadata: Dict[int, Dict] with keys: 'headline', 'timestamp' (datetime), 'source'
        next_id: int
    """

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load or create FAISS index
    if os.path.exists(index_path):
        try:
            faiss_index = faiss.read_index(index_path)
            # Wrap in IndexIDMap if not already
            if not isinstance(faiss_index, faiss.IndexIDMap):
                faiss_index = faiss.IndexIDMap(faiss_index)
        except Exception as e:
            print("Failed to read existing index, creating a new one. Error:", e)
            faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    else:
        faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))

    # Load metadata
    metadata: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                metadata[int(k)] = {
                    "headline": v["headline"],
                    "source": v["source"],
                    "timestamp": (
                        datetime.fromisoformat(v["timestamp"])
                        if v.get("timestamp")
                        else None
                    ),
                }
        except Exception as e:
            print("Failed to load metadata.json, starting empty. Error:", e)

    # Determine next_id
    if metadata:
        next_id = max(metadata.keys()) + 1
    else:
        if faiss_index.ntotal > 0:
            # fallback: pick next_id as total vectors in index
            next_id = int(faiss_index.ntotal)
        else:
            next_id = int(0)

    return faiss_index, metadata, next_id


def l2_to_cosine_dist(l2: float) -> float:
    """Convert L2 distance between two normalized vectors to cosine similarity.

    For two normalized vectors u and v:
    ||u - v||^2 = 2 - 2 * (u . v)
    so cosine = u.v = 1 - l2/2
    """
    return 1.0 - (l2 / 2.0)


def embed_text(model, text: str) -> np.ndarray:
    """Generate normalized text embedding using sentence transformer model.

    Args:
        model: SentenceTransformer model instance
        text: Input text to embed

    Returns:
        np.ndarray: Normalized embedding vector (float32)
    """
    emb = model.encode([text], normalize_embeddings=True)
    return np.array(emb, dtype="float32")


def insert_new(
    emb: np.ndarray,
    headline: str,
    timestamp: datetime,
    source: str,
    next_id,
    faiss_index,
    metadata,
):
    """Insert new headline vector and metadata into FAISS index.

    Args:
        emb: Text embedding vector
        headline: News headline text
        timestamp: Publication timestamp
        source: News source name
        next_id: Next available ID for insertion
        faiss_index: FAISS index instance
        metadata: Metadata dictionary
    """
    id_arr = np.array([next_id], dtype="int64")
    faiss_index.add_with_ids(emb, id_arr)
    metadata[next_id] = {
        "headline": headline,
        "timestamp": timestamp,
        "source": source,
    }
    next_id += 1


def rebuild_index_with_replacement(
    replace_id: int, new_emb: np.ndarray, metadata, model, DIM
):
    """Rebuild FAISS index when vector replacement fails.

    Args:
        replace_id: ID of vector to replace
        new_emb: New embedding vector
        metadata: Current metadata dictionary
        model: SentenceTransformer model
        DIM: Vector dimension

    Returns:
        faiss.IndexIDMap: New FAISS index with replaced vector
    """
    ids = []
    vecs = []
    for id_, meta in metadata.items():
        ids.append(id_)
        if id_ == replace_id:
            vecs.append(new_emb[0])
        else:
            vec = embed_text(model, meta["headline"])
            vecs.append(vec[0])

    new_index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))
    if vecs:
        vecs_np = np.array(vecs, dtype="float32")
        ids_np = np.array(ids, dtype="int64")
        new_index.add_with_ids(vecs_np, ids_np)

    return new_index


def add_or_update(
    headline: str,
    timestamp: datetime,
    source: str,
    embed_func,
    SIM_THRESHOLD: float,
    metadata: dict,
    faiss_index,
    next_id: int,
    model,
    DIM: int,
) -> Tuple[int, faiss.IndexIDMap]:
    """Add new headline or update existing one based on similarity threshold.

    Args:
        headline: News headline text
        timestamp: Publication timestamp
        source: News source name
        embed_func: Function to generate embeddings
        SIM_THRESHOLD: Similarity threshold for duplicate detection
        metadata: Metadata dictionary
        faiss_index: FAISS index instance
        next_id: Next available ID
        model: SentenceTransformer model
        DIM: Vector dimension

    Returns:
        tuple: (updated_next_id, updated_faiss_index)
    """

    emb = embed_func(headline)

    # If index empty → insert first headline
    if faiss_index.ntotal == 0:
        faiss_index.add_with_ids(emb, np.array([next_id], dtype="int64"))
        metadata[next_id] = {
            "headline": headline,
            "timestamp": timestamp,
            "source": source,
        }
        return next_id + 1, faiss_index

    # Search nearest neighbor
    D, I = faiss_index.search(emb, 1)
    nearest_id = int(I[0][0])
    similarity = l2_to_cosine_dist(D[0][0])

    existing = metadata.get(nearest_id)

    # Similar enough → duplicate detected
    if similarity > SIM_THRESHOLD and existing:
        if existing["source"] == source:
            if timestamp > existing["timestamp"]:
                # Update existing with newer timestamp
                metadata[nearest_id] = {
                    "headline": headline,
                    "timestamp": timestamp,
                    "source": source,
                }
                try:
                    faiss_index.remove_ids(np.array([nearest_id], dtype="int64"))
                    faiss_index.add_with_ids(emb, np.array([nearest_id], dtype="int64"))
                except Exception:
                    faiss_index = rebuild_index_with_replacement(
                        nearest_id, emb, metadata, model, DIM
                    )
            # If older or same timestamp, ignore completely
            return next_id, faiss_index
        else:
            # Same headline but different source - ignore duplicate
            return next_id, faiss_index

    # Not similar → insert new
    faiss_index.add_with_ids(emb, np.array([next_id], dtype="int64"))
    metadata[next_id] = {"headline": headline, "timestamp": timestamp, "source": source}
    return next_id + 1, faiss_index


def rebuild_with_unique_headlines(model, faiss_index, metadata, DIM, SIM_THRESHOLD):
    """Rebuild database with only unique headlines sorted by timestamp.

    Args:
        model: SentenceTransformer model
        faiss_index: Current FAISS index
        metadata: Current metadata dictionary
        DIM: Vector dimension
        SIM_THRESHOLD: Similarity threshold for duplicate detection

    Returns:
        tuple: (new_index, new_metadata, next_id)
    """
    unique_headlines = get_sorted_unique_headlines(model, metadata, SIM_THRESHOLD)

    new_index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))
    new_metadata = {}
    next_id = 0

    for item in unique_headlines:
        emb = embed_text(model, item["headline"])
        new_index.add_with_ids(emb, np.array([next_id], dtype="int64"))
        new_metadata[next_id] = {
            "headline": item["headline"],
            "timestamp": item["timestamp"],
            "source": item["source"],
        }
        next_id += 1

    return new_index, new_metadata, next_id
