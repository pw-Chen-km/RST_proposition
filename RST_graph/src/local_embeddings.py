import os
from functools import lru_cache
from typing import List

import numpy as np


DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embedding_model_name(model_name: str = None) -> str:
    return model_name or os.getenv("RST_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embed_texts(texts: List[str], model_name: str = None) -> List[np.ndarray]:
    if not texts:
        return []

    model_name = embedding_model_name(model_name)
    model = _load_model(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [np.asarray(vector, dtype=np.float32) for vector in embeddings]


def embedding_dimension(model_name: str = None) -> int:
    model_name = embedding_model_name(model_name)
    model = _load_model(model_name)
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())
