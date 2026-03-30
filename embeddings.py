"""Embedding model wrapper — runs on GPU."""

from sentence_transformers import SentenceTransformer

import config


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME, device=config.DEVICE
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
