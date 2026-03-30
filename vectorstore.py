"""ChromaDB vector store for document storage and retrieval."""

import chromadb
from chromadb.config import Settings

import config
from embeddings import Embedder


class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=config.VECTOR_STORE_DIR)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None):
        if not texts:
            return
        embeddings = self.embedder.embed_documents(texts)
        ids = [f"doc_{self.collection.count() + i}" for i in range(len(texts))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{}] * len(texts),
        )
        print(f"Added {len(texts)} chunks to the vector store.")

    def query(self, query_text: str, top_k: int = config.TOP_K) -> list[dict]:
        query_embedding = self.embedder.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            documents.append({"text": doc, "metadata": meta, "distance": dist})
        return documents

    @property
    def count(self) -> int:
        return self.collection.count()
