"""RAG pipeline — retrieves context then generates an answer."""

import config
from vectorstore import VectorStore
from llm import LLM

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the question. "
    "If the context does not contain enough information, say so honestly."
)


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    context_str = "\n\n---\n\n".join(chunk["text"] for chunk in context_chunks)
    return (
        f"[INST] {SYSTEM_PROMPT}\n\n"
        f"### Context:\n{context_str}\n\n"
        f"### Question:\n{question} [/INST]"
    )


class RAGPipeline:
    def __init__(self, vector_store: VectorStore, llm: LLM):
        self.vector_store = vector_store
        self.llm = llm

    def answer(self, question: str, top_k: int = config.TOP_K) -> str:
        # 1. Retrieve relevant chunks
        chunks = self.vector_store.query(question, top_k=top_k)
        if not chunks:
            return "No relevant documents found. Please ingest documents first."

        # 2. Build prompt with context
        prompt = build_prompt(question, chunks)

        # 3. Generate answer
        response = self.llm.generate(prompt)

        return response
