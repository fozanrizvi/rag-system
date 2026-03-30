"""Main entry point — ingest documents and start an interactive Q&A loop."""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import sys

import config
from embeddings import Embedder
from vectorstore import VectorStore
from ingest import load_documents, split_documents
from llm import LLM
from rag import RAGPipeline


def ingest(vector_store: VectorStore):
    docs = load_documents()
    if not docs:
        print(f"No documents found in '{config.DOCUMENTS_DIR}'. Add .txt, .md, or .pdf files and retry.")
        return
    texts, metadatas = split_documents(docs)
    vector_store.add_documents(texts, metadatas)


def chat(rag: RAGPipeline):
    print("\n--- RAG Chat (type 'quit' to exit) ---\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        answer = rag.answer(question)
        print(f"\nAssistant: {answer}\n")


def main():
    parser = argparse.ArgumentParser(description="Local RAG System (GPU-accelerated)")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents from the documents/ folder")
    parser.add_argument("--chat", action="store_true", help="Start interactive Q&A")
    args = parser.parse_args()

    if not args.ingest and not args.chat:
        parser.print_help()
        sys.exit(1)

    print(f"Using device: {config.DEVICE}")

    # Always need embedder + vector store
    embedder = Embedder()
    vector_store = VectorStore(embedder)

    if args.ingest:
        ingest(vector_store)

    if args.chat:
        if vector_store.count == 0:
            print("Vector store is empty. Run with --ingest first.")
            sys.exit(1)
        llm = LLM()
        rag = RAGPipeline(vector_store, llm)
        chat(rag)


if __name__ == "__main__":
    main()
