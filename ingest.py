"""Load documents from the documents/ folder and split them into chunks."""

import os
import glob

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_documents(directory: str = config.DOCUMENTS_DIR) -> list[dict]:
    """Load .txt, .md, and .pdf files from the given directory."""
    docs = []
    supported = ("*.txt", "*.md")

    for pattern in supported:
        for filepath in glob.glob(os.path.join(directory, "**", pattern), recursive=True):
            text = _read_file(filepath)
            if text.strip():
                docs.append({"text": text, "metadata": {"source": filepath}})

    # PDF support (requires pypdf)
    for filepath in glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True):
        try:
            from pypdf import PdfReader

            reader = PdfReader(filepath)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                docs.append({"text": text, "metadata": {"source": filepath}})
        except ImportError:
            print(f"Skipping {filepath} — install 'pypdf' for PDF support.")

    print(f"Loaded {len(docs)} document(s) from {directory}")
    return docs


def split_documents(docs: list[dict]) -> tuple[list[str], list[dict]]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    texts, metadatas = [], []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append(doc["metadata"])

    print(f"Split into {len(texts)} chunk(s).")
    return texts, metadatas
