"""Configuration for the RAG system."""

import os
import torch

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Embedding Model ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# --- LLM ---
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
# 4-bit quantization to fit comfortably on a 16 GB GPU
LLM_LOAD_IN_4BIT = True

# --- Text Splitting ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ---
TOP_K = 5
