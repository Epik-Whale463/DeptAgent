"""
Configuration constants for RAG system
"""

# FAISS Vector Store
FAISS_INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL = "qwen3-embedding:8b"  # Ollama embedding model
EMBEDDING_DIMENSION = 1024  # qwen3-embedding:8b outputs 1024 dimensions

# Text Splitting
CHUNK_SIZE = 1000  # Increased for better context
CHUNK_OVERLAP = 150
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Ollama LLM
OLLAMA_MODEL = "gpt-oss:20b"  # or "neural-chat", "orca-mini", etc.
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.6
LLM_TOP_P = 0.9

# Retrieval
TOP_K_RESULTS = 10
SIMILARITY_THRESHOLD = 0.0  # Use all results, filter by score instead

# Data
DATA_FILE = "./teachers.json"
