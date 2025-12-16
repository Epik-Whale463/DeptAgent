"""
Configuration constants for RAG system
"""

# FAISS Vector Store
FAISS_INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL = "mxbai-embed-large:latest"  # Ollama embedding model
EMBEDDING_DIMENSION = 768

# Text Splitting
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Ollama LLM
OLLAMA_MODEL = "gpt-oss:20b"  # or "neural-chat", "orca-mini", etc.
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9

# Retrieval
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.3

# Data
DATA_FILE = "./teachers.json"
