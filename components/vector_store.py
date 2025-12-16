"""
FAISS Vector Store for storing and retrieving embeddings
"""

import os
import numpy as np
from typing import List, Tuple, Optional
import faiss

from .ollama_llm import OllamaEmbeddings
from . import config


class FAISSVectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, embedding_model: Optional[OllamaEmbeddings] = None,
                 index_path: str = config.FAISS_INDEX_PATH):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_model: OllamaEmbeddings instance
            index_path: Path to save/load FAISS index
        """
        self.embedding_model = embedding_model or OllamaEmbeddings()
        self.index_path = index_path
        self.index = None
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        # Load existing index if available
        self._load_index()
    
    def add_documents(self, documents: List[str], metadata: Optional[List[dict]] = None):
        """
        Add documents to vector store
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        print(f"Generating embeddings for {len(documents)} documents...")
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(documents)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        print(f"Created embeddings with shape: {embeddings_array.shape}")
        
        # Initialize or add to index
        if self.index is None:
            # Create new index
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance metric
            print(f"Created new FAISS index with dimension {dimension}")
        else:
            # Verify dimension matches
            current_dim = self.index.d
            new_dim = embeddings_array.shape[1]
            if current_dim != new_dim:
                print(f"⚠ Warning: Dimension mismatch! Index has {current_dim}, new embeddings have {new_dim}")
        
        # Add embeddings to index
        self.index.add(embeddings_array)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])
        
        print(f"Vector store now contains {len(self.documents)} documents")
        
        # Save index
        self._save_index()
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (document, distance, metadata) tuples
        """
        if self.index is None or len(self.documents) == 0:
            raise ValueError("Vector store is empty. Please add documents first.")
        
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_array, min(k, len(self.documents)))
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # Valid result
                results.append((
                    self.documents[idx],
                    float(dist),
                    self.metadata[idx]
                ))
        
        return results
    
    def save_index(self, index_path: Optional[str] = None):
        """Save index to disk"""
        self._save_index(index_path)
    
    def load_index(self, index_path: Optional[str] = None):
        """Load index from disk"""
        self._load_index(index_path)
    
    def _save_index(self, index_path: Optional[str] = None):
        """Internal method to save index"""
        path = index_path or self.index_path
        os.makedirs(path, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
            print(f"Saved FAISS index to {path}")
        
        # Save documents and metadata as pickle for retrieval
        import pickle
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
    
    def _load_index(self, index_path: Optional[str] = None):
        """Internal method to load index"""
        path = index_path or self.index_path
        
        if not os.path.exists(path):
            return
        
        try:
            # Load FAISS index
            index_file = os.path.join(path, "index.faiss")
            if os.path.exists(index_file):
                self.index = faiss.read_index(index_file)
                print(f"Loaded FAISS index from {path}")
            
            # Load documents and metadata
            import pickle
            docs_file = os.path.join(path, "documents.pkl")
            meta_file = os.path.join(path, "metadata.pkl")
            
            if os.path.exists(docs_file):
                with open(docs_file, "rb") as f:
                    self.documents = pickle.load(f)
            
            if os.path.exists(meta_file):
                with open(meta_file, "rb") as f:
                    self.metadata = pickle.load(f)
            
            if self.documents:
                print(f"Loaded {len(self.documents)} documents from index")
        
        except Exception as e:
            print(f"Could not load index: {e}")
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        return {
            "total_documents": len(self.documents),
            "has_index": self.index is not None,
            "index_size": self.index.ntotal if self.index else 0
        }
