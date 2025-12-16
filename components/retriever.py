"""
Retriever component for RAG - handles context retrieval
"""

from typing import List, Tuple, Optional
from .vector_store import FAISSVectorStore
from . import config


class Retriever:
    """Retrieves relevant documents from vector store"""
    
    def __init__(self, vector_store: FAISSVectorStore,
                 top_k: int = config.TOP_K_RESULTS,
                 similarity_threshold: float = config.SIMILARITY_THRESHOLD):
        """
        Initialize retriever
        
        Args:
            vector_store: FAISSVectorStore instance
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1 range for distances)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def retrieve(self, query: str) -> List[str]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            
        Returns:
            List of relevant documents
        """
        results = self.vector_store.search(query, k=self.top_k)
        
        # Filter by similarity threshold and return documents
        retrieved_docs = []
        for doc, distance, metadata in results:
            # Convert L2 distance to similarity score (0-1)
            # Lower distance = higher similarity
            similarity = 1 / (1 + distance)
            
            if similarity >= self.similarity_threshold:
                retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[str, float, dict]]:
        """
        Retrieve relevant documents with similarity scores
        
        Args:
            query: Query text
            
        Returns:
            List of (document, score, metadata) tuples
        """
        results = self.vector_store.search(query, k=self.top_k)
        
        scored_results = []
        for doc, distance, metadata in results:
            # Convert L2 distance to similarity score (0-1)
            similarity = 1 / (1 + distance)
            
            if similarity >= self.similarity_threshold:
                scored_results.append((doc, similarity, metadata))
        
        return scored_results
    
    def format_context(self, documents: List[str]) -> str:
        """
        Format retrieved documents into context for LLM
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found."
        
        context_parts = ["Here is the relevant information:"]
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"\n[Document {i}]")
            context_parts.append(doc)
        
        return "\n".join(context_parts)
    
    def retrieve_and_format(self, query: str) -> str:
        """
        Retrieve documents and format them as context
        
        Args:
            query: Query text
            
        Returns:
            Formatted context string
        """
        documents = self.retrieve(query)
        return self.format_context(documents)
