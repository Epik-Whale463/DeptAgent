"""
Ollama LLM and Embeddings integration
"""

import requests
import json
from typing import List, Optional
from . import config


class OllamaEmbeddings:
    """Generate embeddings using Ollama"""
    
    def __init__(self, model: str = config.EMBEDDING_MODEL,
                 base_url: str = config.OLLAMA_BASE_URL):
        """
        Initialize Ollama embeddings
        
        Args:
            model: Embedding model name
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/embeddings"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Connected to Ollama at {self.base_url}")
                available_models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in available_models]
                print(f"  Available models: {', '.join(set(model_names))}")
            else:
                print(f"✗ Ollama server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"✗ Could not connect to Ollama at {self.base_url}")
            print("  Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"✗ Error testing Ollama connection: {e}")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embed_documents([query])[0]
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents
        
        Args:
            documents: List of document texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for doc in documents:
            try:
                response = requests.post(
                    self.endpoint,
                    json={"model": self.model, "prompt": doc},
                    timeout=30
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except requests.exceptions.RequestException as e:
                print(f"Error generating embedding: {e}")
                raise
        
        return embeddings


class OllamaLLM:
    """Generate responses using Ollama LLM"""
    
    def __init__(self, model: str = config.OLLAMA_MODEL,
                 base_url: str = config.OLLAMA_BASE_URL,
                 temperature: float = config.LLM_TEMPERATURE,
                 top_p: float = config.LLM_TOP_P):
        """
        Initialize Ollama LLM
        
        Args:
            model: Model name
            base_url: Ollama server URL
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.endpoint = f"{base_url}/api/generate"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Connected to Ollama LLM at {self.base_url}")
            else:
                print(f"✗ Ollama server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"✗ Could not connect to Ollama at {self.base_url}")
            print("  Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"✗ Error testing Ollama connection: {e}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate response using Ollama
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        # Build full prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        
        except requests.exceptions.RequestException as e:
            print(f"Error generating response: {e}")
            raise
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        """
        Generate response using streaming
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            
        Yields:
            Response chunks
        """
        # Build full prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "stream": True
                },
                timeout=120,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
        
        except requests.exceptions.RequestException as e:
            print(f"Error generating response: {e}")
            raise
