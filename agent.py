"""
Main RAG Agent - Orchestrates all components
"""

import os
from pathlib import Path
from typing import Optional

from components import (
    DataLoader,
    RecursiveCharacterSplitter,
    FAISSVectorStore,
    OllamaLLM,
    OllamaEmbeddings,
    Retriever,
    DATA_FILE,
    FAISS_INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
)


class RAGAgent:
    """
    Retrieval-Augmented Generation Agent
    
    Combines document retrieval with LLM generation for intelligent Q&A
    """
    
    def __init__(self, data_file: str = DATA_FILE, use_existing_index: bool = True):
        """
        Initialize RAG Agent
        
        Args:
            data_file: Path to data file (teachers.json)
            use_existing_index: Whether to use existing FAISS index if available
        """
        print("\n" + "="*60)
        print("Initializing RAG Agent")
        print("="*60 + "\n")
        
        self.data_file = data_file
        self.use_existing_index = use_existing_index
        
        # Initialize components
        self.embeddings = OllamaEmbeddings()
        self.vector_store = FAISSVectorStore(self.embeddings)
        self.llm = OllamaLLM()
        self.retriever = Retriever(self.vector_store)
        self.text_splitter = RecursiveCharacterSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
        )
        
        # Load and index data if needed
        self._initialize_data()
    
    def _initialize_data(self):
        """Load and index data"""
        index_exists = os.path.exists(FAISS_INDEX_PATH) and \
                      os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss"))
        
        if index_exists and self.use_existing_index:
            print("✓ Using existing FAISS index\n")
            stats = self.vector_store.get_stats()
            print(f"Loaded index with {stats['total_documents']} documents\n")
        else:
            print("Creating new FAISS index...\n")
            self._build_index()
    
    def _build_index(self):
        """Build FAISS index from data"""
        # Load data
        print("Step 1: Loading data...")
        data_loader = DataLoader(self.data_file)
        documents = data_loader.convert_to_documents()
        metadata = data_loader.get_metadata()
        print(f"✓ Loaded {len(documents)} teacher records\n")
        
        # Split text
        print("Step 2: Splitting documents...")
        split_documents = []
        for doc in documents:
            splits = self.text_splitter.split_text(doc)
            split_documents.extend(splits)
        print(f"✓ Split into {len(split_documents)} chunks\n")
        
        # Create vector store
        print("Step 3: Creating embeddings and building FAISS index...")
        self.vector_store.add_documents(split_documents, None)
        print()
    
    def query(self, question: str, verbose: bool = False) -> str:
        """
        Query the RAG system
        
        Args:
            question: User question
            verbose: Whether to print intermediate steps
            
        Returns:
            Generated answer
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {question}")
            print('='*60)
        
        # Retrieve relevant documents
        if verbose:
            print("\n[Retrieval Step]")
        
        retrieved_docs = self.retriever.retrieve(question)
        context = self.retriever.format_context(retrieved_docs)
        
        if verbose:
            print(f"Retrieved {len(retrieved_docs)} relevant documents")
            print(f"\nContext:\n{context}\n")
        
        # Generate answer
        if verbose:
            print("[Generation Step]")
        
        system_prompt = """You are an expert assistant answering questions about teachers and academic staff.
Use the provided context to answer questions accurately. If the information is not available in the context, 
say so clearly. Keep answers concise and informative."""
        
        prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate(prompt, system_prompt)
        
        if verbose:
            print(f"Answer: {answer}\n")
        
        return answer
    
    def query_stream(self, question: str) -> str:
        """
        Query the RAG system with streaming output
        
        Args:
            question: User question
            
        Returns:
            Full generated answer (after streaming completes)
        """
        print(f"\nQuery: {question}\n")
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question)
        context = self.retriever.format_context(retrieved_docs)
        
        # Generate answer with streaming
        system_prompt = """You are an expert assistant answering questions about teachers and academic staff.
Use the provided context to answer questions accurately. If the information is not available in the context, 
say so clearly. Keep answers concise and informative."""
        
        prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        print("Answer: ", end="", flush=True)
        full_answer = ""
        
        for chunk in self.llm.generate_stream(prompt, system_prompt):
            print(chunk, end="", flush=True)
            full_answer += chunk
        
        print("\n")
        return full_answer
    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "="*60)
        print("RAG Agent - Interactive Mode")
        print("="*60)
        print("Ask questions about the teachers database.")
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Query with verbose output
                self.query(question, verbose=True)
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAG Agent for Teacher Database"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Ask a single question"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "-s", "--stream",
        action="store_true",
        help="Use streaming output"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild FAISS index from scratch"
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = RAGAgent(use_existing_index=not args.rebuild)
    
    # Handle commands
    if args.query:
        if args.stream:
            agent.query_stream(args.query)
        else:
            agent.query(args.query, verbose=True)
    elif args.interactive:
        agent.interactive_mode()
    else:
        # Default: interactive mode
        agent.interactive_mode()


if __name__ == "__main__":
    main()
