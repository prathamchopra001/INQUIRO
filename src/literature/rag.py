"""
RAG (Retrieval Augmented Generation) system for Inquiro.

Stores paper chunks as embeddings in ChromaDB and retrieves
relevant chunks for a given question.

Supports both local embeddings (sentence-transformers via shared model)
and API-based embeddings (OpenAI text-embedding-3-small).
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional, List
from pathlib import Path

from src.literature.models import TextChunk
from src.utils.shared_embeddings import get_shared_embedding_model
from config.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# EMBEDDING PROVIDERS
# =============================================================================

class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Convert a list of texts into embedding vectors."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Convert a single query string into an embedding vector."""
        pass


class LocalEmbedder(BaseEmbedder):
    """
    Local embeddings using the shared sentence-transformers model.
    
    Free, runs on your machine, no API key needed.
    Uses the centralized SharedEmbeddingModel to ensure only one
    model instance exists across the application.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Use the shared embedding model singleton
        self._shared_model = get_shared_embedding_model()
        logger.info(f"LocalEmbedder: Using shared embedding model (available={self._shared_model.is_available()})")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._shared_model.encode(texts)
        if embeddings is None:
            raise RuntimeError(
                "Embedding model not available. "
                "Run: pip install sentence-transformers"
            )
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


class OpenAIEmbedder(BaseEmbedder):
    """
    API-based embeddings using OpenAI's text-embedding-3-small.
    
    Requires OPENAI_API_KEY. Tiny cost (~$0.02 per 1M tokens).
    """
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.model = model
        self.api_key = api_key or settings.llm.openai_api_key
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
    
    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


# =============================================================================
# EMBEDDER FACTORY
# =============================================================================

def get_embedder(provider: str = "local", **kwargs) -> BaseEmbedder:
    """
    Create an embedder based on provider name.
    
    Args:
        provider: "local" for sentence-transformers, "openai" for API
    """
    if provider == "local":
        return LocalEmbedder(**kwargs)
    elif provider == "openai":
        return OpenAIEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# =============================================================================
# RAG SYSTEM
# =============================================================================

class RAGSystem:
    """
    Retrieval Augmented Generation system using ChromaDB.
    
    Think of this as a smart filing cabinet:
    - add_chunks(): File away paper chunks with their embeddings
    - query(): Ask a question, get back the most relevant chunks
    - get_collection_stats(): How many chunks are filed away?
    
    Usage:
        rag = RAGSystem()
        rag.add_chunks(chunks)  # Store TextChunks from PDFParser
        results = rag.query("What pathways are affected?", top_k=5)
        for chunk_text, metadata, distance in results:
            print(f"[{metadata['paper_title']}] {chunk_text[:100]}...")
    """
    
    def __init__(
        self,
        collection_name: str = "inquiro_papers",
        persist_dir: str = "./data/chromadb",
        embedder: BaseEmbedder = None,
        embedding_provider: str = "local",
    ):
        """
        Args:
            collection_name: Name for the ChromaDB collection.
            persist_dir: Where ChromaDB stores its data on disk.
            embedder: Pre-configured embedder (overrides embedding_provider).
            embedding_provider: "local" or "openai" (used if embedder is None).
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        
        # Set up embedder
        self.embedder = embedder or get_embedder(embedding_provider)
        
        # Set up ChromaDB
        self._client = None
        self._collection = None
        self._collection_lock = threading.Lock()
    
    def _get_collection(self):
        """Lazy-initialize ChromaDB client and collection (thread-safe)."""
        if self._collection is None:
            with self._collection_lock:
                # Double-check after acquiring lock
                if self._collection is None:
                    try:
                        import chromadb
                    except ImportError:
                        raise ImportError("chromadb not installed. Run: pip install chromadb")
                    
                    Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                    
                    self._client = chromadb.PersistentClient(path=self.persist_dir)
                    self._collection = self._client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(
                        f"ChromaDB collection '{self.collection_name}': "
                        f"{self._collection.count()} existing documents"
                    )
        return self._collection
    
    # =========================================================================
    # === YOUR CODE HERE === (three methods to implement)
    # =========================================================================
    
    def add_chunks(self, chunks: List[TextChunk], batch_size: int = 50) -> int:
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of TextChunk objects from the PDFParser.
            batch_size: How many chunks to embed at once (memory management).
        
        Returns:
            Number of chunks successfully added.
        
        Why batch_size?
            Embedding 500 chunks at once might blow up your RAM.
            Processing in batches of 50 keeps memory usage reasonable.
        
        Algorithm:
            1. Get the ChromaDB collection (self._get_collection())
            2. Process chunks in batches of batch_size
            3. For each batch:
               a. Build lists of: ids, texts, metadatas
                  - id: unique string, e.g., f"{chunk.paper_id}_{chunk.chunk_index}"
                  - text: chunk.text
                  - metadata: chunk.to_metadata()
               b. Generate embeddings: self.embedder.embed_texts(texts)
               c. Add to collection: collection.add(
                      ids=ids,
                      documents=texts,
                      embeddings=embeddings,
                      metadatas=metadatas
                  )
            4. Return total chunks added
        
        Important edge case:
            ChromaDB will error if you try to add a duplicate ID.
            Use collection.get(ids=[...]) to check what already exists,
            OR use collection.upsert() instead of collection.add()
            to handle duplicates gracefully.
        """
        if not chunks:
            return 0
        collection = self._get_collection()
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            ids = [f"{c.paper_id}_{c.chunk_index}" for c in batch]
            texts = [c.text for c in batch]
            metadatas = [c.to_metadata() for c in batch]
            
            # Embedding is thread-safe after model init (sub-task 1B)
            embeddings = self.embedder.embed_texts(texts)
            
            # ChromaDB writes must be serialized
            try:
                with self._collection_lock:
                    collection.upsert(
                        ids=ids,
                        documents=texts,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                total_added += len(batch)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} chunks")
            except Exception as e:
                logger.error(f"ChromaDB upsert failed for batch {i//batch_size + 1}: {e}")
                continue

        return total_added
    
    def query(
        self,
        question: str,
        top_k: int = None,
        filter_paper_id: Optional[str] = None,
    ) -> List[tuple]:
        """
        Query the vector store for chunks relevant to a question.
        
        Args:
            question: The question to search for.
            top_k: Number of results to return. Defaults to settings.rag.top_k
            filter_paper_id: Optional - only search within a specific paper.
        
        Returns:
            List of (text, metadata, distance) tuples, sorted by relevance.
            
            Example:
            [
                ("Nucleotide salvage pathways were found to...",
                 {"paper_id": "abc123", "paper_title": "...", "doi": "..."},
                 0.23),  # cosine distance — lower = more similar
                ...
            ]
        
        Algorithm:
            1. Get the collection
            2. Embed the question: self.embedder.embed_query(question)
            3. Query ChromaDB:
               results = collection.query(
                   query_embeddings=[query_embedding],
                   n_results=top_k,
                   where={"paper_id": filter_paper_id} if filter_paper_id else None,
                   include=["documents", "metadatas", "distances"]
               )
            4. ChromaDB returns a nested structure:
               results["documents"][0]  → list of texts
               results["metadatas"][0]  → list of metadata dicts
               results["distances"][0]  → list of distances
            5. Zip these together into (text, metadata, distance) tuples
            6. Return the list
        """
        collection = self._get_collection()
        top_k = top_k or settings.rag.top_k

        # Guard: ChromaDB raises 'bad parameter' if n_results > collection size
        with self._collection_lock:
            collection_size = collection.count()
        if collection_size == 0:
            return []
        top_k = min(top_k, collection_size)

        # Embedding is thread-safe after model init (sub-task 1B)
        query_embedding = self.embedder.embed_query(question)

        # ChromaDB reads must be serialized
        with self._collection_lock:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"paper_id": filter_paper_id} if filter_paper_id else None,
                include=["documents", "metadatas", "distances"]
            )
        
        # 4 & 5. Parse and zip results
        # ChromaDB results are lists of lists because it supports multiple queries at once.
        # Since we only sent one query_embedding, we take index [0].
        formatted_results = []
        if results["documents"]:
            formatted_results = list(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ))
            
        return formatted_results
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about what's stored in the RAG system.
        
        Returns:
            Dict with:
            - "total_chunks": int — total number of stored chunks
            - "paper_ids": List[str] — unique paper IDs in the store
        
        Hints:
            1. Get the collection
            2. Total count: collection.count()
            3. To get unique paper_ids, you can do:
               all_metadata = collection.get(include=["metadatas"])
               paper_ids = set(m["paper_id"] for m in all_metadata["metadatas"])
               BUT this is expensive for large collections.
               For now, just return count and skip paper_ids if count > 1000.
        """
        collection = self._get_collection()
        with self._collection_lock:
            count = collection.count()
        
            paper_ids = []
            if count <= 1000:
                all_metadata = collection.get(include=["metadatas"])
                paper_ids = list(set(m["paper_id"] for m in all_metadata["metadatas"]))
            
        return {
            "total_chunks": count,
            "paper_ids": paper_ids
        }
    
    # =========================================================================
    # UTILITIES (boilerplate — done for you)
    # =========================================================================
    
    def clear(self):
        """Delete all data in the collection. Use with caution!"""
        collection = self._get_collection()
        # ChromaDB doesn't have a clear() method — delete and recreate
        self._client.delete_collection(self.collection_name)
        self._collection = None
        logger.info(f"Cleared collection: {self.collection_name}")
    
    def delete_paper(self, paper_id: str):
        """Remove all chunks for a specific paper."""
        collection = self._get_collection()
        collection.delete(where={"paper_id": paper_id})
        logger.info(f"Deleted chunks for paper: {paper_id}")