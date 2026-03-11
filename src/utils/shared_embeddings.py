"""
Shared Embedding Model for INQUIRO.

This module provides a centralized, singleton embedding model that is shared
across all components (RAG, SemanticMatcher, etc.) to:
1. Ensure consistent embeddings across the system
2. Avoid loading the same model multiple times
3. Pre-initialize early to catch failures before agents are created

Usage:
    from src.utils.shared_embeddings import get_shared_embedding_model, pre_warm_embeddings
    
    # At startup (before creating agents)
    pre_warm_embeddings()
    
    # In any component that needs embeddings
    model = get_shared_embedding_model()
    if model:
        embeddings = model.encode(["text1", "text2"])
"""

import logging
import threading
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)


class SharedEmbeddingModel:
    """
    Thread-safe singleton for the sentence-transformers embedding model.
    
    This ensures only ONE model instance exists across the entire application,
    saving memory and avoiding race conditions during parallel operations.
    """
    
    _instance = None
    _lock = threading.Lock()
    _model = None
    _model_loaded = False
    _model_failed = False
    _model_name = "all-MiniLM-L6-v2"
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self) -> bool:
        """
        Load the embedding model if not already loaded.
        
        Returns:
            True if model is available, False otherwise.
        """
        if self._model_failed:
            return False
            
        if self._model_loaded:
            return True
            
        with self._lock:
            # Double-check after acquiring lock
            if self._model_loaded:
                return True
            if self._model_failed:
                return False
                
            try:
                logger.info(f"SharedEmbeddingModel: Loading {self._model_name}...")
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                self._model_loaded = True
                logger.info("SharedEmbeddingModel: Model loaded successfully! ✓")
                return True
                
            except ImportError as e:
                logger.warning(
                    f"SharedEmbeddingModel: sentence-transformers not installed: {e}\n"
                    "   To fix: pip install sentence-transformers\n"
                    "   Falling back to keyword-based matching."
                )
                self._model_failed = True
                return False
                
            except Exception as e:
                logger.warning(
                    f"SharedEmbeddingModel: Failed to load model: {e}\n"
                    "   This usually means PyTorch/torchvision is broken.\n"
                    "   Try: pip install --upgrade sentence-transformers\n"
                    "   Falling back to keyword-based matching."
                )
                self._model_failed = True
                return False
    
    def is_available(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model_loaded and self._model is not None
    
    def get_model(self):
        """Get the underlying SentenceTransformer model (or None if unavailable)."""
        if not self._model_loaded:
            self.load_model()
        return self._model
    
    def encode(
        self,
        texts: List[str],
        show_progress_bar: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Encode texts into embedding vectors.
        
        Args:
            texts: List of strings to embed
            show_progress_bar: Whether to show progress (default False)
            
        Returns:
            numpy array of shape (len(texts), embedding_dim), or None if unavailable
        """
        model = self.get_model()
        if model is None:
            return None
        
        try:
            return model.encode(texts, show_progress_bar=show_progress_bar)
        except Exception as e:
            logger.warning(f"SharedEmbeddingModel: Encoding failed: {e}")
            return None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Returns:
            Similarity score (0.0-1.0), or -1.0 if unavailable.
        """
        embeddings = self.encode([text1, text2])
        if embeddings is None:
            return -1.0
        
        try:
            dot_product = np.dot(embeddings[0], embeddings[1])
            norm1 = np.linalg.norm(embeddings[0])
            norm2 = np.linalg.norm(embeddings[1])
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.debug(f"SharedEmbeddingModel: Similarity computation failed: {e}")
            return -1.0


# =============================================================================
# Module-level convenience functions
# =============================================================================

_shared_model: Optional[SharedEmbeddingModel] = None


def get_shared_embedding_model() -> SharedEmbeddingModel:
    """
    Get the shared embedding model singleton.
    
    Returns:
        SharedEmbeddingModel instance (may or may not have loaded successfully)
    """
    global _shared_model
    if _shared_model is None:
        _shared_model = SharedEmbeddingModel()
    return _shared_model


def pre_warm_embeddings() -> bool:
    """
    Pre-initialize the embedding model.
    
    Call this EARLY in application startup, BEFORE creating agents,
    to ensure the model is loaded before any parallel operations.
    
    Returns:
        True if model loaded successfully, False otherwise.
    """
    model = get_shared_embedding_model()
    success = model.load_model()
    
    if success:
        # Do a warm-up encoding to ensure model is fully initialized
        try:
            _ = model.encode(["warm-up text"])
            logger.info("SharedEmbeddingModel: Warm-up complete ✓")
        except Exception as e:
            logger.warning(f"SharedEmbeddingModel: Warm-up encoding failed: {e}")
            return False
    
    return success


def is_embedding_available() -> bool:
    """Check if embeddings are available without trying to load."""
    model = get_shared_embedding_model()
    return model.is_available()
