"""
Neural embedding utilities for AetherSpeak protocol.
"""

import torch
import numpy as np
from typing import Union, List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import warnings


class EmbeddingEngine:
    """
    Handles neural embedding generation for the AetherSpeak protocol.
    
    Provides efficient embedding generation using pre-trained models
    with fallback options for different scenarios.
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: Name of the pre-trained model
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        
        # Initialize the sentence transformer model
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.model_loaded = True
        except Exception as e:
            self.model = None
            self.model_loaded = False
            warnings.warn(f"Could not load {model_name}: {e}. Using fallback embeddings.")
        
        # Fallback embedding parameters
        self.fallback_dim = 384
        self.fallback_seed = 42
        
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embeddings for input text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model_loaded:
            return self._transformer_embed(text)
        else:
            return self._fallback_embed(text)
    
    def _transformer_embed(self, text: str) -> np.ndarray:
        """
        Generate embeddings using the transformer model.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            with torch.no_grad():
                embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            warnings.warn(f"Transformer embedding failed: {e}. Using fallback.")
            return self._fallback_embed(text)
    
    def _fallback_embed(self, text: str) -> np.ndarray:
        """
        Generate fallback embeddings using simple heuristics.
        
        Args:
            text: Input text
            
        Returns:
            Fallback embedding vector
        """
        # Set random seed for reproducibility
        np.random.seed(self.fallback_seed)
        
        # Generate embedding based on text characteristics
        words = text.lower().split()
        word_count = len(words)
        
        # Create embedding based on text length and content
        embedding = np.zeros(self.fallback_dim)
        
        # Length-based features
        embedding[0] = min(word_count / 100.0, 1.0)  # Normalized length
        
        # Content-based features (simple heuristics)
        if any(word in text.lower() for word in ['question', 'what', 'how', 'why']):
            embedding[1] = 1.0  # Question indicator
        
        if any(word in text.lower() for word in ['task', 'work', 'job']):
            embedding[2] = 1.0  # Task indicator
        
        if any(word in text.lower() for word in ['collaborate', 'help', 'assist']):
            embedding[3] = 1.0  # Collaboration indicator
        
        # Fill remaining dimensions with random values
        embedding[4:] = np.random.randn(self.fallback_dim - 4) * 0.1
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Matrix of embeddings (n_texts x embedding_dim)
        """
        if self.model_loaded:
            try:
                with torch.no_grad():
                    embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings
            except Exception as e:
                warnings.warn(f"Batch transformer embedding failed: {e}. Using fallback.")
                return self._fallback_embed_batch(texts)
        else:
            return self._fallback_embed_batch(texts)
    
    def _fallback_embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate fallback embeddings for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Matrix of fallback embeddings
        """
        embeddings = []
        for text in texts:
            embedding = self._fallback_embed(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of generated embeddings.
        
        Returns:
            Embedding dimension
        """
        if self.model_loaded:
            return self.model.get_sentence_embedding_dimension()
        else:
            return self.fallback_dim
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray
    ) -> tuple[int, float]:
        """
        Find the most similar embedding from a set of candidates.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: Matrix of candidate embeddings
            
        Returns:
            Tuple of (best_index, best_similarity)
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.model_loaded,
            "embedding_dimension": self.get_embedding_dimension()
        }
        
        if self.model_loaded:
            info["model_type"] = type(self.model).__name__
            info["max_seq_length"] = getattr(self.model, 'max_seq_length', 'unknown')
        
        return info
    
    def __str__(self) -> str:
        """String representation."""
        status = "loaded" if self.model_loaded else "fallback"
        return f"EmbeddingEngine(model='{self.model_name}', status='{status}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"EmbeddingEngine(model_name='{self.model_name}', "
                f"device='{self.device}', "
                f"model_loaded={self.model_loaded})")
