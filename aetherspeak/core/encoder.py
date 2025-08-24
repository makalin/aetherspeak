"""
Core encoder for AetherSpeak protocol.
"""

import torch
import numpy as np
from typing import Union, List, Dict, Any, Optional
from sympy import symbols, simplify, expand
import json

from ..utils.tokenizer import Tokenizer
from ..utils.embedding import EmbeddingEngine
from ..utils.symbolic import SymbolicProcessor


class Encoder:
    """
    Encodes natural language messages into compact AetherSpeak tokens.
    
    Combines neural embeddings for semantic understanding with symbolic logic
    for precise representation of logical structures.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the encoder.
        
        Args:
            model_name: Pre-trained model for sentence embeddings
            embedding_dim: Dimension of the embedding vectors
            max_length: Maximum sequence length
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Initialize components
        self.tokenizer = Tokenizer(model_name, max_length)
        self.embedding_engine = EmbeddingEngine(model_name, self.device)
        self.symbolic_processor = SymbolicProcessor()
        
        # Protocol state
        self.protocol_version = "0.1.0"
        self.compression_ratio = 0.01  # Target 1% of original size
        
    def encode(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Encode a natural language message into AetherSpeak format.
        
        Args:
            message: Natural language message to encode
            context: Optional context information
            
        Returns:
            Dictionary containing encoded message and metadata
        """
        # Extract symbolic elements
        symbolic_elements = self.symbolic_processor.extract_symbols(message)
        
        # Generate neural embeddings
        embeddings = self.embedding_engine.embed(message)
        
        # Compress embeddings
        compressed_embeddings = self._compress_embeddings(embeddings)
        
        # Create compact token sequence
        tokens = self._create_tokens(compressed_embeddings, symbolic_elements)
        
        # Generate protocol header
        header = self._generate_header(message, context)
        
        return {
            "header": header,
            "tokens": tokens,
            "symbolic_elements": symbolic_elements,
            "compression_ratio": len(tokens) / len(message.split()),
            "protocol_version": self.protocol_version
        }
    
    def _compress_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compress embeddings to achieve target compression ratio.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Compressed embeddings
        """
        # Apply dimensionality reduction
        if embeddings.shape[1] > self.embedding_dim:
            # Use PCA-like compression
            u, s, vt = np.linalg.svd(embeddings, full_matrices=False)
            compressed = u[:, :self.embedding_dim] @ np.diag(s[:self.embedding_dim])
        else:
            compressed = embeddings
            
        # Quantize to reduce precision
        compressed = np.round(compressed * 1000) / 1000
        
        return compressed
    
    def _create_tokens(
        self, 
        embeddings: np.ndarray, 
        symbolic_elements: Dict[str, Any]
    ) -> List[str]:
        """
        Create compact token sequence from compressed embeddings and symbolic elements.
        
        Args:
            embeddings: Compressed embeddings
            symbolic_elements: Extracted symbolic logic
            
        Returns:
            List of compact tokens
        """
        tokens = []
        
        # Add embedding tokens
        for i, embedding in enumerate(embeddings):
            # Convert embedding to compact representation
            token = self._embedding_to_token(embedding, i)
            tokens.append(token)
        
        # Add symbolic tokens
        for symbol_type, symbol_data in symbolic_elements.items():
            token = self._symbolic_to_token(symbol_type, symbol_data)
            tokens.append(token)
        
        return tokens
    
    def _embedding_to_token(self, embedding: np.ndarray, index: int) -> str:
        """
        Convert embedding vector to compact token.
        
        Args:
            embedding: Embedding vector
            index: Position index
            
        Returns:
            Compact token string
        """
        # Convert to base64-like encoding for compactness
        # This is a simplified version - in practice, more sophisticated encoding would be used
        token_parts = []
        
        # Add position marker
        token_parts.append(f"E{index}")
        
        # Add compressed embedding values
        for i, val in enumerate(embedding[:10]):  # Use first 10 dimensions for brevity
            if abs(val) > 0.001:  # Only include significant values
                token_parts.append(f"v{i}:{val:.3f}")
        
        return "|".join(token_parts)
    
    def _symbolic_to_token(self, symbol_type: str, symbol_data: Any) -> str:
        """
        Convert symbolic element to compact token.
        
        Args:
            symbol_type: Type of symbolic element
            symbol_data: Symbolic data
            
        Returns:
            Compact token string
        """
        if symbol_type == "logic":
            return f"S:LOGIC:{symbol_data}"
        elif symbol_type == "math":
            return f"S:MATH:{symbol_data}"
        elif symbol_type == "structure":
            return f"S:STRUCT:{symbol_data}"
        else:
            return f"S:{symbol_type}:{symbol_data}"
    
    def _generate_header(
        self, 
        original_message: str, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate protocol header with metadata.
        
        Args:
            original_message: Original message
            context: Optional context
            
        Returns:
            Header dictionary
        """
        header = {
            "protocol": "AetherSpeak",
            "version": self.protocol_version,
            "timestamp": torch.cuda.Event() if torch.cuda.is_available() else None,
            "original_length": len(original_message),
            "encoding_method": "hybrid_neural_symbolic",
            "compression_target": self.compression_ratio
        }
        
        if context:
            header["context"] = context
            
        return header
    
    def batch_encode(
        self, 
        messages: List[str], 
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Encode multiple messages in batch.
        
        Args:
            messages: List of messages to encode
            contexts: Optional list of contexts
            
        Returns:
            List of encoded messages
        """
        if contexts is None:
            contexts = [None] * len(messages)
            
        return [self.encode(msg, ctx) for msg, ctx in zip(messages, contexts)]
    
    def get_compression_stats(self) -> Dict[str, float]:
        """
        Get compression statistics.
        
        Returns:
            Dictionary with compression metrics
        """
        return {
            "target_compression_ratio": self.compression_ratio,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.max_length
        }
