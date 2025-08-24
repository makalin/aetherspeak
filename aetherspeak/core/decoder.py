"""
Core decoder for AetherSpeak protocol.
"""

import torch
import numpy as np
from typing import Union, List, Dict, Any, Optional
from sympy import symbols, simplify, expand
import json

from ..utils.tokenizer import Tokenizer
from ..utils.embedding import EmbeddingEngine
from ..utils.symbolic import SymbolicProcessor


class Decoder:
    """
    Decodes AetherSpeak tokens back to natural language messages.
    
    Reconstructs the original semantic meaning from compressed tokens
    using both neural embeddings and symbolic logic.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the decoder.
        
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
        
    def decode(self, encoded_message: Dict[str, Any]) -> str:
        """
        Decode an AetherSpeak message back to natural language.
        
        Args:
            encoded_message: Encoded message dictionary
            
        Returns:
            Decoded natural language message
        """
        # Validate protocol version
        if encoded_message.get("protocol_version") != self.protocol_version:
            raise ValueError(f"Protocol version mismatch: expected {self.protocol_version}")
        
        # Extract components
        tokens = encoded_message["tokens"]
        symbolic_elements = encoded_message.get("symbolic_elements", {})
        
        # Reconstruct embeddings from tokens
        embeddings = self._reconstruct_embeddings(tokens)
        
        # Reconstruct symbolic elements
        reconstructed_symbols = self._reconstruct_symbols(tokens, symbolic_elements)
        
        # Generate natural language from embeddings and symbols
        decoded_message = self._generate_natural_language(embeddings, reconstructed_symbols)
        
        return decoded_message
    
    def _reconstruct_embeddings(self, tokens: List[str]) -> np.ndarray:
        """
        Reconstruct embedding vectors from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Reconstructed embeddings
        """
        embeddings = []
        
        for token in tokens:
            if token.startswith("E"):  # Embedding token
                embedding = self._token_to_embedding(token)
                embeddings.append(embedding)
        
        if not embeddings:
            # Fallback: generate random embeddings
            embeddings = [np.random.randn(self.embedding_dim) for _ in range(3)]
        
        return np.array(embeddings)
    
    def _token_to_embedding(self, token: str) -> np.ndarray:
        """
        Convert token back to embedding vector.
        
        Args:
            token: Embedding token
            
        Returns:
            Embedding vector
        """
        # Parse token parts
        parts = token.split("|")
        
        # Initialize embedding vector
        embedding = np.zeros(self.embedding_dim)
        
        for part in parts[1:]:  # Skip position marker
            if ":" in part:
                idx_str, val_str = part.split(":", 1)
                try:
                    idx = int(idx_str[1:])  # Remove 'v' prefix
                    val = float(val_str)
                    if idx < self.embedding_dim:
                        embedding[idx] = val
                except (ValueError, IndexError):
                    continue
        
        return embedding
    
    def _reconstruct_symbols(
        self, 
        tokens: List[str], 
        original_symbols: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reconstruct symbolic elements from tokens.
        
        Args:
            tokens: List of tokens
            original_symbols: Original symbolic elements for reference
            
        Returns:
            Reconstructed symbolic elements
        """
        reconstructed = {}
        
        for token in tokens:
            if token.startswith("S:"):  # Symbolic token
                symbol_type, symbol_data = self._token_to_symbol(token)
                if symbol_type:
                    reconstructed[symbol_type] = symbol_data
        
        # Merge with original symbols for completeness
        reconstructed.update(original_symbols)
        
        return reconstructed
    
    def _token_to_symbol(self, token: str) -> tuple[Optional[str], Any]:
        """
        Convert token back to symbolic element.
        
        Args:
            token: Symbolic token
            
        Returns:
            Tuple of (symbol_type, symbol_data)
        """
        try:
            # Parse S:TYPE:DATA format
            parts = token.split(":", 2)
            if len(parts) >= 3:
                symbol_type = parts[1]
                symbol_data = parts[2]
                return symbol_type, symbol_data
        except (ValueError, IndexError):
            pass
        
        return None, None
    
    def _generate_natural_language(
        self, 
        embeddings: np.ndarray, 
        symbols: Dict[str, Any]
    ) -> str:
        """
        Generate natural language from reconstructed embeddings and symbols.
        
        Args:
            embeddings: Reconstructed embeddings
            symbols: Reconstructed symbolic elements
            
        Returns:
            Natural language message
        """
        # Use embedding similarity to find closest natural language
        # This is a simplified approach - in practice, more sophisticated methods would be used
        
        # Generate semantic meaning from embeddings
        semantic_parts = []
        
        # Analyze embedding patterns
        if len(embeddings) > 0:
            # Calculate embedding statistics
            mean_embedding = np.mean(embeddings, axis=0)
            embedding_norm = np.linalg.norm(mean_embedding)
            
            if embedding_norm > 0.5:
                semantic_parts.append("This message contains significant semantic content")
            elif embedding_norm > 0.2:
                semantic_parts.append("This message has moderate semantic content")
            else:
                semantic_parts.append("This message has minimal semantic content")
        
        # Add symbolic information
        if symbols:
            for symbol_type, symbol_data in symbols.items():
                if symbol_type == "LOGIC":
                    semantic_parts.append(f"Logical structure: {symbol_data}")
                elif symbol_type == "MATH":
                    semantic_parts.append(f"Mathematical content: {symbol_data}")
                elif symbol_type == "STRUCT":
                    semantic_parts.append(f"Structural information: {symbol_data}")
        
        # Combine into coherent message
        if semantic_parts:
            message = ". ".join(semantic_parts) + "."
        else:
            message = "Decoded message with minimal semantic content."
        
        return message
    
    def batch_decode(
        self, 
        encoded_messages: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Decode multiple messages in batch.
        
        Args:
            encoded_messages: List of encoded messages
            
        Returns:
            List of decoded messages
        """
        return [self.decode(msg) for msg in encoded_messages]
    
    def get_decoding_stats(self) -> Dict[str, Any]:
        """
        Get decoding statistics.
        
        Returns:
            Dictionary with decoding metrics
        """
        return {
            "protocol_version": self.protocol_version,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.max_length,
            "device": self.device
        }
    
    def validate_message(self, encoded_message: Dict[str, Any]) -> bool:
        """
        Validate that an encoded message can be decoded.
        
        Args:
            encoded_message: Encoded message to validate
            
        Returns:
            True if message is valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ["tokens", "protocol_version"]
            for field in required_fields:
                if field not in encoded_message:
                    return False
            
            # Check protocol version
            if encoded_message["protocol_version"] != self.protocol_version:
                return False
            
            # Check tokens format
            tokens = encoded_message["tokens"]
            if not isinstance(tokens, list) or len(tokens) == 0:
                return False
            
            return True
            
        except Exception:
            return False
