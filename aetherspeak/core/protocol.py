"""
Main AetherSpeak protocol orchestrator.
"""

import torch
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
import json
import time
from datetime import datetime

from .encoder import Encoder
from .decoder import Decoder
from ..utils.metrics import ProtocolMetrics
from ..utils.adaptation import ProtocolAdaptation


class AetherProtocol:
    """
    Main orchestrator for the AetherSpeak protocol.
    
    Manages the complete encoding/decoding pipeline and provides
    high-level interface for AI-to-AI communication.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_length: int = 512,
        device: Optional[str] = None,
        enable_adaptation: bool = True,
        enable_metrics: bool = True
    ):
        """
        Initialize the AetherSpeak protocol.
        
        Args:
            model_name: Pre-trained model for sentence embeddings
            embedding_dim: Dimension of the embedding vectors
            max_length: Maximum sequence length
            device: Device to run computations on ('cpu' or 'cuda')
            enable_adaptation: Whether to enable protocol adaptation
            enable_metrics: Whether to enable performance metrics
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Initialize core components
        self.encoder = Encoder(model_name, embedding_dim, max_length, device)
        self.decoder = Decoder(model_name, embedding_dim, max_length, device)
        
        # Initialize optional components
        self.adaptation = ProtocolAdaptation() if enable_adaptation else None
        self.metrics = ProtocolMetrics() if enable_metrics else None
        
        # Protocol state
        self.protocol_version = "0.1.0"
        self.session_id = self._generate_session_id()
        self.message_count = 0
        self.adaptation_history = []
        
        # Performance tracking
        self.encoding_times = []
        self.decoding_times = []
        self.compression_ratios = []
        
    def communicate(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None,
        target_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message through the AetherSpeak protocol.
        
        Args:
            message: Natural language message to send
            context: Optional context information
            target_agent: Optional target agent identifier
            
        Returns:
            Dictionary containing communication results
        """
        start_time = time.time()
        
        # Encode message
        encoded = self.encoder.encode(message, context)
        
        # Record encoding metrics
        encoding_time = time.time() - start_time
        self.encoding_times.append(encoding_time)
        
        # Update message count
        self.message_count += 1
        
        # Record compression ratio
        self.compression_ratios.append(encoded["compression_ratio"])
        
        # Add protocol metadata
        encoded["session_id"] = self.session_id
        encoded["message_id"] = self.message_count
        encoded["timestamp"] = datetime.now().isoformat()
        encoded["target_agent"] = target_agent
        
        # Update metrics if enabled
        if self.metrics:
            self.metrics.record_encoding(
                message_length=len(message),
                encoded_length=len(encoded["tokens"]),
                compression_ratio=encoded["compression_ratio"],
                encoding_time=encoding_time
            )
        
        # Adapt protocol if enabled
        if self.adaptation:
            adaptation_result = self.adaptation.adapt(
                message, encoded, self.compression_ratios[-10:]  # Last 10 ratios
            )
            if adaptation_result:
                self.adaptation_history.append(adaptation_result)
        
        return encoded
    
    def receive(
        self, 
        encoded_message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Receive and decode a message through the AetherSpeak protocol.
        
        Args:
            encoded_message: Encoded message to decode
            
        Returns:
            Dictionary containing decoded message and metadata
        """
        start_time = time.time()
        
        # Validate message
        if not self.decoder.validate_message(encoded_message):
            raise ValueError("Invalid encoded message format")
        
        # Decode message
        decoded_message = self.decoder.decode(encoded_message)
        
        # Record decoding metrics
        decoding_time = time.time() - start_time
        self.decoding_times.append(decoding_time)
        
        # Prepare response
        response = {
            "decoded_message": decoded_message,
            "session_id": encoded_message.get("session_id"),
            "message_id": encoded_message.get("message_id"),
            "timestamp": datetime.now().isoformat(),
            "decoding_time": decoding_time,
            "protocol_version": self.protocol_version
        }
        
        # Update metrics if enabled
        if self.metrics:
            self.metrics.record_decoding(
                encoded_length=len(encoded_message["tokens"]),
                decoded_length=len(decoded_message),
                decoding_time=decoding_time
            )
        
        return response
    
    def batch_communicate(
        self, 
        messages: List[str], 
        contexts: Optional[List[Dict[str, Any]]] = None,
        target_agents: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Send multiple messages in batch.
        
        Args:
            messages: List of messages to send
            contexts: Optional list of contexts
            target_agents: Optional list of target agents
            
        Returns:
            List of communication results
        """
        if contexts is None:
            contexts = [None] * len(messages)
        if target_agents is None:
            target_agents = [None] * len(messages)
            
        results = []
        for msg, ctx, agent in zip(messages, contexts, target_agents):
            result = self.communicate(msg, ctx, agent)
            results.append(result)
            
        return results
    
    def batch_receive(
        self, 
        encoded_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Receive and decode multiple messages in batch.
        
        Args:
            encoded_messages: List of encoded messages
            
        Returns:
            List of decoded messages
        """
        return [self.receive(msg) for msg in encoded_messages]
    
    def get_protocol_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive protocol statistics.
        
        Returns:
            Dictionary with protocol metrics and state
        """
        stats = {
            "protocol_version": self.protocol_version,
            "session_id": self.session_id,
            "message_count": self.message_count,
            "device": self.device,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.max_length,
            "adaptation_enabled": self.adaptation is not None,
            "metrics_enabled": self.metrics is not None
        }
        
        # Add performance metrics
        if self.encoding_times:
            stats["avg_encoding_time"] = np.mean(self.encoding_times)
            stats["total_encoding_time"] = np.sum(self.encoding_times)
            
        if self.decoding_times:
            stats["avg_decoding_time"] = np.mean(self.decoding_times)
            stats["total_decoding_time"] = np.sum(self.decoding_times)
            
        if self.compression_ratios:
            stats["avg_compression_ratio"] = np.mean(self.compression_ratios)
            stats["best_compression_ratio"] = min(self.compression_ratios)
            stats["worst_compression_ratio"] = max(self.compression_ratios)
        
        # Add adaptation history
        if self.adaptation_history:
            stats["adaptation_count"] = len(self.adaptation_history)
            stats["recent_adaptations"] = self.adaptation_history[-5:]  # Last 5
        
        # Add detailed metrics if available
        if self.metrics:
            stats["detailed_metrics"] = self.metrics.get_summary()
        
        return stats
    
    def reset_session(self) -> None:
        """
        Reset the current session and clear metrics.
        """
        self.session_id = self._generate_session_id()
        self.message_count = 0
        self.encoding_times = []
        self.decoding_times = []
        self.compression_ratios = []
        self.adaptation_history = []
        
        if self.metrics:
            self.metrics.reset()
    
    def export_session(self, filepath: str) -> None:
        """
        Export session data to a file.
        
        Args:
            filepath: Path to export file
        """
        session_data = {
            "session_id": self.session_id,
            "protocol_version": self.protocol_version,
            "timestamp": datetime.now().isoformat(),
            "stats": self.get_protocol_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def _generate_session_id(self) -> str:
        """
        Generate a unique session identifier.
        
        Returns:
            Session ID string
        """
        timestamp = int(time.time() * 1000)
        random_suffix = np.random.randint(1000, 9999)
        return f"aether_{timestamp}_{random_suffix}"
    
    def __str__(self) -> str:
        """String representation of the protocol."""
        return f"AetherSpeak Protocol v{self.protocol_version} (Session: {self.session_id})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"AetherProtocol(version='{self.protocol_version}', "
                f"device='{self.device}', messages={self.message_count})")
