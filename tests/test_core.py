"""
Tests for core AetherSpeak functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import the modules to test
from aetherspeak.core.encoder import Encoder
from aetherspeak.core.decoder import Decoder
from aetherspeak.core.protocol import AetherProtocol


class TestEncoder:
    """Test the Encoder class."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        encoder = Encoder()
        assert encoder.protocol_version == "0.1.0"
        assert encoder.compression_ratio == 0.01
        assert encoder.embedding_dim == 384
        assert encoder.max_length == 512
    
    def test_encoder_with_custom_params(self):
        """Test encoder with custom parameters."""
        encoder = Encoder(
            model_name="test-model",
            embedding_dim=256,
            max_length=128,
            device="cpu"
        )
        assert encoder.embedding_dim == 256
        assert encoder.max_length == 128
        assert encoder.device == "cpu"
    
    @patch('aetherspeak.utils.embedding.EmbeddingEngine')
    @patch('aetherspeak.utils.tokenizer.Tokenizer')
    @patch('aetherspeak.utils.symbolic.SymbolicProcessor')
    def test_encode_message(self, mock_symbolic, mock_tokenizer, mock_embedding):
        """Test message encoding."""
        # Mock the dependencies
        mock_embedding_instance = Mock()
        mock_embedding_instance.embed.return_value = np.random.randn(1, 384)
        mock_embedding.return_value = mock_embedding_instance
        
        mock_symbolic_instance = Mock()
        mock_symbolic_instance.extract_symbols.return_value = {"logic": []}
        mock_symbolic.return_value = mock_symbolic_instance
        
        encoder = Encoder()
        
        # Test encoding
        message = "Test message"
        result = encoder.encode(message)
        
        assert "header" in result
        assert "tokens" in result
        assert "symbolic_elements" in result
        assert "compression_ratio" in result
        assert result["protocol_version"] == "0.1.0"
        assert result["header"]["original_length"] == len(message)
    
    def test_compress_embeddings(self):
        """Test embedding compression."""
        encoder = Encoder(embedding_dim=256)
        
        # Create test embeddings
        embeddings = np.random.randn(5, 384)
        
        # Compress
        compressed = encoder._compress_embeddings(embeddings)
        
        assert compressed.shape[1] <= 256
        assert compressed.shape[0] == 5
    
    def test_create_tokens(self):
        """Test token creation."""
        encoder = Encoder()
        
        # Mock embeddings and symbols
        embeddings = np.random.randn(3, 384)
        symbolic_elements = {"logic": ["test"], "math": ["x+y"]}
        
        tokens = encoder._create_tokens(embeddings, symbolic_elements)
        
        assert len(tokens) > 0
        assert any(token.startswith("E") for token in tokens)  # Embedding tokens
        assert any(token.startswith("S:") for token in tokens)  # Symbolic tokens
    
    def test_batch_encode(self):
        """Test batch encoding."""
        encoder = Encoder()
        
        messages = ["Message 1", "Message 2", "Message 3"]
        contexts = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        results = encoder.batch_encode(messages, contexts)
        
        assert len(results) == 3
        for result in results:
            assert "header" in result
            assert "tokens" in result


class TestDecoder:
    """Test the Decoder class."""
    
    def test_decoder_initialization(self):
        """Test decoder initialization."""
        decoder = Decoder()
        assert decoder.protocol_version == "0.1.0"
        assert decoder.embedding_dim == 384
        assert decoder.max_length == 512
    
    def test_decode_message(self):
        """Test message decoding."""
        decoder = Decoder()
        
        # Create a mock encoded message
        encoded_message = {
            "protocol_version": "0.1.0",
            "tokens": ["E0|v0:0.123|v1:0.456", "S:LOGIC:test"],
            "symbolic_elements": {"logic": ["test"]}
        }
        
        result = decoder.decode(encoded_message)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_validate_message(self):
        """Test message validation."""
        decoder = Decoder()
        
        # Valid message
        valid_message = {
            "protocol_version": "0.1.0",
            "tokens": ["E0|v0:0.123"]
        }
        assert decoder.validate_message(valid_message) is True
        
        # Invalid message - missing required fields
        invalid_message = {"tokens": []}
        assert decoder.validate_message(invalid_message) is False
        
        # Invalid message - wrong protocol version
        wrong_version_message = {
            "protocol_version": "0.2.0",
            "tokens": ["E0|v0:0.123"]
        }
        assert decoder.validate_message(wrong_version_message) is False
    
    def test_reconstruct_embeddings(self):
        """Test embedding reconstruction."""
        decoder = Decoder()
        
        # Test tokens with embedding data
        tokens = ["E0|v0:0.123|v1:0.456", "E1|v0:0.789"]
        
        embeddings = decoder._reconstruct_embeddings(tokens)
        
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) > 0
    
    def test_batch_decode(self):
        """Test batch decoding."""
        decoder = Decoder()
        
        # Create mock encoded messages
        encoded_messages = [
            {
                "protocol_version": "0.1.0",
                "tokens": ["E0|v0:0.123"],
                "symbolic_elements": {}
            },
            {
                "protocol_version": "0.1.0",
                "tokens": ["E0|v0:0.456"],
                "symbolic_elements": {}
            }
        ]
        
        results = decoder.batch_decode(encoded_messages)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result, str)


class TestAetherProtocol:
    """Test the AetherProtocol class."""
    
    def test_protocol_initialization(self):
        """Test protocol initialization."""
        protocol = AetherProtocol()
        assert protocol.protocol_version == "0.1.0"
        assert protocol.message_count == 0
        assert protocol.adaptation is not None
        assert protocol.metrics is not None
    
    def test_protocol_without_adaptation(self):
        """Test protocol without adaptation."""
        protocol = AetherProtocol(enable_adaptation=False)
        assert protocol.adaptation is None
    
    def test_protocol_without_metrics(self):
        """Test protocol without metrics."""
        protocol = AetherProtocol(enable_metrics=False)
        assert protocol.metrics is None
    
    def test_communicate(self):
        """Test message communication."""
        protocol = AetherProtocol()
        
        message = "Test communication message"
        result = protocol.communicate(message)
        
        assert "session_id" in result
        assert "message_id" in result
        assert "timestamp" in result
        assert protocol.message_count == 1
    
    def test_receive(self):
        """Test message reception."""
        protocol = AetherProtocol()
        
        # First encode a message
        encoded = protocol.communicate("Test message")
        
        # Then receive/decode it
        result = protocol.receive(encoded)
        
        assert "decoded_message" in result
        assert "session_id" in result
        assert "message_id" in result
    
    def test_batch_communicate(self):
        """Test batch communication."""
        protocol = AetherProtocol()
        
        messages = ["Message 1", "Message 2", "Message 3"]
        results = protocol.batch_communicate(messages)
        
        assert len(results) == 3
        assert protocol.message_count == 3
    
    def test_batch_receive(self):
        """Test batch reception."""
        protocol = AetherProtocol()
        
        # First encode messages
        encoded = protocol.batch_communicate(["Test 1", "Test 2"])
        
        # Then receive them
        results = protocol.batch_receive(encoded)
        
        assert len(results) == 2
    
    def test_get_protocol_stats(self):
        """Test protocol statistics."""
        protocol = AetherProtocol()
        
        # Process some messages
        protocol.communicate("Test message 1")
        protocol.communicate("Test message 2")
        
        stats = protocol.get_protocol_stats()
        
        assert stats["message_count"] == 2
        assert stats["protocol_version"] == "0.1.0"
        assert "avg_encoding_time" in stats
        assert "avg_compression_ratio" in stats
    
    def test_reset_session(self):
        """Test session reset."""
        protocol = AetherProtocol()
        
        # Process some messages
        protocol.communicate("Test message")
        original_session_id = protocol.session_id
        
        # Reset session
        protocol.reset_session()
        
        assert protocol.message_count == 0
        assert protocol.session_id != original_session_id
        assert len(protocol.encoding_times) == 0
    
    def test_export_session(self, tmp_path):
        """Test session export."""
        protocol = AetherProtocol()
        
        # Process a message
        protocol.communicate("Test message")
        
        # Export session
        export_file = tmp_path / "session_export.json"
        protocol.export_session(str(export_file))
        
        assert export_file.exists()
        assert export_file.stat().st_size > 0


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_encode_decode_pipeline(self):
        """Test the complete encode-decode pipeline."""
        protocol = AetherProtocol()
        
        # Test message
        original_message = "This is a test message with logical structure: if A then B, and C implies D."
        
        # Encode
        encoded = protocol.communicate(original_message)
        
        # Verify encoding
        assert "tokens" in encoded
        assert "symbolic_elements" in encoded
        assert encoded["compression_ratio"] < 1.0  # Should compress
        
        # Decode
        decoded = protocol.receive(encoded)
        
        # Verify decoding
        assert "decoded_message" in decoded
        assert isinstance(decoded["decoded_message"], str)
        assert len(decoded["decoded_message"]) > 0
    
    def test_batch_pipeline(self):
        """Test the complete batch pipeline."""
        protocol = AetherProtocol()
        
        # Test messages
        messages = [
            "Simple message",
            "Message with logic: A and B",
            "Mathematical: x + y = z",
            "Structural: first step, then second, finally third"
        ]
        
        # Batch encode
        encoded_batch = protocol.batch_communicate(messages)
        
        # Verify batch encoding
        assert len(encoded_batch) == len(messages)
        for encoded in encoded_batch:
            assert "tokens" in encoded
            assert encoded["compression_ratio"] < 1.0
        
        # Batch decode
        decoded_batch = protocol.batch_receive(encoded_batch)
        
        # Verify batch decoding
        assert len(decoded_batch) == len(messages)
        for decoded in decoded_batch:
            assert isinstance(decoded, str)
            assert len(decoded) > 0


if __name__ == "__main__":
    pytest.main([__file__])
