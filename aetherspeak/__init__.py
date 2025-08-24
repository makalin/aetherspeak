"""
AetherSpeak: An adaptive language protocol for efficient AI-to-AI communication.

This package provides a hybrid approach combining neural embeddings for semantic nuance
and symbolic elements for logical precision, enabling compact data exchange between AI agents.
"""

__version__ = "0.1.0"
__author__ = "makalin"
__email__ = ""

from .core.encoder import Encoder
from .core.decoder import Decoder
from .core.protocol import AetherProtocol
from .utils.tokenizer import Tokenizer
from .utils.embedding import EmbeddingEngine

__all__ = [
    "Encoder",
    "Decoder", 
    "AetherProtocol",
    "Tokenizer",
    "EmbeddingEngine",
    "__version__",
    "__author__",
    "__email__",
]
