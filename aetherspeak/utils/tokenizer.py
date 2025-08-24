"""
Text tokenization utilities for AetherSpeak protocol.
"""

import re
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer


class Tokenizer:
    """
    Handles text tokenization and preprocessing for the AetherSpeak protocol.
    
    Provides both standard tokenization and custom preprocessing for
    optimal embedding generation.
    """
    
    def __init__(self, model_name: str, max_length: int = 512):
        """
        Initialize the tokenizer.
        
        Args:
            model_name: Name of the pre-trained model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Initialize the transformer tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            # Fallback to basic tokenization
            self.tokenizer = None
            print(f"Warning: Could not load {model_name}, using basic tokenization: {e}")
        
        # Custom preprocessing patterns
        self.preprocessing_patterns = [
            (r'\s+', ' '),  # Normalize whitespace
            (r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', ''),  # Remove special chars
            (r'\.{2,}', '...'),  # Normalize ellipsis
        ]
        
    def tokenize(self, text: str, return_tensors: bool = False) -> Union[List[str], Dict[str, Any]]:
        """
        Tokenize input text.
        
        Args:
            text: Input text to tokenize
            return_tensors: Whether to return PyTorch tensors
            
        Returns:
            Tokenized text (list of tokens or tokenizer output)
        """
        if self.tokenizer:
            return self._transformer_tokenize(text, return_tensors)
        else:
            return self._basic_tokenize(text)
    
    def _transformer_tokenize(self, text: str, return_tensors: bool) -> Union[List[str], Dict[str, Any]]:
        """
        Use transformer tokenizer for tokenization.
        
        Args:
            text: Input text
            return_tensors: Whether to return tensors
            
        Returns:
            Tokenizer output
        """
        # Preprocess text
        preprocessed = self.preprocess_text(text)
        
        # Tokenize
        tokens = self.tokenizer(
            preprocessed,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt" if return_tensors else None
        )
        
        return tokens
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """
        Basic tokenization fallback.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Preprocess text
        preprocessed = self.preprocess_text(text)
        
        # Simple word-based tokenization
        tokens = preprocessed.split()
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for optimal tokenization.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        # Apply preprocessing patterns
        for pattern, replacement in self.preprocessing_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Reconstructed text
        """
        if self.tokenizer:
            return self.tokenizer.convert_tokens_to_string(tokens)
        else:
            return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Number of tokens in vocabulary
        """
        if self.tokenizer:
            return self.tokenizer.vocab_size
        else:
            return 50000  # Default estimate
    
    def get_special_tokens(self) -> Dict[str, str]:
        """
        Get special tokens used by the tokenizer.
        
        Returns:
            Dictionary of special token types and values
        """
        if self.tokenizer:
            return {
                'pad_token': self.tokenizer.pad_token,
                'unk_token': self.tokenizer.unk_token,
                'sep_token': self.tokenizer.sep_token,
                'cls_token': self.tokenizer.cls_token,
                'mask_token': self.tokenizer.mask_token
            }
        else:
            return {}
    
    def encode_batch(
        self, 
        texts: List[str], 
        return_tensors: bool = False
    ) -> Union[List[List[str]], Dict[str, Any]]:
        """
        Tokenize multiple texts in batch.
        
        Args:
            texts: List of texts to tokenize
            return_tensors: Whether to return tensors
            
        Returns:
            Batch tokenization results
        """
        if self.tokenizer:
            # Use batch processing
            preprocessed_texts = [self.preprocess_text(text) for text in texts]
            return self.tokenizer(
                preprocessed_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt" if return_tensors else None
            )
        else:
            # Process individually
            return [self.tokenize(text, return_tensors) for text in texts]
    
    def __str__(self) -> str:
        """String representation."""
        return f"Tokenizer(model='{self.model_name}', max_length={self.max_length})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Tokenizer(model_name='{self.model_name}', "
                f"max_length={self.max_length}, "
                f"transformer_loaded={self.tokenizer is not None})")
