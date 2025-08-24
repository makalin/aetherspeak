#!/usr/bin/env python3
"""
Basic usage example for AetherSpeak protocol.

This example demonstrates the core functionality of encoding and decoding
messages using the AetherSpeak protocol.
"""

import sys
import os

# Add the parent directory to the path so we can import aetherspeak
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aetherspeak import AetherProtocol, Encoder, Decoder


def basic_encoding_decoding():
    """Demonstrate basic encoding and decoding."""
    print("=== Basic Encoding and Decoding ===")
    
    # Initialize the protocol
    protocol = AetherProtocol(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        enable_adaptation=True,
        enable_metrics=True
    )
    
    # Example messages
    messages = [
        "Hello, how are you today?",
        "Please help me with this task.",
        "If the temperature is above 25°C, then turn on the air conditioning.",
        "The sum of x and y equals 10, and x is greater than y.",
        "First, collect the data. Then, analyze the results. Finally, draw conclusions."
    ]
    
    print(f"Protocol initialized: {protocol}")
    print(f"Device: {protocol.device}")
    print(f"Embedding dimension: {protocol.embedding_dim}")
    print()
    
    # Process each message
    for i, message in enumerate(messages, 1):
        print(f"Message {i}: {message}")
        print(f"Length: {len(message)} characters")
        
        # Encode the message
        encoded = protocol.communicate(message)
        print(f"Encoded tokens: {len(encoded['tokens'])}")
        print(f"Compression ratio: {encoded['compression_ratio']:.3f}")
        
        # Decode the message
        decoded = protocol.receive(encoded)
        print(f"Decoded: {decoded['decoded_message']}")
        
        # Show symbolic elements if any
        if encoded.get('symbolic_elements'):
            print(f"Symbolic elements: {len(encoded['symbolic_elements'])}")
            for elem_type, elements in encoded['symbolic_elements'].items():
                print(f"  {elem_type}: {len(elements)} elements")
        
        print("-" * 50)
    
    # Show protocol statistics
    print("\n=== Protocol Statistics ===")
    stats = protocol.get_protocol_stats()
    print(f"Total messages: {stats['message_count']}")
    print(f"Average compression ratio: {stats.get('avg_compression_ratio', 'N/A')}")
    print(f"Best compression ratio: {stats.get('best_compression_ratio', 'N/A')}")
    print(f"Average encoding time: {stats.get('avg_encoding_time', 'N/A'):.4f}s")
    print(f"Average decoding time: {stats.get('avg_decoding_time', 'N/A'):.4f}s")
    
    if stats.get('adaptation_count', 0) > 0:
        print(f"Adaptations performed: {stats['adaptation_count']}")
        print("Recent adaptations:")
        for adaptation in stats.get('recent_adaptations', [])[:3]:
            print(f"  - {adaptation['trigger']}: {adaptation['changes']}")


def standalone_encoder_decoder():
    """Demonstrate standalone encoder and decoder usage."""
    print("\n=== Standalone Encoder/Decoder ===")
    
    # Initialize encoder and decoder separately
    encoder = Encoder()
    decoder = Decoder()
    
    print(f"Encoder: {encoder}")
    print(f"Decoder: {decoder}")
    print()
    
    # Test message
    test_message = "Collaborate on task X with agent Y"
    print(f"Test message: {test_message}")
    
    # Encode
    encoded = encoder.encode(test_message)
    print(f"Encoded result: {len(encoded['tokens'])} tokens")
    print(f"Compression ratio: {encoded['compression_ratio']:.3f}")
    
    # Decode
    decoded = decoder.decode(encoded)
    print(f"Decoded message: {decoded}")
    
    # Show compression stats
    stats = encoder.get_compression_stats()
    print(f"Target compression ratio: {stats['target_compression_ratio']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")


def batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n=== Batch Processing ===")
    
    protocol = AetherProtocol()
    
    # Batch of messages
    batch_messages = [
        "Process data from sensor A",
        "Analyze temperature readings",
        "Generate summary report",
        "Send results to monitoring system"
    ]
    
    print(f"Processing {len(batch_messages)} messages in batch...")
    
    # Batch encode
    encoded_batch = protocol.batch_communicate(batch_messages)
    print(f"Batch encoded: {len(encoded_batch)} results")
    
    # Show compression for each
    for i, (msg, encoded) in enumerate(zip(batch_messages, encoded_batch)):
        print(f"  {i+1}. {msg[:30]}... -> {len(encoded['tokens'])} tokens "
              f"(ratio: {encoded['compression_ratio']:.3f})")
    
    # Batch decode
    decoded_batch = protocol.batch_receive(encoded_batch)
    print(f"Batch decoded: {len(decoded_batch)} results")


def main():
    """Run all examples."""
    print("AetherSpeak Protocol Examples")
    print("=" * 50)
    
    try:
        basic_encoding_decoding()
        standalone_encoder_decoder()
        batch_processing()
        
        print("\n=== Example Completed Successfully ===")
        print("The AetherSpeak protocol is working correctly!")
        
    except Exception as e:
        print(f"\nError during example execution: {e}")
        print("This might be due to missing dependencies or model files.")
        print("Please ensure all requirements are installed:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
