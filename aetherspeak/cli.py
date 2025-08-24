#!/usr/bin/env python3
"""
Command-line interface for AetherSpeak protocol.
"""

import argparse
import sys
import json
from typing import Optional

from .core.protocol import AetherProtocol
from .api.server import create_server


def encode_message(args):
    """Encode a message using AetherSpeak."""
    protocol = AetherProtocol(
        model_name=args.model,
        embedding_dim=args.embedding_dim,
        max_length=args.max_length,
        device=args.device
    )
    
    # Encode the message
    result = protocol.communicate(args.message, args.context)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Encoded message saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))
    
    return result


def decode_message(args):
    """Decode a message using AetherSpeak."""
    protocol = AetherProtocol(
        model_name=args.model,
        embedding_dim=args.embedding_dim,
        max_length=args.max_length,
        device=args.device
    )
    
    # Load encoded message
    if args.input:
        with open(args.input, 'r') as f:
            encoded_message = json.load(f)
    else:
        encoded_message = json.loads(args.message)
    
    # Decode the message
    result = protocol.receive(encoded_message)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Decoded message saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))
    
    return result


def run_server(args):
    """Run the AetherSpeak API server."""
    server = create_server(
        model_name=args.model,
        embedding_dim=args.embedding_dim,
        max_length=args.max_length,
        device=args.device,
        enable_adaptation=args.enable_adaptation,
        enable_metrics=args.enable_metrics,
        host=args.host,
        port=args.port
    )
    
    print(f"Starting AetherSpeak API server on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Adaptation: {'enabled' if args.enable_adaptation else 'disabled'}")
    print(f"Metrics: {'enabled' if args.enable_metrics else 'disabled'}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    
    server.run()


def show_stats(args):
    """Show protocol statistics."""
    protocol = AetherProtocol(
        model_name=args.model,
        embedding_dim=args.embedding_dim,
        max_length=args.max_length,
        device=args.device,
        enable_metrics=True
    )
    
    # Process some test messages to generate stats
    test_messages = [
        "Hello, this is a test message.",
        "Another test message for statistics.",
        "Final test message to complete the set."
    ]
    
    for msg in test_messages:
        protocol.communicate(msg)
    
    stats = protocol.get_protocol_stats()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {args.output}")
    else:
        print(json.dumps(stats, indent=2))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AetherSpeak Protocol - Efficient AI-to-AI Communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode a message
  aetherspeak encode "Hello, how are you?" --output encoded.json
  
  # Decode a message
  aetherspeak decode --input encoded.json
  
  # Run API server
  aetherspeak server --port 8000
  
  # Show statistics
  aetherspeak stats --output stats.json
        """
    )
    
    # Global options
    parser.add_argument(
        '--model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Model to use for embeddings (default: sentence-transformers/all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=384,
        help='Embedding dimension (default: 384)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        help='Device to use (default: auto-detect)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Encode command
    encode_parser = subparsers.add_parser('encode', help='Encode a message')
    encode_parser.add_argument('message', help='Message to encode')
    encode_parser.add_argument('--context', help='Context as JSON string')
    encode_parser.add_argument('--output', help='Output file for encoded message')
    encode_parser.set_defaults(func=encode_message)
    
    # Decode command
    decode_parser = subparsers.add_parser('decode', help='Decode a message')
    decode_parser.add_argument('--message', help='Encoded message as JSON string')
    decode_parser.add_argument('--input', help='Input file with encoded message')
    decode_parser.add_argument('--output', help='Output file for decoded message')
    decode_parser.set_defaults(func=decode_message)
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Run API server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    server_parser.add_argument('--enable-adaptation', action='store_true', help='Enable protocol adaptation')
    server_parser.add_argument('--enable-metrics', action='store_true', help='Enable performance metrics')
    server_parser.set_defaults(func=run_server)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show protocol statistics')
    stats_parser.add_argument('--output', help='Output file for statistics')
    stats_parser.set_defaults(func=show_stats)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle context parsing
    if hasattr(args, 'context') and args.context:
        try:
            args.context = json.loads(args.context)
        except json.JSONDecodeError:
            print("Error: Invalid JSON context string", file=sys.stderr)
            sys.exit(1)
    
    # Execute command
    if args.command:
        try:
            args.func(args)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
