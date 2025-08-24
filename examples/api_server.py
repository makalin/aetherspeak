#!/usr/bin/env python3
"""
API Server example for AetherSpeak protocol.

This example demonstrates how to run the AetherSpeak API server
and interact with it using HTTP requests.
"""

import sys
import os
import time
import requests
import json

# Add the parent directory to the path so we can import aetherspeak
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aetherspeak.api.server import create_server


def run_api_server():
    """Run the AetherSpeak API server."""
    print("Starting AetherSpeak API Server...")
    
    # Create and configure the server
    server = create_server(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        max_length=512,
        enable_adaptation=True,
        enable_metrics=True,
        host="127.0.0.1",
        port=8000
    )
    
    print(f"Server configured:")
    print(f"  Host: {server.host}")
    print(f"  Port: {server.port}")
    print(f"  Model: {server.protocol.encoder.embedding_engine.model_name}")
    print(f"  Device: {server.protocol.device}")
    print(f"  Adaptation: {'Enabled' if server.protocol.adaptation else 'Disabled'}")
    print(f"  Metrics: {'Enabled' if server.protocol.metrics else 'Disabled'}")
    print()
    
    # Run the server
    print("Starting server... (Press Ctrl+C to stop)")
    try:
        server.run(host=server.host, port=server.port, log_level="info")
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")


def test_api_endpoints():
    """Test the API endpoints."""
    base_url = "http://127.0.0.1:8000"
    
    print("Testing API endpoints...")
    print(f"Base URL: {base_url}")
    print()
    
    # Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✓ Health check passed: {health_data['status']}")
            print(f"   Version: {health_data['version']}")
            print(f"   Model status: {health_data['model_status']}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ✗ Could not connect to server. Is it running?")
        return
    except Exception as e:
        print(f"   ✗ Health check error: {e}")
    
    # Test info endpoint
    print("\n2. Testing info endpoint...")
    try:
        response = requests.get(f"{base_url}/info")
        if response.status_code == 200:
            info_data = response.json()
            print(f"   ✓ Info retrieved: {info_data['name']} v{info_data['version']}")
            print(f"   Description: {info_data['description']}")
            print(f"   Features: {', '.join(info_data['features'])}")
        else:
            print(f"   ✗ Info endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Info endpoint error: {e}")
    
    # Test message encoding
    print("\n3. Testing message encoding...")
    test_message = "Hello, this is a test message for the AetherSpeak protocol."
    encode_data = {
        "message": test_message,
        "operation": "encode",
        "context": {"test": True, "timestamp": time.time()}
    }
    
    try:
        response = requests.post(f"{base_url}/message", json=encode_data)
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"   ✓ Message encoded successfully")
                print(f"   Original length: {len(test_message)} characters")
                print(f"   Encoded tokens: {len(result['result']['tokens'])}")
                print(f"   Compression ratio: {result['result']['compression_ratio']:.3f}")
                
                # Test decoding
                print("\n4. Testing message decoding...")
                decode_data = {
                    "message": json.dumps(result['result']),
                    "operation": "decode"
                }
                
                decode_response = requests.post(f"{base_url}/message", json=decode_data)
                if decode_response.status_code == 200:
                    decode_result = decode_response.json()
                    if decode_result['success']:
                        print(f"   ✓ Message decoded successfully")
                        print(f"   Decoded message: {decode_result['result']['decoded_message']}")
                    else:
                        print(f"   ✗ Decoding failed: {decode_result['error']}")
                else:
                    print(f"   ✗ Decode request failed: {decode_response.status_code}")
            else:
                print(f"   ✗ Encoding failed: {result['error']}")
        else:
            print(f"   ✗ Encode request failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Encoding error: {e}")
    
    # Test batch processing
    print("\n5. Testing batch processing...")
    batch_messages = [
        "First message in batch",
        "Second message with different content",
        "Third message for testing purposes"
    ]
    
    batch_data = {
        "messages": batch_messages,
        "operation": "encode",
        "contexts": [{"batch_id": i} for i in range(len(batch_messages))]
    }
    
    try:
        response = requests.post(f"{base_url}/batch", json=batch_data)
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"   ✓ Batch processing successful")
                print(f"   Processed: {result['successful_messages']}/{result['total_messages']} messages")
                
                # Show compression for each
                for i, (msg, encoded) in enumerate(zip(batch_messages, result['results'])):
                    print(f"   Message {i+1}: {len(msg)} chars -> {len(encoded['tokens'])} tokens "
                          f"(ratio: {encoded['compression_ratio']:.3f})")
            else:
                print(f"   ✗ Batch processing failed: {result['errors']}")
        else:
            print(f"   ✗ Batch request failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Batch processing error: {e}")
    
    # Test statistics
    print("\n6. Testing statistics endpoint...")
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✓ Statistics retrieved")
            print(f"   Session ID: {stats['session_id']}")
            print(f"   Message count: {stats['message_count']}")
            print(f"   Average compression: {stats.get('avg_compression_ratio', 'N/A')}")
            print(f"   Adaptation count: {stats.get('adaptation_count', 0)}")
        else:
            print(f"   ✗ Statistics request failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Statistics error: {e}")
    
    print("\n=== API Testing Complete ===")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AetherSpeak API Server Example")
    parser.add_argument(
        "--test-only", 
        action="store_true", 
        help="Only test the API endpoints (assumes server is running)"
    )
    parser.add_argument(
        "--server-only", 
        action="store_true", 
        help="Only run the server (don't test endpoints)"
    )
    
    args = parser.parse_args()
    
    if args.test_only:
        test_api_endpoints()
    elif args.server_only:
        run_api_server()
    else:
        print("AetherSpeak API Server Example")
        print("=" * 50)
        print("This example will start the API server and then test the endpoints.")
        print("You can also use --test-only to test an existing server or --server-only to just run the server.")
        print()
        
        # Start server in background thread
        import threading
        server_thread = threading.Thread(target=run_api_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(5)
        
        # Test endpoints
        test_api_endpoints()


if __name__ == "__main__":
    main()
