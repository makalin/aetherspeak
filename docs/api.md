# AetherSpeak API Documentation

## Overview

The AetherSpeak API provides a RESTful interface for efficient AI-to-AI communication using the AetherSpeak protocol. The API supports message encoding, decoding, batch processing, and protocol management.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. All endpoints are publicly accessible.

## Endpoints

### Health Check

#### GET /health

Check the health status of the service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "version": "0.1.0",
  "uptime": "0:00:05.123456",
  "model_status": "loaded"
}
```

### Service Information

#### GET /info

Get information about the AetherSpeak service.

**Response:**
```json
{
  "name": "AetherSpeak",
  "version": "0.1.0",
  "description": "An adaptive language protocol for efficient AI-to-AI communication",
  "features": [
    "Neural embedding compression",
    "Symbolic logic extraction",
    "Self-adapting protocol",
    "Performance metrics",
    "Batch processing"
  ],
  "model_info": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",
    "model_loaded": true,
    "embedding_dimension": 384
  }
}
```

### Message Processing

#### POST /message

Process a single message (encode or decode).

**Request Body:**
```json
{
  "message": "Hello, how are you today?",
  "context": {
    "user_id": "12345",
    "session": "chat_001"
  },
  "target_agent": "assistant_01",
  "operation": "encode"
}
```

**Parameters:**
- `message` (string, required): The message to process
- `context` (object, optional): Additional context information
- `target_agent` (string, optional): Target agent identifier
- `operation` (string, required): Either "encode" or "decode"

**Response (Encode):**
```json
{
  "success": true,
  "result": {
    "header": {
      "protocol": "AetherSpeak",
      "version": "0.1.0",
      "original_length": 25,
      "encoding_method": "hybrid_neural_symbolic",
      "compression_target": 0.01
    },
    "tokens": ["E0|v0:0.123|v1:0.456", "S:LOGIC:greeting"],
    "symbolic_elements": {
      "logic": ["greeting"]
    },
    "compression_ratio": 0.08,
    "protocol_version": "0.1.0",
    "session_id": "aether_1704110400000_1234",
    "message_id": 1,
    "timestamp": "2024-01-01T12:00:00",
    "target_agent": "assistant_01"
  },
  "operation": "encode",
  "protocol_version": "0.1.0",
  "timestamp": "2024-01-01T12:00:00"
}
```

**Response (Decode):**
```json
{
  "success": true,
  "result": {
    "decoded_message": "This message contains significant semantic content. Logical structure: greeting",
    "session_id": "aether_1704110400000_1234",
    "message_id": 1,
    "timestamp": "2024-01-01T12:00:00",
    "decoding_time": 0.0012,
    "protocol_version": "0.1.0"
  },
  "operation": "decode",
  "protocol_version": "0.1.0",
  "timestamp": "2024-01-01T12:00:00"
}
```

### Batch Processing

#### POST /batch

Process multiple messages in batch.

**Request Body:**
```json
{
  "messages": [
    "Process data from sensor A",
    "Analyze temperature readings",
    "Generate summary report"
  ],
  "contexts": [
    {"sensor_id": "A", "priority": "high"},
    {"data_type": "temperature", "timeframe": "24h"},
    {"report_type": "summary", "format": "json"}
  ],
  "target_agents": ["processor_01", "analyzer_01", "reporter_01"],
  "operation": "encode"
}
```

**Parameters:**
- `messages` (array of strings, required): List of messages to process
- `contexts` (array of objects, optional): Context for each message
- `target_agents` (array of strings, optional): Target agent for each message
- `operation` (string, required): Either "encode" or "decode"

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "header": {...},
      "tokens": [...],
      "compression_ratio": 0.06,
      "session_id": "aether_1704110400000_1234",
      "message_id": 1
    },
    {
      "header": {...},
      "tokens": [...],
      "compression_ratio": 0.08,
      "session_id": "aether_1704110400000_1234",
      "message_id": 2
    },
    {
      "header": {...},
      "tokens": [...],
      "compression_ratio": 0.07,
      "session_id": "aether_1704110400000_1234",
      "message_id": 3
    }
  ],
  "operation": "encode",
  "total_messages": 3,
  "successful_messages": 3,
  "timestamp": "2024-01-01T12:00:00"
}
```

### Protocol Statistics

#### GET /stats

Get comprehensive protocol statistics.

**Response:**
```json
{
  "protocol_version": "0.1.0",
  "session_id": "aether_1704110400000_1234",
  "message_count": 15,
  "device": "cpu",
  "embedding_dimension": 384,
  "max_sequence_length": 512,
  "adaptation_enabled": true,
  "metrics_enabled": true,
  "avg_encoding_time": 0.0023,
  "avg_decoding_time": 0.0018,
  "avg_compression_ratio": 0.075,
  "best_compression_ratio": 0.04,
  "worst_compression_ratio": 0.12,
  "adaptation_count": 2,
  "recent_adaptations": [
    {
      "timestamp": "2024-01-01T11:55:00",
      "trigger": "performance_deviation",
      "changes": {
        "compression_aggressiveness": {
          "old": 0.5,
          "new": 0.6
        }
      }
    }
  ]
}
```

### Protocol Adaptation

#### POST /adaptation

Control protocol adaptation behavior.

**Request Body:**
```json
{
  "action": "get_params"
}
```

**Actions:**
- `reset`: Reset adaptation parameters to defaults
- `export`: Export adaptation data to a file
- `get_params`: Get current adaptation parameters

**Response:**
```json
{
  "success": true,
  "action": "get_params",
  "result": {
    "embedding_dimension": 384,
    "compression_aggressiveness": 0.6,
    "symbolic_weight": 0.3,
    "neural_weight": 0.7
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### Session Management

#### POST /reset

Reset the current session and clear all metrics.

**Response:**
```json
{
  "message": "Session reset successfully"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes and error messages in case of failure.

**Error Response Format:**
```json
{
  "success": false,
  "error": "Error description",
  "operation": "encode",
  "protocol_version": "0.1.0",
  "timestamp": "2024-01-01T12:00:00"
}
```

**Common HTTP Status Codes:**
- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `500 Internal Server Error`: Server-side error

## Rate Limiting

Currently, no rate limiting is implemented. All requests are processed as they arrive.

## Examples

### Python Client Example

```python
import requests
import json

# Encode a message
response = requests.post("http://localhost:8000/message", json={
    "message": "Hello, how are you?",
    "operation": "encode"
})

if response.status_code == 200:
    result = response.json()
    print(f"Compression ratio: {result['result']['compression_ratio']}")
    
    # Decode the message
    decode_response = requests.post("http://localhost:8000/message", json={
        "message": json.dumps(result['result']),
        "operation": "decode"
    })
    
    if decode_response.status_code == 200:
        decoded = decode_response.json()
        print(f"Decoded: {decoded['result']['decoded_message']}")
```

### cURL Examples

```bash
# Encode a message
curl -X POST "http://localhost:8000/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello world", "operation": "encode"}'

# Get statistics
curl "http://localhost:8000/stats"

# Run batch encoding
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{"messages": ["Msg 1", "Msg 2"], "operation": "encode"}'
```

## WebSocket Support

WebSocket support is planned for future versions to enable real-time communication.

## Versioning

The API version is included in all responses. Breaking changes will be introduced in new major versions with appropriate migration guides.
