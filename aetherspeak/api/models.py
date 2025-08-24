"""
Pydantic models for AetherSpeak API.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime


class MessageRequest(BaseModel):
    """Request model for message encoding/decoding."""
    
    message: str = Field(..., description="Message to process")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context information")
    target_agent: Optional[str] = Field(None, description="Target agent identifier")
    operation: str = Field("encode", description="Operation to perform: 'encode' or 'decode'")


class MessageResponse(BaseModel):
    """Response model for message processing."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    result: Optional[Dict[str, Any]] = Field(None, description="Processing result")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    operation: str = Field(..., description="Operation performed")
    protocol_version: str = Field(..., description="Protocol version used")


class ProtocolStats(BaseModel):
    """Model for protocol statistics."""
    
    protocol_version: str = Field(..., description="Protocol version")
    session_id: str = Field(..., description="Current session identifier")
    message_count: int = Field(..., description="Total messages processed")
    device: str = Field(..., description="Device being used")
    embedding_dimension: int = Field(..., description="Embedding dimension")
    max_sequence_length: int = Field(..., description="Maximum sequence length")
    adaptation_enabled: bool = Field(..., description="Whether adaptation is enabled")
    metrics_enabled: bool = Field(..., description="Whether metrics are enabled")
    avg_encoding_time: Optional[float] = Field(None, description="Average encoding time")
    avg_decoding_time: Optional[float] = Field(None, description="Average decoding time")
    avg_compression_ratio: Optional[float] = Field(None, description="Average compression ratio")
    best_compression_ratio: Optional[float] = Field(None, description="Best compression ratio achieved")
    worst_compression_ratio: Optional[float] = Field(None, description="Worst compression ratio achieved")
    adaptation_count: Optional[int] = Field(None, description="Number of adaptations performed")
    recent_adaptations: Optional[List[Dict[str, Any]]] = Field(None, description="Recent adaptation events")


class BatchRequest(BaseModel):
    """Request model for batch operations."""
    
    messages: List[str] = Field(..., description="List of messages to process")
    contexts: Optional[List[Dict[str, Any]]] = Field(None, description="Optional contexts for each message")
    target_agents: Optional[List[str]] = Field(None, description="Optional target agents for each message")
    operation: str = Field("encode", description="Operation to perform: 'encode' or 'decode'")


class BatchResponse(BaseModel):
    """Response model for batch operations."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    results: List[Dict[str, Any]] = Field(..., description="Results for each message")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during processing")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    operation: str = Field(..., description="Operation performed")
    total_messages: int = Field(..., description="Total number of messages processed")
    successful_messages: int = Field(..., description="Number of successfully processed messages")


class HealthCheck(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(..., description="Protocol version")
    uptime: str = Field(..., description="Service uptime")
    model_status: str = Field(..., description="Model loading status")


class AdaptationRequest(BaseModel):
    """Request model for adaptation control."""
    
    action: str = Field(..., description="Action to perform: 'reset', 'export', 'get_params'")
    filepath: Optional[str] = Field(None, description="File path for export action")


class AdaptationResponse(BaseModel):
    """Response model for adaptation operations."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    action: str = Field(..., description="Action performed")
    result: Optional[Dict[str, Any]] = Field(None, description="Operation result")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
