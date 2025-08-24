"""
FastAPI server for AetherSpeak protocol.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any, List
import time
import os

from ..core.protocol import AetherProtocol
from .models import (
    MessageRequest, MessageResponse, ProtocolStats, BatchRequest, BatchResponse,
    HealthCheck, AdaptationRequest, AdaptationResponse
)


class AetherSpeakAPI:
    """
    FastAPI server for AetherSpeak protocol.
    
    Provides REST API endpoints for message encoding/decoding,
    protocol statistics, and adaptation control.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_length: int = 512,
        device: str = None,
        enable_adaptation: bool = True,
        enable_metrics: bool = True,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        """
        Initialize the API server.
        
        Args:
            model_name: Pre-trained model name
            embedding_dim: Embedding dimension
            max_length: Maximum sequence length
            device: Device to use ('cpu' or 'cuda')
            enable_adaptation: Whether to enable protocol adaptation
            enable_metrics: Whether to enable performance metrics
            host: Host to bind to
            port: Port to bind to
        """
        self.host = host
        self.port = port
        
        # Initialize the protocol
        self.protocol = AetherProtocol(
            model_name=model_name,
            embedding_dim=embedding_dim,
            max_length=max_length,
            device=device,
            enable_adaptation=enable_adaptation,
            enable_metrics=enable_metrics
        )
        
        # Create FastAPI app
        self.app = FastAPI(
            title="AetherSpeak API",
            description="API for efficient AI-to-AI communication using the AetherSpeak protocol",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/", response_model=HealthCheck)
        async def root():
            """Root endpoint with health check."""
            return await self._health_check()
        
        @self.app.get("/health", response_model=HealthCheck)
        async def health():
            """Health check endpoint."""
            return await self._health_check()
        
        @self.app.post("/message", response_model=MessageResponse)
        async def process_message(request: MessageRequest):
            """Process a single message (encode or decode)."""
            return await self._process_message(request)
        
        @self.app.post("/batch", response_model=BatchResponse)
        async def process_batch(request: BatchRequest):
            """Process multiple messages in batch."""
            return await self._process_batch(request)
        
        @self.app.get("/stats", response_model=ProtocolStats)
        async def get_stats():
            """Get protocol statistics."""
            return await self._get_stats()
        
        @self.app.post("/adaptation", response_model=AdaptationResponse)
        async def control_adaptation(request: AdaptationRequest):
            """Control protocol adaptation."""
            return await self._control_adaptation(request)
        
        @self.app.post("/reset")
        async def reset_session():
            """Reset the current session."""
            return await self._reset_session()
        
        @self.app.get("/info")
        async def get_info():
            """Get basic protocol information."""
            return await self._get_info()
    
    async def _health_check(self) -> HealthCheck:
        """Perform health check."""
        try:
            # Check if protocol is working
            test_message = "Health check test"
            encoded = self.protocol.communicate(test_message)
            
            return HealthCheck(
                status="healthy",
                version=self.protocol.protocol_version,
                uptime=str(time.time() - time.time()),  # Placeholder
                model_status="loaded" if encoded else "error"
            )
        except Exception as e:
            return HealthCheck(
                status="unhealthy",
                version=self.protocol.protocol_version,
                uptime="0",
                model_status=f"error: {str(e)}"
            )
    
    async def _process_message(self, request: MessageRequest) -> MessageResponse:
        """Process a single message."""
        try:
            if request.operation == "encode":
                result = self.protocol.communicate(
                    request.message, 
                    request.context, 
                    request.target_agent
                )
                operation = "encode"
            elif request.operation == "decode":
                # For decode, we expect the message to be an encoded message dict
                try:
                    import json
                    encoded_message = json.loads(request.message) if isinstance(request.message, str) else request.message
                    result = self.protocol.receive(encoded_message)
                    operation = "decode"
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid encoded message format: {str(e)}")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid operation: {request.operation}")
            
            return MessageResponse(
                success=True,
                result=result,
                operation=operation,
                protocol_version=self.protocol.protocol_version
            )
            
        except Exception as e:
            return MessageResponse(
                success=False,
                error=str(e),
                operation=request.operation,
                protocol_version=self.protocol.protocol_version
            )
    
    async def _process_batch(self, request: BatchRequest) -> BatchResponse:
        """Process multiple messages in batch."""
        try:
            if request.operation == "encode":
                results = self.protocol.batch_communicate(
                    request.messages,
                    request.contexts,
                    request.target_agents
                )
                operation = "encode"
            elif request.operation == "decode":
                # For decode, we expect messages to be encoded message dicts
                try:
                    import json
                    encoded_messages = []
                    for msg in request.messages:
                        if isinstance(msg, str):
                            encoded_messages.append(json.loads(msg))
                        else:
                            encoded_messages.append(msg)
                    
                    results = self.protocol.batch_receive(encoded_messages)
                    operation = "decode"
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid encoded message format: {str(e)}")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid operation: {request.operation}")
            
            return BatchResponse(
                success=True,
                results=results,
                operation=operation,
                total_messages=len(request.messages),
                successful_messages=len(results)
            )
            
        except Exception as e:
            return BatchResponse(
                success=False,
                results=[],
                errors=[str(e)],
                operation=request.operation,
                total_messages=len(request.messages),
                successful_messages=0
            )
    
    async def _get_stats(self) -> ProtocolStats:
        """Get protocol statistics."""
        try:
            stats = self.protocol.get_protocol_stats()
            
            return ProtocolStats(
                protocol_version=stats.get('protocol_version'),
                session_id=stats.get('session_id'),
                message_count=stats.get('message_count'),
                device=stats.get('device'),
                embedding_dimension=stats.get('embedding_dimension'),
                max_sequence_length=stats.get('max_sequence_length'),
                adaptation_enabled=stats.get('adaptation_enabled'),
                metrics_enabled=stats.get('metrics_enabled'),
                avg_encoding_time=stats.get('avg_encoding_time'),
                avg_decoding_time=stats.get('avg_decoding_time'),
                avg_compression_ratio=stats.get('avg_compression_ratio'),
                best_compression_ratio=stats.get('best_compression_ratio'),
                worst_compression_ratio=stats.get('worst_compression_ratio'),
                adaptation_count=stats.get('adaptation_count'),
                recent_adaptations=stats.get('recent_adaptations')
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
    
    async def _control_adaptation(self, request: AdaptationRequest) -> AdaptationResponse:
        """Control protocol adaptation."""
        try:
            if request.action == "reset":
                if self.protocol.adaptation:
                    self.protocol.adaptation.reset_adaptation()
                    result = {"message": "Adaptation parameters reset to defaults"}
                else:
                    result = {"message": "Adaptation not enabled"}
                
            elif request.action == "export":
                if not request.filepath:
                    raise HTTPException(status_code=400, detail="Filepath required for export action")
                
                if self.protocol.adaptation:
                    self.protocol.adaptation.export_adaptation_data(request.filepath)
                    result = {"message": f"Adaptation data exported to {request.filepath}"}
                else:
                    result = {"message": "Adaptation not enabled"}
                
            elif request.action == "get_params":
                if self.protocol.adaptation:
                    result = self.protocol.adaptation.get_current_params()
                else:
                    result = {"message": "Adaptation not enabled"}
                
            else:
                raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
            
            return AdaptationResponse(
                success=True,
                action=request.action,
                result=result
            )
            
        except Exception as e:
            return AdaptationResponse(
                success=False,
                action=request.action,
                error=str(e)
            )
    
    async def _reset_session(self) -> Dict[str, Any]:
        """Reset the current session."""
        try:
            self.protocol.reset_session()
            return {"message": "Session reset successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reset session: {str(e)}")
    
    async def _get_info(self) -> Dict[str, Any]:
        """Get basic protocol information."""
        return {
            "name": "AetherSpeak",
            "version": self.protocol.protocol_version,
            "description": "An adaptive language protocol for efficient AI-to-AI communication",
            "features": [
                "Neural embedding compression",
                "Symbolic logic extraction",
                "Self-adapting protocol",
                "Performance metrics",
                "Batch processing"
            ],
            "model_info": self.protocol.encoder.embedding_engine.get_model_info()
        }
    
    def run(self, **kwargs):
        """Run the API server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            **kwargs
        )
    
    def get_app(self):
        """Get the FastAPI app instance."""
        return self.app


# Convenience function to create and run the server
def create_server(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim: int = 384,
    max_length: int = 512,
    device: str = None,
    enable_adaptation: bool = True,
    enable_metrics: bool = True,
    host: str = "0.0.0.0",
    port: int = 8000
) -> AetherSpeakAPI:
    """
    Create and configure an AetherSpeak API server.
    
    Args:
        model_name: Pre-trained model name
        embedding_dim: Embedding dimension
        max_length: Maximum sequence length
        device: Device to use ('cpu' or 'cuda')
        enable_adaptation: Whether to enable protocol adaptation
        enable_metrics: Whether to enable performance metrics
        host: Host to bind to
        port: Port to bind to
        
    Returns:
        Configured API server instance
    """
    return AetherSpeakAPI(
        model_name=model_name,
        embedding_dim=embedding_dim,
        max_length=max_length,
        device=device,
        enable_adaptation=enable_adaptation,
        enable_metrics=enable_metrics,
        host=host,
        port=port
    )
