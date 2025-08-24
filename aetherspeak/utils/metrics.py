"""
Performance metrics tracking for AetherSpeak protocol.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta


class ProtocolMetrics:
    """
    Tracks and analyzes performance metrics for the AetherSpeak protocol.
    
    Provides comprehensive monitoring of encoding/decoding performance,
    compression ratios, and system efficiency.
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize the metrics tracker.
        
        Args:
            history_size: Maximum number of historical records to keep
        """
        self.history_size = history_size
        
        # Performance metrics
        self.encoding_metrics = {
            'count': 0,
            'total_time': 0.0,
            'total_input_length': 0,
            'total_output_length': 0,
            'times': deque(maxlen=history_size),
            'input_lengths': deque(maxlen=history_size),
            'output_lengths': deque(maxlen=history_size),
            'compression_ratios': deque(maxlen=history_size)
        }
        
        self.decoding_metrics = {
            'count': 0,
            'total_time': 0.0,
            'total_input_length': 0,
            'total_output_length': 0,
            'times': deque(maxlen=history_size),
            'input_lengths': deque(maxlen=history_size),
            'output_lengths': deque(maxlen=history_size)
        }
        
        # System metrics
        self.system_metrics = {
            'start_time': datetime.now(),
            'last_reset': datetime.now(),
            'peak_memory': 0.0,
            'total_messages': 0
        }
        
        # Error tracking
        self.error_metrics = {
            'encoding_errors': 0,
            'decoding_errors': 0,
            'validation_errors': 0,
            'error_history': deque(maxlen=history_size)
        }
    
    def record_encoding(
        self, 
        message_length: int, 
        encoded_length: int, 
        compression_ratio: float, 
        encoding_time: float
    ) -> None:
        """
        Record encoding performance metrics.
        
        Args:
            message_length: Length of input message
            encoded_length: Length of encoded output
            compression_ratio: Achieved compression ratio
            encoding_time: Time taken for encoding
        """
        # Update counters
        self.encoding_metrics['count'] += 1
        self.encoding_metrics['total_time'] += encoding_time
        self.encoding_metrics['total_input_length'] += message_length
        self.encoding_metrics['total_output_length'] += encoded_length
        
        # Store historical data
        self.encoding_metrics['times'].append(encoding_time)
        self.encoding_metrics['input_lengths'].append(message_length)
        self.encoding_metrics['output_lengths'].append(encoded_length)
        self.encoding_metrics['compression_ratios'].append(compression_ratio)
        
        # Update system metrics
        self.system_metrics['total_messages'] += 1
    
    def record_decoding(
        self, 
        encoded_length: int, 
        decoded_length: int, 
        decoding_time: float
    ) -> None:
        """
        Record decoding performance metrics.
        
        Args:
            encoded_length: Length of encoded input
            decoded_length: Length of decoded output
            decoding_time: Time taken for decoding
        """
        # Update counters
        self.decoding_metrics['count'] += 1
        self.decoding_metrics['total_time'] += decoding_time
        self.decoding_metrics['total_input_length'] += encoded_length
        self.decoding_metrics['total_output_length'] += decoded_length
        
        # Store historical data
        self.decoding_metrics['times'].append(decoding_time)
        self.decoding_metrics['input_lengths'].append(encoded_length)
        self.decoding_metrics['output_lengths'].append(decoded_length)
    
    def record_error(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an error occurrence.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Optional error context
        """
        if error_type == 'encoding':
            self.error_metrics['encoding_errors'] += 1
        elif error_type == 'decoding':
            self.error_metrics['decoding_errors'] += 1
        elif error_type == 'validation':
            self.error_metrics['validation_errors'] += 1
        
        # Store error details
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'context': context or {}
        }
        self.error_metrics['error_history'].append(error_record)
    
    def get_encoding_stats(self) -> Dict[str, Any]:
        """
        Get encoding performance statistics.
        
        Returns:
            Dictionary with encoding metrics
        """
        if self.encoding_metrics['count'] == 0:
            return {'count': 0}
        
        times = list(self.encoding_metrics['times'])
        input_lengths = list(self.encoding_metrics['input_lengths'])
        output_lengths = list(self.encoding_metrics['output_lengths'])
        compression_ratios = list(self.encoding_metrics['compression_ratios'])
        
        return {
            'count': self.encoding_metrics['count'],
            'total_time': self.encoding_metrics['total_time'],
            'avg_time': np.mean(times) if times else 0.0,
            'min_time': np.min(times) if times else 0.0,
            'max_time': np.max(times) if times else 0.0,
            'std_time': np.std(times) if times else 0.0,
            'total_input_length': self.encoding_metrics['total_input_length'],
            'total_output_length': self.encoding_metrics['total_output_length'],
            'avg_input_length': np.mean(input_lengths) if input_lengths else 0.0,
            'avg_output_length': np.mean(output_lengths) if output_lengths else 0.0,
            'avg_compression_ratio': np.mean(compression_ratios) if compression_ratios else 0.0,
            'best_compression_ratio': np.min(compression_ratios) if compression_ratios else 0.0,
            'worst_compression_ratio': np.max(compression_ratios) if compression_ratios else 0.0
        }
    
    def get_decoding_stats(self) -> Dict[str, Any]:
        """
        Get decoding performance statistics.
        
        Returns:
            Dictionary with decoding metrics
        """
        if self.decoding_metrics['count'] == 0:
            return {'count': 0}
        
        times = list(self.decoding_metrics['times'])
        input_lengths = list(self.decoding_metrics['input_lengths'])
        output_lengths = list(self.decoding_metrics['output_lengths'])
        
        return {
            'count': self.decoding_metrics['count'],
            'total_time': self.decoding_metrics['total_time'],
            'avg_time': np.mean(times) if times else 0.0,
            'min_time': np.min(times) if times else 0.0,
            'max_time': np.max(times) if times else 0.0,
            'std_time': np.std(times) if times else 0.0,
            'total_input_length': self.decoding_metrics['total_input_length'],
            'total_output_length': self.decoding_metrics['total_output_length'],
            'avg_input_length': np.mean(input_lengths) if input_lengths else 0.0,
            'avg_output_length': np.mean(output_lengths) if output_lengths else 0.0
        }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dictionary with error metrics
        """
        return {
            'total_errors': sum([
                self.error_metrics['encoding_errors'],
                self.error_metrics['decoding_errors'],
                self.error_metrics['validation_errors']
            ]),
            'encoding_errors': self.error_metrics['encoding_errors'],
            'decoding_errors': self.error_metrics['decoding_errors'],
            'validation_errors': self.error_metrics['validation_errors'],
            'error_rate': self._calculate_error_rate(),
            'recent_errors': list(self.error_metrics['error_history'])[-10:]  # Last 10 errors
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system-level statistics.
        
        Returns:
            Dictionary with system metrics
        """
        uptime = datetime.now() - self.system_metrics['start_time']
        time_since_reset = datetime.now() - self.system_metrics['last_reset']
        
        return {
            'start_time': self.system_metrics['start_time'].isoformat(),
            'uptime': str(uptime),
            'time_since_reset': str(time_since_reset),
            'total_messages': self.system_metrics['total_messages'],
            'messages_per_minute': self._calculate_messages_per_minute(),
            'peak_memory': self.system_metrics['peak_memory']
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            'encoding': self.get_encoding_stats(),
            'decoding': self.get_decoding_stats(),
            'errors': self.get_error_stats(),
            'system': self.get_system_stats(),
            'overall_performance': self._calculate_overall_performance()
        }
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        # Reset encoding metrics
        for key in self.encoding_metrics:
            if isinstance(self.encoding_metrics[key], (int, float)):
                self.encoding_metrics[key] = 0
            elif isinstance(self.encoding_metrics[key], deque):
                self.encoding_metrics[key].clear()
        
        # Reset decoding metrics
        for key in self.decoding_metrics:
            if isinstance(self.decoding_metrics[key], (int, float)):
                self.decoding_metrics[key] = 0
            elif isinstance(self.decoding_metrics[key], deque):
                self.decoding_metrics[key].clear()
        
        # Reset error metrics
        for key in self.error_metrics:
            if isinstance(self.error_metrics[key], int):
                self.error_metrics[key] = 0
            elif isinstance(self.error_metrics[key], deque):
                self.error_metrics[key].clear()
        
        # Update system metrics
        self.system_metrics['last_reset'] = datetime.now()
    
    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate."""
        total_operations = self.encoding_metrics['count'] + self.decoding_metrics['count']
        if total_operations == 0:
            return 0.0
        
        total_errors = sum([
            self.error_metrics['encoding_errors'],
            self.error_metrics['decoding_errors'],
            self.error_metrics['validation_errors']
        ])
        
        return total_errors / total_operations
    
    def _calculate_messages_per_minute(self) -> float:
        """Calculate messages processed per minute."""
        uptime = datetime.now() - self.system_metrics['start_time']
        uptime_minutes = uptime.total_seconds() / 60
        
        if uptime_minutes == 0:
            return 0.0
        
        return self.system_metrics['total_messages'] / uptime_minutes
    
    def _calculate_overall_performance(self) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        encoding_stats = self.get_encoding_stats()
        decoding_stats = self.get_decoding_stats()
        
        # Calculate efficiency metrics
        total_time = encoding_stats.get('total_time', 0) + decoding_stats.get('total_time', 0)
        total_messages = encoding_stats.get('count', 0) + decoding_stats.get('count', 0)
        
        avg_time_per_message = total_time / total_messages if total_messages > 0 else 0.0
        
        # Calculate compression efficiency
        avg_compression = encoding_stats.get('avg_compression_ratio', 1.0)
        compression_efficiency = (1.0 - avg_compression) * 100  # Percentage improvement
        
        return {
            'avg_time_per_message': avg_time_per_message,
            'compression_efficiency_percent': compression_efficiency,
            'total_throughput': total_messages / total_time if total_time > 0 else 0.0
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"ProtocolMetrics(encoding={self.encoding_metrics['count']}, decoding={self.decoding_metrics['count']})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"ProtocolMetrics(history_size={self.history_size}, "
                f"total_messages={self.system_metrics['total_messages']})")
