"""
Protocol adaptation utilities for AetherSpeak protocol.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json


class ProtocolAdaptation:
    """
    Handles self-adaptation of the AetherSpeak protocol based on performance feedback.
    
    Monitors protocol performance and automatically adjusts parameters
    to optimize compression, speed, and accuracy.
    """
    
    def __init__(self, adaptation_threshold: float = 0.1):
        """
        Initialize the adaptation engine.
        
        Args:
            adaptation_threshold: Threshold for triggering adaptations
        """
        self.adaptation_threshold = adaptation_threshold
        
        # Adaptation history
        self.adaptation_history = []
        
        # Performance baselines
        self.baseline_compression = 0.01  # Target 1% compression
        self.baseline_speed = 0.1  # Target 100ms per message
        
        # Adaptation parameters
        self.adaptation_params = {
            'embedding_dimension': 384,
            'compression_aggressiveness': 0.5,
            'symbolic_weight': 0.3,
            'neural_weight': 0.7
        }
        
        # Performance tracking
        self.performance_window = []
        self.window_size = 20
        
    def adapt(
        self, 
        original_message: str, 
        encoded_result: Dict[str, Any], 
        recent_ratios: List[float]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze performance and adapt protocol parameters if needed.
        
        Args:
            original_message: Original message
            encoded_result: Result of encoding
            recent_ratios: Recent compression ratios
            
        Returns:
            Adaptation result if adaptation occurred, None otherwise
        """
        # Update performance window
        self._update_performance_window(original_message, encoded_result)
        
        # Check if adaptation is needed
        if not self._should_adapt(recent_ratios):
            return None
        
        # Perform adaptation
        adaptation_result = self._perform_adaptation(recent_ratios)
        
        # Record adaptation
        self._record_adaptation(adaptation_result)
        
        return adaptation_result
    
    def _update_performance_window(
        self, 
        message: str, 
        result: Dict[str, Any]
    ) -> None:
        """
        Update the performance tracking window.
        
        Args:
            message: Input message
            result: Encoding result
        """
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'message_length': len(message),
            'encoded_length': len(result.get('tokens', [])),
            'compression_ratio': result.get('compression_ratio', 1.0),
            'symbolic_elements': len(result.get('symbolic_elements', {}))
        }
        
        self.performance_window.append(performance_record)
        
        # Maintain window size
        if len(self.performance_window) > self.window_size:
            self.performance_window.pop(0)
    
    def _should_adapt(self, recent_ratios: List[float]) -> bool:
        """
        Determine if adaptation is needed.
        
        Args:
            recent_ratios: Recent compression ratios
            
        Returns:
            True if adaptation should occur
        """
        if len(recent_ratios) < 5:
            return False
        
        # Check if performance is degrading
        recent_avg = np.mean(recent_ratios[-5:])
        baseline = self.baseline_compression
        
        # Adaptation needed if performance deviates significantly from baseline
        deviation = abs(recent_avg - baseline) / baseline
        
        return deviation > self.adaptation_threshold
    
    def _perform_adaptation(self, recent_ratios: List[float]) -> Dict[str, Any]:
        """
        Perform protocol adaptation.
        
        Args:
            recent_ratios: Recent compression ratios
            
        Returns:
            Adaptation result
        """
        recent_avg = np.mean(recent_ratios[-5:])
        baseline = self.baseline_compression
        
        adaptation_result = {
            'timestamp': datetime.now().isoformat(),
            'trigger': 'performance_deviation',
            'old_compression': recent_avg,
            'target_compression': baseline,
            'changes': {}
        }
        
        # Adjust compression aggressiveness
        if recent_avg > baseline:
            # Compression is worse than target, increase aggressiveness
            old_aggressive = self.adaptation_params['compression_aggressiveness']
            new_aggressive = min(old_aggressive * 1.2, 1.0)
            
            self.adaptation_params['compression_aggressiveness'] = new_aggressive
            adaptation_result['changes']['compression_aggressiveness'] = {
                'old': old_aggressive,
                'new': new_aggressive
            }
        
        # Adjust symbolic vs neural weights
        if len(self.performance_window) >= 10:
            symbolic_performance = self._analyze_symbolic_performance()
            neural_performance = self._analyze_neural_performance()
            
            if symbolic_performance > neural_performance:
                # Symbolic processing is more effective, increase its weight
                old_symbolic = self.adaptation_params['symbolic_weight']
                new_symbolic = min(old_symbolic * 1.1, 0.8)
                
                self.adaptation_params['symbolic_weight'] = new_symbolic
                self.adaptation_params['neural_weight'] = 1.0 - new_symbolic
                
                adaptation_result['changes']['symbolic_weight'] = {
                    'old': old_symbolic,
                    'new': new_symbolic
                }
        
        # Adjust embedding dimension if needed
        if len(self.performance_window) >= 15:
            avg_encoding_time = np.mean([
                record.get('encoding_time', 0) 
                for record in self.performance_window[-10:]
            ])
            
            if avg_encoding_time > self.baseline_speed:
                # Encoding is too slow, reduce embedding dimension
                old_dim = self.adaptation_params['embedding_dimension']
                new_dim = max(int(old_dim * 0.9), 256)
                
                self.adaptation_params['embedding_dimension'] = new_dim
                adaptation_result['changes']['embedding_dimension'] = {
                    'old': old_dim,
                    'new': new_dim
                }
        
        return adaptation_result
    
    def _analyze_symbolic_performance(self) -> float:
        """
        Analyze the effectiveness of symbolic processing.
        
        Returns:
            Symbolic performance score
        """
        if len(self.performance_window) < 5:
            return 0.5
        
        # Calculate correlation between symbolic elements and compression
        symbolic_counts = [r['symbolic_elements'] for r in self.performance_window[-10:]]
        compression_ratios = [r['compression_ratio'] for r in self.performance_window[-10:]]
        
        if len(set(symbolic_counts)) < 2:
            return 0.5
        
        try:
            correlation = np.corrcoef(symbolic_counts, compression_ratios)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.5
        except:
            return 0.5
    
    def _analyze_neural_performance(self) -> float:
        """
        Analyze the effectiveness of neural processing.
        
        Returns:
            Neural performance score
        """
        if len(self.performance_window) < 5:
            return 0.5
        
        # Calculate consistency of compression ratios
        compression_ratios = [r['compression_ratio'] for r in self.performance_window[-10:]]
        
        if len(set(compression_ratios)) < 2:
            return 0.5
        
        try:
            std_dev = np.std(compression_ratios)
            mean_ratio = np.mean(compression_ratios)
            
            # Lower standard deviation indicates more consistent performance
            consistency = 1.0 / (1.0 + std_dev)
            
            # Factor in how close to target
            target_proximity = 1.0 - abs(mean_ratio - self.baseline_compression)
            
            return (consistency + target_proximity) / 2.0
        except:
            return 0.5
    
    def _record_adaptation(self, adaptation_result: Dict[str, Any]) -> None:
        """
        Record an adaptation event.
        
        Args:
            adaptation_result: Result of adaptation
        """
        self.adaptation_history.append(adaptation_result)
        
        # Keep only recent adaptations
        if len(self.adaptation_history) > 50:
            self.adaptation_history = self.adaptation_history[-50:]
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """
        Get adaptation history.
        
        Returns:
            List of adaptation events
        """
        return self.adaptation_history.copy()
    
    def get_current_params(self) -> Dict[str, Any]:
        """
        Get current adaptation parameters.
        
        Returns:
            Current parameter values
        """
        return self.adaptation_params.copy()
    
    def reset_adaptation(self) -> None:
        """Reset adaptation parameters to defaults."""
        self.adaptation_params = {
            'embedding_dimension': 384,
            'compression_aggressiveness': 0.5,
            'symbolic_weight': 0.3,
            'neural_weight': 0.7
        }
        
        self.performance_window = []
        self.adaptation_history = []
    
    def export_adaptation_data(self, filepath: str) -> None:
        """
        Export adaptation data to a file.
        
        Args:
            filepath: Path to export file
        """
        export_data = {
            'adaptation_history': self.adaptation_history,
            'current_params': self.adaptation_params,
            'performance_window': self.performance_window,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def __str__(self) -> str:
        """String representation."""
        return f"ProtocolAdaptation(threshold={self.adaptation_threshold})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"ProtocolAdaptation(adaptation_threshold={self.adaptation_threshold}, "
                f"adaptations={len(self.adaptation_history)})")
