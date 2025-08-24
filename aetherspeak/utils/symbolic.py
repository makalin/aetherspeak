"""
Symbolic logic processing utilities for AetherSpeak protocol.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from sympy import symbols, simplify, expand, solve, Eq
import json


class SymbolicProcessor:
    """
    Handles symbolic logic extraction and processing for the AetherSpeak protocol.
    
    Identifies and extracts logical structures, mathematical expressions,
    and symbolic patterns from natural language text.
    """
    
    def __init__(self):
        """Initialize the symbolic processor."""
        # Logical patterns
        self.logical_patterns = {
            'if_then': r'if\s+(.+?)\s+then\s+(.+)',
            'and_pattern': r'(.+?)\s+and\s+(.+)',
            'or_pattern': r'(.+?)\s+or\s+(.+)',
            'not_pattern': r'not\s+(.+)',
            'implies': r'(.+?)\s+implies\s+(.+)',
            'equivalent': r'(.+?)\s+is\s+equivalent\s+to\s+(.+)',
        }
        
        # Mathematical patterns
        self.math_patterns = {
            'equation': r'([a-zA-Z]\s*[+\-*/]\s*[a-zA-Z0-9]+\s*=\s*[a-zA-Z0-9]+)',
            'inequality': r'([a-zA-Z]\s*[<>≤≥]\s*[a-zA-Z0-9]+)',
            'function': r'([a-zA-Z]+\s*\([^)]+\))',
            'sum': r'sum\s+of\s+(.+)',
            'product': r'product\s+of\s+(.+)',
        }
        
        # Structural patterns
        self.structural_patterns = {
            'list': r'([0-9]+\.\s+.+?)(?=\n[0-9]+\.|\n*$)',
            'sequence': r'first\s+(.+?),\s+then\s+(.+?),\s+finally\s+(.+)',
            'hierarchy': r'(.+?)\s+contains\s+(.+)',
            'dependency': r'(.+?)\s+depends\s+on\s+(.+)',
        }
        
        # Initialize SymPy symbols for common variables
        self.common_symbols = {
            'x': symbols('x'),
            'y': symbols('y'),
            'z': symbols('z'),
            'n': symbols('n'),
            't': symbols('t'),
            'i': symbols('i'),
        }
    
    def extract_symbols(self, text: str) -> Dict[str, Any]:
        """
        Extract symbolic elements from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing extracted symbolic elements
        """
        symbols = {}
        
        # Extract logical structures
        logical_elements = self._extract_logical_structures(text)
        if logical_elements:
            symbols['logic'] = logical_elements
        
        # Extract mathematical expressions
        math_elements = self._extract_mathematical_expressions(text)
        if math_elements:
            symbols['math'] = math_elements
        
        # Extract structural patterns
        structural_elements = self._extract_structural_patterns(text)
        if structural_elements:
            symbols['structure'] = structural_elements
        
        return symbols
    
    def _extract_logical_structures(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract logical structures from text.
        
        Args:
            text: Input text
            
        Returns:
            List of logical structures
        """
        logical_structures = []
        
        for pattern_name, pattern in self.logical_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                structure = {
                    'type': pattern_name,
                    'pattern': pattern,
                    'groups': match.groups(),
                    'span': match.span(),
                    'text': match.group(0)
                }
                logical_structures.append(structure)
        
        return logical_structures
    
    def _extract_mathematical_expressions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical expressions from text.
        
        Args:
            text: Input text
            
        Returns:
            List of mathematical expressions
        """
        math_expressions = []
        
        for pattern_name, pattern in self.math_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                expression = {
                    'type': pattern_name,
                    'pattern': pattern,
                    'expression': match.group(0),
                    'span': match.span(),
                    'parsed': self._parse_math_expression(match.group(0))
                }
                math_expressions.append(expression)
        
        return math_expressions
    
    def _extract_structural_patterns(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract structural patterns from text.
        
        Args:
            text: Input text
            
        Returns:
            List of structural patterns
        """
        structural_patterns = []
        
        for pattern_name, pattern in self.structural_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                pattern_info = {
                    'type': pattern_name,
                    'pattern': pattern,
                    'groups': match.groups(),
                    'span': match.span(),
                    'text': match.group(0)
                }
                structural_patterns.append(pattern_info)
        
        return structural_patterns
    
    def _parse_math_expression(self, expression: str) -> Dict[str, Any]:
        """
        Parse mathematical expression using SymPy.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Parsed expression information
        """
        try:
            # Clean up expression
            clean_expr = re.sub(r'\s+', '', expression)
            
            # Try to parse as equation
            if '=' in clean_expr:
                left, right = clean_expr.split('=', 1)
                try:
                    left_sym = simplify(left)
                    right_sym = simplify(right)
                    equation = Eq(left_sym, right_sym)
                    
                    return {
                        'type': 'equation',
                        'left': str(left_sym),
                        'right': str(right_sym),
                        'equation': str(equation),
                        'variables': list(equation.free_symbols)
                    }
                except:
                    pass
            
            # Try to parse as expression
            try:
                parsed = simplify(clean_expr)
                return {
                    'type': 'expression',
                    'parsed': str(parsed),
                    'variables': list(parsed.free_symbols)
                }
            except:
                pass
            
        except Exception as e:
            pass
        
        # Return basic info if parsing fails
        return {
            'type': 'raw',
            'expression': expression,
            'parsing_error': True
        }
    
    def create_symbolic_representation(self, text: str) -> str:
        """
        Create a symbolic representation of the text.
        
        Args:
            text: Input text
            
        Returns:
            Symbolic representation string
        """
        symbols = self.extract_symbols(text)
        
        if not symbols:
            return "NO_SYMBOLS"
        
        representation_parts = []
        
        for symbol_type, elements in symbols.items():
            if symbol_type == 'logic':
                for elem in elements:
                    representation_parts.append(f"L:{elem['type']}:{len(elem['groups'])}")
            
            elif symbol_type == 'math':
                for elem in elements:
                    representation_parts.append(f"M:{elem['type']}:{elem['parsed'].get('type', 'unknown')}")
            
            elif symbol_type == 'structure':
                for elem in elements:
                    representation_parts.append(f"S:{elem['type']}:{len(elem['groups'])}")
        
        return "|".join(representation_parts) if representation_parts else "NO_SYMBOLS"
    
    def validate_symbolic_structure(self, symbolic_repr: str) -> bool:
        """
        Validate a symbolic representation.
        
        Args:
            symbolic_repr: Symbolic representation string
            
        Returns:
            True if valid, False otherwise
        """
        if not symbolic_repr or symbolic_repr == "NO_SYMBOLS":
            return True
        
        try:
            parts = symbolic_repr.split("|")
            for part in parts:
                if ":" not in part:
                    return False
                
                category, type_name, data = part.split(":", 2)
                if category not in ['L', 'M', 'S']:
                    return False
                
        except Exception:
            return False
        
        return True
    
    def get_symbolic_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about symbolic elements in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with symbolic statistics
        """
        symbols = self.extract_symbols(text)
        
        stats = {
            'total_symbols': 0,
            'logic_count': 0,
            'math_count': 0,
            'structure_count': 0,
            'symbolic_density': 0.0
        }
        
        for symbol_type, elements in symbols.items():
            count = len(elements)
            stats[f'{symbol_type}_count'] = count
            stats['total_symbols'] += count
        
        # Calculate symbolic density
        word_count = len(text.split())
        if word_count > 0:
            stats['symbolic_density'] = stats['total_symbols'] / word_count
        
        return stats
    
    def __str__(self) -> str:
        """String representation."""
        return "SymbolicProcessor()"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return "SymbolicProcessor()"
