"""
AWEsomeScoring CLI package.

This package contains all command-line interface components.
"""

# Make parser easily accessible
from .parser import create_parser

__all__ = ['create_parser']
