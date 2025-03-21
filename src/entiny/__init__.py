"""
Entiny - A high-performance Python package for Information-Based Optimal Subdata Selection (IBOSS)
"""

__version__ = "0.1.0"

from .cli import cli
from .core import entiny

__all__ = ["cli", "entiny"]
