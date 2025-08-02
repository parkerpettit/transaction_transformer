"""
Training module for feature prediction transformer.

Contains training scripts, trainers, and training utilities.
"""

# Import the main training function
from .pretrain import main

# Import model factory for external use

__all__ = [
    "main",
] 