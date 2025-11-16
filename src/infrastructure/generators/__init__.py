"""
Embedding Generator Implementations
===================================

Infrastructure implementations of embedding generator interfaces.
"""

from .resnet_embedding_generator import ResNetEmbeddingGenerator
from .efficientnet_embedding_generator import EfficientNetEmbeddingGenerator
from .clip_embedding_generator import CLIPEmbeddingGenerator

__all__ = [
    'ResNetEmbeddingGenerator',
    'EfficientNetEmbeddingGenerator',
    'CLIPEmbeddingGenerator',
]