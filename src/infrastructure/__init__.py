"""
Infrastructure Layer
====================

Infrastructure implementations for DDD components.
"""

from .repositories.vector_repository import (
    SQLiteVectorRepository,
    ChromaDBVectorRepository
)
from .generators.embedding_generator import (
    ResNetEmbeddingGenerator,
    EfficientNetEmbeddingGenerator,
    CLIPEmbeddingGenerator
)
from .processors.image_processor import PILImageProcessor
from .dependency_injection.service_container import ServiceContainer

__all__ = [
    # Repositories
    'SQLiteVectorRepository',
    'ChromaDBVectorRepository',
    
    # Generators
    'ResNetEmbeddingGenerator',
    'EfficientNetEmbeddingGenerator',
    'CLIPEmbeddingGenerator',
    
    # Processors
    'PILImageProcessor',
    
    # Dependency Injection
    'ServiceContainer',
]