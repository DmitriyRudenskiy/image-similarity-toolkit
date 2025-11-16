"""
Domain Layer
============

Contains the core business logic and rules of the Image Similarity domain.

Domain Bounded Contexts:
- ImageProcessing: Core image processing and embedding extraction
- VectorStorage: Vector representation storage and management
- SimilaritySearch: Semantic similarity search functionality
- DatabaseManagement: Database abstraction and management
- Configuration: Configuration management domain
"""

from .image_processing import *
from .vector_storage import *
from .similarity_search import *
from .database_management import *
from .configuration import *

__all__ = [
    # Image Processing
    "Image",
    "ImageProcessor",
    "EmbeddingExtractor",
    
    # Vector Storage
    "VectorEmbedding",
    "VectorRepository",
    "VectorStore",
    
    # Similarity Search
    "SimilarityQuery",
    "SimilarityResult",
    "SimilarityCalculator",
    "DuplicateDetector",
    
    # Database Management
    "DatabaseConnection",
    "DatabaseConfiguration",
    "Repository",
    
    # Configuration
    "Configuration",
    "ModelConfiguration",
]