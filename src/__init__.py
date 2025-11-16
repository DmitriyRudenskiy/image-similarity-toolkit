"""
Image Similarity Toolkit - DDD Architecture
==========================================

A professional toolkit for image similarity comparison using deep learning models.

DDD Structure:
- Domain Layer: Core business logic and rules
- Application Layer: Use cases and application services
- Infrastructure Layer: External dependencies and implementations
- Interfaces Layer: External API interfaces

Supported Models:
- ResNet50
- EfficientNet-B0  
- CLIP (Vision-Language)

Database Backends:
- SQLite (traditional, embedded)
- ChromaDB (modern vector database)
- PostgreSQL (network database)

Example Usage:
    
    # Load configuration
    from src_ddd.domain.configuration import Configuration
    config = Configuration.default()
    
    # Create infrastructure
    from src_ddd.infrastructure.database import SQLiteRepository
    repository = SQLiteRepository(config.database)
    vector_store = VectorStore(repository)
    
    # Process image
    from src_ddd.domain.image_processing import Image
    from src_ddd.application.use_cases import AddImageUseCase
    from src_ddd.application.interfaces import DefaultImageProcessor
    
    image = Image.from_file(Path("image.jpg"))
    use_case = AddImageUseCase(vector_store, DefaultImageProcessor(), EmbeddingGenerator())
    
    # Add image
    from src_ddd.application.use_cases.add_image_use_case import AddImageRequest
    request = AddImageRequest(image, ModelConfiguration.resnet50())
    response = use_case.execute(request)
    
    # Search similar images
    from src_ddd.application.use_cases.search_similar_images_use_case import SearchSimilarImagesRequest
    search_request = SearchSimilarImagesRequest.from_image(Path("query.jpg"))
    search_use_case = SearchSimilarImagesUseCase(vector_store, DefaultImageProcessor(), EmbeddingGenerator())
    search_response = search_use_case.execute(search_request)
    
    print(f"Found {search_response.total_found} similar images")
"""

__version__ = "0.3.0"
__author__ = "MiniMax Agent"
__email__ = "your.email@example.com"

# Domain Layer
from .domain import (
    # Image Processing
    Image,
    
    # Vector Storage
    VectorEmbedding,
    VectorRepository,
    VectorStore,
    
    # Similarity Search
    SimilarityQuery,
    SimilarityResult,
    SimilarityCalculator,
    DuplicateDetector,
    
    # Database Management
    DatabaseConnection,
    DatabaseConfiguration,
    Repository,
    
    # Configuration
    Configuration,
    ModelConfiguration,
)

# Application Layer
from .application.use_cases import (
    AddImageRequest,
    AddImageResponse,
    AddImageUseCase,
    SearchSimilarImagesRequest,
    SearchSimilarImagesResponse,
    SearchSimilarImagesUseCase,
    BatchProcessImagesRequest,
    BatchProcessImagesResponse,
    BatchProcessImagesUseCase,
    FindDuplicatesRequest,
    FindDuplicatesResponse,
    FindDuplicatesUseCase,
)

# Exception Classes
class ImageSimilarityError(Exception):
    """Base exception for image similarity toolkit."""
    pass


class ImageProcessingError(ImageSimilarityError):
    """Exception raised during image processing."""
    pass


class EmbeddingGenerationError(ImageSimilarityError):
    """Exception raised during embedding generation."""
    pass


class VectorStorageError(ImageSimilarityError):
    """Exception raised during vector storage operations."""
    pass


class SimilaritySearchError(ImageSimilarityError):
    """Exception raised during similarity search."""
    pass


class DuplicateDetectionError(ImageSimilarityError):
    """Exception raised during duplicate detection."""
    pass


class ConfigurationError(ImageSimilarityError):
    """Exception raised during configuration issues."""
    pass


# Define public API
__all__ = [
    # Domain Objects
    "Image",
    "VectorEmbedding",
    "VectorRepository",
    "VectorStore",
    "SimilarityQuery",
    "SimilarityResult",
    "SimilarityCalculator",
    "DuplicateDetector",
    "DatabaseConnection",
    "DatabaseConfiguration",
    "Repository",
    "Configuration",
    "ModelConfiguration",
    
    # Use Cases
    "AddImageRequest",
    "AddImageResponse",
    "AddImageUseCase",
    "SearchSimilarImagesRequest",
    "SearchSimilarImagesResponse",
    "SearchSimilarImagesUseCase",
    "BatchProcessImagesRequest",
    "BatchProcessImagesResponse",
    "BatchProcessImagesUseCase",
    "FindDuplicatesRequest",
    "FindDuplicatesResponse",
    "FindDuplicatesUseCase",
    
    # Exception Classes
    "ImageSimilarityError",
    "ImageProcessingError",
    "EmbeddingGenerationError",
    "VectorStorageError",
    "SimilaritySearchError",
    "DuplicateDetectionError",
    "ConfigurationError",
]