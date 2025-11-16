"""
Image Similarity Toolkit
========================

A professional toolkit for image similarity comparison using deep learning models.

Supported models:
- ResNet50
- EfficientNet-B0
- CLIP

Database backends:
- SQLite (traditional, embedded)
- ChromaDB (modern vector database)

Example usage:
    >>> from image_similarity import ImageSimilarity
    >>> checker = ImageSimilarity(model_name='efficientnet')
    >>> results = checker.compare_images('image1.jpg', 'image2.jpg')
    >>> print(f"Similarity: {results['cosine_similarity']:.4f}")
    
    >>> # Traditional SQLite backend
    >>> from image_similarity import EmbeddingDatabase
    >>> db = EmbeddingDatabase('embeddings.db')
    >>> db.add_image('image.jpg', embedding)
    
    >>> # Modern ChromaDB backend
    >>> from image_similarity import ChromaDBBackend
    >>> backend = ChromaDBBackend()
    >>> backend.add_image('image.jpg', embedding)
    >>> results = backend.find_similar(query_text="cat photo", top_k=5)
"""

__version__ = "0.2.1"
__author__ = "MiniMax Agent"
__email__ = "your.email@example.com"

from .core import ImageSimilarity
from .database import EmbeddingDatabase

# ChromaDB backend (optional import)
try:
    from .chromadb_backend import ChromaDBBackend
    __all__ = ["ImageSimilarity", "EmbeddingDatabase", "ChromaDBBackend"]
except ImportError:
    # ChromaDB dependencies not installed
    __all__ = ["ImageSimilarity", "EmbeddingDatabase"]
