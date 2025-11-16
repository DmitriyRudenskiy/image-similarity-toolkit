"""
Image Similarity Toolkit
========================

A professional toolkit for image similarity comparison using deep learning models.

Supported models:
- ResNet50
- EfficientNet-B0
- CLIP

Example usage:
    >>> from image_similarity import ImageSimilarity
    >>> checker = ImageSimilarity(model_name='efficientnet')
    >>> results = checker.compare_images('image1.jpg', 'image2.jpg')
    >>> print(f"Similarity: {results['cosine_similarity']:.4f}")
    
    >>> from image_similarity import EmbeddingDatabase
    >>> db = EmbeddingDatabase('embeddings.db')
    >>> db.add_image('image.jpg', embedding)
"""

__version__ = "0.2.0"
__author__ = "MiniMax Agent"
__email__ = "your.email@example.com"

from .core import ImageSimilarity
from .database import EmbeddingDatabase

__all__ = ["ImageSimilarity", "EmbeddingDatabase"]
