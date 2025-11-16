"""
Add Image Use Case
=================

Use case for adding images with embeddings to the system.
"""

import time
from typing import Dict, Any
from uuid import UUID
from pathlib import Path

from ...domain.image_processing import Image
from ...domain.vector_storage import VectorEmbedding, VectorStore
from ...domain.configuration import ModelConfiguration
from ...domain.similarity_search import SimilarityResult


class AddImageRequest:
    """
    Request object for adding an image.
    
    Attributes:
        image_path: Path to the image file
        model_config: Configuration for the model to use
        metadata: Additional metadata to store
        force_add: Whether to force add even if similar images exist
        similarity_threshold: Threshold for considering images similar
    """
    
    def __init__(
        self,
        image_path: Path,
        model_config: ModelConfiguration,
        metadata: Dict[str, Any] = None,
        force_add: bool = False,
        similarity_threshold: float = 0.95
    ):
        self.image_path = image_path
        self.model_config = model_config
        self.metadata = metadata or {}
        self.force_add = force_add
        self.similarity_threshold = similarity_threshold


class AddImageResponse:
    """
    Response object for adding an image.
    
    Attributes:
        embedding_id: Unique identifier of the added embedding
        image: The processed image object
        embedding: The generated embedding vector
        processing_time: Time taken to process the image
        warnings: List of warnings during processing
        similar_images: Similar images found (if any)
    """
    
    def __init__(
        self,
        embedding_id: UUID,
        image: Image,
        embedding: VectorEmbedding,
        processing_time: float,
        warnings: list = None,
        similar_images: list = None
    ):
        self.embedding_id = embedding_id
        self.image = image
        self.embedding = embedding
        self.processing_time = processing_time
        self.warnings = warnings or []
        self.similar_images = similar_images or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "embedding_id": str(self.embedding_id),
            "image_info": {
                "path": str(self.image.path),
                "filename": self.image.filename,
                "dimensions": self.image.dimensions,
                "format": self.image.format,
                "file_hash": self.image.file_hash,
                "size_mb": self.image.size_mb
            },
            "embedding_info": {
                "model": self.embedding.model_name,
                "dimension": self.embedding.dimension,
                "created_at": self.embedding.created_at.isoformat(),
                "norm": self.embedding.norm
            },
            "processing_time": self.processing_time,
            "warnings": self.warnings,
            "similar_images": [
                {
                    "image_path": str(img.path),
                    "similarity_score": sim_score
                }
                for img, sim_score in self.similar_images
            ]
        }


class AddImageUseCase:
    """
    Use case for adding images with embeddings to the system.
    
    Coordinates image processing, embedding generation, and storage.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        image_processor: "ImageProcessor",
        embedding_generator: "EmbeddingGenerator"
    ):
        """
        Initialize the use case.
        
        Args:
            vector_store: Vector storage aggregate root
            image_processor: Image processing service
            embedding_generator: Embedding generation service
        """
        self.vector_store = vector_store
        self.image_processor = image_processor
        self.embedding_generator = embedding_generator
    
    def execute(self, request: AddImageRequest) -> AddImageResponse:
        """
        Execute the add image use case.
        
        Args:
            request: Add image request
            
        Returns:
            Add image response
            
        Raises:
            ValueError: If image file doesn't exist or is invalid
            RuntimeError: If image processing fails
        """
        start_time = time.time()
        warnings = []
        
        # Step 1: Load and validate image
        try:
            image = self.image_processor.load_image(request.image_path)
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
        
        # Step 2: Generate embedding
        try:
            embedding = self.embedding_generator.generate_embedding(
                image, request.model_config
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
        
        # Step 3: Check for similar images (unless forced)
        similar_images = []
        if not request.force_add:
            try:
                similar_results = self.vector_store.find_similar_images(
                    embedding, limit=5, threshold=request.similarity_threshold
                )
                similar_images = [
                    (result["image"], result["similarity_score"])
                    for result in similar_results
                ]
                
                if similar_images:
                    warnings.append(
                        f"Found {len(similar_images)} similar images "
                        f"(similarity >= {request.similarity_threshold})"
                    )
            except Exception as e:
                warnings.append(f"Similarity check failed: {str(e)}")
        
        # Step 4: Add to vector store
        try:
            embedding_id = self.vector_store.add_image(image, embedding)
        except Exception as e:
            raise RuntimeError(f"Failed to store embedding: {str(e)}")
        
        # Step 5: Calculate processing time
        processing_time = time.time() - start_time
        
        return AddImageResponse(
            embedding_id=embedding_id,
            image=image,
            embedding=embedding,
            processing_time=processing_time,
            warnings=warnings,
            similar_images=similar_images
        )


# Placeholder interfaces for services
class ImageProcessor:
    """Placeholder for image processing service."""
    
    def load_image(self, image_path: Path) -> Image:
        """Load and validate an image."""
        # This would be implemented with actual image processing logic
        raise NotImplementedError()


class EmbeddingGenerator:
    """Placeholder for embedding generation service."""
    
    def generate_embedding(self, image: Image, config: ModelConfiguration) -> VectorEmbedding:
        """Generate embedding for an image."""
        # This would be implemented with actual ML model logic
        raise NotImplementedError()