"""
Embedding Generator Interface
============================

Interface for generating embeddings from images.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional

from ...domain.image_processing import Image
from ...domain.vector_storage import VectorEmbedding
from ...domain.configuration import ModelConfiguration


class EmbeddingGenerator(Protocol):
    """
    Protocol for embedding generation operations.
    
    Defines the interface that all embedding generator implementations must provide.
    """
    
    def generate_embedding(
        self, 
        image: Image, 
        model_config: Optional[ModelConfiguration] = None
    ) -> VectorEmbedding:
        """
        Generate embedding vector for an image.
        
        Args:
            image: Image to generate embedding for
            model_config: Optional model configuration
            
        Returns:
            VectorEmbedding object
            
        Raises:
            ValueError: If image is invalid
            RuntimeError: If embedding generation fails
        """
        ...
    
    def generate_batch_embeddings(
        self, 
        images: list[Image], 
        model_config: Optional[ModelConfiguration] = None
    ) -> list[VectorEmbedding]:
        """
        Generate embeddings for multiple images.
        
        Args:
            images: List of images to generate embeddings for
            model_config: Optional model configuration
            
        Returns:
            List of VectorEmbedding objects
        """
        ...
    
    def supports_model(self, model_config: ModelConfiguration) -> bool:
        """
        Check if the generator supports a specific model.
        
        Args:
            model_config: Model configuration to check
            
        Returns:
            True if supported, False otherwise
        """
        ...
    
    def get_supported_models(self) -> list[str]:
        """
        Get list of supported model names.
        
        Returns:
            List of supported model names
        """
        ...
    
    def get_model_info(self, model_name: str) -> dict:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        ...


class EmbeddingGeneratorBase(ABC):
    """
    Abstract base class for embedding generator implementations.
    
    Provides common functionality for embedding generation.
    """
    
    def __init__(self):
        """Initialize the embedding generator."""
        self._model_cache = {}
    
    @abstractmethod
    def generate_embedding(
        self, 
        image: Image, 
        model_config: Optional[ModelConfiguration] = None
    ) -> VectorEmbedding:
        """Generate embedding for an image."""
        pass
    
    def generate_batch_embeddings(
        self, 
        images: list[Image], 
        model_config: Optional[ModelConfiguration] = None
    ) -> list[VectorEmbedding]:
        """Generate embeddings for multiple images."""
        embeddings = []
        for image in images:
            embedding = self.generate_embedding(image, model_config)
            embeddings.append(embedding)
        return embeddings
    
    @abstractmethod
    def supports_model(self, model_config: ModelConfiguration) -> bool:
        """Check if model is supported."""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """Get supported model names."""
        pass
    
    def get_model_info(self, model_name: str) -> dict:
        """Get model information."""
        return {
            "model_name": model_name,
            "supported": model_name in self.get_supported_models(),
            "generator_type": self.__class__.__name__
        }
    
    def _get_model_from_config(
        self, 
        model_config: Optional[ModelConfiguration] = None
    ) -> ModelConfiguration:
        """
        Get model configuration with defaults.
        
        Args:
            model_config: Optional model configuration
            
        Returns:
            ModelConfiguration instance
        """
        if model_config:
            return model_config
        
        # Return default configuration
        from ...domain.configuration import ModelConfiguration as Config
        return Config.efficientnet_b0()
    
    def _cache_model(self, model_config: ModelConfiguration, model: any) -> None:
        """
        Cache a loaded model.
        
        Args:
            model_config: Model configuration
            model: Loaded model object
        """
        cache_key = f"{model_config.model_name}_{model_config.model_size}"
        self._model_cache[cache_key] = model
    
    def _get_cached_model(self, model_config: ModelConfiguration) -> Optional[any]:
        """
        Get cached model if available.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Cached model or None
        """
        cache_key = f"{model_config.model_name}_{model_config.model_size}"
        return self._model_cache.get(cache_key)
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache.clear()