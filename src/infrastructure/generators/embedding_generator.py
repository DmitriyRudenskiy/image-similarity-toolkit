"""
Embedding Generator Base
========================

Base functionality for embedding generators.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ....domain.vector_storage.vector_embedding import VectorEmbedding
from ....domain.image_processing.image_processing import Image
from ....domain.configuration.model_configuration import ModelConfiguration


class BaseEmbeddingGenerator(ABC):
    """
    Base class for embedding generators.
    
    Provides common functionality for embedding generation.
    """
    
    def __init__(self):
        """Initialize the embedding generator."""
        self._model_cache = {}
        self.device = 'cuda' if self._check_cuda_available() else 'cpu'
    
    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
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
        from ....domain.configuration.model_configuration import ModelConfiguration as Config
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
    
    @abstractmethod
    def supports_model(self, model_config: ModelConfiguration) -> bool:
        """Check if model is supported."""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """Get supported model names."""
        pass
    
    @abstractmethod
    def get_model_info(self, model_name: str) -> dict:
        """Get model information."""
        pass