"""
EfficientNet Embedding Generator
================================

EfficientNet-B0-based embedding generator implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image as PILImage
from typing import List, Optional
from datetime import datetime

from .embedding_generator import BaseEmbeddingGenerator
from ....domain.vector_storage.vector_embedding import VectorEmbedding
from ....domain.image_processing.image_processing import Image
from ....domain.configuration.model_configuration import ModelConfiguration


class EfficientNetEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    EfficientNet-B0-based embedding generator.
    
    Uses EfficientNet-B0 pre-trained model to generate image embeddings.
    Provides good balance between accuracy and computational efficiency.
    
    Example:
        >>> generator = EfficientNetEmbeddingGenerator()
        >>> config = ModelConfiguration.efficientnet_b0()
        >>> embedding = generator.generate_embedding(image, config)
    """
    
    def __init__(self):
        """Initialize EfficientNet embedding generator."""
        super().__init__()
        self.embedding_size = 1280
    
    def supports_model(self, model_config: ModelConfiguration) -> bool:
        """Check if EfficientNet model is supported."""
        return model_config.model_name == "efficientnet_b0"
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported model names."""
        return ["efficientnet_b0"]
    
    def get_model_info(self, model_name: str) -> dict:
        """Get EfficientNet model information."""
        return {
            "model_name": model_name,
            "supported": model_name == "efficientnet_b0",
            "generator_type": "EfficientNet-B0",
            "embedding_size": self.embedding_size,
            "description": "EfficientNet-B0 pre-trained model with classification head removed",
            "architecture": "Mobile Inverted Bottleneck Convolution",
            "parameters": "~5.3M",
            "flops": "~0.4G",
            "accuracy": "~77.1% top-1 ImageNet"
        }
    
    def _load_efficientnet_model(self, model_config: ModelConfiguration) -> tuple[nn.Module, any]:
        """Load EfficientNet-B0 model."""
        try:
            from torchvision.models import (
                efficientnet_b0, 
                EfficientNet_B0_Weights
            )
        except ImportError as e:
            raise ImportError(
                "torchvision is not installed. Install with:\n"
                "pip install torchvision"
            ) from e
        
        # Load pre-trained weights
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        
        # Remove the classification head to get feature representations
        model = nn.Sequential(*(list(model.children())[:-1]))
        
        # Load preprocessing transforms
        preprocess = weights.transforms()
        
        model = model.to(self.device)
        model.eval()
        
        return model, preprocess
    
    def generate_embedding(
        self, 
        image: Image, 
        model_config: Optional[ModelConfiguration] = None
    ) -> VectorEmbedding:
        """
        Generate embedding vector for an image using EfficientNet-B0.
        
        Args:
            image: Image to generate embedding for
            model_config: EfficientNet-B0 model configuration
            
        Returns:
            VectorEmbedding object
            
        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If embedding generation fails
        """
        try:
            # Validate and get model configuration
            model_config = self._get_model_from_config(model_config)
            
            if not self.supports_model(model_config):
                raise ValueError(f"Model {model_config.model_name} is not supported by EfficientNetEmbeddingGenerator")
            
            # Load or get cached model
            cached_model = self._get_cached_model(model_config)
            if cached_model:
                model, preprocess = cached_model
            else:
                model, preprocess = self._load_efficientnet_model(model_config)
                self._cache_model(model_config, (model, preprocess))
            
            # Load and preprocess image
            pil_image = PILImage.open(image.path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Apply preprocessing transforms
            input_tensor = preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = model(input_tensor)
                
                # Squeeze to remove batch dimension
                embedding = embedding.squeeze()
                
                # Convert to numpy and normalize
                embedding_np = embedding.cpu().numpy()
                
                # Optional: apply L2 normalization
                if model_config.normalize_embeddings:
                    norm = np.linalg.norm(embedding_np)
                    if norm > 0:
                        embedding_np = embedding_np / norm
            
            return VectorEmbedding(
                vector=embedding_np,
                model_name=model_config.model_name,
                metadata={
                    'generator': 'EfficientNet-B0',
                    'input_size': model_config.input_size,
                    'normalize': model_config.normalize_embeddings,
                    'device': self.device
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate EfficientNet embedding: {e}")
    
    def generate_batch_embeddings(
        self, 
        images: List[Image], 
        model_config: Optional[ModelConfiguration] = None
    ) -> List[VectorEmbedding]:
        """
        Generate embeddings for multiple images using EfficientNet-B0.
        
        Args:
            images: List of images to generate embeddings for
            model_config: EfficientNet-B0 model configuration
            
        Returns:
            List of VectorEmbedding objects
            
        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If batch processing fails
        """
        try:
            # Validate and get model configuration
            model_config = self._get_model_from_config(model_config)
            
            if not self.supports_model(model_config):
                raise ValueError(f"Model {model_config.model_name} is not supported by EfficientNetEmbeddingGenerator")
            
            # Load or get cached model
            cached_model = self._get_cached_model(model_config)
            if cached_model:
                model, preprocess = cached_model
            else:
                model, preprocess = self._load_efficientnet_model(model_config)
                self._cache_model(model_config, (model, preprocess))
            
            embeddings = []
            
            # Process images in batches for efficiency
            batch_size = 32
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                
                # Prepare batch input
                batch_tensors = []
                valid_images = []
                
                for img in batch_images:
                    try:
                        pil_image = PILImage.open(img.path)
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        tensor = preprocess(pil_image).to(self.device)
                        batch_tensors.append(tensor)
                        valid_images.append(img)
                        
                    except Exception as e:
                        print(f"Warning: Failed to process image {img.path}: {e}")
                        continue
                
                if not batch_tensors:
                    continue
                
                # Stack tensors into batch
                batch_input = torch.stack(batch_tensors)
                
                # Generate batch embeddings
                with torch.no_grad():
                    batch_embeddings = model(batch_input)
                    
                    # Squeeze to remove batch dimension and spatial dimensions
                    batch_embeddings = batch_embeddings.squeeze(-1).squeeze(-1)
                    
                    # Convert to numpy and normalize
                    embeddings_np = batch_embeddings.cpu().numpy()
                    
                    if model_config.normalize_embeddings:
                        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
                        norms[norms == 0] = 1  # Avoid division by zero
                        embeddings_np = embeddings_np / norms
                    
                    # Create VectorEmbedding objects
                    for j, embedding_np in enumerate(embeddings_np):
                        if j < len(valid_images):
                            embeddings.append(VectorEmbedding(
                                vector=embedding_np,
                                model_name=model_config.model_name,
                                metadata={
                                    'generator': 'EfficientNet-B0',
                                    'input_size': model_config.input_size,
                                    'normalize': model_config.normalize_embeddings,
                                    'device': self.device,
                                    'batch_index': i + j
                                },
                                created_at=datetime.now()
                            ))
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate EfficientNet batch embeddings: {e}")