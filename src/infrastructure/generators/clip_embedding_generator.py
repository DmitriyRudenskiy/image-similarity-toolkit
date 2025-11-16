"""
CLIP Embedding Generator
========================

CLIP-based embedding generator implementation.
"""

import torch
import numpy as np
from PIL import Image as PILImage
from typing import List, Optional
from datetime import datetime

from .embedding_generator import BaseEmbeddingGenerator
from ....domain.vector_storage.vector_embedding import VectorEmbedding
from ....domain.image_processing.image_processing import Image
from ....domain.configuration.model_configuration import ModelConfiguration


class CLIPEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    CLIP-based embedding generator.
    
    Uses OpenAI's CLIP model to generate image embeddings.
    Supports both image and text embeddings with shared representation space.
    
    Example:
        >>> generator = CLIPEmbeddingGenerator()
        >>> config = ModelConfiguration.clip_vit_b32()
        >>> embedding = generator.generate_embedding(image, config)
    """
    
    def __init__(self):
        """Initialize CLIP embedding generator."""
        super().__init__()
        self.embedding_size = 512  # CLIP ViT-B/32 default size
    
    def supports_model(self, model_config: ModelConfiguration) -> bool:
        """Check if CLIP model is supported."""
        return model_config.model_name.startswith("clip")
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported model names."""
        return ["clip_vit_b32", "clip_vit_l14", "clip_custom"]
    
    def get_model_info(self, model_name: str) -> dict:
        """Get CLIP model information."""
        model_info = {
            "clip_vit_b32": {
                "supported": True,
                "generator_type": "CLIP ViT-B/32",
                "embedding_size": 512,
                "description": "CLIP Vision Transformer with 32x32 patches",
                "architecture": "Vision Transformer",
                "parameters": "~151M",
                "input_resolution": 224,
                "accuracy": "~67.0% top-1 ImageNet"
            },
            "clip_vit_l14": {
                "supported": True,
                "generator_type": "CLIP ViT-L/14",
                "embedding_size": 768,
                "description": "CLIP Vision Transformer with 14x14 patches",
                "architecture": "Vision Transformer",
                "parameters": "~428M",
                "input_resolution": 224,
                "accuracy": "~75.4% top-1 ImageNet"
            }
        }
        
        base_info = {
            "model_name": model_name,
            "generator_type": "CLIP",
            "description": "OpenAI CLIP model for vision-language understanding",
            "supports_text": True,
            "supports_image": True,
            "multimodal": True
        }
        
        if model_name in model_info:
            base_info.update(model_info[model_name])
        else:
            base_info.update({
                "supported": False,
                "embedding_size": 512,
                "architecture": "Custom CLIP"
            })
        
        return base_info
    
    def _load_clip_model(self, model_config: ModelConfiguration) -> tuple[any, any]:
        """Load CLIP model."""
        try:
            import clip
        except ImportError as e:
            raise ImportError(
                "CLIP is not installed. Install with:\n"
                "pip install git+https://github.com/openai/CLIP.git"
            ) from e
        
        # Determine model name
        if model_config.model_name == "clip_vit_l14":
            clip_model_name = "ViT-L/14"
            self.embedding_size = 768
        else:  # Default to ViT-B/32
            clip_model_name = "ViT-B/32"
            self.embedding_size = 512
        
        # Load model and preprocessing
        model, preprocess = clip.load(clip_model_name, device=self.device)
        model.eval()
        
        return model, preprocess
    
    def generate_embedding(
        self, 
        image: Image, 
        model_config: Optional[ModelConfiguration] = None
    ) -> VectorEmbedding:
        """
        Generate embedding vector for an image using CLIP.
        
        Args:
            image: Image to generate embedding for
            model_config: CLIP model configuration
            
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
                raise ValueError(f"Model {model_config.model_name} is not supported by CLIPEmbeddingGenerator")
            
            # Load or get cached model
            cached_model = self._get_cached_model(model_config)
            if cached_model:
                model, preprocess = cached_model
            else:
                model, preprocess = self._load_clip_model(model_config)
                self._cache_model(model_config, (model, preprocess))
            
            # Load and preprocess image
            pil_image = PILImage.open(image.path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Apply CLIP preprocessing
            image_input = preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                # Get image features
                image_features = model.encode_image(image_input)
                
                # Convert to numpy and normalize
                embedding_np = image_features.cpu().numpy().squeeze()
                
                # CLIP embeddings are already normalized, but we can re-normalize
                if model_config.normalize_embeddings:
                    norm = np.linalg.norm(embedding_np)
                    if norm > 0:
                        embedding_np = embedding_np / norm
            
            return VectorEmbedding(
                vector=embedding_np,
                model_name=model_config.model_name,
                metadata={
                    'generator': 'CLIP',
                    'input_size': model_config.input_size,
                    'normalize': model_config.normalize_embeddings,
                    'device': self.device,
                    'multimodal': True,
                    'supports_text': True
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate CLIP embedding: {e}")
    
    def generate_text_embedding(
        self,
        text: str,
        model_config: Optional[ModelConfiguration] = None
    ) -> VectorEmbedding:
        """
        Generate embedding vector for text using CLIP.
        
        Args:
            text: Text to generate embedding for
            model_config: CLIP model configuration
            
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
                raise ValueError(f"Model {model_config.model_name} is not supported by CLIPEmbeddingGenerator")
            
            # Load or get cached model
            cached_model = self._get_cached_model(model_config)
            if cached_model:
                model, preprocess = cached_model
            else:
                model, preprocess = self._load_clip_model(model_config)
                self._cache_model(model_config, (model, preprocess))
            
            # Process text
            with torch.no_grad():
                # Tokenize and encode text
                text_tokens = model.encode_text(model.tokenize([text]).to(self.device))
                
                # Convert to numpy and normalize
                embedding_np = text_tokens.cpu().numpy().squeeze()
                
                # CLIP embeddings are already normalized, but we can re-normalize
                if model_config.normalize_embeddings:
                    norm = np.linalg.norm(embedding_np)
                    if norm > 0:
                        embedding_np = embedding_np / norm
            
            return VectorEmbedding(
                vector=embedding_np,
                model_name=model_config.model_name,
                metadata={
                    'generator': 'CLIP',
                    'text_input': text,
                    'normalize': model_config.normalize_embeddings,
                    'device': self.device,
                    'multimodal': True,
                    'supports_image': True
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate CLIP text embedding: {e}")
    
    def generate_batch_embeddings(
        self, 
        images: List[Image], 
        model_config: Optional[ModelConfiguration] = None
    ) -> List[VectorEmbedding]:
        """
        Generate embeddings for multiple images using CLIP.
        
        Args:
            images: List of images to generate embeddings for
            model_config: CLIP model configuration
            
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
                raise ValueError(f"Model {model_config.model_name} is not supported by CLIPEmbeddingGenerator")
            
            # Load or get cached model
            cached_model = self._get_cached_model(model_config)
            if cached_model:
                model, preprocess = cached_model
            else:
                model, preprocess = self._load_clip_model(model_config)
                self._cache_model(model_config, (model, preprocess))
            
            embeddings = []
            
            # Process images in batches for efficiency
            batch_size = 32
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                
                # Prepare batch input
                valid_images = []
                image_inputs = []
                
                for img in batch_images:
                    try:
                        pil_image = PILImage.open(img.path)
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        # Convert to tensor using CLIP preprocessing
                        image_input = preprocess(pil_image).to(self.device)
                        image_inputs.append(image_input)
                        valid_images.append(img)
                        
                    except Exception as e:
                        print(f"Warning: Failed to process image {img.path}: {e}")
                        continue
                
                if not image_inputs:
                    continue
                
                # Stack tensors into batch
                batch_input = torch.stack(image_inputs)
                
                # Generate batch embeddings
                with torch.no_grad():
                    batch_features = model.encode_image(batch_input)
                    
                    # Convert to numpy and normalize
                    embeddings_np = batch_features.cpu().numpy()
                    
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
                                    'generator': 'CLIP',
                                    'input_size': model_config.input_size,
                                    'normalize': model_config.normalize_embeddings,
                                    'device': self.device,
                                    'multimodal': True,
                                    'supports_text': True,
                                    'batch_index': i + j
                                },
                                created_at=datetime.now()
                            ))
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate CLIP batch embeddings: {e}")