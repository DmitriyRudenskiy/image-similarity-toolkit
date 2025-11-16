"""
Model Configuration Value Object
===============================

Configuration for machine learning models used in image processing.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

from .configuration import ProcessingMode


class ModelType(Enum):
    """Supported model types."""
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"
    CLIP = "clip"
    CUSTOM = "custom"


class ModelSize(Enum):
    """Model size options."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True)
class ModelConfiguration:
    """
    Value Object representing model configuration.
    
    Immutable by design - once created, cannot be modified.
    
    Attributes:
        model_type: Type of model to use
        model_name: Specific model name/identifier
        model_size: Size of the model
        processing_mode: CPU/GPU processing mode
        device: Specific device to use
        batch_size: Batch size for inference
        max_sequence_length: Maximum sequence length (for text models)
        embedding_dimension: Output embedding dimension
        normalization: Whether to normalize embeddings
        cache_enabled: Whether to cache model outputs
        quantization: Model quantization settings
        custom_config: Custom model-specific configuration
        metadata: Additional metadata
    """
    model_type: ModelType
    model_name: str
    model_size: Optional[ModelSize] = None
    processing_mode: ProcessingMode = ProcessingMode.AUTO
    device: Optional[str] = None
    batch_size: int = 32
    max_sequence_length: Optional[int] = None
    embedding_dimension: Optional[int] = None
    normalization: bool = True
    cache_enabled: bool = True
    quantization: Optional[Dict[str, Any]] = None
    custom_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate model configuration after initialization."""
        self._validate_model_settings()
    
    @classmethod
    def resnet50(cls, **kwargs) -> "ModelConfiguration":
        """
        Create ResNet50 model configuration.
        
        Args:
            **kwargs: Additional configuration options
            
        Returns:
            ModelConfiguration instance
        """
        return cls(
            model_type=ModelType.RESNET,
            model_name="resnet50",
            model_size=ModelSize.LARGE,
            embedding_dimension=2048,
            **kwargs
        )
    
    @classmethod
    def efficientnet_b0(cls, **kwargs) -> "ModelConfiguration":
        """
        Create EfficientNet-B0 model configuration.
        
        Args:
            **kwargs: Additional configuration options
            
        Returns:
            ModelConfiguration instance
        """
        return cls(
            model_type=ModelType.EFFICIENTNET,
            model_name="efficientnet_b0",
            model_size=ModelSize.MEDIUM,
            embedding_dimension=1280,
            **kwargs
        )
    
    @classmethod
    def clip_vit_b32(cls, **kwargs) -> "ModelConfiguration":
        """
        Create CLIP ViT-B/32 model configuration.
        
        Args:
            **kwargs: Additional configuration options
            
        Returns:
            ModelConfiguration instance
        """
        return cls(
            model_type=ModelType.CLIP,
            model_name="clip_vit_b32",
            model_size=ModelSize.MEDIUM,
            embedding_dimension=512,
            max_sequence_length=77,
            **kwargs
        )
    
    @classmethod
    def clip_vit_l14(cls, **kwargs) -> "ModelConfiguration":
        """
        Create CLIP ViT-L/14 model configuration.
        
        Args:
            **kwargs: Additional configuration options
            
        Returns:
            ModelConfiguration instance
        """
        return cls(
            model_type=ModelType.CLIP,
            model_name="clip_vit_l14",
            model_size=ModelSize.LARGE,
            embedding_dimension=768,
            max_sequence_length=77,
            **kwargs
        )
    
    def _validate_model_settings(self) -> None:
        """Validate model-specific settings."""
        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Validate max sequence length for text models
        if self.model_type == ModelType.CLIP and self.max_sequence_length is None:
            raise ValueError("CLIP models require max_sequence_length")
        
        # Validate embedding dimension
        if self.embedding_dimension and self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        
        # Validate quantization settings
        if self.quantization:
            valid_quantization_types = ["int8", "int4", "dynamic"]
            quant_type = self.quantization.get("type")
            if quant_type and quant_type not in valid_quantization_types:
                raise ValueError(f"Invalid quantization type: {quant_type}")
    
    @property
    def is_image_model(self) -> bool:
        """Check if model is primarily for image processing."""
        return self.model_type in [ModelType.RESNET, ModelType.EFFICIENTNET]
    
    @property
    def is_multimodal(self) -> bool:
        """Check if model supports multiple modalities (e.g., CLIP)."""
        return self.model_type == ModelType.CLIP
    
    @property
    def supports_text(self) -> bool:
        """Check if model supports text processing."""
        return self.model_type == ModelType.CLIP
    
    @property
    def supports_image(self) -> bool:
        """Check if model supports image processing."""
        return self.model_type in [ModelType.RESNET, ModelType.EFFICIENTNET, ModelType.CLIP]
    
    @property
    def memory_requirement_mb(self) -> int:
        """
        Estimate memory requirement in MB.
        
        Returns:
            Estimated memory usage in megabytes
        """
        # Rough estimates based on model type and size
        if self.model_type == ModelType.RESNET:
            if self.model_size == ModelSize.SMALL:
                return 50
            elif self.model_size == ModelSize.MEDIUM:
                return 100
            else:  # LARGE
                return 200
        elif self.model_type == ModelType.EFFICIENTNET:
            if self.model_size == ModelSize.SMALL:
                return 30
            elif self.model_size == ModelSize.MEDIUM:
                return 50
            else:  # LARGE
                return 100
        elif self.model_type == ModelType.CLIP:
            if self.model_size == ModelSize.SMALL:
                return 200
            elif self.model_size == ModelSize.MEDIUM:
                return 400
            else:  # LARGE
                return 800
        else:  # CUSTOM
            return 100  # Default estimate
    
    @property
    def expected_throughput_images_per_second(self) -> float:
        """
        Estimate processing throughput.
        
        Returns:
            Expected images per second processing rate
        """
        # Rough estimates based on model complexity and device
        base_throughput = 1.0
        
        if self.model_type == ModelType.RESNET:
            if self.device == "cuda":
                base_throughput = 100.0
            else:
                base_throughput = 10.0
        elif self.model_type == ModelType.EFFICIENTNET:
            if self.device == "cuda":
                base_throughput = 80.0
            else:
                base_throughput = 8.0
        elif self.model_type == ModelType.CLIP:
            if self.device == "cuda":
                base_throughput = 20.0
            else:
                base_throughput = 2.0
        
        # Adjust for batch size
        adjusted_throughput = base_throughput * (self.batch_size / 32)
        
        # Adjust for model size
        if self.model_size == ModelSize.SMALL:
            adjusted_throughput *= 2.0
        elif self.model_size == ModelSize.LARGE:
            adjusted_throughput *= 0.5
        
        return adjusted_throughput
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "model_size": self.model_size.value if self.model_size else None,
            "processing_mode": self.processing_mode.value,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_sequence_length": self.max_sequence_length,
            "embedding_dimension": self.embedding_dimension,
            "normalization": self.normalization,
            "cache_enabled": self.cache_enabled,
            "supports_image": self.supports_image,
            "supports_text": self.supports_text,
            "is_multimodal": self.is_multimodal,
            "memory_requirement_mb": self.memory_requirement_mb,
            "expected_throughput": self.expected_throughput_images_per_second
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model configuration to dictionary."""
        result = {
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "model_size": self.model_size.value if self.model_size else None,
            "processing_mode": self.processing_mode.value,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_sequence_length": self.max_sequence_length,
            "embedding_dimension": self.embedding_dimension,
            "normalization": self.normalization,
            "cache_enabled": self.cache_enabled,
            "quantization": self.quantization,
            "custom_config": self.custom_config,
            "metadata": self.metadata
        }
        return result
    
    def __str__(self) -> str:
        return (
            f"ModelConfiguration(type={self.model_type.value}, "
            f"name={self.model_name}, "
            f"size={self.model_size.value if self.model_size else 'unknown'})"
        )
    
    def __repr__(self) -> str:
        return (
            f"ModelConfiguration("
            f"model_type={self.model_type}, "
            f"model_name={self.model_name}, "
            f"device={self.device}, "
            f"batch_size={self.batch_size}"
            f")"
        )