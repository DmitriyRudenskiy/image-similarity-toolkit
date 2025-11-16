"""
Model loading and management module.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, 
    ResNet50_Weights,
    efficientnet_b0, 
    EfficientNet_B0_Weights
)
from typing import Tuple, Any


def load_model(model_name: str, device: str) -> Tuple[nn.Module, Any]:
    """
    Load a pre-trained model for image similarity comparison.
    
    Args:
        model_name: Name of the model ('resnet', 'efficientnet', or 'clip')
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, preprocessing_transform)
        
    Raises:
        ValueError: If an unsupported model name is provided
        ImportError: If CLIP is selected but not installed
        
    Example:
        >>> model, preprocess = load_model('efficientnet', 'cuda')
    """
    if model_name == 'resnet':
        return _load_resnet50(device)
    elif model_name == 'efficientnet':
        return _load_efficientnet_b0(device)
    elif model_name == 'clip':
        return _load_clip(device)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            "Choose from: 'resnet', 'efficientnet', 'clip'"
        )


def _load_resnet50(device: str) -> Tuple[nn.Module, Any]:
    """
    Load ResNet50 model.
    
    Args:
        device: Device to load the model on
        
    Returns:
        Tuple of (model, preprocessing_transform)
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    # Remove the classification head
    model = nn.Sequential(*(list(model.children())[:-1]))
    
    preprocess = weights.transforms()
    
    model = model.to(device)
    model.eval()
    
    return model, preprocess


def _load_efficientnet_b0(device: str) -> Tuple[nn.Module, Any]:
    """
    Load EfficientNet-B0 model.
    
    Args:
        device: Device to load the model on
        
    Returns:
        Tuple of (model, preprocessing_transform)
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    
    # Remove the classification head
    model = nn.Sequential(*(list(model.children())[:-1]))
    
    preprocess = weights.transforms()
    
    model = model.to(device)
    model.eval()
    
    return model, preprocess


def _load_clip(device: str) -> Tuple[Any, Any]:
    """
    Load CLIP model from OpenAI.
    
    Args:
        device: Device to load the model on
        
    Returns:
        Tuple of (model, preprocessing_transform)
        
    Raises:
        ImportError: If CLIP is not installed
    """
    try:
        import clip
    except ImportError:
        raise ImportError(
            "CLIP is not installed. Install it with:\n"
            "pip install git+https://github.com/openai/CLIP.git"
        )
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    return model, preprocess


def get_embedding_size(model_name: str) -> int:
    """
    Get the embedding size for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Size of the embedding vector
    """
    embedding_sizes = {
        'resnet': 2048,
        'efficientnet': 1280,
        'clip': 512
    }
    
    return embedding_sizes.get(model_name, 0)
