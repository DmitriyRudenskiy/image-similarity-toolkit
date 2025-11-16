"""
Image Processor Interface
========================

Interface for image processing operations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from ...domain.image_processing import Image


class ImageProcessor(Protocol):
    """
    Protocol for image processing operations.
    
    Defines the interface that all image processor implementations must provide.
    """
    
    def load_image(self, image_path: Path) -> Image:
        """
        Load and validate an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image value object
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file is not a valid image
        """
        ...
    
    def validate_image(self, image_path: Path) -> bool:
        """
        Validate if a file is a valid image.
        
        Args:
            image_path: Path to the file
            
        Returns:
            True if valid image, False otherwise
        """
        ...
    
    def get_image_info(self, image_path: Path) -> dict:
        """
        Get image information without loading the full image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image information
        """
        ...
    
    def preprocess_image(self, image: Image) -> Image:
        """
        Preprocess image for model input.
        
        Args:
            image: Image to preprocess
            
        Returns:
            Preprocessed image
        """
        ...


class ImageProcessorBase(ABC):
    """
    Abstract base class for image processor implementations.
    
    Provides common functionality for image processing.
    """
    
    @abstractmethod
    def load_image(self, image_path: Path) -> Image:
        """Load and validate an image."""
        pass
    
    def validate_image(self, image_path: Path) -> bool:
        """Validate if a file is a valid image."""
        try:
            self.load_image(image_path)
            return True
        except Exception:
            return False
    
    @abstractmethod
    def get_image_info(self, image_path: Path) -> dict:
        """Get image information."""
        pass
    
    def preprocess_image(self, image: Image) -> Image:
        """Preprocess image for model input."""
        # Default implementation returns the image as-is
        # Subclasses should override for specific preprocessing
        return image