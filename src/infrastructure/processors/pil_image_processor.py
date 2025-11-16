"""
PIL Image Processor
===================

Pillow-based implementation of ImageProcessor interface.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from PIL import Image as PILImage, ImageStat
from ....domain.image_processing.image_processing import Image
from ....application.interfaces.image_processor import ImageProcessorBase


class PILImageProcessor(ImageProcessorBase):
    """
    Pillow-based image processor.
    
    Provides image loading, validation, and preprocessing capabilities
    using the PIL/Pillow library for broad format support and reliability.
    
    Example:
        >>> processor = PILImageProcessor()
        >>> image = processor.load_image(Path('cat.jpg'))
        >>> info = processor.get_image_info(Path('cat.jpg'))
        >>> is_valid = processor.validate_image(Path('image.jpg'))
    """
    
    SUPPORTED_FORMATS = {
        'JPEG', 'JPG', 'PNG', 'BMP', 'GIF', 'TIFF', 'WEBP', 'ICO'
    }
    
    def __init__(self):
        """Initialize PIL image processor."""
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.ico'}
    
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
        # Check if file exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check if file has supported extension
        if image_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        try:
            # Open image with PIL
            pil_image = PILImage.open(image_path)
            
            # Get basic information
            dimensions = pil_image.size
            format_name = pil_image.format
            mode = pil_image.mode
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(image_path)
            
            # Get file size
            file_size = image_path.stat().st_size
            
            # Get additional metadata
            metadata = {
                'file_size': file_size,
                'mode': mode,
                'has_transparency': self._has_transparency(pil_image),
                'aspect_ratio': dimensions[0] / dimensions[1]
            }
            
            # Get image statistics
            try:
                stat = ImageStat.Stat(pil_image)
                metadata.update({
                    'mean_brightness': float(stat.mean[0]) if len(stat.mean) > 0 else None,
                    'std_brightness': float(stat.stddev[0]) if len(stat.stddev) > 0 else None
                })
            except Exception:
                # Image statistics might not be available for some formats
                pass
            
            return Image(
                path=image_path,
                dimensions=dimensions,
                format=format_name,
                file_hash=file_hash,
                metadata=metadata
            )
            
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    def validate_image(self, image_path: Path) -> bool:
        """
        Validate if a file is a valid image.
        
        Args:
            image_path: Path to the file
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            self.load_image(image_path)
            return True
        except Exception:
            return False
    
    def get_image_info(self, image_path: Path) -> Dict:
        """
        Get image information without loading the full image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image information
        """
        info = {
            'exists': image_path.exists(),
            'extension': image_path.suffix.lower(),
            'size': None,
            'dimensions': None,
            'format': None,
            'mode': None,
            'file_size': None,
            'file_hash': None
        }
        
        if not image_path.exists():
            return info
        
        try:
            # Get file size
            info['file_size'] = image_path.stat().st_size
            
            # Try to open with PIL for additional info
            pil_image = PILImage.open(image_path)
            info.update({
                'dimensions': pil_image.size,
                'format': pil_image.format,
                'mode': pil_image.mode,
                'has_transparency': self._has_transparency(pil_image)
            })
            
            # Calculate hash
            info['file_hash'] = self._calculate_file_hash(image_path)
            
            pil_image.close()
            
        except Exception:
            # If PIL fails, just return basic info
            pass
        
        return info
    
    def preprocess_image(self, image: Image) -> Image:
        """
        Preprocess image for model input.
        
        This implementation returns the image as-is, but subclasses can override
        for specific preprocessing requirements (resize, normalize, etc.).
        
        Args:
            image: Image to preprocess
            
        Returns:
            Preprocessed image
        """
        # Basic preprocessing: ensure RGB format for better compatibility
        try:
            pil_image = PILImage.open(image.path)
            
            if pil_image.mode != 'RGB':
                # Convert to RGB if necessary
                pil_image = pil_image.convert('RGB')
                
                # Update metadata
                updated_metadata = dict(image.metadata) if image.metadata else {}
                updated_metadata['original_mode'] = image.metadata.get('mode', pil_image.mode) if image.metadata else pil_image.mode
                updated_metadata['converted_to_rgb'] = True
                
                return Image(
                    path=image.path,
                    dimensions=image.dimensions,
                    format='RGB',
                    file_hash=image.file_hash,
                    metadata=updated_metadata
                )
            
            pil_image.close()
            
        except Exception:
            # If preprocessing fails, return original image
            pass
        
        return image
    
    def resize_image(
        self, 
        image: Image, 
        target_size: tuple[int, int], 
        maintain_aspect_ratio: bool = True
    ) -> PILImage.Image:
        """
        Resize an image to target size.
        
        Args:
            image: Image value object
            target_size: Target size (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized PIL Image
        """
        try:
            pil_image = PILImage.open(image.path)
            
            if maintain_aspect_ratio:
                # Calculate resize ratio
                ratio = min(target_size[0] / image.dimensions[0], target_size[1] / image.dimensions[1])
                new_size = (int(image.dimensions[0] * ratio), int(image.dimensions[1] * ratio))
                
                # Resize with high quality
                resized = pil_image.resize(new_size, PILImage.Resampling.LANCZOS)
            else:
                # Direct resize
                resized = pil_image.resize(target_size, PILImage.Resampling.LANCZOS)
            
            pil_image.close()
            return resized
            
        except Exception as e:
            raise RuntimeError(f"Failed to resize image {image.path}: {e}")
    
    def convert_to_thumbnail(
        self, 
        image: Image, 
        max_size: int = 256
    ) -> PILImage.Image:
        """
        Create a thumbnail of the image.
        
        Args:
            image: Image value object
            max_size: Maximum size for thumbnail
            
        Returns:
            Thumbnail PIL Image
        """
        try:
            pil_image = PILImage.open(image.path)
            
            # Create thumbnail
            pil_image.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
            
            return pil_image
            
        except Exception as e:
            raise RuntimeError(f"Failed to create thumbnail for {image.path}: {e}")
    
    def extract_exif_data(self, image_path: Path) -> Dict:
        """
        Extract EXIF data from image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with EXIF data
        """
        try:
            pil_image = PILImage.open(image_path)
            exif_data = {}
            
            # Get EXIF data if available
            if hasattr(pil_image, '_getexif') and pil_image._getexif():
                exif_dict = {}
                for tag_id, value in pil_image._getexif().items():
                    tag = pil_image.getexif().get_label(tag_id, tag_id)
                    exif_dict[tag] = value
                exif_data = exif_dict
            
            pil_image.close()
            return exif_data
            
        except Exception:
            # Return empty dict if EXIF extraction fails
            return {}
    
    def get_color_histogram(self, image: Image) -> Dict[str, list]:
        """
        Get color histogram of the image.
        
        Args:
            image: Image value object
            
        Returns:
            Dictionary with color histogram data
        """
        try:
            pil_image = PILImage.open(image.path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Calculate histogram
            histogram = pil_image.histogram()
            
            # Split into R, G, B channels
            histogram_dict = {
                'red': histogram[0:256],
                'green': histogram[256:512],
                'blue': histogram[512:768]
            }
            
            pil_image.close()
            return histogram_dict
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate histogram for {image.path}: {e}")
    
    def _calculate_file_hash(self, image_path: Path) -> str:
        """
        Calculate MD5 hash of the image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _has_transparency(self, pil_image: PILImage.Image) -> bool:
        """
        Check if image has transparency.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            True if image has transparency, False otherwise
        """
        if 'A' in pil_image.getbands():
            return True
        
        # Check for transparency in palette mode
        if pil_image.mode == 'P':
            transparency = pil_image.info.get('transparency')
            return transparency is not None
        
        return False