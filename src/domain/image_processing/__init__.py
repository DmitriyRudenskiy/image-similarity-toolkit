"""
Image Value Object
==================

Immutable value object representing an image in the domain.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL import Image as PILImage


@dataclass(frozen=True)
class Image:
    """
    Value Object representing an image.
    
    Immutable by design - once created, cannot be modified.
    
    Attributes:
        path: Path to the image file
        filename: Name of the image file
        dimensions: Tuple of (width, height)
        format: Image format (JPEG, PNG, etc.)
        file_hash: MD5 hash of the file content
        metadata: Additional image metadata
    """
    path: Path
    dimensions: tuple[int, int]
    format: str
    file_hash: str
    metadata: Optional[dict] = None
    
    @classmethod
    def from_file(cls, file_path: Path, metadata: Optional[dict] = None) -> "Image":
        """
        Create Image from file.
        
        Args:
            file_path: Path to image file
            metadata: Optional metadata dictionary
            
        Returns:
            Image instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid image
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Validate it's an image
        with PILImage.open(file_path) as img:
            width, height = img.size
            format_str = img.format or file_path.suffix.upper()
        
        # Calculate file hash
        file_hash = cls._calculate_file_hash(file_path)
        
        return cls(
            path=file_path,
            filename=file_path.name,
            dimensions=(width, height),
            format=format_str,
            file_hash=file_hash,
            metadata=metadata
        )
    
    @property
    def width(self) -> int:
        """Image width."""
        return self.dimensions[0]
    
    @property
    def height(self) -> int:
        """Image height."""
        return self.dimensions[1]
    
    @property
    def aspect_ratio(self) -> float:
        """Image aspect ratio (width / height)."""
        return self.width / self.height if self.height > 0 else 0.0
    
    @property
    def size_mb(self) -> float:
        """File size in megabytes."""
        return self.path.stat().st_size / (1024 * 1024)
    
    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """Calculate MD5 hash of file content."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def __str__(self) -> str:
        return f"Image({self.filename}, {self.dimensions}, {self.format})"
    
    def __repr__(self) -> str:
        return (
            f"Image(path={self.path}, "
            f"dimensions={self.dimensions}, "
            f"format={self.format}, "
            f"hash={self.file_hash[:8]}...)"
        )