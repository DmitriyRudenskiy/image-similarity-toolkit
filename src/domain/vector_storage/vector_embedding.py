"""
Vector Embedding Value Object
============================

Immutable representation of an image embedding vector.
"""

import numpy as np
from dataclasses import dataclass
from typing import Union, Dict, Optional
from datetime import datetime


@dataclass(frozen=True)
class VectorEmbedding:
    """
    Value Object representing an image embedding vector.
    
    Immutable by design - once created, cannot be modified.
    
    Attributes:
        vector: The embedding vector (numpy array)
        dimension: Vector dimensionality
        model_name: Name of the model used to generate the embedding
        created_at: Timestamp of creation
        metadata: Additional metadata
        normalization: Vector normalization status
    """
    vector: np.ndarray
    model_name: str
    created_at: datetime
    metadata: Optional[Dict] = None
    normalized: bool = False
    
    def __post_init__(self):
        """Validate embedding after initialization."""
        if not isinstance(self.vector, np.ndarray):
            raise TypeError("Vector must be a numpy array")
        
        if self.vector.ndim != 1:
            raise ValueError("Vector must be 1-dimensional")
        
        if len(self.vector) == 0:
            raise ValueError("Vector cannot be empty")
    
    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        return len(self.vector)
    
    @property
    def norm(self) -> float:
        """L2 norm of the vector."""
        return np.linalg.norm(self.vector)
    
    def cosine_similarity(self, other: "VectorEmbedding") -> float:
        """
        Calculate cosine similarity with another embedding.
        
        Args:
            other: Another VectorEmbedding to compare with
            
        Returns:
            Cosine similarity score between 0 and 1
            
        Raises:
            ValueError: If embeddings have different dimensions
            ValueError: If embeddings use different models
        """
        if self.dimension != other.dimension:
            raise ValueError(
                f"Cannot compare embeddings of different dimensions: "
                f"{self.dimension} vs {other.dimension}"
            )
        
        if self.model_name != other.model_name:
            raise ValueError(
                f"Cannot compare embeddings from different models: "
                f"{self.model_name} vs {other.model_name}"
            )
        
        # Ensure vectors are normalized for cosine similarity
        self_vec = self._get_normalized_vector()
        other_vec = other._get_normalized_vector()
        
        return float(np.dot(self_vec, other_vec))
    
    def euclidean_distance(self, other: "VectorEmbedding") -> float:
        """
        Calculate Euclidean distance to another embedding.
        
        Args:
            other: Another VectorEmbedding to compare with
            
        Returns:
            Euclidean distance
            
        Raises:
            ValueError: If embeddings have different dimensions
            ValueError: If embeddings use different models
        """
        if self.dimension != other.dimension:
            raise ValueError(
                f"Cannot compare embeddings of different dimensions: "
                f"{self.dimension} vs {other.dimension}"
            )
        
        if self.model_name != other.model_name:
            raise ValueError(
                f"Cannot compare embeddings from different models: "
                f"{self.model_name} vs {other.model_name}"
            )
        
        return float(np.linalg.norm(self.vector - other.vector))
    
    def is_similar_to(self, other: "VectorEmbedding", threshold: float = 0.8) -> bool:
        """
        Check if this embedding is similar to another based on cosine similarity.
        
        Args:
            other: Another VectorEmbedding to compare with
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            True if similar, False otherwise
        """
        return self.cosine_similarity(other) >= threshold
    
    def to_dict(self) -> Dict:
        """
        Convert embedding to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "vector": self.vector.tolist(),
            "dimension": self.dimension,
            "model_name": self.model_name,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata or {},
            "normalized": self.normalized,
            "norm": self.norm
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "VectorEmbedding":
        """
        Create embedding from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            VectorEmbedding instance
        """
        return cls(
            vector=np.array(data["vector"]),
            model_name=data["model_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata"),
            normalized=data.get("normalized", False)
        )
    
    def _get_normalized_vector(self) -> np.ndarray:
        """Get L2-normalized version of the vector."""
        if self.normalized:
            return self.vector
        
        norm = self.norm
        if norm > 0:
            return self.vector / norm
        return self.vector.copy()
    
    def normalize(self) -> "VectorEmbedding":
        """
        Create a normalized version of this embedding.
        
        Returns:
            New VectorEmbedding instance with normalized vector
        """
        if self.normalized:
            return self
        
        normalized_vector = self._get_normalized_vector()
        
        # Create new instance with normalized vector
        return VectorEmbedding(
            vector=normalized_vector,
            model_name=self.model_name,
            created_at=self.created_at,
            metadata=self.metadata.copy() if self.metadata else None,
            normalized=True
        )
    
    def __str__(self) -> str:
        return f"VectorEmbedding(model={self.model_name}, dim={self.dimension}, norm={self.norm:.4f})"
    
    def __repr__(self) -> str:
        return (
            f"VectorEmbedding(vector_shape={self.vector.shape}, "
            f"model={self.model_name}, "
            f"created_at={self.created_at}, "
            f"metadata={self.metadata})"
        )