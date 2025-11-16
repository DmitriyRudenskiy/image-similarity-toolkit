"""
Similarity Query Value Object
============================

Immutable query for similarity search operations.
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Protocol
from enum import Enum
from pathlib import Path

from ..vector_storage.vector_embedding import VectorEmbedding
from ..image_processing import Image


class SearchMode(Enum):
    """Search mode enumeration."""
    COSINE_SIMILARITY = "cosine"
    EUCLIDEAN_DISTANCE = "euclidean"
    MANHATTAN_DISTANCE = "manhattan"


class SearchType(Enum):
    """Search type enumeration."""
    IMAGE_TO_IMAGE = "image_to_image"
    EMBEDDING_TO_IMAGE = "embedding_to_image"
    TEXT_TO_IMAGE = "text_to_image"


@dataclass(frozen=True)
class SimilarityQuery:
    """
    Value Object representing a similarity search query.
    
    Immutable by design - once created, cannot be modified.
    
    Attributes:
        query_embedding: Vector embedding for the query
        query_image: Optional query image (alternative to embedding)
        query_text: Optional query text (alternative to embedding)
        search_mode: How to calculate similarity
        search_type: Type of search to perform
        limit: Maximum number of results
        threshold: Minimum similarity threshold
        model_name: Expected model name for results
        filters: Additional filters to apply
    """
    query_embedding: Optional[VectorEmbedding] = None
    query_image: Optional[Image] = None
    query_text: Optional[str] = None
    search_mode: SearchMode = SearchMode.COSINE_SIMILARITY
    search_type: SearchType = SearchType.EMBEDDING_TO_IMAGE
    limit: int = 10
    threshold: float = 0.0
    model_name: Optional[str] = None
    filters: Optional[dict] = None
    
    def __post_init__(self):
        """Validate query after initialization."""
        # Check that at least one query source is provided
        if not any([self.query_embedding, self.query_image, self.query_text]):
            raise ValueError(
                "At least one query source must be provided: "
                "query_embedding, query_image, or query_text"
            )
        
        # Validate limit
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        
        # Validate threshold
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        # Validate query consistency
        if self.query_image and self.query_embedding:
            # If both image and embedding are provided, they should match
            pass  # Validation would require more context
    
    @classmethod
    def from_image(
        cls,
        image: Image,
        limit: int = 10,
        threshold: float = 0.0,
        model_name: Optional[str] = None,
        filters: Optional[dict] = None
    ) -> "SimilarityQuery":
        """
        Create query from image.
        
        Args:
            image: Query image
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            model_name: Expected model name
            filters: Additional filters
            
        Returns:
            SimilarityQuery instance
        """
        return cls(
            query_image=image,
            search_type=SearchType.IMAGE_TO_IMAGE,
            limit=limit,
            threshold=threshold,
            model_name=model_name,
            filters=filters
        )
    
    @classmethod
    def from_embedding(
        cls,
        embedding: VectorEmbedding,
        limit: int = 10,
        threshold: float = 0.0,
        filters: Optional[dict] = None
    ) -> "SimilarityQuery":
        """
        Create query from embedding.
        
        Args:
            embedding: Query embedding
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            filters: Additional filters
            
        Returns:
            SimilarityQuery instance
        """
        return cls(
            query_embedding=embedding,
            search_type=SearchType.EMBEDDING_TO_IMAGE,
            limit=limit,
            threshold=threshold,
            model_name=embedding.model_name,
            filters=filters
        )
    
    @classmethod
    def from_text(
        cls,
        text: str,
        limit: int = 10,
        threshold: float = 0.0,
        model_name: Optional[str] = None,
        filters: Optional[dict] = None
    ) -> "SimilarityQuery":
        """
        Create query from text.
        
        Args:
            text: Query text
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            model_name: Expected model name
            filters: Additional filters
            
        Returns:
            SimilarityQuery instance
        """
        return cls(
            query_text=text,
            search_type=SearchType.TEXT_TO_IMAGE,
            limit=limit,
            threshold=threshold,
            model_name=model_name,
            filters=filters
        )
    
    @property
    def has_embedding(self) -> bool:
        """Check if query has embedding."""
        return self.query_embedding is not None
    
    @property
    def has_image(self) -> bool:
        """Check if query has image."""
        return self.query_image is not None
    
    @property
    def has_text(self) -> bool:
        """Check if query has text."""
        return self.query_text is not None
    
    def with_cosine_similarity(self) -> "SimilarityQuery":
        """Create copy with cosine similarity mode."""
        return SimilarityQuery(
            query_embedding=self.query_embedding,
            query_image=self.query_image,
            query_text=self.query_text,
            search_mode=SearchMode.COSINE_SIMILARITY,
            search_type=self.search_type,
            limit=self.limit,
            threshold=self.threshold,
            model_name=self.model_name,
            filters=self.filters
        )
    
    def with_euclidean_distance(self) -> "SimilarityQuery":
        """Create copy with euclidean distance mode."""
        return SimilarityQuery(
            query_embedding=self.query_embedding,
            query_image=self.query_image,
            query_text=self.query_text,
            search_mode=SearchMode.EUCLIDEAN_DISTANCE,
            search_type=self.search_type,
            limit=self.limit,
            threshold=self.threshold,
            model_name=self.model_name,
            filters=self.filters
        )
    
    def with_limit(self, limit: int) -> "SimilarityQuery":
        """Create copy with different limit."""
        return SimilarityQuery(
            query_embedding=self.query_embedding,
            query_image=self.query_image,
            query_text=self.query_text,
            search_mode=self.search_mode,
            search_type=self.search_type,
            limit=limit,
            threshold=self.threshold,
            model_name=self.model_name,
            filters=self.filters
        )
    
    def with_threshold(self, threshold: float) -> "SimilarityQuery":
        """Create copy with different threshold."""
        return SimilarityQuery(
            query_embedding=self.query_embedding,
            query_image=self.query_image,
            query_text=self.query_text,
            search_mode=self.search_mode,
            search_type=self.search_type,
            limit=self.limit,
            threshold=threshold,
            model_name=self.model_name,
            filters=self.filters
        )
    
    def to_dict(self) -> dict:
        """Convert query to dictionary representation."""
        return {
            "search_type": self.search_type.value,
            "search_mode": self.search_mode.value,
            "limit": self.limit,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "filters": self.filters,
            "has_embedding": self.has_embedding,
            "has_image": self.has_image,
            "has_text": self.has_text
        }
    
    def __str__(self) -> str:
        return (
            f"SimilarityQuery(type={self.search_type.value}, "
            f"mode={self.search_mode.value}, "
            f"limit={self.limit}, "
            f"threshold={self.threshold})"
        )
    
    def __repr__(self) -> str:
        return (
            f"SimilarityQuery("
            f"search_type={self.search_type}, "
            f"search_mode={self.search_mode}, "
            f"limit={self.limit}, "
            f"threshold={self.threshold}, "
            f"model_name={self.model_name}, "
            f"filters={self.filters}"
            f")"
        )