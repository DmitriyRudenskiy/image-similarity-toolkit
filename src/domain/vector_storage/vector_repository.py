"""
Vector Repository Interface
==========================

Repository pattern for vector storage operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Protocol
from uuid import UUID
from datetime import datetime

from ..vector_storage.vector_embedding import VectorEmbedding
from ..image_processing import Image


class VectorRepository(Protocol):
    """
    Protocol for vector repository operations.
    
    Defines the interface that all vector storage implementations must provide.
    """
    
    def save(self, embedding: VectorEmbedding, image: Image) -> UUID:
        """
        Save an embedding with associated image.
        
        Args:
            embedding: The vector embedding to save
            image: The associated image
            
        Returns:
            Unique identifier of the saved embedding
        """
        ...
    
    def find_by_id(self, embedding_id: UUID) -> Optional[VectorEmbedding]:
        """
        Find embedding by ID.
        
        Args:
            embedding_id: Unique identifier
            
        Returns:
            Embedding if found, None otherwise
        """
        ...
    
    def find_by_image_hash(self, image_hash: str) -> Optional[VectorEmbedding]:
        """
        Find embedding by image file hash.
        
        Args:
            image_hash: MD5 hash of the image file
            
        Returns:
            Embedding if found, None otherwise
        """
        ...
    
    def find_similar(
        self, 
        query_embedding: VectorEmbedding, 
        limit: int = 10,
        threshold: float = 0.0
    ) -> List[tuple[VectorEmbedding, float]]:
        """
        Find similar embeddings using vector similarity search.
        
        Args:
            query_embedding: Query embedding to search with
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (embedding, similarity_score) tuples
        """
        ...
    
    def find_duplicates(
        self, 
        threshold: float = 0.95
    ) -> List[List[VectorEmbedding]]:
        """
        Find duplicate or near-duplicate embeddings.
        
        Args:
            threshold: Similarity threshold for duplicates
            
        Returns:
            List of duplicate groups (list of embeddings)
        """
        ...
    
    def delete_by_id(self, embedding_id: UUID) -> bool:
        """
        Delete embedding by ID.
        
        Args:
            embedding_id: Unique identifier
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    def get_all(self) -> List[VectorEmbedding]:
        """
        Get all embeddings.
        
        Returns:
            List of all embeddings
        """
        ...
    
    def get_stats(self) -> dict:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with statistics
        """
        ...
    
    def clear(self) -> None:
        """Clear all embeddings from repository."""
        ...


class VectorRepositoryBase(ABC):
    """
    Abstract base class for vector repository implementations.
    
    Provides common functionality for all repository implementations.
    """
    
    @abstractmethod
    def save(self, embedding: VectorEmbedding, image: Image) -> UUID:
        """Save embedding with associated image."""
        pass
    
    @abstractmethod
    def find_by_id(self, embedding_id: UUID) -> Optional[VectorEmbedding]:
        """Find embedding by ID."""
        pass
    
    @abstractmethod
    def find_by_image_hash(self, image_hash: str) -> Optional[VectorEmbedding]:
        """Find embedding by image hash."""
        pass
    
    @abstractmethod
    def find_similar(
        self, 
        query_embedding: VectorEmbedding, 
        limit: int = 10,
        threshold: float = 0.0
    ) -> List[tuple[VectorEmbedding, float]]:
        """Find similar embeddings."""
        pass
    
    @abstractmethod
    def find_duplicates(
        self, 
        threshold: float = 0.95
    ) -> List[List[VectorEmbedding]]:
        """Find duplicate embeddings."""
        pass
    
    @abstractmethod
    def delete_by_id(self, embedding_id: UUID) -> bool:
        """Delete embedding by ID."""
        pass
    
    @abstractmethod
    def get_all(self) -> List[VectorEmbedding]:
        """Get all embeddings."""
        pass
    
    @abstractmethod
    def get_stats(self) -> dict:
        """Get repository statistics."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all embeddings."""
        pass