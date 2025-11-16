"""
Repository Base Class
====================

Abstract base class for repository implementations following the Repository pattern.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Protocol
from uuid import UUID

from ..database_management.database_connection import DatabaseConnection
from ..database_management.database_configuration import DatabaseConfiguration
from ..vector_storage.vector_embedding import VectorEmbedding
from ..image_processing import Image

T = TypeVar('T')


class Repository(Protocol, Generic[T]):
    """
    Protocol for repository operations.
    
    Defines the interface that all repository implementations must provide.
    """
    
    def save(self, entity: T) -> UUID:
        """
        Save an entity to the repository.
        
        Args:
            entity: Entity to save
            
        Returns:
            Unique identifier of the saved entity
        """
        ...
    
    def find_by_id(self, entity_id: UUID) -> T:
        """
        Find entity by ID.
        
        Args:
            entity_id: Unique identifier
            
        Returns:
            Entity if found
            
        Raises:
            EntityNotFoundError: If entity not found
        """
        ...
    
    def find_all(self) -> list[T]:
        """Find all entities."""
        ...
    
    def delete(self, entity_id: UUID) -> bool:
        """
        Delete entity by ID.
        
        Args:
            entity_id: Unique identifier
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    def count(self) -> int:
        """Get total number of entities."""
        ...
    
    def exists(self, entity_id: UUID) -> bool:
        """
        Check if entity exists.
        
        Args:
            entity_id: Unique identifier
            
        Returns:
            True if entity exists, False otherwise
        """
        ...
    
    def clear(self) -> None:
        """Clear all entities from repository."""
        ...


class RepositoryBase(ABC, Repository[T]):
    """
    Abstract base class for repository implementations.
    
    Provides common functionality for all repository implementations.
    """
    
    def __init__(self, connection: DatabaseConnection):
        """
        Initialize repository.
        
        Args:
            connection: Database connection
        """
        self._connection = connection
        self._connected = False
    
    @property
    def connection(self) -> DatabaseConnection:
        """Get database connection."""
        return self._connection
    
    @property
    def configuration(self) -> DatabaseConfiguration:
        """Get database configuration."""
        return self._connection.configuration
    
    def ensure_connected(self) -> None:
        """Ensure database connection is active."""
        if not self._connection.is_connected:
            self._connection.connect()
            self._connected = True
    
    def save(self, entity: T) -> UUID:
        """
        Save an entity to the repository.
        
        Args:
            entity: Entity to save
            
        Returns:
            Unique identifier of the saved entity
        """
        self.ensure_connected()
        return self._save_entity(entity)
    
    def find_by_id(self, entity_id: UUID) -> T:
        """
        Find entity by ID.
        
        Args:
            entity_id: Unique identifier
            
        Returns:
            Entity if found
            
        Raises:
            EntityNotFoundError: If entity not found
        """
        self.ensure_connected()
        return self._find_entity_by_id(entity_id)
    
    def find_all(self) -> list[T]:
        """Find all entities."""
        self.ensure_connected()
        return self._find_all_entities()
    
    def delete(self, entity_id: UUID) -> bool:
        """
        Delete entity by ID.
        
        Args:
            entity_id: Unique identifier
            
        Returns:
            True if deleted, False if not found
        """
        self.ensure_connected()
        return self._delete_entity(entity_id)
    
    def count(self) -> int:
        """Get total number of entities."""
        self.ensure_connected()
        return self._count_entities()
    
    def exists(self, entity_id: UUID) -> bool:
        """
        Check if entity exists.
        
        Args:
            entity_id: Unique identifier
            
        Returns:
            True if entity exists, False otherwise
        """
        self.ensure_connected()
        return self._entity_exists(entity_id)
    
    def clear(self) -> None:
        """Clear all entities from repository."""
        self.ensure_connected()
        self._clear_entities()
    
    def get_repository_info(self) -> dict:
        """
        Get repository information.
        
        Returns:
            Dictionary with repository information
        """
        return {
            "repository_type": self.__class__.__name__,
            "connection_info": self._connection.get_connection_info(),
            "entity_count": self.count(),
            "connected": self._connection.is_connected
        }
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def _save_entity(self, entity: T) -> UUID:
        """Save entity to database."""
        pass
    
    @abstractmethod
    def _find_entity_by_id(self, entity_id: UUID) -> T:
        """Find entity by ID from database."""
        pass
    
    @abstractmethod
    def _find_all_entities(self) -> list[T]:
        """Find all entities from database."""
        pass
    
    @abstractmethod
    def _delete_entity(self, entity_id: UUID) -> bool:
        """Delete entity from database."""
        pass
    
    @abstractmethod
    def _count_entities(self) -> int:
        """Count entities in database."""
        pass
    
    @abstractmethod
    def _entity_exists(self, entity_id: UUID) -> bool:
        """Check if entity exists in database."""
        pass
    
    @abstractmethod
    def _clear_entities(self) -> None:
        """Clear all entities from database."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.ensure_connected()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(connection={self._connection})"