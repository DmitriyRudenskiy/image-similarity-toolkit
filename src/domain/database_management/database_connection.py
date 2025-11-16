"""
Database Connection Value Object
===============================

Immutable representation of a database connection.
"""

from abc import ABC, abstractmethod
from typing import Protocol
from datetime import datetime

from .database_configuration import DatabaseConfiguration


class DatabaseConnection(Protocol):
    """
    Protocol for database connection operations.
    
    Defines the interface that all database connection implementations must provide.
    """
    
    def connect(self) -> None:
        """Establish connection to the database."""
        ...
    
    def disconnect(self) -> None:
        """Close connection to the database."""
        ...
    
    def is_connected(self) -> bool:
        """Check if connection is active."""
        ...
    
    def execute_query(self, query: str, parameters: dict = None) -> dict:
        """Execute a database query."""
        ...
    
    def transaction(self, operations: list) -> dict:
        """Execute multiple operations in a transaction."""
        ...
    
    def get_connection_info(self) -> dict:
        """Get connection information."""
        ...


class DatabaseConnectionBase(ABC):
    """
    Abstract base class for database connection implementations.
    
    Provides common functionality for all database connections.
    """
    
    def __init__(self, configuration: DatabaseConfiguration):
        """
        Initialize database connection.
        
        Args:
            configuration: Database configuration
        """
        self._configuration = configuration
        self._connected = False
        self._connection = None
        self._connected_at = None
    
    @property
    def configuration(self) -> DatabaseConfiguration:
        """Get database configuration."""
        return self._configuration
    
    @property
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._connected
    
    @property
    def connection_time(self) -> Optional[datetime]:
        """Get connection establishment time."""
        return self._connected_at
    
    @property
    def uptime_seconds(self) -> float:
        """Get connection uptime in seconds."""
        if not self._connected_at:
            return 0.0
        return (datetime.now() - self._connected_at).total_seconds()
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the database."""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, parameters: dict = None) -> dict:
        """Execute a database query."""
        pass
    
    @abstractmethod
    def transaction(self, operations: list) -> dict:
        """Execute multiple operations in a transaction."""
        pass
    
    def get_connection_info(self) -> dict:
        """Get safe connection information."""
        return {
            "db_type": self._configuration.db_type.value,
            "database_name": self._configuration.database_name,
            "host": self._configuration.host,
            "port": self._configuration.port,
            "is_local": self._configuration.is_local,
            "is_network": self._configuration.is_network,
            "requires_auth": self._configuration.requires_auth,
            "connected": self._connected,
            "connected_at": self._connected_at.isoformat() if self._connected_at else None,
            "uptime_seconds": self.uptime_seconds,
            "configuration": self._configuration.get_connection_info()
        }
    
    def test_connection(self) -> dict:
        """
        Test database connection.
        
        Returns:
            Dictionary with test results
        """
        try:
            # Try to connect
            self.connect()
            
            # Try a simple query
            test_result = self.execute_query("SELECT 1 as test")
            
            # Disconnect
            self.disconnect()
            
            return {
                "success": True,
                "message": "Connection test successful",
                "test_result": test_result,
                "connection_info": self.get_connection_info()
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection test failed: {str(e)}",
                "error_type": type(e).__name__,
                "connection_info": self.get_connection_info()
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __repr__(self) -> str:
        return (
            f"DatabaseConnection("
            f"type={self._configuration.db_type.value}, "
            f"database={self._configuration.database_name}, "
            f"connected={self._connected}"
            f")"
        )