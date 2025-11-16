"""
Database Configuration Value Object
==================================

Immutable configuration for database connections and operations.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
from pathlib import Path


class DatabaseType(Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    CHROMADB = "chromadb"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"


@dataclass(frozen=True)
class DatabaseConfiguration:
    """
    Value Object representing database configuration.
    
    Immutable by design - once created, cannot be modified.
    
    Attributes:
        db_type: Type of database
        connection_string: Database connection string
        database_name: Name of the database
        host: Database host (for network databases)
        port: Database port (for network databases)
        username: Database username
        password: Database password
        timeout: Connection timeout in seconds
        max_connections: Maximum number of connections
        options: Additional database options
        metadata: Additional configuration metadata
    """
    db_type: DatabaseType
    connection_string: Optional[str] = None
    database_name: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30
    max_connections: int = 10
    options: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")
        
        # Validate specific database type requirements
        self._validate_db_specific_requirements()
    
    @classmethod
    def sqlite(
        cls,
        database_path: Path,
        timeout: int = 30,
        options: Optional[Dict[str, Any]] = None
    ) -> "DatabaseConfiguration":
        """
        Create SQLite database configuration.
        
        Args:
            database_path: Path to SQLite database file
            timeout: Connection timeout
            options: Additional options
            
        Returns:
            DatabaseConfiguration instance
        """
        return cls(
            db_type=DatabaseType.SQLITE,
            connection_string=str(database_path),
            database_name=database_path.stem,
            timeout=timeout,
            options=options or {}
        )
    
    @classmethod
    def chromadb(
        cls,
        collection_name: str = "image_embeddings",
        persist_directory: Optional[Path] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: int = 30,
        options: Optional[Dict[str, Any]] = None
    ) -> "DatabaseConfiguration":
        """
        Create ChromaDB database configuration.
        
        Args:
            collection_name: Name of the vector collection
            persist_directory: Directory for persistent storage
            host: ChromaDB server host
            port: ChromaDB server port
            timeout: Connection timeout
            options: Additional options
            
        Returns:
            DatabaseConfiguration instance
        """
        if persist_directory:
            connection_string = str(persist_directory)
        else:
            connection_string = "memory://"
        
        return cls(
            db_type=DatabaseType.CHROMADB,
            connection_string=connection_string,
            database_name=collection_name,
            host=host,
            port=port,
            timeout=timeout,
            options=options or {}
        )
    
    @classmethod
    def postgresql(
        cls,
        database_name: str,
        host: str = "localhost",
        port: int = 5432,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30,
        max_connections: int = 10,
        options: Optional[Dict[str, Any]] = None
    ) -> "DatabaseConfiguration":
        """
        Create PostgreSQL database configuration.
        
        Args:
            database_name: Name of the database
            host: Database host
            port: Database port
            username: Database username
            password: Database password
            timeout: Connection timeout
            max_connections: Maximum connections
            options: Additional options
            
        Returns:
            DatabaseConfiguration instance
        """
        connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
        
        return cls(
            db_type=DatabaseType.POSTGRESQL,
            connection_string=connection_string,
            database_name=database_name,
            host=host,
            port=port,
            username=username,
            password=password,
            timeout=timeout,
            max_connections=max_connections,
            options=options or {}
        )
    
    @classmethod
    def mongodb(
        cls,
        database_name: str,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30,
        max_connections: int = 10,
        options: Optional[Dict[str, Any]] = None
    ) -> "DatabaseConfiguration":
        """
        Create MongoDB database configuration.
        
        Args:
            database_name: Name of the database
            host: Database host
            port: Database port
            username: Database username
            password: Database password
            timeout: Connection timeout
            max_connections: Maximum connections
            options: Additional options
            
        Returns:
            DatabaseConfiguration instance
        """
        auth_part = f"{username}:{password}@" if username and password else ""
        connection_string = f"mongodb://{auth_part}{host}:{port}/{database_name}"
        
        return cls(
            db_type=DatabaseType.MONGODB,
            connection_string=connection_string,
            database_name=database_name,
            host=host,
            port=port,
            username=username,
            password=password,
            timeout=timeout,
            max_connections=max_connections,
            options=options or {}
        )
    
    def _validate_db_specific_requirements(self) -> None:
        """Validate database-specific requirements."""
        if self.db_type == DatabaseType.SQLITE:
            if not self.connection_string:
                raise ValueError("SQLite requires connection_string (database path)")
        
        elif self.db_type == DatabaseType.CHROMADB:
            if not self.database_name:
                raise ValueError("ChromaDB requires database_name (collection name)")
        
        elif self.db_type == DatabaseType.POSTGRESQL:
            if not self.database_name:
                raise ValueError("PostgreSQL requires database_name")
            if not self.host:
                raise ValueError("PostgreSQL requires host")
        
        elif self.db_type == DatabaseType.MONGODB:
            if not self.database_name:
                raise ValueError("MongoDB requires database_name")
            if not self.host:
                raise ValueError("MongoDB requires host")
    
    @property
    def is_local(self) -> bool:
        """Check if database is local (file-based)."""
        return self.db_type in [DatabaseType.SQLITE, DatabaseType.CHROMADB]
    
    @property
    def is_network(self) -> bool:
        """Check if database requires network connection."""
        return self.db_type in [DatabaseType.POSTGRESQL, DatabaseType.MONGODB]
    
    @property
    def requires_auth(self) -> bool:
        """Check if database requires authentication."""
        return bool(self.username or self.password)
    
    def get_connection_info(self) -> dict:
        """Get safe connection information (without sensitive data)."""
        return {
            "db_type": self.db_type.value,
            "database_name": self.database_name,
            "host": self.host,
            "port": self.port,
            "is_local": self.is_local,
            "is_network": self.is_network,
            "requires_auth": self.requires_auth,
            "timeout": self.timeout,
            "max_connections": self.max_connections,
            "options": self.options or {}
        }
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary (without sensitive data)."""
        return {
            "db_type": self.db_type.value,
            "connection_string": self.connection_string,
            "database_name": self.database_name,
            "host": self.host,
            "port": self.port,
            "username": self.username,  # Note: This might be sensitive
            "timeout": self.timeout,
            "max_connections": self.max_connections,
            "options": self.options or {},
            "metadata": self.metadata or {},
            "is_local": self.is_local,
            "is_network": self.is_network,
            "requires_auth": self.requires_auth
        }
    
    def __str__(self) -> str:
        return (
            f"DatabaseConfiguration(type={self.db_type.value}, "
            f"name={self.database_name}, "
            f"host={self.host}, "
            f"local={self.is_local})"
        )
    
    def __repr__(self) -> str:
        return (
            f"DatabaseConfiguration("
            f"db_type={self.db_type}, "
            f"database_name={self.database_name}, "
            f"host={self.host}, "
            f"port={self.port}"
            f")"
        )