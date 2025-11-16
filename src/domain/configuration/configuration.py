"""
Configuration Value Object
=========================

Immutable configuration for the entire Image Similarity Toolkit.
"""

import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum

from ..database_management.database_configuration import DatabaseConfiguration, DatabaseType


class ProcessingMode(Enum):
    """Processing mode enumeration."""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Configuration:
    """
    Value Object representing the complete application configuration.
    
    Immutable by design - once created, cannot be modified.
    
    Attributes:
        application: Application-level settings
        processing: Processing configuration
        database: Database configuration
        logging: Logging configuration
        cache: Cache configuration
        security: Security configuration
        extensions: Extension configuration
        metadata: Additional configuration metadata
    """
    application: Dict[str, Any]
    processing: Dict[str, Any]
    database: Optional[DatabaseConfiguration] = None
    logging: Dict[str, Any] = None
    cache: Dict[str, Any] = None
    security: Dict[str, Any] = None
    extensions: Dict[str, Any] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_application_settings()
        self._validate_processing_settings()
        self._validate_database_settings()
        self._validate_logging_settings()
        self._validate_cache_settings()
        self._validate_security_settings()
    
    @classmethod
    def default(cls) -> "Configuration":
        """
        Create default configuration.
        
        Returns:
            Configuration with default settings
        """
        return cls(
            application={
                "name": "Image Similarity Toolkit",
                "version": "0.2.1",
                "debug": False,
                "max_image_size": "10MB",
                "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"],
                "temp_directory": "temp"
            },
            processing={
                "mode": ProcessingMode.AUTO.value,
                "batch_size": 32,
                "max_workers": 4,
                "model_cache_size": 100,
                "embedding_dimension": 512,
                "similarity_threshold": 0.8
            },
            logging={
                "level": LogLevel.INFO.value,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/image_similarity.log",
                "max_file_size": "10MB",
                "backup_count": 5,
                "console_output": True
            },
            cache={
                "enabled": True,
                "type": "memory",
                "size_limit": "1GB",
                "ttl": 3600,
                "cleanup_interval": 1800
            },
            security={
                "max_file_size": "50MB",
                "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
                "sanitize_filenames": True,
                "validate_images": True
            }
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> "Configuration":
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        suffix = config_path.suffix.lower()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if suffix == '.json':
                config_data = json.load(f)
            elif suffix in ['.yml', '.yaml']:
                config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {suffix}")
        
        # Parse database configuration if present
        database_config = None
        if 'database' in config_data:
            db_data = config_data['database']
            db_type = DatabaseType(db_data.get('type', 'sqlite'))
            
            if db_type == DatabaseType.SQLITE:
                database_config = DatabaseConfiguration.sqlite(
                    Path(db_data.get('path', 'embeddings.db'))
                )
            elif db_type == DatabaseType.CHROMADB:
                database_config = DatabaseConfiguration.chromadb(
                    collection_name=db_data.get('collection_name', 'image_embeddings'),
                    persist_directory=Path(db_data.get('persist_directory', './chromadb'))
                )
        
        return cls(
            application=config_data.get('application', {}),
            processing=config_data.get('processing', {}),
            database=database_config,
            logging=config_data.get('logging', {}),
            cache=config_data.get('cache', {}),
            security=config_data.get('security', {}),
            extensions=config_data.get('extensions', {}),
            metadata=config_data.get('metadata', {})
        )
    
    def _validate_application_settings(self) -> None:
        """Validate application settings."""
        app = self.application
        
        if 'max_image_size' in app:
            # Parse size string (e.g., "10MB", "1GB")
            size_str = str(app['max_image_size']).upper()
            if not any(suffix in size_str for suffix in ['B', 'KB', 'MB', 'GB']):
                raise ValueError("Invalid max_image_size format. Use formats like '10MB', '1GB'")
        
        if 'supported_formats' in app:
            if not isinstance(app['supported_formats'], list):
                raise ValueError("supported_formats must be a list")
    
    def _validate_processing_settings(self) -> None:
        """Validate processing settings."""
        processing = self.processing
        
        # Validate mode
        if 'mode' in processing:
            try:
                ProcessingMode(processing['mode'])
            except ValueError:
                raise ValueError(f"Invalid processing mode: {processing['mode']}")
        
        # Validate numeric values
        if 'batch_size' in processing:
            if not isinstance(processing['batch_size'], int) or processing['batch_size'] <= 0:
                raise ValueError("batch_size must be a positive integer")
        
        if 'max_workers' in processing:
            if not isinstance(processing['max_workers'], int) or processing['max_workers'] <= 0:
                raise ValueError("max_workers must be a positive integer")
        
        if 'embedding_dimension' in processing:
            if not isinstance(processing['embedding_dimension'], int) or processing['embedding_dimension'] <= 0:
                raise ValueError("embedding_dimension must be a positive integer")
        
        if 'similarity_threshold' in processing:
            threshold = processing['similarity_threshold']
            if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
                raise ValueError("similarity_threshold must be between 0.0 and 1.0")
    
    def _validate_database_settings(self) -> None:
        """Validate database settings."""
        # Database configuration validation is handled by DatabaseConfiguration class
        pass
    
    def _validate_logging_settings(self) -> None:
        """Validate logging settings."""
        if not self.logging:
            return
        
        logging = self.logging
        
        # Validate log level
        if 'level' in logging:
            try:
                LogLevel(logging['level'])
            except ValueError:
                raise ValueError(f"Invalid log level: {logging['level']}")
        
        # Validate console_output
        if 'console_output' in logging and not isinstance(logging['console_output'], bool):
            raise ValueError("console_output must be a boolean")
    
    def _validate_cache_settings(self) -> None:
        """Validate cache settings."""
        if not self.cache:
            return
        
        cache = self.cache
        
        # Validate enabled
        if 'enabled' in cache and not isinstance(cache['enabled'], bool):
            raise ValueError("cache.enabled must be a boolean")
        
        # Validate ttl
        if 'ttl' in cache:
            if not isinstance(cache['ttl'], int) or cache['ttl'] <= 0:
                raise ValueError("cache.ttl must be a positive integer")
    
    def _validate_security_settings(self) -> None:
        """Validate security settings."""
        if not self.security:
            return
        
        security = self.security
        
        # Validate max_file_size
        if 'max_file_size' in security:
            size_str = str(security['max_file_size']).upper()
            if not any(suffix in size_str for suffix in ['B', 'KB', 'MB', 'GB']):
                raise ValueError("Invalid max_file_size format. Use formats like '10MB', '1GB'")
        
        # Validate allowed_extensions
        if 'allowed_extensions' in security:
            if not isinstance(security['allowed_extensions'], list):
                raise ValueError("allowed_extensions must be a list")
    
    def get_processing_mode(self) -> ProcessingMode:
        """Get processing mode enum."""
        mode_str = self.processing.get('mode', 'auto')
        return ProcessingMode(mode_str)
    
    def get_log_level(self) -> LogLevel:
        """Get log level enum."""
        level_str = self.logging.get('level', 'INFO')
        return LogLevel(level_str)
    
    def get_cache_enabled(self) -> bool:
        """Get cache enabled status."""
        return self.cache.get('enabled', True) if self.cache else True
    
    def get_security_settings(self) -> Dict[str, Any]:
        """Get security settings with defaults."""
        default_security = {
            "max_file_size": "50MB",
            "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
            "sanitize_filenames": True,
            "validate_images": True
        }
        
        if self.security:
            default_security.update(self.security)
        
        return default_security
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {
            "application": self.application,
            "processing": self.processing
        }
        
        if self.database:
            result["database"] = self.database.to_dict()
        
        if self.logging:
            result["logging"] = self.logging
        
        if self.cache:
            result["cache"] = self.cache
        
        if self.security:
            result["security"] = self.security
        
        if self.extensions:
            result["extensions"] = self.extensions
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    def to_file(self, config_path: Path, format: str = "json") -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration
            format: File format ('json' or 'yaml')
            
        Raises:
            ValueError: If format is not supported
        """
        config_dict = self.to_dict()
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            elif format.lower() in ['yml', 'yaml']:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def merge_with(self, other: "Configuration") -> "Configuration":
        """
        Merge this configuration with another.
        
        Args:
            other: Configuration to merge with
            
        Returns:
            New Configuration with merged settings
        """
        # Deep merge dictionaries
        def deep_merge(base: Dict, update: Dict) -> Dict:
            result = base.copy()
            for key, value in update.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_app = deep_merge(self.application, other.application)
        merged_processing = deep_merge(self.processing, other.processing)
        
        merged_logging = self.logging.copy() if self.logging else {}
        if other.logging:
            merged_logging.update(other.logging)
        
        merged_cache = self.cache.copy() if self.cache else {}
        if other.cache:
            merged_cache.update(other.cache)
        
        merged_security = self.security.copy() if self.security else {}
        if other.security:
            merged_security.update(other.security)
        
        # Database configuration: prefer other if different types
        merged_database = self.database
        if other.database and self.database != other.database:
            merged_database = other.database
        
        return Configuration(
            application=merged_app,
            processing=merged_processing,
            database=merged_database,
            logging=merged_logging,
            cache=merged_cache,
            security=merged_security,
            extensions=self.extensions.copy() if self.extensions else {},
            metadata=self.metadata.copy() if self.metadata else {}
        )
    
    def __str__(self) -> str:
        db_info = f"database={self.database.db_type.value}" if self.database else "database=None"
        return f"Configuration({db_info}, processing={self.processing.get('mode', 'unknown')})"
    
    def __repr__(self) -> str:
        return (
            f"Configuration("
            f"application={self.application}, "
            f"processing={self.processing}, "
            f"database={self.database}, "
            f"logging={self.logging}"
            f")"
        )