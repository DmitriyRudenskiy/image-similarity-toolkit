"""
Service Container
=================

Dependency injection container for managing service lifecycle.
"""

from typing import Type, TypeVar, Any, Optional, Dict
from abc import ABC, abstractmethod

from ...domain.vector_storage.vector_repository import VectorRepository
from ...application.interfaces.embedding_generator import EmbeddingGenerator
from ...application.interfaces.image_processor import ImageProcessor
from ...domain.vector_storage.vector_store import VectorStore

T = TypeVar('T')


class ServiceFactory(ABC):
    """Abstract base class for service factories."""
    
    @abstractmethod
    def create(self, **kwargs: Any) -> Any:
        """Create a service instance."""
        pass


class ServiceContainer:
    """
    Dependency injection container for managing service lifecycle.
    
    Provides factory pattern for creating and managing service instances
    with proper dependency injection and lifecycle management.
    
    Example:
        >>> container = ServiceContainer()
        >>> container.register_repository('sqlite', SQLiteVectorRepository)
        >>> container.register_generator('resnet', ResNetEmbeddingGenerator)
        >>> 
        >>> repository = container.get_repository('sqlite', db_path='embeddings.db')
        >>> generator = container.get_generator('resnet')
        >>> processor = container.get_image_processor()
    """
    
    def __init__(self):
        """Initialize service container."""
        self._repositories: Dict[str, Type[VectorRepository]] = {}
        self._generators: Dict[str, Type[EmbeddingGenerator]] = {}
        self._processors: Dict[str, Type[ImageProcessor]] = {}
        self._service_instances: Dict[str, Any] = {}
        self._factories: Dict[str, ServiceFactory] = {}
    
    def register_repository(
        self,
        name: str,
        repository_class: Type[VectorRepository]
    ) -> 'ServiceContainer':
        """
        Register a vector repository implementation.
        
        Args:
            name: Name to register the repository under
            repository_class: Repository class to register
            
        Returns:
            Self for method chaining
        """
        self._repositories[name] = repository_class
        return self
    
    def register_generator(
        self,
        name: str,
        generator_class: Type[EmbeddingGenerator]
    ) -> 'ServiceContainer':
        """
        Register an embedding generator implementation.
        
        Args:
            name: Name to register the generator under
            generator_class: Generator class to register
            
        Returns:
            Self for method chaining
        """
        self._generators[name] = generator_class
        return self
    
    def register_processor(
        self,
        name: str,
        processor_class: Type[ImageProcessor]
    ) -> 'ServiceContainer':
        """
        Register an image processor implementation.
        
        Args:
            name: Name to register the processor under
            processor_class: Processor class to register
            
        Returns:
            Self for method chaining
        """
        self._processors[name] = processor_class
        return self
    
    def register_factory(
        self,
        name: str,
        factory: ServiceFactory
    ) -> 'ServiceContainer':
        """
        Register a custom service factory.
        
        Args:
            name: Name to register the factory under
            factory: Factory instance
            
        Returns:
            Self for method chaining
        """
        self._factories[name] = factory
        return self
    
    def get_repository(
        self,
        name: str,
        **kwargs: Any
    ) -> VectorRepository:
        """
        Get a repository instance.
        
        Args:
            name: Name of the repository to get
            **kwargs: Arguments to pass to repository constructor
            
        Returns:
            Repository instance
            
        Raises:
            ValueError: If repository name is not registered
        """
        if name not in self._repositories:
            raise ValueError(f"Repository '{name}' is not registered")
        
        # Try to get from cache first
        instance_key = f"repository_{name}_{hash(str(kwargs))}"
        if instance_key in self._service_instances:
            return self._service_instances[instance_key]
        
        # Create new instance
        repository_class = self._repositories[name]
        instance = repository_class(**kwargs)
        
        # Cache instance
        self._service_instances[instance_key] = instance
        
        return instance
    
    def get_generator(
        self,
        name: str,
        **kwargs: Any
    ) -> EmbeddingGenerator:
        """
        Get an embedding generator instance.
        
        Args:
            name: Name of the generator to get
            **kwargs: Arguments to pass to generator constructor
            
        Returns:
            Generator instance
            
        Raises:
            ValueError: If generator name is not registered
        """
        if name not in self._generators:
            raise ValueError(f"Generator '{name}' is not registered")
        
        # Try to get from cache first
        instance_key = f"generator_{name}"
        if instance_key in self._service_instances:
            return self._service_instances[instance_key]
        
        # Create new instance
        generator_class = self._generators[name]
        instance = generator_class(**kwargs)
        
        # Cache instance
        self._service_instances[instance_key] = instance
        
        return instance
    
    def get_image_processor(
        self,
        name: str = 'pil',
        **kwargs: Any
    ) -> ImageProcessor:
        """
        Get an image processor instance.
        
        Args:
            name: Name of the processor to get (default: 'pil')
            **kwargs: Arguments to pass to processor constructor
            
        Returns:
            Processor instance
            
        Raises:
            ValueError: If processor name is not registered
        """
        if name not in self._processors:
            raise ValueError(f"Processor '{name}' is not registered")
        
        # Try to get from cache first
        instance_key = f"processor_{name}"
        if instance_key in self._service_instances:
            return self._service_instances[instance_key]
        
        # Create new instance
        processor_class = self._processors[name]
        instance = processor_class(**kwargs)
        
        # Cache instance
        self._service_instances[instance_key] = instance
        
        return instance
    
    def get_service(self, name: str, **kwargs: Any) -> Any:
        """
        Get a service from factory.
        
        Args:
            name: Name of the service to get
            **kwargs: Arguments to pass to factory
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service name is not registered
        """
        if name not in self._factories:
            raise ValueError(f"Service '{name}' is not registered")
        
        # Get from factory
        factory = self._factories[name]
        return factory.create(**kwargs)
    
    def create_vector_store(
        self,
        repository_name: str,
        **repository_kwargs: Any
    ) -> VectorStore:
        """
        Create a VectorStore with the specified repository.
        
        Args:
            repository_name: Name of the repository to use
            **repository_kwargs: Arguments for repository creation
            
        Returns:
            VectorStore instance
        """
        repository = self.get_repository(repository_name, **repository_kwargs)
        return VectorStore(repository)
    
    def create_full_pipeline(
        self,
        repository_name: str,
        generator_name: str,
        processor_name: str = 'pil',
        **kwargs: Any
    ) -> tuple[VectorRepository, EmbeddingGenerator, ImageProcessor]:
        """
        Create a complete processing pipeline.
        
        Args:
            repository_name: Name of the repository to use
            generator_name: Name of the generator to use
            processor_name: Name of the processor to use (default: 'pil')
            **kwargs: Additional arguments for services
            
        Returns:
            Tuple of (repository, generator, processor)
        """
        repository = self.get_repository(repository_name, **kwargs.get('repository', {}))
        generator = self.get_generator(generator_name, **kwargs.get('generator', {}))
        processor = self.get_image_processor(processor_name, **kwargs.get('processor', {}))
        
        return repository, generator, processor
    
    def clear_cache(self) -> None:
        """Clear all cached service instances."""
        self._service_instances.clear()
    
    def clear_cached_instance(self, instance_key: str) -> None:
        """
        Clear a specific cached instance.
        
        Args:
            instance_key: Key of the instance to clear
        """
        if instance_key in self._service_instances:
            del self._service_instances[instance_key]
    
    def get_registered_services(self) -> Dict[str, Dict[str, Type]]:
        """
        Get all registered services.
        
        Returns:
            Dictionary with all registered services
        """
        return {
            'repositories': dict(self._repositories),
            'generators': dict(self._generators),
            'processors': dict(self._processors),
            'factories': {name: factory.__class__.__name__ 
                         for name, factory in self._factories.items()}
        }
    
    def configure_defaults(self) -> 'ServiceContainer':
        """
        Configure default service registrations.
        
        Returns:
            Self for method chaining
        """
        # Import here to avoid circular imports
        from ..repositories.sqlite_vector_repository import SQLiteVectorRepository
        from ..repositories.chromadb_vector_repository import ChromaDBVectorRepository
        from ..generators.resnet_embedding_generator import ResNetEmbeddingGenerator
        from ..generators.efficientnet_embedding_generator import EfficientNetEmbeddingGenerator
        from ..generators.clip_embedding_generator import CLIPEmbeddingGenerator
        from ..processors.pil_image_processor import PILImageProcessor
        
        # Register default services
        self.register_repository('sqlite', SQLiteVectorRepository)
        self.register_repository('chromadb', ChromaDBVectorRepository)
        
        self.register_generator('resnet50', ResNetEmbeddingGenerator)
        self.register_generator('efficientnet', EfficientNetEmbeddingGenerator)
        self.register_generator('clip', CLIPEmbeddingGenerator)
        
        self.register_processor('pil', PILImageProcessor)
        
        return self
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear_cache()


class SingletonServiceContainer(ServiceContainer):
    """
    Singleton service container with global instance management.
    
    Ensures only one instance of each service type exists across the application.
    """
    
    _instance: Optional['SingletonServiceContainer'] = None
    _lock = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            super().__init__()
            self._initialized = True
            self.configure_defaults()
    
    @classmethod
    def get_instance(cls) -> 'SingletonServiceContainer':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> 'SingletonServiceContainer':
        """Reset the singleton instance."""
        cls._instance = None
        return cls.get_instance()