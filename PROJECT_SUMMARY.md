# Image Similarity Toolkit - DDD Version

**Version**: DDD Architecture  
**Date**: 2025-11-16  
**Author**: MiniMax Agent

## Key Features

- **Domain-Driven Architecture**: Clean DDD structure with Domain, Application, Infrastructure layers
- **Multiple Models**: ResNet50, EfficientNet-B0, CLIP
- **Vector Databases**: SQLite, ChromaDB support
- **Advanced Search**: Similarity search, duplicate detection
- **Dependency Injection**: ServiceContainer for proper lifecycle management

## Architecture

```
src/
├── domain/              # Domain layer (business logic)
│   ├── image_processing/   # Value Objects
│   ├── vector_storage/     # Aggregates, Repositories
│   ├── similarity_search/  # Domain Services
│   └── configuration/      # Configuration
├── application/          # Application layer (Use Cases)
│   ├── use_cases/         # Use Cases
│   └── interfaces/        # Application interfaces
└── infrastructure/       # Infrastructure layer (implementations)
    ├── repositories/      # Concrete repositories
    ├── generators/        # Model implementations
    ├── processors/        # Image processors
    └── dependency_injection/ # DI container
```

## Quick Start

```python
from src import ServiceContainer

# Configure services
container = ServiceContainer().configure_defaults()

# Use cases
from src.application.use_cases import AddImageUseCase, SearchSimilarImagesUseCase
```

## Testing

```bash
# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Generate coverage report
coverage report
```

Full documentation in `/docs/`.