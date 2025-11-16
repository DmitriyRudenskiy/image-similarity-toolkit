# Image Similarity Toolkit - Project Information DDD

**Name**: Image Similarity Toolkit DDD  
**Version**: DDD Architecture  
**License**: MIT  
**Author**: MiniMax Agent

## Description

Modern Python toolkit for image similarity search using domain-driven design principles.

## Architecture

- **Domain Layer**: Business logic and rules
- **Application Layer**: Use Cases and interfaces
- **Infrastructure Layer**: Concrete implementations

## Key Features

- Domain-Driven Architecture
- Multiple ML models (ResNet50, EfficientNet-B0, CLIP)
- Vector database support (SQLite, ChromaDB)
- Dependency injection container
- Comprehensive test coverage

## Quick Start

```python
from src import ServiceContainer
container = ServiceContainer().configure_defaults()
```

## Documentation

See `/docs/` for detailed documentation.

## License

MIT License