# Changelog

## [DDD] - 2025-11-16

### Added
- **Domain-Driven Architecture**: Complete DDD refactoring
  - Domain Layer: Business logic, Value Objects, Domain Services
  - Application Layer: Use Cases, Application interfaces
  - Infrastructure Layer: Concrete implementations
- **Dependency Injection**: ServiceContainer with factory pattern
- **Infrastructure Implementations**:
  - SQLiteVectorRepository
  - ChromaDBVectorRepository
  - ResNetEmbeddingGenerator
  - EfficientNetEmbeddingGenerator
  - CLIPEmbeddingGenerator
  - PILImageProcessor

### Changed
- Complete codebase restructure following DDD principles
- Improved separation of concerns
- Better testability and maintainability

## [0.2.0] - 2025-11-15

### Added
- SQLite database integration
- Similarity search in database
- Duplicate detection
- Database statistics

## [0.1.0] - 2025-11-15

### Added
- Initial release with basic image similarity functionality
- Support for ResNet50, EfficientNet-B0, CLIP models
- Visualization features
- Basic examples and documentation