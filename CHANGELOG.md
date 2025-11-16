# Changelog

All notable changes to the Image Similarity Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-15

### Added
- **Database Module**: New `EmbeddingDatabase` class for storing and managing embeddings
  - SQLite-based storage for image embeddings
  - Automatic indexing of images
  - Metadata storage (file size, dimensions)
  - Update existing embeddings
- **Similarity Search**: Fast search for similar images in database
  - Configurable top-k results
  - Similarity threshold filtering
  - Efficient cosine similarity computation
- **Duplicate Detection**: Automatic detection of duplicate and near-duplicate images
  - Configurable similarity threshold
  - Save duplicates to database table
  - Retrieve saved duplicates
- **Database Statistics**: Get comprehensive database statistics
  - Total images count
  - Duplicate pairs count
  - Average file sizes and dimensions
- **New Examples**:
  - `database_example.py`: Complete database operations workflow
  - `duplicate_cleaner.py`: Interactive and automatic duplicate removal
- **Additional Tests**: Comprehensive test suite for database module
  - Database CRUD operations
  - Similarity search functionality
  - Duplicate detection
  - Context manager support

### Features
- Store embeddings permanently for fast repeated comparisons
- Build searchable image databases
- Find and manage duplicate images
- Batch indexing of image directories
- HTML report generation for duplicates
- Interactive duplicate review mode
- Automatic duplicate removal with backup

### Technical Details
- SQLite database integration
- Efficient JSON serialization for embeddings
- Indexed database queries for performance
- Foreign key constraints for data integrity
- Transaction support for data consistency
- Context manager support for database connections

### Breaking Changes
- Version updated to 0.2.0
- New dependency: sqlite3 (built-in with Python)

## [0.1.0] - 2025-11-15

### Added
- Initial release of Image Similarity Toolkit
- Support for three pre-trained models:
  - ResNet50
  - EfficientNet-B0
  - CLIP (OpenAI)
- Core functionality:
  - Image embedding extraction
  - Cosine similarity calculation
  - Euclidean distance calculation
  - Normalized similarity metric
- Visualization features:
  - Side-by-side image comparison
  - Metrics display with color coding
  - Batch comparison charts
- Documentation:
  - Comprehensive README
  - Detailed usage guide
  - API documentation in docstrings
- Examples:
  - Basic usage example
  - Batch comparison example
- Testing:
  - Unit tests for core functionality
  - Pytest configuration
- Project infrastructure:
  - setup.py for package installation
  - requirements.txt for dependencies
  - .gitignore for version control
  - MIT License
  - Proper project structure

### Features
- Automatic GPU/CPU detection and usage
- Support for multiple image formats (JPG, PNG, BMP, WEBP)
- Automatic RGB conversion
- Configurable visualization output
- Human-readable similarity interpretation

### Technical Details
- Python 3.8+ support
- PyTorch 2.0+ compatibility
- Modular architecture for easy extension
- Type hints throughout codebase
- Comprehensive error handling

## [Unreleased]

### Planned Features
- [ ] Support for additional models (ViT, DINO)
- [ ] Batch processing with parallel execution
- [ ] Web interface for demonstrations
- [ ] Database integration for image search
- [ ] Export results to various formats (JSON, CSV)
- [ ] CLI tool for command-line usage
- [ ] Performance benchmarking tools
- [ ] Docker containerization
- [ ] REST API server
- [ ] Image augmentation support

---

[0.1.0]: https://github.com/yourusername/image-similarity-toolkit/releases/tag/v0.1.0
