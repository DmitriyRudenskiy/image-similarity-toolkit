# Contributing to Image Similarity Toolkit DDD

Thank you for contributing! Follow these guidelines:

## Development Setup

```bash
git clone <repo-url>
cd image-similarity-toolkit
pip install -r requirements.txt
pip install -e .
```

## Code Quality

```bash
# Run formatting and linting
./scripts/format_and_lint.sh

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## DDD Architecture Guidelines

- **Domain Layer**: Business logic, no dependencies on infrastructure
- **Application Layer**: Use Cases, orchestrates domain objects
- **Infrastructure Layer**: Concrete implementations, databases, external services

## Testing Standards

- Minimum 80% code coverage
- Unit tests for all Use Cases
- Integration tests for repositories
- Error handling tests for all exception paths

## Pull Request Process

1. Fork repository and create feature branch
2. Write tests for new functionality
3. Run quality checks
4. Update documentation
5. Submit PR with clear description

See `/docs/DEVELOPMENT_GUIDE.md` for detailed guidelines.