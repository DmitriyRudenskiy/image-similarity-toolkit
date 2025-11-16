# Quick Start Guide

Get started with Image Similarity Toolkit DDD in 5 minutes!

## Installation

```bash
git clone <repo-url>
cd image-similarity-toolkit
pip install -r requirements.txt
pip install -e .
```

## Basic Usage

```python
from src import ServiceContainer, Configuration

# Configure services
container = ServiceContainer().configure_defaults()

# Create pipeline
repository, generator, processor = container.create_full_pipeline(
    repository_name='sqlite',
    generator_name='efficientnet',
    processor_name='pil'
)

# Use cases available
from src.application.use_cases import (
    AddImageUseCase,
    SearchSimilarImagesUseCase,
    FindDuplicatesUseCase
)
```

## Examples

See `/examples/ddd_example.py` for complete examples.

## Documentation

See `/docs/` for detailed documentation.