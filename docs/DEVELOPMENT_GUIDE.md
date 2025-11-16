# Development Guide

## Overview

This guide covers the complete development workflow for the Image Similarity Toolkit, including code quality tools, formatting, linting, and best practices for maintaining high code standards.

## Table of Contents

1. [Installation](#installation)
2. [Development Workflow](#development-workflow)
3. [Code Quality Tools](#code-quality-tools)
4. [Configuration](#configuration)
5. [Pre-commit Hooks](#pre-commit-hooks)
6. [IDE Integration](#ide-integration)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Install Development Dependencies

```bash
pip install mypy ruff pylint black isort autoflake docformatter pre-commit
```

Or add to your existing requirements:

```bash
# Add to requirements-dev.txt
mypy
ruff
pylint
black
isort
autoflake
docformatter
pre-commit
```

```bash
pip install -r requirements-dev.txt
```

### Verify Installation

```bash
# Check versions
python --version  # Python 3.8+
ruff --version    # Latest
black --version   # Latest
mypy --version    # Latest
pylint --version  # Latest
```

---

## Development Workflow

### Recommended Order

To avoid conflicts and ensure optimal results, follow this sequence:

1. **Fix syntax errors** → `ruff` (automatic fixes)
2. **Format code** → `black` + `isort`
3. **Remove unused code** → `autoflake`
4. **Format docstrings** → `docformatter`
5. **Type checking** → `mypy`
6. **Deep linting** → `pylint`

### Complete Automation Script

Create `scripts/format_and_lint.sh`:

```bash
#!/bin/bash
set -e

echo "🔧 Step 1: Fix syntax errors with ruff"
ruff check --fix . --exclude .venv

echo "🧹 Step 2: Remove unused code"
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 autoflake \
  --in-place --remove-all-unused-imports --remove-unused-variables

echo "🎨 Step 3: Format code with black"
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 black

echo "📦 Step 4: Sort imports with isort"
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 isort

echo "📚 Step 5: Format docstrings"
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 docformatter \
  --in-place --wrap-summaries=100 --wrap-description=100

echo "✅ Step 6: Type checking with mypy"
mypy . --exclude ".venv" --show-error-codes --pretty

echo "🔍 Step 7: Deep linting with pylint"
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 pylint \
  --rcfile=pyproject.toml

echo "🎉 All checks completed!"
```

Make executable:
```bash
chmod +x scripts/format_and_lint.sh
```

Run with:
```bash
./scripts/format_and_lint.sh
```

---

## Code Quality Tools

### 1. Ruff - Fast Python Linter

Ruff is our primary linting tool, replacing multiple tools (autopep8, pyflakes, flake8, etc.).

#### Basic Commands

```bash
# Check for issues
ruff check .

# Fix automatically fixable issues
ruff check --fix .

# Check specific files
ruff check src/image_similarity/

# Show help
ruff check --help
```

#### Advanced Configuration

Edit `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py38"
line-length = 88
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "UP",   # pyupgrade
    "C4",   # flake8-comprehensions
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "I",    # isort
    "N",    # pep8-naming
]
ignore = [
    "E501", # line too long (handled by black)
    "B008", # do not perform function calls in argument defaults
    "C901", # too complex
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]
"tests/*" = ["B018", "B017"]
```

### 2. Black - Code Formatter

Black enforces consistent code formatting.

#### Basic Commands

```bash
# Format all Python files
black .

# Check without modifying
black --check .

# Show diff
black --diff .

# Format specific files
black src/image_similarity/core.py

# Show help
black --help
```

#### Configuration

```toml
[tool.black]
target-version = ["py38", "py39", "py310", "py311"]
line-length = 88
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

### 3. isort - Import Sorter

Organizes imports alphabetically within sections.

#### Basic Commands

```bash
# Sort imports
isort .

# Check without modifying
isort --check-only .

# Show diff
isort --diff .

# Use black profile (recommended)
isort --profile black .

# Show help
isort --help
```

#### Configuration

```toml
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
src_paths = ["src", "tests"]

[tool.isort.sections]
"FUTURE" = ["__future__"]
"STDLIB" = ["os", "sys", "collections", "itertools", "operator", "random", "re", "math"]
"THIRDPARTY" = ["numpy", "torch", "transformers", "PIL", "sklearn"]
"FIRSTPARTY" = ["image_similarity"]
"LOCALFOLDER" = [".", ".."]
```

### 4. Autoflake - Remove Unused Code

Removes unused imports and variables.

#### Basic Commands

```bash
# Remove unused imports and variables
autoflake --in-place --remove-all-unused-imports --remove-unused-variables *.py

# For entire project
find . -name "*.py" -not -path "./.venv/*" -exec autoflake --in-place --remove-all-unused-imports --remove-unused-variables {} +

# Show help
autoflake --help
```

### 5. MyPy - Type Checker

Static type checking for Python code.

#### Basic Commands

```bash
# Type check entire project
mypy .

# Type check specific module
mypy src/image_similarity/database.py

# Show error codes
mypy --show-error-codes .

# Show colored output
mypy --pretty .

# Show help
mypy --help
```

#### Configuration

```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "torch.*",
    "numpy.*", 
    "PIL.*",
    "sklearn.*",
    "transformers.*",
]
ignore_missing_imports = true
```

### 6. Pylint - Deep Code Analysis

Comprehensive code quality analysis.

#### Basic Commands

```bash
# Check specific file
pylint src/image_similarity/core.py

# Check entire project
find . -name "*.py" -not -path "./.venv/*" | xargs pylint

# Use custom config
pylint --rcfile=pyproject.toml .

# Generate report
pylint --output-format=json . > pylint_report.json

# Show help
pylint --help
```

#### Configuration

Add to `pyproject.toml`:

```toml
[tool.pylint.messages_control]
disable = [
    "too-few-public-methods",
    "too-many-arguments",
    "too-many-instance-attributes",
    "too-many-locals",
    "too-many-branches",
    "too-many-statements",
    "missing-docstring",
    "invalid-name",
    "broad-except",
]

[tool.pylint.format]
max-line-length = 88
good-names = ["i", "j", "k", "ex", "Run", "_"]

[tool.pylint.design]
max-complexity = 10

[tool.pylint.typecheck]
ignored-modules = [
    "torch",
    "numpy",
    "PIL",
    "sklearn",
    "transformers",
]
```

### 7. Docformatter - Docstring Formatter

Formats docstrings to follow PEP 257 conventions.

#### Basic Commands

```bash
# Format docstrings
docformatter --in-place --wrap-summaries=100 --wrap-descriptions=100 *.py

# For entire project
find . -name "*.py" -not -path "./.venv/*" -exec docformatter --in-place --wrap-summaries=100 --wrap-descriptions=100 {} +

# Show help
docformatter --help
```

---

## Configuration

### PyProject.toml Configuration

Create or update `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.2"]

[project]
name = "image-similarity-toolkit"
description = "Professional image similarity toolkit with deep learning models"
version = "0.2.0"
authors = [
    {name = "MiniMax Agent", email = "agent@minimax.com"},
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.21.0",
    "Pillow>=9.0.0",
    "scikit-learn>=1.1.0",
    "matplotlib>=3.5.0",
    "tqdm>=4.64.0",
    "pathlib",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
    "pylint>=2.17.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "autoflake>=2.0.0",
    "docformatter>=1.5.0",
    "pre-commit>=3.0.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/image-similarity-toolkit"
Repository = "https://github.com/yourusername/image-similarity-toolkit.git"
Documentation = "https://github.com/yourusername/image-similarity-toolkit/blob/main/docs"
"Bug Tracker" = "https://github.com/yourusername/image-similarity-toolkit/issues"

[project.scripts]
image-similarity = "image_similarity.cli:main"
database-example = "image_similarity.examples.database_example:main"
duplicate-cleaner = "image_similarity.examples.duplicate_cleaner:main"

# Tool configurations

[tool.ruff]
target-version = "py38"
line-length = 88
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "UP",   # pyupgrade
    "C4",   # flake8-comprehensions
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "I",    # isort
    "N",    # pep8-naming
    "PT",   # flake8-pytest-style
]
ignore = [
    "E501", # line too long (handled by black)
    "B008", # do not perform function calls in argument defaults
    "C901", # too complex
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]
"tests/*" = ["B018", "B017", "PT022"]

[tool.black]
target-version = ["py38", "py39", "py310", "py311"]
line-length = 88
include = '\.pyi?$'
extend-exclude = '''
/(
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
src_paths = ["src", "tests"]

[tool.isort.sections]
"FUTURE" = ["__future__"]
"STDLIB" = ["os", "sys", "collections", "itertools", "operator", "random", "re", "math"]
"THIRDPARTY" = ["numpy", "torch", "transformers", "PIL", "sklearn", "matplotlib"]
"FIRSTPARTY" = ["image_similarity"]
"LOCALFOLDER" = [".", ".."]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "torch.*",
    "numpy.*", 
    "PIL.*",
    "sklearn.*",
    "matplotlib.*",
    "transformers.*",
]
ignore_missing_imports = true

[tool.pylint.messages_control]
disable = [
    "too-few-public-methods",
    "too-many-arguments",
    "too-many-instance-attributes",
    "too-many-locals",
    "too-many-branches",
    "too-many-statements",
    "missing-docstring",
    "invalid-name",
    "broad-except",
]

[tool.pylint.format]
max-line-length = 88
good-names = ["i", "j", "k", "ex", "Run", "_"]

[tool.pylint.design]
max-complexity = 10

[tool.pylint.typecheck]
ignored-modules = [
    "torch",
    "numpy",
    "PIL",
    "sklearn",
    "matplotlib",
    "transformers",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src/image_similarity",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
]

[tool.coverage.run]
source = ["src/image_similarity"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/migrations/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

### Ignore Patterns

Create `.ruffignore` or `.gitignore` patterns:

```
# Ruff ignore patterns
__pycache__/
.pytest_cache/
.venv/
.eggs/
*.egg-info/
build/
dist/
.mypy_cache/
.coverage
htmlcov/
.cache/
```

---

## Pre-commit Hooks

Pre-commit hooks ensure code quality before each commit.

### Setup Pre-commit

1. Install pre-commit:
```bash
pip install pre-commit
```

2. Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/myint/autoflake
    rev: v2.2.1
    hooks:
      - id: autoflake
        args:
          - --in-place
          - --remove-all-unused-imports
          - --remove-unused-variables

  - repo: https://github.com/myint/docformatter
    rev: v1.7.5
    hooks:
      - id: docformatter
        args:
          - --in-place
          - --wrap-summaries=100
          - --wrap-descriptions=100

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/codespell-project/codespell
    rev: v2.2.6
    hooks:
      - id: codespell
        args: [--ignore-words-list=nd,ned,te,thems,ur,wich]
```

3. Install hooks:
```bash
pre-commit install
```

4. Run on all files (first time):
```bash
pre-commit run --all-files
```

### Usage

Now every commit automatically:
- Fixes linting issues
- Formats code
- Sorts imports
- Removes unused imports
- Formats docstrings
- Checks file formatting

Skip hooks for specific commit:
```bash
git commit --no-verify -m "Emergency commit"
```

---

## IDE Integration

### VS Code

Create `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.pylintEnabled": false,
  "python.formatting.provider": "black",
  "python.sortImports.args": ["--profile", "black"],
  "python.analysis.autoImportCompletions": true,
  "python.analysis.typeCheckingMode": "strict",
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.mypy_cache": true,
    "**/.pytest_cache": true
  }
}
```

Install VS Code extensions:
- Python
- Black Formatter
- isort
- Ruff

### PyCharm

1. Enable Black: Settings → Tools → External Tools → Add Black
2. Enable isort: Settings → Tools → External Tools → Add isort
3. Configure Ruff: Settings → Tools → Ruff
4. Enable auto-save formatting

### Vim/Neovim

Add to `.vimrc` or `init.vim`:

```vim
" Python formatting
autocmd FileType python setlocal formatprg=black\ --quiet\ -
autocmd BufWritePost *.py silent !ruff check --fix %

" Autoformat on save
autocmd BufWritePre *.py execute '%!black -q -'
autocmd BufWritePre *.py execute '%!isort --quiet -'

" Quickfix for linting errors
autocmd QuickFixCmdPost [^l]* nested cwindow
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Ruff vs Black Conflicts

**Problem**: Ruff removes trailing commas that Black adds

**Solution**: Use compatible configuration:

```toml
[tool.ruff]
line-length = 88  # Same as black

[tool.ruff.format]
quote-style = "double"  # Same as black
```

#### 2. MyPy Import Errors

**Problem**: Cannot find installed packages

**Solution**: Install in development mode:

```bash
pip install -e .
```

Or configure paths:

```toml
[tool.mypy]
mypy_path = ["src"]
```

#### 3. Pylint Too Strict

**Problem**: Too many warnings about code style

**Solution**: Adjust configuration:

```toml
[tool.pylint.messages_control]
disable = [
    "missing-docstring",
    "invalid-name",
    "too-few-public-methods",
]
```

#### 4. Pre-commit Hook Failures

**Problem**: Hook modifies files unexpectedly

**Solution**: Update hooks and re-run:

```bash
pre-commit autoupdate
pre-commit run --all-files
```

#### 5. Performance Issues

**Problem**: Tools are too slow

**Solution**: Use Ruff which is faster:

```bash
# Replace flake8, autopep8, pyflakes with ruff
ruff check --fix .
black .
```

### Performance Optimization

#### For Large Projects

1. Use Ruff instead of flake8 + pyflakes
2. Parallel execution:
```bash
find . -name "*.py" -not -path "./.venv/*" | xargs -P 4 pylint
```

3. Cache results:
```bash
# Ruff automatically caches
# MyPy cache in .mypy_cache/
```

#### Skip Expensive Checks

```bash
# Skip mypy for quick commits
git commit -m "Quick fix" && mypy . || echo "Type check failed"

# Skip pylint for small changes
git commit -m "Minor change" && pylint src/file.py || echo "Pylint failed"
```

---

## Best Practices

### Code Quality Workflow

1. **Write code** with type hints
2. **Run locally** before committing:
```bash
ruff check --fix .
black .
mypy .
```
3. **Use pre-commit hooks** for automated checks
4. **Review** pylint output carefully
5. **Document** complex functions with docstrings

### Type Hints Best Practices

```python
from typing import List, Dict, Optional, Union, Any
import numpy as np
from pathlib import Path

def process_images(
    image_paths: List[Union[str, Path]],
    model_name: str = "efficientnet",
    threshold: float = 0.8
) -> Dict[str, float]:
    """Process list of images and return similarity scores.
    
    Args:
        image_paths: List of image file paths
        model_name: Name of the model to use
        threshold: Similarity threshold for filtering
        
    Returns:
        Dictionary mapping image paths to similarity scores
        
    Raises:
        FileNotFoundError: If any image file doesn't exist
        ValueError: If model_name is not supported
    """
    results: Dict[str, float] = {}
    
    for path in image_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image not found: {path}")
            
        # Process image
        score = calculate_similarity(path_obj, model_name, threshold)
        results[str(path)] = score
        
    return results
```

### Documentation Standards

1. **Google Style Docstrings**:

```python
def embedding_function(image_path: str) -> Optional[np.ndarray]:
    """Generate embedding for an image.
    
    This function loads an image from the given path and generates
    a feature embedding using the specified model.
    
    Args:
        image_path: Path to the image file. Must be a valid
            image format (JPEG, PNG, etc.).
            
    Returns:
        A numpy array containing the image embedding, or None
        if the image could not be loaded or processed.
        
    Raises:
        FileNotFoundError: If the image file doesn't exist.
        ValueError: If the image file format is unsupported.
        
    Example:
        >>> embedding = embedding_function("cat.jpg")
        >>> print(embedding.shape)
        (1280,)
    """
    pass
```

2. **Module Docstrings**:

```python
"""Image similarity toolkit core functionality.

This module provides the main classes and functions for computing
image similarities using various deep learning models.

Classes:
    ImageSimilarity: Main class for similarity computations.
    EmbeddingDatabase: Database operations for embeddings.

Functions:
    load_model: Load a specific deep learning model.
    preprocess_image: Standard image preprocessing pipeline.
"""

from .core import ImageSimilarity
from .database import EmbeddingDatabase
from .models import load_model, preprocess_image

__all__ = [
    "ImageSimilarity",
    "EmbeddingDatabase", 
    "load_model",
    "preprocess_image",
]
```

---

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install -r requirements-dev.txt
        
    - name: Run Ruff
      run: ruff check .
      
    - name: Run Black
      run: black --check .
      
    - name: Run isort
      run: isort --check-only .
      
    - name: Run MyPy
      run: mypy .
      
    - name: Run Pylint
      run: pylint --rcfile=pyproject.toml src/
      
    - name: Run tests
      run: pytest --cov=src/image_similarity
      
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## Summary

This development guide provides a complete workflow for maintaining high code quality in the Image Similarity Toolkit. Key takeaways:

1. **Use Ruff** as primary linter (replaces flake8, pyflakes, etc.)
2. **Follow the order**: Ruff → Autoflake → Black → isort → docformatter → MyPy → Pylint
3. **Set up pre-commit hooks** for automated quality checks
4. **Configure your IDE** for seamless integration
5. **Write comprehensive type hints** and docstrings
6. **Use CI/CD** for automated quality assurance

The tools work together to ensure:
- ✅ Syntax correctness
- ✅ Code formatting consistency
- ✅ Import organization
- ✅ Type safety
- ✅ Code quality analysis
- ✅ Documentation standards

Regular use of these tools will result in cleaner, more maintainable, and more reliable code.

---

Last updated: 2025-11-16
Version: 0.2.0
Author: MiniMax Agent