# Contributing to Image Similarity Toolkit

Thank you for your interest in contributing to the Image Similarity Toolkit! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Pull Request Process](#pull-request-process)

## Code of Conduct

This project follows a code of conduct to ensure a welcoming and inclusive environment:

- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Accept criticism gracefully
- Prioritize the community's best interests

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of deep learning and image processing
- Familiarity with PyTorch

### Finding Issues to Work On

- Check the [Issues](https://github.com/yourusername/image-similarity-toolkit/issues) page
- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are ready for contributions
- Feel free to propose new features or improvements

## Development Setup

1. **Fork the repository**
   ```bash
   # Click 'Fork' on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/image-similarity-toolkit.git
   cd image-similarity-toolkit
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

When reporting bugs, include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: OS, Python version, PyTorch version, etc.
- **Screenshots**: If applicable
- **Code Sample**: Minimal code to reproduce the issue

Example:
```markdown
**Bug Description**
Comparison fails when using CLIP model with grayscale images

**Steps to Reproduce**
1. Initialize ImageSimilarity with model_name='clip'
2. Call compare_images with a grayscale image
3. Error occurs

**Environment**
- OS: Ubuntu 20.04
- Python: 3.9.5
- PyTorch: 2.0.1
```

### Suggesting Enhancements

When suggesting features:

- **Use Case**: Explain why this feature would be useful
- **Proposed Solution**: Describe how it might work
- **Alternatives**: Consider alternative approaches
- **Impact**: Who would benefit from this feature

### Code Contributions

Types of contributions we welcome:

- Bug fixes
- New features
- Documentation improvements
- Performance optimizations
- Test coverage improvements
- Code refactoring

## Coding Standards

### Python Style

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines:

```python
# Good
def calculate_similarity(image1: str, image2: str) -> float:
    """
    Calculate cosine similarity between two images.
    
    Args:
        image1: Path to first image
        image2: Path to second image
        
    Returns:
        Cosine similarity value
    """
    pass

# Bad
def calc_sim(img1,img2):
    pass
```

### Code Formatting

Use automated tools:

```bash
# Format code with black
black src/ tests/ examples/

# Check style with flake8
flake8 src/ tests/ examples/

# Type checking with mypy
mypy src/
```

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type hints
- Provide usage examples in docstrings

Example:
```python
def compare_images(
    self, 
    image_path1: str, 
    image_path2: str
) -> Dict[str, float]:
    """
    Compare two images and return similarity metrics.
    
    Args:
        image_path1: Path to the first image
        image_path2: Path to the second image
        
    Returns:
        Dictionary containing similarity metrics
        
    Raises:
        FileNotFoundError: If image files don't exist
        ValueError: If images cannot be processed
        
    Example:
        >>> checker = ImageSimilarity()
        >>> results = checker.compare_images('cat.jpg', 'dog.jpg')
        >>> print(results['cosine_similarity'])
        0.4523
    """
    pass
```

## Testing Guidelines

### Writing Tests

- Write tests for all new features
- Maintain or improve test coverage
- Use pytest for testing
- Follow the Arrange-Act-Assert pattern

Example:
```python
def test_compare_similar_images():
    # Arrange
    checker = ImageSimilarity(model_name='efficientnet')
    img1 = 'tests/data/cat1.jpg'
    img2 = 'tests/data/cat2.jpg'
    
    # Act
    results = checker.compare_images(img1, img2)
    
    # Assert
    assert results['cosine_similarity'] > 0.8
    assert 'euclidean_distance' in results
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_similarity.py

# Run specific test
pytest tests/test_similarity.py::test_compare_images
```

### Test Coverage

Aim for:
- Minimum 80% code coverage
- 100% coverage for critical functions
- Tests for edge cases and error conditions

## Pull Request Process

### Before Submitting

1. **Update your fork**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**
   ```bash
   pytest tests/
   ```

3. **Check code style**
   ```bash
   black src/ tests/ examples/
   flake8 src/ tests/ examples/
   ```

4. **Update documentation**
   - Update README if needed
   - Add/update docstrings
   - Update CHANGELOG.md

### Submitting PR

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Go to GitHub and create a PR
   - Use a clear, descriptive title
   - Fill out the PR template
   - Link related issues

3. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] All tests pass
   - [ ] Added new tests
   - [ ] Updated documentation
   
   ## Related Issues
   Fixes #123
   
   ## Screenshots (if applicable)
   ```

### Review Process

- Maintainers will review your PR
- Address feedback promptly
- Keep PR focused and atomic
- Be patient and respectful

### After Merge

- Delete your feature branch
- Update your local repository
- Celebrate your contribution!

## Development Tips

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use breakpoint for debugging
def compare_images(self, img1, img2):
    breakpoint()  # Debugger will stop here
    ...
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## Questions?

- Open an [issue](https://github.com/yourusername/image-similarity-toolkit/issues)
- Email: your.email@example.com
- Check [documentation](docs/usage.md)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Image Similarity Toolkit! 🎉
