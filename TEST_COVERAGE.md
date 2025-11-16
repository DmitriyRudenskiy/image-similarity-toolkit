# Code Coverage Guide

Guide for measuring and improving test coverage in the Image Similarity Toolkit.

## Running Coverage Tests

### Basic Coverage Report
```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML report
pytest tests/ --cov=src --cov-report=html
```

### Coverage Reports

#### Terminal Report
```bash
pytest tests/ --cov=src --cov-report=term-missing
```
Shows coverage percentage and missing lines.

#### HTML Report
```bash
pytest tests/ --cov=src --cov-report=html
pytest tests/ --cov=src --cov-report=html --cov-report=term
```
Generates detailed HTML report in `htmlcov/index.html`.

#### XML Report
```bash
pytest tests/ --cov=src --cov-report=xml
```
Generates XML report for CI/CD integration.

## Coverage Configuration

### pytest.ini Configuration
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80
    --strict-markers
    --verbose
```

### .coveragerc Configuration
```ini
[run]
source = src
omit = 
    */tests/*
    */test_*
    */conftest.py
    */setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:

[html]
directory = htmlcov
```

## Coverage Goals

### Minimum Coverage Targets
- **Overall**: 80% minimum
- **Domain Layer**: 90% minimum (critical business logic)
- **Application Layer**: 85% minimum (use cases)
- **Infrastructure Layer**: 75% minimum (implementation details)

### DDD Layer Coverage Focus

#### Domain Layer (90%)
- Value Objects validation
- Domain Services logic
- Aggregate behavior
- Business rules

#### Application Layer (85%)
- Use Case execution
- Input validation
- Error handling
- Output formatting

#### Infrastructure Layer (75%)
- Repository implementations
- Generator implementations
- Processor functionality
- Integration patterns

## Writing Coverage-Friendly Tests

### Test Structure
```python
import pytest
from src.domain.vector_storage import VectorEmbedding
from src.application.use_cases import AddImageUseCase

class TestAddImageUseCase:
    """Test AddImageUseCase with comprehensive coverage."""
    
    def test_successful_add_image(self, mock_repository, mock_generator, mock_processor):
        """Test successful image addition - happy path."""
        # Arrange
        use_case = AddImageUseCase(mock_repository, mock_generator, mock_processor)
        image_path = Path("test.jpg")
        
        # Act
        result = use_case.execute(image_path)
        
        # Assert
        assert result.is_success
        assert result.embedding_id is not None
    
    def test_invalid_image_path(self, mock_repository, mock_generator, mock_processor):
        """Test handling of invalid image path."""
        # Arrange
        use_case = AddImageUseCase(mock_repository, mock_generator, mock_processor)
        invalid_path = Path("nonexistent.jpg")
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            use_case.execute(invalid_path)
    
    def test_repository_failure(self, mock_repository, mock_generator, mock_processor):
        """Test handling of repository failure."""
        # Arrange
        mock_repository.save.side_effect = Exception("Database error")
        use_case = AddImageUseCase(mock_repository, mock_generator, mock_processor)
        
        # Act
        result = use_case.execute(Path("test.jpg"))
        
        # Assert
        assert not result.is_success
        assert "Database error" in result.error_message
```

### Coverage Testing Patterns

#### Path Coverage
```python
def test_all_code_paths():
    """Test all branches and paths."""
    # Test if-elif-else branches
    # Test try-except blocks
    # Test loop iterations (0, 1, many)
    # Test boundary conditions
```

#### Exception Coverage
```python
def test_exception_handling():
    """Test all exception paths."""
    with pytest.raises(SpecificException):
        function_that_raises()
```

#### Edge Case Coverage
```python
def test_edge_cases():
    """Test boundary conditions."""
    # Test empty collections
    # Test None values
    # Test zero/negative values
    # Test maximum values
```

## Measuring Coverage

### Manual Coverage Check
```bash
# Run specific test module
pytest tests/test_vector_repository.py --cov=src.infrastructure.repositories --cov-report=term-missing

# Run single test with coverage
pytest tests/test_add_image_use_case.py::TestAddImageUseCase::test_successful_add_image --cov=src --cov-report=term-missing
```

### CI Integration
```yaml
# .github/workflows/test.yml
- name: Run tests with coverage
  run: |
    pytest tests/ --cov=src --cov-report=xml --cov-fail-under=80
    
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Coverage Improvement Workflow

### 1. Run Initial Coverage
```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

### 2. Identify Low Coverage Areas
```bash
# Check specific modules
pytest tests/ --cov=src.domain.vector_storage --cov-report=term-missing
```

### 3. Write Missing Tests
```bash
# Run specific test
pytest tests/test_low_coverage_module.py -v
```

### 4. Re-measure Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## Coverage Metrics

### What to Measure
- **Line Coverage**: Percentage of lines executed
- **Branch Coverage**: Percentage of branches taken
- **Function Coverage**: Percentage of functions called
- **Statement Coverage**: Percentage of statements executed

### Coverage Reports
- **Term**: Terminal output
- **HTML**: Detailed web report
- **XML**: CI/CD integration
- **JSON**: Programmatic access

## Best Practices

### Test Pyramid
- **Unit Tests**: 70% of tests (fast, isolated)
- **Integration Tests**: 20% of tests (components working together)
- **E2E Tests**: 10% of tests (full workflow)

### Coverage Strategy
- **Business Logic**: 100% coverage (domain layer)
- **Use Cases**: 90% coverage (application layer)
- **Infrastructure**: 80% coverage (implementation)
- **Error Handling**: 100% coverage (all exception paths)

### DDD Coverage Focus
- **Value Objects**: Immutable, all methods tested
- **Domain Services**: Pure functions, all scenarios
- **Aggregates**: State transitions, invariants
- **Use Cases**: Happy path, error paths, edge cases

## Common Coverage Issues

### Low Coverage Areas
1. **Error Handling**: Often not tested
2. **Edge Cases**: Boundary conditions
3. **Integration Points**: External dependencies
4. **Configuration**: Initialization code

### Coverage Anti-patterns
- Testing implementation details
- Using `@patch` without need
- Ignoring slow tests
- Focusing only on percentage

## Tools and Commands

### Essential Commands
```bash
# Basic coverage
pytest --cov=src

# Detailed report
pytest --cov=src --cov-report=term-missing

# HTML report
pytest --cov=src --cov-report=html

# Fail under threshold
pytest --cov=src --cov-fail-under=80

# Specific module
pytest --cov=src.domain.vector_storage

# With markers
pytest -m "not slow" --cov=src
```

### Coverage Tools
- **pytest-cov**: pytest coverage plugin
- **coverage.py**: Coverage measurement tool
- **codecov**: Coverage reporting service
- **coveralls**: Coverage tracking service

## Monitoring Coverage

### Continuous Integration
- Fail builds under threshold
- Track coverage trends
- Generate coverage reports
- Alert on coverage drops

### Local Development
- Run coverage before committing
- Check coverage in pull requests
- Monitor coverage metrics
- Set coverage goals

Coverage testing ensures code quality and reliability across all DDD layers.