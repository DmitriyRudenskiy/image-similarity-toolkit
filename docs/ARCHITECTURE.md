# Project Architecture

## Overview

Image Similarity Toolkit is a modular Python library designed for comparing images using deep learning models. The architecture follows a clean separation of concerns with distinct modules for different functionalities.

## Directory Structure

```
image-similarity-toolkit/
├── src/image_similarity/      # Main package
│   ├── __init__.py           # Package initialization and exports
│   ├── core.py               # Core ImageSimilarity class
│   ├── models.py             # Model loading and management
│   └── visualization.py      # Visualization utilities
├── examples/                  # Usage examples
│   ├── basic_usage.py        # Simple comparison example
│   └── batch_comparison.py   # Batch processing example
├── tests/                     # Test suite
│   ├── __init__.py
│   └── test_similarity.py    # Unit tests
├── docs/                      # Documentation
│   └── usage.md              # Detailed usage guide
├── data/                      # Data directory
│   ├── input/                # Input images
│   └── output/               # Generated outputs
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup configuration
├── README.md                 # Project overview
├── CHANGELOG.md              # Version history
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # MIT License
└── .gitignore               # Git ignore rules
```

## Component Architecture

### 1. Core Module (`core.py`)

**Purpose**: Main business logic for image similarity comparison

**Key Components**:
- `ImageSimilarity` class: Primary interface for users
  - Initialization with model selection
  - Embedding extraction from images
  - Similarity metric calculation
  - Result interpretation

**Design Patterns**:
- Facade Pattern: Provides simple interface to complex subsystems
- Strategy Pattern: Different models can be selected at runtime

**Dependencies**:
- PyTorch for deep learning operations
- NumPy for numerical computations
- PIL for image loading
- scikit-learn for similarity metrics

### 2. Models Module (`models.py`)

**Purpose**: Model loading, initialization, and management

**Key Functions**:
- `load_model()`: Factory function for model creation
- `_load_resnet50()`: ResNet50 model loader
- `_load_efficientnet_b0()`: EfficientNet-B0 model loader
- `_load_clip()`: CLIP model loader
- `get_embedding_size()`: Utility for embedding dimensions

**Design Patterns**:
- Factory Pattern: Dynamic model creation based on string identifier
- Singleton-like: Models loaded once and reused

**Features**:
- Automatic device selection (GPU/CPU)
- Lazy loading of CLIP to avoid mandatory dependency
- Standardized preprocessing pipelines

### 3. Visualization Module (`visualization.py`)

**Purpose**: Result visualization and presentation

**Key Functions**:
- `setup_matplotlib_for_plotting()`: Environment configuration
- `visualize_comparison()`: Single comparison visualization
- `visualize_batch_results()`: Batch results chart

**Design Patterns**:
- Template Method: Standardized plotting workflow
- Decorator Pattern: Plot customization based on metrics

**Features**:
- Cross-platform font support
- Color-coded results based on similarity thresholds
- Automatic figure management and cleanup

## Data Flow

```
┌─────────────┐
│ User Input  │
│ (2 images)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ ImageSimilarity     │
│ .compare_images()   │
└──────┬──────────────┘
       │
       ├─────────────────────┐
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Load Image 1 │      │ Load Image 2 │
│ (PIL)        │      │ (PIL)        │
└──────┬───────┘      └──────┬───────┘
       │                     │
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Preprocess 1 │      │ Preprocess 2 │
│ (transforms) │      │ (transforms) │
└──────┬───────┘      └──────┬───────┘
       │                     │
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Model        │      │ Model        │
│ Forward Pass │      │ Forward Pass │
└──────┬───────┘      └──────┬───────┘
       │                     │
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Embedding 1  │      │ Embedding 2  │
│ (numpy array)│      │ (numpy array)│
└──────┬───────┘      └──────┬───────┘
       │                     │
       └──────────┬──────────┘
                  ▼
         ┌────────────────┐
         │ Calculate      │
         │ Similarities   │
         │ - Cosine       │
         │ - Euclidean    │
         │ - Normalized   │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ Return Results │
         │ (Dictionary)   │
         └────────────────┘
```

## Model Architecture

### Supported Models

#### 1. ResNet50
```
Input Image (H×W×3)
    ↓
Preprocessing (224×224)
    ↓
ResNet50 Backbone
    ↓
Remove Classification Head
    ↓
Global Average Pooling
    ↓
Embedding (2048-dim)
```

#### 2. EfficientNet-B0
```
Input Image (H×W×3)
    ↓
Preprocessing (224×224)
    ↓
EfficientNet-B0 Backbone
    ↓
Remove Classification Head
    ↓
Global Average Pooling
    ↓
Embedding (1280-dim)
```

#### 3. CLIP
```
Input Image (H×W×3)
    ↓
Preprocessing (224×224)
    ↓
Vision Transformer (ViT-B/32)
    ↓
L2 Normalization
    ↓
Embedding (512-dim)
```

## Similarity Metrics

### Cosine Similarity
```python
similarity = (A · B) / (||A|| × ||B||)
```
- Measures angle between vectors
- Range: [-1, 1] (typically [0, 1] for images)
- Invariant to magnitude

### Euclidean Distance
```python
distance = √(Σ(A[i] - B[i])²)
```
- Direct distance in embedding space
- Range: [0, ∞)
- Sensitive to magnitude

### Normalized Similarity
```python
similarity = 1 / (1 + euclidean_distance)
```
- Normalized version of Euclidean distance
- Range: [0, 1]
- Easier to interpret

## Extension Points

### Adding New Models

1. Create loader function in `models.py`:
```python
def _load_new_model(device: str):
    model = YourModel()
    preprocess = YourPreprocess()
    model.to(device)
    model.eval()
    return model, preprocess
```

2. Register in `load_model()`:
```python
elif model_name == 'new_model':
    return _load_new_model(device)
```

3. Update `get_embedding_size()`:
```python
embedding_sizes = {
    # ...
    'new_model': YOUR_SIZE
}
```

### Adding New Metrics

1. Extend `compare_images()` in `core.py`:
```python
# Add your metric calculation
new_metric = your_calculation(embedding1, embedding2)

return {
    # ... existing metrics
    'new_metric': new_metric
}
```

### Custom Visualizations

1. Create new function in `visualization.py`:
```python
def visualize_custom(data, save_path=None):
    setup_matplotlib_for_plotting()
    # Your visualization code
    plt.savefig(save_path)
    plt.close()
```

## Performance Considerations

### Memory Management
- Models are loaded once per instance
- Embeddings can be cached for repeated comparisons
- GPU memory is cleared after inference

### Optimization Strategies
1. **Batch Processing**: Pre-compute embeddings for multiple images
2. **GPU Utilization**: Automatic CUDA detection and usage
3. **Model Selection**: EfficientNet for speed, ResNet for accuracy
4. **Caching**: Store embeddings to avoid re-computation

### Scalability
- **Small Scale** (<100 images): Direct comparison works well
- **Medium Scale** (100-10,000): Use batch processing
- **Large Scale** (>10,000): Consider vector databases (FAISS, Milvus)

## Testing Strategy

### Unit Tests
- Test each function independently
- Mock external dependencies
- Cover edge cases and error conditions

### Integration Tests
- Test complete workflows
- Use real images when possible
- Verify output format and accuracy

### Performance Tests
- Benchmark different models
- Measure memory usage
- Profile bottlenecks

## Security Considerations

1. **Input Validation**: Verify image files before processing
2. **Path Sanitization**: Prevent directory traversal attacks
3. **Resource Limits**: Prevent memory exhaustion with large images
4. **Error Handling**: Don't expose internal paths in error messages

## Future Architecture Plans

1. **Plugin System**: Dynamic model loading from external modules
2. **REST API**: Web service for remote comparisons
3. **Database Integration**: Efficient storage and retrieval
4. **Distributed Processing**: Multi-GPU and multi-node support
5. **Streaming Pipeline**: Real-time image comparison

---

Last updated: 2025-11-15
Version: 0.1.0
