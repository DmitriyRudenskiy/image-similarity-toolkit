# Image Similarity Toolkit - Usage Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Detailed Usage](#detailed-usage)
4. [Model Selection](#model-selection)
5. [Understanding Metrics](#understanding-metrics)
6. [Advanced Examples](#advanced-examples)
7. [Performance Tips](#performance-tips)
8. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA support for faster processing

### Step-by-step Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-similarity-toolkit.git
cd image-similarity-toolkit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install CLIP model:
```bash
pip install git+https://github.com/openai/CLIP.git
```

4. Install the package:
```bash
pip install -e .
```

## Quick Start

### Basic Comparison

```python
from image_similarity import ImageSimilarity

# Initialize
checker = ImageSimilarity(model_name='efficientnet')

# Compare two images
results = checker.compare_images('image1.jpg', 'image2.jpg')

# Print results
print(f"Similarity: {results['cosine_similarity']:.4f}")
```

## Detailed Usage

### Initializing the Checker

```python
from image_similarity import ImageSimilarity

# Using EfficientNet (recommended for speed)
checker = ImageSimilarity(model_name='efficientnet')

# Using ResNet50 (good accuracy)
checker = ImageSimilarity(model_name='resnet')

# Using CLIP (best for semantic understanding)
checker = ImageSimilarity(model_name='clip')
```

### Comparing Images

```python
# Compare two images
results = checker.compare_images('cat1.jpg', 'cat2.jpg')

# Access different metrics
cosine_sim = results['cosine_similarity']
euclidean_dist = results['euclidean_distance']
normalized_sim = results['normalized_similarity']

# Get raw embeddings
embedding1 = results['embedding1']
embedding2 = results['embedding2']
```

### Visualizing Results

```python
# Create and save visualization
checker.visualize_comparison(
    'image1.jpg',
    'image2.jpg',
    results,
    save_path='comparison.png'
)
```

### Interpreting Results

```python
# Get human-readable interpretation
interpretation = checker.interpret_similarity(results['cosine_similarity'])
print(interpretation)  # e.g., "Very similar images"
```

## Model Selection

### ResNet50
- **Pros**: Well-established, good accuracy
- **Cons**: Larger model size
- **Best for**: General purpose image comparison
- **Embedding size**: 2048

```python
checker = ImageSimilarity(model_name='resnet')
```

### EfficientNet-B0
- **Pros**: Fast, efficient, good balance
- **Cons**: Slightly less accurate than larger models
- **Best for**: Real-time applications, production use
- **Embedding size**: 1280

```python
checker = ImageSimilarity(model_name='efficientnet')
```

### CLIP
- **Pros**: Excellent semantic understanding
- **Cons**: Requires additional installation
- **Best for**: Cross-modal comparisons, semantic similarity
- **Embedding size**: 512

```python
checker = ImageSimilarity(model_name='clip')
```

## Understanding Metrics

### Cosine Similarity

**Range**: -1 to 1 (typically 0 to 1 for images)

**Interpretation**:
- 0.85 - 1.0: Very similar (same object, similar scenes)
- 0.70 - 0.85: Moderately similar (similar objects or themes)
- 0.50 - 0.70: Weakly similar (some common features)
- 0.00 - 0.50: Different (distinct images)

### Euclidean Distance

**Range**: 0 to infinity

**Interpretation**:
- Lower values = more similar
- Higher values = more different
- Absolute values depend on embedding size

### Normalized Similarity

**Range**: 0 to 1

**Interpretation**:
- 1.0: Identical
- 0.5: Moderately similar
- 0.0: Very different

## Advanced Examples

### Batch Processing

```python
import os
from pathlib import Path

checker = ImageSimilarity(model_name='efficientnet')

# Get all images in directory
image_dir = Path('images')
images = list(image_dir.glob('*.jpg'))

# Compare all pairs
results = []
reference = images[0]

for img in images[1:]:
    result = checker.compare_images(str(reference), str(img))
    results.append((img.name, result['cosine_similarity']))

# Sort by similarity
results.sort(key=lambda x: x[1], reverse=True)

# Print top 5 most similar
for name, similarity in results[:5]:
    print(f"{name}: {similarity:.4f}")
```

### Finding Duplicates

```python
def find_duplicates(image_dir, threshold=0.95):
    """Find duplicate or near-duplicate images."""
    checker = ImageSimilarity(model_name='efficientnet')
    
    images = list(Path(image_dir).glob('*.jpg'))
    duplicates = []
    
    for i, img1 in enumerate(images):
        for img2 in images[i+1:]:
            results = checker.compare_images(str(img1), str(img2))
            
            if results['cosine_similarity'] > threshold:
                duplicates.append((img1.name, img2.name, results['cosine_similarity']))
    
    return duplicates

# Usage
duplicates = find_duplicates('my_images', threshold=0.95)
for img1, img2, sim in duplicates:
    print(f"{img1} and {img2}: {sim:.4f}")
```

### Building an Image Search System

```python
import numpy as np

class ImageSearchEngine:
    def __init__(self, model_name='efficientnet'):
        self.checker = ImageSimilarity(model_name=model_name)
        self.database = {}
    
    def index_images(self, image_dir):
        """Index all images in a directory."""
        for img_path in Path(image_dir).glob('*.jpg'):
            embedding = self.checker.get_embedding(str(img_path))
            self.database[str(img_path)] = embedding
    
    def search(self, query_image, top_k=5):
        """Find most similar images to query."""
        query_embedding = self.checker.get_embedding(query_image)
        
        similarities = []
        for path, embedding in self.database.items():
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((path, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Usage
engine = ImageSearchEngine()
engine.index_images('database_images')
results = engine.search('query.jpg', top_k=5)

for path, similarity in results:
    print(f"{path}: {similarity:.4f}")
```

## Performance Tips

### GPU Acceleration

```python
# Check which device is being used
checker = ImageSimilarity(model_name='efficientnet')
print(f"Using device: {checker.device}")

# Force CPU (not recommended)
import torch
torch.cuda.is_available = lambda: False
checker = ImageSimilarity(model_name='efficientnet')
```

### Batch Processing Optimization

```python
# Compute embeddings once, then compare
embeddings = {}
for img_path in image_paths:
    embeddings[img_path] = checker.get_embedding(img_path)

# Now compare using pre-computed embeddings
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embeddings[img1]], [embeddings[img2]])[0][0]
```

### Memory Management

```python
import gc
import torch

# Clear CUDA cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Force garbage collection
gc.collect()
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```python
# Use CPU instead
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

checker = ImageSimilarity(model_name='efficientnet')
```

#### 2. CLIP Model Not Found

```bash
# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

#### 3. Image Loading Errors

```python
# Ensure image is RGB
from PIL import Image
img = Image.open('image.jpg').convert('RGB')
img.save('image_rgb.jpg')
```

#### 4. Slow Performance

- Use GPU if available
- Use EfficientNet instead of ResNet for faster processing
- Process images in batches
- Pre-compute embeddings for repeated comparisons

### Getting Help

- Check [GitHub Issues](https://github.com/yourusername/image-similarity-toolkit/issues)
- Read the [README](../README.md)
- Contact: your.email@example.com

---

Last updated: 2025-11-15
