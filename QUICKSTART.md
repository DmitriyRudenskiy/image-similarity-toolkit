# Quick Start Guide

Get started with Image Similarity Toolkit in less than 5 minutes!

## Installation

### Option 1: Clone and Install
```bash
git clone https://github.com/yourusername/image-similarity-toolkit.git
cd image-similarity-toolkit
pip install -r requirements.txt
pip install -e .
```

### Option 2: Install from PyPI (future)
```bash
pip install image-similarity-toolkit
```

## Your First Comparison

### Step 1: Import the Library
```python
from image_similarity import ImageSimilarity
```

### Step 2: Initialize
```python
# Choose a model: 'resnet', 'efficientnet', or 'clip'
checker = ImageSimilarity(model_name='efficientnet')
```

### Step 3: Compare Images
```python
# Compare two images
results = checker.compare_images('image1.jpg', 'image2.jpg')

# Print similarity score
print(f"Similarity: {results['cosine_similarity']:.4f}")
```

### Step 4: Visualize (Optional)
```python
# Create a comparison visualization
checker.visualize_comparison(
    'image1.jpg',
    'image2.jpg',
    results,
    save_path='comparison.png'
)
```

## Complete Example

```python
from image_similarity import ImageSimilarity

# Initialize
checker = ImageSimilarity(model_name='efficientnet')

# Compare
results = checker.compare_images('cat1.jpg', 'cat2.jpg')

# Interpret
if results['cosine_similarity'] > 0.85:
    print("Very similar images!")
elif results['cosine_similarity'] > 0.7:
    print("Moderately similar")
else:
    print("Different images")

# Get detailed metrics
print(f"Cosine Similarity: {results['cosine_similarity']:.4f}")
print(f"Euclidean Distance: {results['euclidean_distance']:.4f}")
```

## Understanding Results

### Cosine Similarity Scale
- **0.85 - 1.0**: Very similar (same object/scene)
- **0.70 - 0.85**: Moderately similar (related objects/themes)
- **0.50 - 0.70**: Weakly similar (some common features)
- **0.00 - 0.50**: Different (distinct images)

## Model Selection Guide

### EfficientNet-B0 (Recommended for most cases)
```python
checker = ImageSimilarity(model_name='efficientnet')
```
- **Best for**: Production, real-time applications
- **Speed**: Fast
- **Accuracy**: Good

### ResNet50 (Higher accuracy)
```python
checker = ImageSimilarity(model_name='resnet')
```
- **Best for**: High-precision requirements
- **Speed**: Moderate
- **Accuracy**: Excellent

### CLIP (Semantic understanding)
```python
checker = ImageSimilarity(model_name='clip')
```
- **Best for**: Semantic similarity, cross-modal tasks
- **Speed**: Fast
- **Accuracy**: Excellent for semantic tasks
- **Note**: Requires `pip install git+https://github.com/openai/CLIP.git`

## Common Use Cases

### 1. Finding Duplicates
```python
checker = ImageSimilarity(model_name='efficientnet')

# Check if images are duplicates
results = checker.compare_images('photo1.jpg', 'photo2.jpg')

if results['cosine_similarity'] > 0.95:
    print("These are likely duplicates!")
```

### 2. Image Search
```python
checker = ImageSimilarity(model_name='efficientnet')

# Find images similar to a query
query = 'query_image.jpg'
candidates = ['img1.jpg', 'img2.jpg', 'img3.jpg']

similarities = []
for img in candidates:
    result = checker.compare_images(query, img)
    similarities.append((img, result['cosine_similarity']))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

# Print top results
print("Top matches:")
for img, sim in similarities[:3]:
    print(f"{img}: {sim:.4f}")
```

### 3. Quality Control
```python
checker = ImageSimilarity(model_name='efficientnet')

# Compare product photo with reference
reference = 'reference_product.jpg'
sample = 'product_sample.jpg'

results = checker.compare_images(reference, sample)

if results['cosine_similarity'] < 0.7:
    print("Warning: Product image differs from reference!")
```

## Batch Processing Example

```python
from pathlib import Path
from image_similarity import ImageSimilarity

checker = ImageSimilarity(model_name='efficientnet')

# Get all images in directory
image_dir = Path('my_images')
images = list(image_dir.glob('*.jpg'))

# Compare all with reference
reference = images[0]
for img in images[1:]:
    results = checker.compare_images(str(reference), str(img))
    print(f"{img.name}: {results['cosine_similarity']:.4f}")
```

## GPU Acceleration

The toolkit automatically uses GPU if available:

```python
checker = ImageSimilarity(model_name='efficientnet')
print(f"Using device: {checker.device}")  # Will show 'cuda' or 'cpu'
```

## Next Steps

1. **Read the full documentation**: [docs/usage.md](docs/usage.md)
2. **Explore examples**: Check `examples/` directory
3. **Try different models**: Experiment with ResNet, EfficientNet, and CLIP
4. **Build something cool**: Create your own image search engine!

## Troubleshooting

### Import Error
```python
# Make sure you're in the project directory
import sys
sys.path.insert(0, 'src')
from image_similarity import ImageSimilarity
```

### CUDA Out of Memory
```python
# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Image Loading Error
```python
# Ensure image is RGB
from PIL import Image
img = Image.open('image.jpg').convert('RGB')
img.save('image_rgb.jpg')
```

## Need Help?

- **Documentation**: [README.md](README.md)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/image-similarity-toolkit/issues)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

---

Happy comparing! 🚀
