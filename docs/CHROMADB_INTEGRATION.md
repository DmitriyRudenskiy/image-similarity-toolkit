# ChromaDB Integration Guide

## Overview

ChromaDB integration provides a modern vector database backend for the Image Similarity Toolkit, offering significant advantages over traditional SQLite storage:

- **Modern Vector Database**: Built specifically for AI/ML workloads with advanced similarity search
- **Embedded Operation**: No server required - runs directly in your application
- **High Performance**: Optimized for vector operations with GPU support
- **Rich Metadata**: Advanced filtering and metadata queries
- **Persistence**: Automatic data persistence with optional persistence directory
- **Production Ready**: Suitable for applications with up to 1M+ vectors

## Quick Start

### Installation

```bash
# Install with new dependencies
pip install -r requirements.txt

# Or install individually
pip install chromadb sentence-transformers
```

### Basic Usage

```python
from image_similarity.chromadb_backend import ChromaDBBackend

# Initialize backend
backend = ChromaDBBackend(
    collection_name="my_images",
    persist_directory="./my_vector_db"
)

# Add images from directory
backend.add_images_from_directory("./images", max_images=100)

# Find similar images
results = backend.find_similar(
    query_text="cat photo",
    top_k=5,
    threshold=0.7
)

for result in results:
    print(f"{result['filename']}: {result['similarity']:.3f}")
```

## Features

### 1. Embedded Vector Database

ChromaDB runs embedded in your application without requiring a separate server:

```python
# No server setup required
backend = ChromaDBBackend()  # Uses in-memory by default
backend = ChromaDBBackend(persist_directory="./db")  # Persisted storage
```

### 2. Advanced Similarity Search

Multiple search modes supported:

```python
# Text-based search
results = backend.find_similar(query_text="beautiful landscape")

# Image-based search
results = backend.find_similar(query_image_path="./query.jpg")

# Direct embedding search
results = backend.find_similar(query_embedding=my_embedding_vector)
```

### 3. Rich Metadata Support

Store and query rich metadata alongside embeddings:

```python
metadata = {
    "category": "landscape",
    "camera": "Canon EOS R5",
    "location": "Swiss Alps",
    "date_taken": "2023-07-15",
    "tags": ["mountain", "snow", "alpine"]
}

backend.add_image("landscape.jpg", embedding, metadata=metadata)

# Search with metadata filtering
results = backend.find_similar(
    query_text="mountain landscape",
    filter_metadata={"category": "landscape"}
)
```

### 4. Batch Operations

Efficiently process large collections:

```python
# Add all images from directory
count = backend.add_images_from_directory(
    "./photo_collection",
    max_images=1000,
    recursive=True,
    extensions=['.jpg', '.jpeg', '.png']
)

print(f"Added {count} images")
```

### 5. Duplicate Detection

Find duplicate or near-duplicate images:

```python
duplicates = backend.find_duplicates(similarity_threshold=0.95)

for dup in duplicates:
    print(f"Duplicate: {dup['image1_filename']} ≈ {dup['image2_filename']}")
    print(f"Similarity: {dup['similarity']:.3f}")
```

## Configuration Options

### Embedding Models

Choose from various pre-trained models:

```python
# Fast and lightweight
backend = ChromaDBBackend(embedding_model="sentence-transformers/all-MiniLM-L6-v2")

# High quality
backend = ChromaDBBackend(embedding_model="sentence-transformers/all-mpnet-base-v2")

# CLIP for image-text tasks
backend = ChromaDBBackend(embedding_model="clip-ViT-B-32")
```

### Persistence Options

```python
# In-memory (data lost on restart)
backend = ChromaDBBackend()

# Persistent storage
backend = ChromaDBBackend(persist_directory="./vector_db")

# Custom collection
backend = ChromaDBBackend(
    collection_name="my_collection",
    persist_directory="./custom_db"
)
```

## Performance Optimization

### 1. Batch Processing

```python
# Process in batches for large collections
for batch_start in range(0, total_images, batch_size):
    batch_end = min(batch_start + batch_size, total_images)
    backend.add_images_from_directory(
        directory_path,
        max_images=batch_size,
        recursive=False
    )
```

### 2. Metadata Filtering

Use metadata filters to narrow search space:

```python
# Search only in landscape photos
results = backend.find_similar(
    query_text="mountain view",
    filter_metadata={
        "category": "landscape",
        "camera_model": {"$in": ["Canon", "Nikon"]}
    }
)
```

### 3. Similarity Thresholds

Adjust thresholds for your use case:

```python
# Strict matching for exact duplicates
exact_matches = backend.find_similar(
    query_image_path="./reference.jpg",
    threshold=0.95
)

# Loose matching for similar concepts
similar_concepts = backend.find_similar(
    query_text="vehicle",
    threshold=0.6
)
```

## Migration from SQLite

If you're migrating from the SQLite backend:

```python
# SQLite approach
from image_similarity.database import EmbeddingDatabase

sqlite_db = EmbeddingDatabase('embeddings.db')

# ChromaDB approach
chromadb_backend = ChromaDBBackend(collection_name="images")

# Similar APIs
# sqlite_db.add_image(...) -> chromadb_backend.add_image(...)
# sqlite_db.find_similar(...) -> chromadb_backend.find_similar(...)
# sqlite_db.get_stats() -> chromadb_backend.get_stats()
```

## Integration Examples

### 1. Web API Integration

```python
from flask import Flask, request, jsonify
from image_similarity.chromadb_backend import ChromaDBBackend

app = Flask(__name__)
backend = ChromaDBBackend()

@app.route('/search', methods=['POST'])
def search_images():
    data = request.json
    query = data.get('query')
    results = backend.find_similar(query_text=query, top_k=10)
    
    return jsonify([{
        'filename': r['filename'],
        'path': r['path'],
        'similarity': r['similarity']
    } for r in results])

@app.route('/upload', methods=['POST'])
def upload_image():
    # Handle image upload and embedding generation
    # Implementation depends on your upload mechanism
    pass
```

### 2. Jupyter Notebook Integration

```python
# In Jupyter notebook
%matplotlib inline
from image_similarity.chromadb_backend import ChromaDBBackend
import matplotlib.pyplot as plt

backend = ChromaDBBackend()

# Interactive image search
def search_and_display(query, top_k=6):
    results = backend.find_similar(query_text=query, top_k=top_k)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results[:6]):
        # Load and display image
        # Implementation depends on your image loading
        pass
    
    plt.tight_layout()
    plt.show()

# Use in notebook
search_and_display("sunset photo")
```

## Advanced Features

### 1. Collection Export/Import

```python
# Export collection
backend.export_collection("my_collection_backup.json")

# Import collection (create new backend)
new_backend = ChromaDBBackend(collection_name="restored_collection")
# Note: Import functionality can be added based on export format
```

### 2. Custom Embeddings

```python
# Use your own embeddings instead of automatic generation
import numpy as np

# Add image with pre-computed embedding
my_embedding = np.random.random(384)  # Match your model dimension
backend.add_image("my_image.jpg", embedding=my_embedding)
```

### 3. Collection Statistics

```python
# Get detailed collection statistics
stats = backend.get_stats()

print(f"Total images: {stats['total_images']}")
print(f"Average width: {stats.get('avg_width', 'N/A')}")
print(f"Average height: {stats.get('avg_height', 'N/A')}")
print(f"Embedding model: {stats['embedding_model']}")
```

## Troubleshooting

### Common Issues

1. **Model Download Issues**
   ```python
   # Check model availability
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('your-model-name')
   ```

2. **Memory Issues with Large Collections**
   ```python
   # Process in smaller batches
   backend.add_images_from_directory(directory, max_images=100)
   ```

3. **Permission Issues with Persistence**
   ```python
   # Use appropriate directory
   backend = ChromaDBBackend(persist_directory="/tmp/chroma_db")
   ```

### Performance Monitoring

```python
import time

# Monitor operation times
start_time = time.time()
backend.add_images_from_directory("./large_collection")
print(f"Insertion time: {time.time() - start_time:.2f}s")

start_time = time.time()
results = backend.find_similar(query_text="test", top_k=100)
print(f"Search time: {time.time() - start_time:.2f}s")
```

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -f Dockerfile.chromadb -t image-similarity-chromadb .

# Run container
docker run -it --rm -v $(pwd)/data:/app/data image-similarity-chromadb
```

### Docker Compose

```bash
# Start with Docker Compose
docker-compose -f docker-compose.chromadb.yml up image-similarity-chromadb

# Interactive shell
docker-compose -f docker-compose.chromadb.yml exec image-similarity-chromadb bash

# Run specific example
docker-compose -f docker-compose.chromadb.yml exec image-similarity-chromadb python examples/chromadb_example.py
```

## Best Practices

1. **Choose Appropriate Models**
   - Use lightweight models for fast prototyping
   - Use high-quality models for production

2. **Organize Data**
   - Use clear naming conventions
   - Include relevant metadata
   - Regular collection maintenance

3. **Performance Optimization**
   - Use batch operations for large datasets
   - Implement appropriate similarity thresholds
   - Use metadata filtering to narrow search space

4. **Persistence Strategy**
   - Always use persistence for production
   - Regular backups of collection data
   - Monitor disk usage

## Comparison with SQLite

| Feature | SQLite | ChromaDB |
|---------|--------|----------|
| **Setup Complexity** | Simple | Simple |
| **Vector Operations** | Basic | Advanced |
| **Similarity Search** | Manual implementation | Built-in optimized |
| **Metadata Support** | Limited | Rich and flexible |
| **Performance** | Good for small datasets | Excellent for vector data |
| **Scalability** | Limited | Up to 1M+ vectors |
| **Production Readiness** | Basic | Production-ready |
| **Modern AI Integration** | Limited | Native |

## Getting Help

- **Documentation**: [ChromaDB Official Docs](https://docs.trychroma.com/)
- **Community**: [ChromaDB GitHub](https://github.com/chroma-core/chroma)
- **Examples**: See `examples/chromadb_example.py`
- **Issues**: Report issues in the project repository

---

*This integration guide covers ChromaDB v0.4+ with sentence-transformers support.*