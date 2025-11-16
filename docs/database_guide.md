# Database Guide

## Overview

The Image Similarity Toolkit includes a powerful database module for storing, indexing, and searching image embeddings. This guide covers everything you need to know about using the database features.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Database Operations](#database-operations)
4. [Searching for Similar Images](#searching-for-similar-images)
5. [Duplicate Detection](#duplicate-detection)
6. [Advanced Usage](#advanced-usage)
7. [Performance Tips](#performance-tips)
8. [Best Practices](#best-practices)

## Introduction

### Why Use the Database?

- **Persistent Storage**: Store embeddings once, reuse many times
- **Fast Search**: Quickly find similar images without re-computing embeddings
- **Duplicate Detection**: Automatically find and manage duplicate images
- **Scalability**: Handle thousands of images efficiently
- **Metadata**: Store additional information about images

### Database Backend

The toolkit uses **SQLite** as the database backend:
- No separate server required
- File-based storage
- ACID compliant
- Built into Python
- Cross-platform compatible

## Getting Started

### Basic Setup

```python
from image_similarity import ImageSimilarity, EmbeddingDatabase

# Initialize the similarity checker
checker = ImageSimilarity(model_name='efficientnet')

# Create/open a database
db = EmbeddingDatabase(
    db_path='my_images.db',
    model_name='efficientnet'
)
```

### First Image Indexing

```python
# Index a single image
image_path = 'photos/cat.jpg'

# Get embedding
embedding = checker.get_embedding(image_path)

# Add to database
db.add_image(image_path, embedding)

print("Image indexed successfully!")
```

### Closing the Database

```python
# Always close when done
db.close()

# Or use context manager (recommended)
with EmbeddingDatabase('my_images.db') as db:
    # Your operations here
    db.add_image(path, embedding)
# Database automatically closed
```

## Database Operations

### Adding Images with Metadata

```python
from PIL import Image

# Get image info
img = Image.open(image_path)
metadata = {
    'file_size': Path(image_path).stat().st_size,
    'width': img.width,
    'height': img.height
}

# Add with metadata
db.add_image(image_path, embedding, metadata=metadata)
```

### Batch Indexing

```python
from pathlib import Path

# Index entire directory
image_dir = Path('my_photos')
image_extensions = {'.jpg', '.jpeg', '.png'}

for img_path in image_dir.rglob('*'):
    if img_path.suffix.lower() in image_extensions:
        try:
            # Get embedding
            embedding = checker.get_embedding(str(img_path))
            
            # Get metadata
            img = Image.open(img_path)
            metadata = {
                'file_size': img_path.stat().st_size,
                'width': img.width,
                'height': img.height
            }
            
            # Add to database
            db.add_image(str(img_path), embedding, metadata)
            print(f"✓ Indexed: {img_path.name}")
            
        except Exception as e:
            print(f"✗ Error: {img_path.name}: {e}")
```

### Retrieving Embeddings

```python
# Get embedding for a specific image
embedding = db.get_embedding('photos/cat.jpg')

if embedding is not None:
    print(f"Embedding size: {len(embedding)}")
else:
    print("Image not found in database")

# Get all embeddings
all_embeddings = db.get_all_embeddings()
print(f"Total images in database: {len(all_embeddings)}")

for img_id, img_path, embedding in all_embeddings:
    print(f"ID: {img_id}, Path: {img_path}")
```

### Updating Images

```python
# If you add an image that already exists, it will be updated
new_embedding = checker.get_embedding('photos/cat.jpg')
db.add_image('photos/cat.jpg', new_embedding)
# Old embedding is replaced with new one
```

### Removing Images

```python
# Remove a single image
removed = db.remove_image('photos/old_photo.jpg')

if removed:
    print("Image removed from database")
else:
    print("Image not found")

# Clear entire database
db.clear_all()
print("Database cleared")
```

## Searching for Similar Images

### Basic Search

```python
# Get query embedding
query_embedding = checker.get_embedding('query_image.jpg')

# Search for top 5 similar images
results = db.find_similar(
    query_embedding,
    top_k=5,
    threshold=0.5  # Minimum similarity
)

# Display results
for result in results:
    print(f"{result['image_name']}: {result['similarity']:.4f}")
```

### Filtering by Threshold

```python
# Only find highly similar images
results = db.find_similar(
    query_embedding,
    top_k=10,
    threshold=0.8  # Only similarity >= 0.8
)

if results:
    print(f"Found {len(results)} similar images")
else:
    print("No similar images found")
```

### Building an Image Search Engine

```python
class ImageSearchEngine:
    def __init__(self, db_path='search_engine.db'):
        self.checker = ImageSimilarity(model_name='efficientnet')
        self.db = EmbeddingDatabase(db_path)
    
    def index_directory(self, directory):
        """Index all images in a directory."""
        for img_path in Path(directory).rglob('*.jpg'):
            embedding = self.checker.get_embedding(str(img_path))
            self.db.add_image(str(img_path), embedding)
    
    def search(self, query_image, top_k=10):
        """Search for similar images."""
        query_embedding = self.checker.get_embedding(query_image)
        return self.db.find_similar(query_embedding, top_k=top_k)
    
    def close(self):
        self.db.close()

# Usage
engine = ImageSearchEngine()
engine.index_directory('my_photos')
results = engine.search('query.jpg', top_k=5)

for r in results:
    print(f"{r['image_name']}: {r['similarity']:.4f}")

engine.close()
```

## Duplicate Detection

### Finding Duplicates

```python
# Find duplicates with default threshold (0.95)
duplicates = db.find_duplicates(
    similarity_threshold=0.95,
    save_to_table=True  # Save to database
)

print(f"Found {len(duplicates)} duplicate pairs")

# Display duplicates
for dup in duplicates:
    print(f"\nDuplicate pair:")
    print(f"  Image 1: {dup['image1_path']}")
    print(f"  Image 2: {dup['image2_path']}")
    print(f"  Similarity: {dup['similarity']:.4f}")
```

### Adjusting Sensitivity

```python
# Very strict (almost identical)
strict_duplicates = db.find_duplicates(similarity_threshold=0.99)

# Moderate (similar images)
moderate_duplicates = db.find_duplicates(similarity_threshold=0.90)

# Loose (broadly similar)
loose_duplicates = db.find_duplicates(similarity_threshold=0.85)
```

### Retrieving Saved Duplicates

```python
# Get previously detected duplicates
saved_duplicates = db.get_saved_duplicates()

for dup in saved_duplicates:
    print(f"Found on: {dup['detected_at']}")
    print(f"Images: {dup['image1_name']}, {dup['image2_name']}")
    print(f"Similarity: {dup['similarity']:.4f}")
```

### Managing Duplicates

```python
import os
import shutil

def remove_duplicates(duplicates, keep='larger'):
    """
    Remove duplicate images.
    
    Args:
        duplicates: List of duplicate pairs
        keep: 'larger' or 'smaller' - which file to keep
    """
    backup_dir = 'duplicates_backup'
    os.makedirs(backup_dir, exist_ok=True)
    
    for dup in duplicates:
        img1 = dup['image1_path']
        img2 = dup['image2_path']
        
        # Determine which to remove
        size1 = Path(img1).stat().st_size
        size2 = Path(img2).stat().st_size
        
        if keep == 'larger':
            to_remove = img1 if size1 < size2 else img2
        else:
            to_remove = img1 if size1 > size2 else img2
        
        # Backup before removing
        backup_path = Path(backup_dir) / Path(to_remove).name
        shutil.copy2(to_remove, backup_path)
        
        # Remove
        os.remove(to_remove)
        print(f"Removed: {to_remove}")
        print(f"Backup: {backup_path}")

# Usage
duplicates = db.find_duplicates(similarity_threshold=0.95)
remove_duplicates(duplicates, keep='larger')
```

## Advanced Usage

### Database Statistics

```python
# Get comprehensive statistics
stats = db.get_stats()

print(f"Total images: {stats['total_images']}")
print(f"Total duplicates: {stats['total_duplicates']}")
print(f"Model: {stats['model_name']}")
print(f"Database: {stats['database_path']}")

if stats['avg_file_size']:
    print(f"Average file size: {stats['avg_file_size']/1024:.1f} KB")
    
if stats['avg_width']:
    print(f"Average dimensions: {stats['avg_width']:.0f}x{stats['avg_height']:.0f}")
```

### Multiple Databases

```python
# Use different databases for different models
db_efficientnet = EmbeddingDatabase('embeddings_efficientnet.db', model_name='efficientnet')
db_clip = EmbeddingDatabase('embeddings_clip.db', model_name='clip')

# Or different databases for different purposes
db_photos = EmbeddingDatabase('photos.db')
db_products = EmbeddingDatabase('products.db')
db_artwork = EmbeddingDatabase('artwork.db')
```

### Custom Similarity Metrics

```python
import numpy as np

# Get embeddings
all_embeddings = db.get_all_embeddings()

# Calculate custom similarity
def custom_similarity(emb1, emb2):
    """Custom similarity metric."""
    # Your custom calculation
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Compare images
for i, (id1, path1, emb1) in enumerate(all_embeddings):
    for id2, path2, emb2 in all_embeddings[i+1:]:
        sim = custom_similarity(emb1, emb2)
        if sim > 0.9:
            print(f"{path1} <-> {path2}: {sim:.4f}")
```

## Performance Tips

### 1. Batch Operations

```python
# Good: Batch indexing
embeddings = []
for img_path in image_paths:
    emb = checker.get_embedding(img_path)
    embeddings.append((img_path, emb))

for img_path, emb in embeddings:
    db.add_image(img_path, emb)

# Better: Use transactions (coming in future version)
```

### 2. Caching Embeddings

```python
# Cache embeddings in memory for repeated searches
embedding_cache = {}

for img_id, img_path, embedding in db.get_all_embeddings():
    embedding_cache[img_path] = embedding

# Now use cache for fast access
query_embedding = embedding_cache.get('query.jpg')
```

### 3. Indexing Strategy

```python
# Check before adding (avoid unnecessary updates)
for img_path in image_paths:
    if db.get_embedding(img_path) is None:
        # Only process if not already indexed
        embedding = checker.get_embedding(img_path)
        db.add_image(img_path, embedding)
```

### 4. Database Maintenance

```python
# Periodically clean up database
import sqlite3

conn = sqlite3.connect('embeddings.db')
conn.execute('VACUUM')  # Optimize database file
conn.close()
```

## Best Practices

### 1. Always Use Context Managers

```python
# Good
with EmbeddingDatabase('images.db') as db:
    db.add_image(path, embedding)

# Avoid
db = EmbeddingDatabase('images.db')
db.add_image(path, embedding)
# Forgot to close!
```

### 2. Handle Errors Gracefully

```python
with EmbeddingDatabase('images.db') as db:
    for img_path in image_paths:
        try:
            embedding = checker.get_embedding(img_path)
            db.add_image(img_path, embedding)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
```

### 3. Use Appropriate Thresholds

```python
# For exact duplicates
duplicates = db.find_duplicates(threshold=0.99)

# For near-duplicates (edited versions)
near_duplicates = db.find_duplicates(threshold=0.95)

# For similar images (same object)
similar = db.find_duplicates(threshold=0.85)
```

### 4. Regular Backups

```python
import shutil
from datetime import datetime

# Backup database
backup_name = f"embeddings_backup_{datetime.now():%Y%m%d_%H%M%S}.db"
shutil.copy2('embeddings.db', f'backups/{backup_name}')
```

### 5. Monitor Database Size

```python
import os

db_size = os.path.getsize('embeddings.db')
print(f"Database size: {db_size / 1024 / 1024:.2f} MB")

# If too large, consider splitting into multiple databases
if db_size > 100 * 1024 * 1024:  # 100 MB
    print("Consider splitting database")
```

---

## Example Workflows

### Workflow 1: Photo Library Deduplication

```python
# 1. Index entire photo library
with EmbeddingDatabase('photos.db') as db:
    checker = ImageSimilarity(model_name='efficientnet')
    
    for photo in Path('~/Photos').rglob('*.jpg'):
        embedding = checker.get_embedding(str(photo))
        db.add_image(str(photo), embedding)
    
    # 2. Find duplicates
    duplicates = db.find_duplicates(threshold=0.95)
    
    # 3. Review and remove
    for dup in duplicates:
        print(f"Duplicate: {dup['image1_name']} ↔ {dup['image2_name']}")
```

### Workflow 2: Image Search Service

```python
# Build searchable database
with EmbeddingDatabase('search.db') as db:
    checker = ImageSimilarity(model_name='clip')
    
    # Index product images
    for product_img in Path('products').glob('*.jpg'):
        embedding = checker.get_embedding(str(product_img))
        db.add_image(str(product_img), embedding)
    
    # Search for similar products
    query = checker.get_embedding('customer_upload.jpg')
    results = db.find_similar(query, top_k=10, threshold=0.7)
    
    return results
```

---

Last updated: 2025-11-15
Version: 0.2.0
