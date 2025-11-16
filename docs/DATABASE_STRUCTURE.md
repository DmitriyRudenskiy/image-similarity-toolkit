# Database Structure Guide

## Overview

This document provides a detailed overview of the Image Similarity Toolkit's project structure and database schema, helping developers understand the internal architecture and data organization.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Database Schema](#database-schema)
3. [Table Relationships](#table-relationships)
4. [Database Operations Flow](#database-operations-flow)
5. [File Organization](#file-organization)
6. [Model Integration](#model-integration)
7. [API Structure](#api-structure)

---

## Project Structure

```
image-similarity-toolkit/
├── .git/                    # Git version control
├── .gitignore              # Git ignore patterns
├── CHANGELOG.md            # Version history
├── CONTRIBUTING.md         # Contribution guidelines
├── GETTING_STARTED_v0.2.md # Quick start guide
├── LICENSE                 # Project license
├── PROJECT_SUMMARY.md      # Project overview
├── QUICKSTART.md          # Quick start instructions
├── README.md              # Main documentation
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation
├── data/                  # Data directories
│   ├── input/            # Input images
│   ├── output/           # Results and outputs
│   └── README.md         # Data usage guide
├── src/                   # Source code
│   └── image_similarity/  # Main package
│       ├── __init__.py      # Package initialization
│       ├── core.py          # Core similarity functionality
│       ├── database.py      # Database operations
│       ├── models.py        # ML model integration
│       └── visualization.py # Result visualization
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_database.py    # Database tests
│   └── test_similarity.py  # Core functionality tests
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md     # System architecture
│   ├── CONTRIBUTING.md     # Contribution guide
│   ├── database_guide.md   # Database usage guide
│   └── DATABASE_STRUCTURE.md # This file
└── examples/              # Usage examples
    ├── basic_usage.py      # Basic usage example
    ├── batch_comparison.py # Batch processing example
    ├── database_example.py # Database operations example
    └── duplicate_cleaner.py # Duplicate detection tool
```

---

## Database Schema

### Main Tables

#### 1. `embeddings` Table

Stores image embeddings and metadata:

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT UNIQUE NOT NULL,
    image_path TEXT NOT NULL,
    image_name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    format TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Field Descriptions:**
- `id`: Unique identifier (auto-generated)
- `image_id`: Unique image identifier (file hash or path-based)
- `image_path`: Full filesystem path to image
- `image_name`: Filename only
- `embedding`: Binary representation of image embedding vector
- `model_name`: Name of ML model used (efficientnet, resnet50, clip)
- `file_size`: Size in bytes
- `width`: Image width in pixels
- `height`: Image height in pixels
- `format`: Image format (JPEG, PNG, etc.)
- `created_at`: When the record was created
- `updated_at`: When the record was last updated

#### 2. `duplicates` Table

Stores detected duplicate pairs:

```sql
CREATE TABLE duplicates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image1_id TEXT NOT NULL,
    image2_id TEXT NOT NULL,
    image1_path TEXT NOT NULL,
    image2_path TEXT NOT NULL,
    image1_name TEXT NOT NULL,
    image2_name TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image1_id) REFERENCES embeddings (image_id),
    FOREIGN KEY (image2_id) REFERENCES embeddings (image_id)
);
```

**Field Descriptions:**
- `id`: Unique identifier (auto-generated)
- `image1_id`: Reference to first duplicate image
- `image2_id`: Reference to second duplicate image
- `image1_path`: Path to first image
- `image2_path`: Path to second image
- `image1_name`: Name of first image
- `image2_name`: Name of second image
- `similarity_score`: Calculated similarity (0.0 to 1.0)
- `detected_at`: When duplicates were detected

#### 3. `model_metadata` Table

Stores model-specific information:

```sql
CREATE TABLE model_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT UNIQUE NOT NULL,
    model_type TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Field Descriptions:**
- `model_name`: Name of the model (efficientnet, resnet50, clip)
- `model_type`: Type of model architecture
- `embedding_dim`: Dimensionality of output vectors
- `description`: Model description
- `created_at`: When model info was added

### Database Indexes

For optimal performance, the following indexes are created:

```sql
-- Fast lookup by image path
CREATE INDEX idx_image_path ON embeddings (image_path);

-- Fast lookup by image ID
CREATE INDEX idx_image_id ON embeddings (image_id);

-- Fast duplicate retrieval
CREATE INDEX idx_duplicates_score ON duplicates (similarity_score DESC);

-- Model-specific queries
CREATE INDEX idx_model_name ON embeddings (model_name);
```

---

## Table Relationships

```
┌─────────────────┐     ┌─────────────────┐
│   embeddings    │     │    duplicates   │
├─────────────────┤     ├─────────────────┤
│ id              │     │ id              │
│ image_id        │◄────┤ image1_id       │
│ image_path      │     │ image2_id       │
│ embedding       │     │ similarity      │
│ model_name      │     │ detected_at     │
└─────────────────┘     └─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ model_metadata  │
├─────────────────┤
│ model_name      │
│ model_type      │
│ embedding_dim   │
└─────────────────┘
```

### Relationship Details

1. **embeddings → duplicates**: One-to-many relationship
   - Each embedding can have multiple duplicate pairs
   - Enforced through foreign key constraints

2. **embeddings → model_metadata**: Many-to-one relationship
   - Multiple images can use the same model
   - Model metadata provides context for embeddings

---

## Database Operations Flow

### 1. Adding New Image

```
User Image → Preprocessing → Model Inference → Embedding Storage
                                           │
                                           ▼
                                    Database Insertion
```

**Process Steps:**
1. Validate image format and path
2. Load and preprocess image
3. Generate embedding using selected model
4. Extract metadata (size, dimensions, format)
5. Insert into `embeddings` table
6. Return success status

### 2. Searching Similar Images

```
Query Image → Preprocessing → Model Inference → Embedding Comparison
                                                │
                                                ▼
                                         Database Query
```

**Process Steps:**
1. Generate query embedding
2. Load all stored embeddings from database
3. Calculate similarity scores using vector operations
4. Filter by threshold and limit to top-k results
5. Return ranked results with metadata

### 3. Duplicate Detection

```
All Images → Pairwise Comparison → Similarity Filtering → Duplicate Storage
```

**Process Steps:**
1. Retrieve all embeddings from database
2. Generate all possible image pairs
3. Calculate similarity for each pair
4. Filter pairs above similarity threshold
5. Store results in `duplicates` table

---

## File Organization

### Source Code Structure

#### Core Module (`src/image_similarity/`)

**`core.py`** - Main functionality
- `ImageSimilarity` class
- Image preprocessing methods
- Embedding generation
- Similarity calculation algorithms

**`database.py`** - Database operations
- `EmbeddingDatabase` class
- CRUD operations for embeddings
- Search and duplicate detection
- Database management methods

**`models.py`** - ML model integration
- Model loading and initialization
- Preprocessing pipelines
- Model-specific configurations
- Device selection (CPU/GPU)

**`visualization.py`** - Result visualization
- Comparison result display
- Similarity heatmaps
- Graph plotting utilities
- Result export functions

### Example Applications

**`examples/database_example.py`**
- Command-line interface for database operations
- Batch indexing functionality
- Search and duplicate detection
- Database statistics and maintenance

**`examples/duplicate_cleaner.py`**
- Interactive duplicate management
- Batch operations for duplicate removal
- Backup and recovery features
- Safety confirmations and logging

### Test Coverage

**`tests/test_database.py`**
- Database CRUD operation tests
- Search accuracy validation
- Duplicate detection testing
- Performance and stress tests

**`tests/test_similarity.py`**
- Model inference tests
- Similarity calculation tests
- Edge case handling
- Cross-model compatibility

---

## Model Integration

### Supported Models

#### 1. ResNet50
```python
{
    "name": "resnet50",
    "architecture": "CNN",
    "embedding_dim": 2048,
    "input_size": (224, 224),
    "preprocessing": "torchvision_transforms"
}
```

#### 2. EfficientNet-B0
```python
{
    "name": "efficientnet",
    "architecture": "CNN",
    "embedding_dim": 1280,
    "input_size": (224, 224),
    "preprocessing": "efficientnet_transforms"
}
```

#### 3. CLIP ViT-B/32
```python
{
    "name": "clip",
    "architecture": "Vision Transformer",
    "embedding_dim": 512,
    "input_size": (224, 224),
    "preprocessing": "clip_transforms"
}
```

### Model Selection Criteria

- **ResNet50**: Good general-purpose embeddings, widely tested
- **EfficientNet**: Compact embeddings, faster inference
- **CLIP**: Multi-modal capabilities, text-image understanding

### Embedding Storage Format

Embeddings are stored as binary blobs in the database:

```python
# Python representation
embedding = np.array([0.1, 0.2, 0.3, ...], dtype=np.float32)

# Storage format
embedding_blob = embedding.tobytes()  # 4 bytes per float32
```

---

## API Structure

### EmbeddingDatabase Class API

#### Core Operations
```python
class EmbeddingDatabase:
    def __init__(self, db_path: str, model_name: str)
    def add_image(self, image_path: str, embedding: np.ndarray, metadata: dict = None)
    def get_embedding(self, image_path: str) -> Optional[np.ndarray]
    def remove_image(self, image_path: str) -> bool
    def clear_all(self) -> None
```

#### Search Operations
```python
def find_similar(self, query_embedding: np.ndarray, top_k: int = 10, 
                threshold: float = 0.0) -> List[Dict]
def find_duplicates(self, similarity_threshold: float = 0.95, 
                   save_to_table: bool = True) -> List[Dict]
```

#### Utility Operations
```python
def get_stats(self) -> Dict
def export_to_json(self, output_path: str) -> None
def import_from_json(self, input_path: str) -> None
def close(self) -> None
```

### ImageSimilarity Class API

#### Model Management
```python
class ImageSimilarity:
    def __init__(self, model_name: str = 'efficientnet', device: str = 'auto')
    def get_embedding(self, image_path: str) -> np.ndarray
    def compare_images(self, img1_path: str, img2_path: str) -> float
```

---

## Performance Considerations

### Database Optimization

1. **Indexing Strategy**
   - Primary keys on auto-increment IDs
   - Secondary indexes on frequently queried fields
   - Composite indexes for complex queries

2. **Data Types**
   - Embeddings stored as BLOB for efficient binary storage
   - Metadata stored as individual columns for filtering
   - TIMESTAMP for temporal queries

3. **Connection Management**
   - Context managers for automatic cleanup
   - Connection pooling for high-frequency operations
   - Transaction support for batch operations

### Memory Management

1. **Embedding Caching**
   - In-memory cache for frequently accessed embeddings
   - LRU eviction for large datasets
   - Configurable cache size

2. **Batch Processing**
   - Chunked processing for large datasets
   - Progress tracking for long operations
   - Memory usage monitoring

---

## Migration and Versioning

### Database Version Management

```sql
-- Version tracking table
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Initial version
INSERT INTO schema_version (version, description) VALUES (1, 'Initial database schema');
```

### Schema Evolution

Future database schema changes will be tracked with:
1. Version number increments
2. Migration scripts for upgrades
3. Backward compatibility support
4. Data migration utilities

---

## Security and Data Integrity

### Data Validation

1. **Input Validation**
   - File path validation
   - Image format verification
   - Embedding dimension checking
   - Metadata type validation

2. **Error Handling**
   - Graceful degradation for corrupted images
   - Rollback support for failed operations
   - Comprehensive logging
   - User-friendly error messages

### Backup and Recovery

1. **Automatic Backups**
   - Periodic database snapshots
   - Incremental backup support
   - Backup integrity verification

2. **Recovery Procedures**
   - Point-in-time recovery
   - Partial database restoration
   - Data consistency checking

---

## Troubleshooting

### Common Issues

1. **Database Lock Issues**
   - Proper connection handling
   - Lock timeout configuration
   - Concurrent access patterns

2. **Memory Issues**
   - Large embedding batches
   - Memory-efficient search algorithms
   - Garbage collection optimization

3. **Performance Issues**
   - Index optimization
   - Query plan analysis
   - Hardware recommendations

### Diagnostic Queries

```sql
-- Check database size
SELECT name, size FROM database_stats;

-- Verify table integrity
PRAGMA integrity_check;

-- Analyze query performance
EXPLAIN QUERY PLAN SELECT * FROM embeddings WHERE model_name = 'efficientnet';
```

---

Last updated: 2025-11-16
Version: 0.2.0
Author: MiniMax Agent