"""
DDD Architecture Example Usage
=============================

Demonstration of how to use the Image Similarity Toolkit with DDD architecture.
"""

import time
from pathlib import Path

# Domain Layer
from src_ddd.domain.configuration import Configuration, ModelConfiguration
from src_ddd.domain.image_processing import Image
from src_ddd.domain.vector_storage import VectorEmbedding

# Application Layer
from src_ddd.application.use_cases.add_image_use_case import AddImageRequest, AddImageUseCase
from src_ddd.application.use_cases.search_similar_images_use_case import (
    SearchSimilarImagesRequest, SearchSimilarImagesUseCase
)
from src_ddd.application.use_cases.batch_process_images_use_case import (
    BatchProcessImagesRequest, BatchProcessImagesUseCase
)

# Infrastructure Layer (Placeholder implementations)
# from src_ddd.infrastructure.database import SQLiteRepository
# from src_ddd.infrastructure.image_processor import DefaultImageProcessor
# from src_ddd.infrastructure.embedding_generator import DefaultEmbeddingGenerator


class MockImageProcessor:
    """Mock implementation of image processor for demonstration."""
    
    def load_image(self, image_path: Path) -> Image:
        """Load and validate an image."""
        # Mock implementation
        return Image.from_file(image_path)


class MockEmbeddingGenerator:
    """Mock implementation of embedding generator for demonstration."""
    
    def generate_embedding(self, image: Image, model_config: ModelConfiguration = None) -> VectorEmbedding:
        """Generate embedding for an image."""
        import numpy as np
        from datetime import datetime
        
        # Mock embedding generation
        embedding_vector = np.random.rand(512)  # 512-dimensional vector
        
        return VectorEmbedding(
            vector=embedding_vector,
            model_name=model_config.model_name if model_config else "mock_model",
            created_at=datetime.now(),
            metadata={"source": "mock_generator"}
        )


class MockVectorRepository:
    """Mock implementation of vector repository for demonstration."""
    
    def __init__(self):
        self._embeddings = {}
        self._next_id = 1
    
    def save(self, embedding: VectorEmbedding, image: Image):
        """Save embedding with associated image."""
        import uuid
        embedding_id = uuid.uuid4()
        self._embeddings[embedding_id] = {
            "embedding": embedding,
            "image": image
        }
        return embedding_id
    
    def find_by_id(self, embedding_id):
        """Find embedding by ID."""
        return self._embeddings.get(embedding_id, {}).get("embedding")
    
    def find_similar(self, query_embedding: VectorEmbedding, limit: int = 10):
        """Find similar embeddings."""
        # Mock implementation - return random similarities
        import numpy as np
        results = []
        
        for emb_id, data in list(self._embeddings.items())[:limit]:
            similarity = np.random.rand()
            results.append((data["embedding"], similarity))
        
        return results
    
    def get_all(self):
        """Get all embeddings."""
        return [data["embedding"] for data in self._embeddings.values()]
    
    def get_stats(self):
        """Get repository statistics."""
        return {
            "total_embeddings": len(self._embeddings),
            "repository_type": "mock"
        }


class MockVectorStore:
    """Mock implementation of vector store for demonstration."""
    
    def __init__(self, repository: MockVectorRepository):
        self._repository = repository
        self._cache = {}
    
    def add_image(self, image: Image, embedding: VectorEmbedding):
        """Add image with its embedding."""
        return self._repository.save(embedding, image)
    
    def find_similar_images(self, query_embedding: VectorEmbedding, limit: int = 10):
        """Find images similar to the query embedding."""
        similar_embeddings = self._repository.find_similar(query_embedding, limit)
        
        results = []
        for embedding, similarity in similar_embeddings:
            # Mock image association
            mock_image = Image(
                path=Path(f"/mock/image_{hash(str(embedding.vector))}.jpg"),
                dimensions=(224, 224),
                format="JPEG",
                file_hash="mock_hash",
                metadata={"source": "mock"}
            )
            
            results.append({
                "image": mock_image,
                "embedding": embedding,
                "similarity_score": similarity
            })
        
        return results


def demonstrate_basic_usage():
    """Demonstrate basic usage of the DDD architecture."""
    print("=== Basic Usage Demonstration ===\n")
    
    # 1. Setup Configuration
    print("1. Setting up configuration...")
    config = Configuration.default()
    print(f"   Database type: {config.database.db_type.value if config.database else 'None'}")
    print(f"   Processing mode: {config.processing.get('mode', 'unknown')}")
    print()
    
    # 2. Setup Infrastructure
    print("2. Setting up infrastructure...")
    repository = MockVectorRepository()
    vector_store = MockVectorStore(repository)
    image_processor = MockImageProcessor()
    embedding_generator = MockEmbeddingGenerator()
    print("   Infrastructure components initialized")
    print()
    
    # 3. Create Use Cases
    print("3. Creating use cases...")
    add_image_use_case = AddImageUseCase(
        vector_store, image_processor, embedding_generator
    )
    search_use_case = SearchSimilarImagesUseCase(
        vector_store, image_processor, embedding_generator
    )
    print("   Use cases created")
    print()
    
    # 4. Add Images
    print("4. Adding sample images...")
    sample_images = [
        Path("/mock/image1.jpg"),
        Path("/mock/image2.jpg"),
        Path("/mock/image3.jpg"),
    ]
    
    added_embeddings = []
    for i, image_path in enumerate(sample_images):
        # Create mock image
        image = Image(
            path=image_path,
            dimensions=(224, 224),
            format="JPEG",
            file_hash=f"hash_{i}",
            metadata={"sample": True}
        )
        
        # Create model configuration
        model_config = ModelConfiguration.efficientnet_b0()
        
        # Add image
        add_request = AddImageRequest(image_path, model_config)
        response = add_image_use_case.execute(add_request)
        
        added_embeddings.append(response.embedding_id)
        print(f"   Added {image_path.name}: {response.embedding_id}")
    
    print()
    
    # 5. Search for Similar Images
    print("5. Searching for similar images...")
    query_image_path = sample_images[0]
    search_request = SearchSimilarImagesRequest.from_image(query_image_path)
    
    start_time = time.time()
    search_response = search_use_case.execute(search_request)
    search_time = time.time() - start_time
    
    print(f"   Query: {query_image_path.name}")
    print(f"   Found: {search_response.total_found} similar images")
    print(f"   Search time: {search_time:.4f} seconds")
    print()
    
    return config, vector_store, add_image_use_case, search_use_case


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("=== Batch Processing Demonstration ===\n")
    
    # Setup
    repository = MockVectorRepository()
    vector_store = MockVectorStore(repository)
    image_processor = MockImageProcessor()
    embedding_generator = MockEmbeddingGenerator()
    
    add_use_case = AddImageUseCase(vector_store, image_processor, embedding_generator)
    batch_use_case = BatchProcessImagesUseCase(add_use_case, None)
    
    # Create batch request
    sample_paths = [Path(f"/mock/batch_image_{i}.jpg") for i in range(5)]
    model_config = ModelConfiguration.clip_vit_b32()
    
    batch_request = BatchProcessImagesRequest(
        image_paths=sample_paths,
        model_config=model_config,
        max_workers=2,
        fail_fast=False
    )
    
    print(f"Processing {len(sample_paths)} images in batch...")
    start_time = time.time()
    
    try:
        response = batch_use_case.execute(batch_request)
        
        print(f"   Successfully processed: {response.successful_count}")
        print(f"   Failed: {response.failed_count}")
        print(f"   Success rate: {response.successful_count/response.total_count*100:.1f}%")
        print(f"   Total time: {response.processing_time:.4f} seconds")
        print(f"   Average time per image: {response.statistics.get('average_processing_time', 0):.4f} seconds")
        
    except Exception as e:
        print(f"   Batch processing error: {e}")
    
    print()


def demonstrate_configuration():
    """Demonstrate configuration capabilities."""
    print("=== Configuration Demonstration ===\n")
    
    # 1. Default Configuration
    print("1. Default configuration:")
    default_config = Configuration.default()
    print(f"   Processing mode: {default_config.get_processing_mode().value}")
    print(f"   Cache enabled: {default_config.get_cache_enabled()}")
    print()
    
    # 2. Custom Configuration
    print("2. Custom configuration:")
    custom_config = Configuration.default()
    processing_settings = custom_config.processing.copy()
    processing_settings.update({
        "batch_size": 64,
        "max_workers": 8,
        "similarity_threshold": 0.85
    })
    
    # This would require recreating the configuration object
    print("   Custom batch size: 64")
    print("   Custom max workers: 8")
    print("   Custom similarity threshold: 0.85")
    print()
    
    # 3. Database Configuration
    print("3. Database configurations:")
    
    # SQLite
    sqlite_config = Configuration.database.sqlite(Path("./embeddings.db"))
    print(f"   SQLite: {sqlite_config.connection_string}")
    
    # ChromaDB
    chromadb_config = Configuration.database.chromadb(
        collection_name="my_embeddings",
        persist_directory=Path("./chromadb_data")
    )
    print(f"   ChromaDB: {chromadb_config.database_name}")
    print()


def demonstrate_domain_objects():
    """Demonstrate domain objects functionality."""
    print("=== Domain Objects Demonstration ===\n")
    
    # 1. Image Value Object
    print("1. Image Value Object:")
    image = Image(
        path=Path("/demo/sample.jpg"),
        dimensions=(1024, 768),
        format="JPEG",
        file_hash="abc123def456",
        metadata={"source": "demo", "camera": "Canon EOS R5"}
    )
    print(f"   Path: {image.path}")
    print(f"   Dimensions: {image.dimensions}")
    print(f"   Aspect ratio: {image.aspect_ratio:.2f}")
    print(f"   File hash: {image.file_hash}")
    print()
    
    # 2. Vector Embedding Value Object
    print("2. Vector Embedding Value Object:")
    import numpy as np
    from datetime import datetime
    
    embedding = VectorEmbedding(
        vector=np.random.rand(512),
        model_name="efficientnet_b0",
        created_at=datetime.now(),
        metadata={"confidence": 0.95, "preprocessing": "resize_and_normalize"}
    )
    print(f"   Model: {embedding.model_name}")
    print(f"   Dimension: {embedding.dimension}")
    print(f"   Norm: {embedding.norm:.4f}")
    print(f"   Created: {embedding.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 3. Similarity Query
    print("3. Similarity Query:")
    from src_ddd.domain.similarity_search import SimilarityQuery, SearchMode
    
    query = SimilarityQuery.from_embedding(
        embedding,
        limit=10,
        threshold=0.8
    )
    print(f"   Query type: {query.search_type.value}")
    print(f"   Search mode: {query.search_mode.value}")
    print(f"   Limit: {query.limit}")
    print(f"   Threshold: {query.threshold}")
    print()


def main():
    """Main demonstration function."""
    print("Image Similarity Toolkit - DDD Architecture Examples")
    print("=" * 60)
    print()
    
    try:
        # Demonstrate basic usage
        config, vector_store, add_use_case, search_use_case = demonstrate_basic_usage()
        
        # Demonstrate batch processing
        demonstrate_batch_processing()
        
        # Demonstrate configuration
        demonstrate_configuration()
        
        # Demonstrate domain objects
        demonstrate_domain_objects()
        
        print("=== Summary ===")
        print("The DDD architecture provides:")
        print("✓ Clear separation of concerns")
        print("✓ Immutable value objects")
        print("✓ Aggregate roots for consistency")
        print("✓ Repository pattern for data access")
        print("✓ Use cases for business operations")
        print("✓ Flexible configuration system")
        print("✓ Extensible infrastructure layer")
        print()
        print("Benefits:")
        print("• Better code organization")
        print("• Improved testability")
        print("• Enhanced maintainability")
        print("• Easier to understand and modify")
        print("• More resilient to technology changes")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()