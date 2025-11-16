"""
Tests for ChromaDB backend integration.

Tests cover basic operations, similarity search, metadata handling,
and performance characteristics of the ChromaDB backend.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from image_similarity.chromadb_backend import ChromaDBBackend


class TestChromaDBBackend:
    """Test suite for ChromaDB backend."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def backend(self, temp_dir):
        """Create ChromaDB backend for testing."""
        backend = ChromaDBBackend(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        yield backend
    
    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample images for testing."""
        images = []
        
        # Create test images with different colors
        colors = ["red", "green", "blue", "yellow", "purple"]
        
        for i, color in enumerate(colors):
            img = Image.new("RGB", (100, 100), color)
            img_path = Path(temp_dir) / f"{color}_image_{i}.jpg"
            img.save(img_path)
            images.append(str(img_path))
        
        return images
    
    def test_backend_initialization(self, temp_dir):
        """Test backend initialization."""
        backend = ChromaDBBackend(
            collection_name="test_init",
            persist_directory=temp_dir
        )
        
        assert backend.collection_name == "test_init"
        assert backend.persist_directory == temp_dir
        assert backend.collection is not None
        assert backend.model is not None
    
    def test_add_single_image(self, backend, sample_images):
        """Test adding a single image."""
        image_path = sample_images[0]
        
        # Test adding image
        result = backend.add_image(image_path)
        assert result is True
        
        # Verify image was added
        stats = backend.get_stats()
        assert stats['total_images'] == 1
    
    def test_add_images_from_directory(self, backend, sample_images):
        """Test adding images from directory."""
        temp_dir = Path(sample_images[0]).parent
        
        # Add images from directory
        count = backend.add_images_from_directory(
            str(temp_dir),
            max_images=3,
            recursive=False
        )
        
        assert count == 3
        
        # Verify all were added
        stats = backend.get_stats()
        assert stats['total_images'] == 3
    
    def test_get_embedding(self, backend, sample_images):
        """Test retrieving embeddings."""
        image_path = sample_images[0]
        
        # Add image first
        backend.add_image(image_path)
        
        # Get embedding
        embedding = backend.get_embedding(image_path)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_find_similar_by_text(self, backend, sample_images):
        """Test text-based similarity search."""
        # Add sample images
        backend.add_images_from_directory(
            Path(sample_images[0]).parent,
            max_images=2,
            recursive=False
        )
        
        # Search by text
        results = backend.find_similar(
            query_text="red image",
            top_k=5
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check result structure
        result = results[0]
        assert 'filename' in result
        assert 'similarity' in result
        assert 'path' in result
        assert isinstance(result['similarity'], float)
    
    def test_find_similar_by_image(self, backend, sample_images):
        """Test image-based similarity search."""
        # Add images
        backend.add_images_from_directory(
            Path(sample_images[0]).parent,
            max_images=2,
            recursive=False
        )
        
        # Use first image as query
        query_image = sample_images[0]
        
        # Search by image
        results = backend.find_similar(
            query_image_path=query_image,
            top_k=5
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # First result should be the query image itself (highest similarity)
        if len(results) > 0:
            first_result = results[0]
            assert first_result['similarity'] > 0.9  # Should be very similar to itself
    
    def test_find_similar_with_threshold(self, backend, sample_images):
        """Test similarity search with threshold."""
        # Add images
        backend.add_images_from_directory(
            Path(sample_images[0]).parent,
            max_images=3,
            recursive=False
        )
        
        # Search with high threshold
        results = backend.find_similar(
            query_text="colorful image",
            top_k=5,
            threshold=0.95
        )
        
        # All results should meet threshold
        for result in results:
            assert result['similarity'] >= 0.95
    
    def test_metadata_storage_and_retrieval(self, backend, sample_images):
        """Test metadata storage and retrieval."""
        image_path = sample_images[0]
        metadata = {
            "category": "test",
            "color": "red",
            "size": "100x100",
            "test_metadata": True
        }
        
        # Add image with metadata
        result = backend.add_image(image_path, metadata=metadata)
        assert result is True
        
        # Search and verify metadata
        results = backend.find_similar(
            query_text="red image",
            top_k=1
        )
        
        assert len(results) > 0
        result = results[0]
        
        # Verify metadata was stored correctly
        assert result['metadata']['category'] == "test"
        assert result['metadata']['color'] == "red"
        assert result['metadata']['size'] == "100x100"
    
    def test_metadata_filtering(self, backend, sample_images):
        """Test search with metadata filters."""
        # Add images with different metadata
        temp_dir = Path(sample_images[0]).parent
        
        # Add images and modify metadata for filtering
        backend.add_images_from_directory(
            str(temp_dir),
            max_images=2,
            recursive=False
        )
        
        # Search with metadata filter
        results = backend.find_similar(
            query_text="image",
            filter_metadata={"color": "red"},
            top_k=5
        )
        
        # All results should match the filter
        for result in results:
            if 'color' in result['metadata']:
                assert result['metadata']['color'] == "red"
    
    def test_find_duplicates(self, backend, temp_dir):
        """Test duplicate detection."""
        # Create identical images
        identical_images = []
        for i in range(3):
            img = Image.new("RGB", (50, 50), "red")
            img_path = Path(temp_dir) / f"red_identical_{i}.jpg"
            img.save(img_path)
            identical_images.append(str(img_path))
        
        # Add identical images
        for image_path in identical_images:
            backend.add_image(image_path)
        
        # Find duplicates (should find pairs since images are identical)
        duplicates = backend.find_duplicates(similarity_threshold=0.95)
        
        assert isinstance(duplicates, list)
        # Should find duplicate pairs
        assert len(duplicates) > 0
    
    def test_search_by_metadata(self, backend, sample_images):
        """Test metadata-only search."""
        # Add images
        backend.add_images_from_directory(
            Path(sample_images[0]).parent,
            max_images=2,
            recursive=False
        )
        
        # Search by metadata
        results = backend.search_by_metadata(
            {"format": "JPEG"},
            limit=10
        )
        
        assert isinstance(results, list)
        
        # All results should be JPEG images
        for result in results:
            assert result['metadata']['format'] == "JPEG"
    
    def test_remove_image(self, backend, sample_images):
        """Test image removal."""
        image_path = sample_images[0]
        
        # Add image
        backend.add_image(image_path)
        
        # Verify it was added
        stats_before = backend.get_stats()
        assert stats_before['total_images'] == 1
        
        # Remove image
        result = backend.remove_image(image_path)
        assert result is True
        
        # Verify it was removed
        stats_after = backend.get_stats()
        assert stats_after['total_images'] == 0
    
    def test_get_stats(self, backend, sample_images):
        """Test collection statistics."""
        # Add some images
        backend.add_images_from_directory(
            Path(sample_images[0]).parent,
            max_images=2,
            recursive=False
        )
        
        # Get stats
        stats = backend.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_images' in stats
        assert 'collection_name' in stats
        assert 'embedding_model' in stats
        assert stats['total_images'] == 2
        assert stats['collection_name'] == "test_collection"
    
    def test_export_collection(self, backend, sample_images):
        """Test collection export."""
        # Add some images
        backend.add_images_from_directory(
            Path(sample_images[0]).parent,
            max_images=2,
            recursive=False
        )
        
        # Export collection
        export_file = os.path.join(backend.persist_directory, "export.json")
        result = backend.export_collection(export_file)
        
        assert result is True
        assert os.path.exists(export_file)
        
        # Verify export file content
        import json
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        assert 'collection_name' in export_data
        assert 'embedding_model' in export_data
        assert 'data' in export_data
        assert len(export_data['data']) == 2
    
    def test_context_manager(self, temp_dir):
        """Test context manager usage."""
        with ChromaDBBackend(
            collection_name="context_test",
            persist_directory=temp_dir
        ) as backend:
            assert backend.collection is not None
        
        # Backend should be properly closed (no explicit cleanup needed for ChromaDB)
    
    def test_large_batch_processing(self, backend, temp_dir):
        """Test processing large batches of images."""
        # Create many test images
        num_images = 50
        image_paths = []
        
        for i in range(num_images):
            img = Image.new("RGB", (50, 50), (i % 255, (i * 2) % 255, (i * 3) % 255))
            img_path = Path(temp_dir) / f"batch_image_{i:03d}.jpg"
            img.save(img_path)
            image_paths.append(str(img_path))
        
        # Add all images
        count = backend.add_images_from_directory(
            str(temp_dir),
            max_images=num_images,
            recursive=False
        )
        
        assert count == num_images
        
        # Verify all were added
        stats = backend.get_stats()
        assert stats['total_images'] == num_images
    
    def test_similarity_search_performance(self, backend, temp_dir):
        """Test search performance with moderate dataset."""
        import time
        
        # Create moderate dataset
        num_images = 20
        for i in range(num_images):
            img = Image.new("RGB", (100, 100), (i, i * 2, i * 3))
            img_path = Path(temp_dir) / f"perf_image_{i:03d}.jpg"
            img.save(img_path)
            backend.add_image(str(img_path))
        
        # Time search operations
        start_time = time.time()
        results = backend.find_similar(query_text="test image", top_k=10)
        search_time = time.time() - start_time
        
        assert len(results) > 0
        assert search_time < 5.0  # Should be fast even with 20 images
    
    def test_error_handling_nonexistent_image(self, backend):
        """Test error handling for non-existent images."""
        # Try to add non-existent image
        result = backend.add_image("/nonexistent/image.jpg")
        assert result is False
        
        # Try to get embedding for non-existent image
        embedding = backend.get_embedding("/nonexistent/image.jpg")
        assert embedding is None
    
    def test_error_handling_invalid_query(self, backend, sample_images):
        """Test error handling for invalid queries."""
        # Add some images first
        backend.add_images_from_directory(
            Path(sample_images[0]).parent,
            max_images=1,
            recursive=False
        )
        
        # Test with no query provided
        results = backend.find_similar()
        assert results == []
        
        # Test with invalid query image path
        results = backend.find_similar(query_image_path="/nonexistent.jpg")
        assert results == []


@pytest.mark.integration
class TestChromaDBIntegration:
    """Integration tests for ChromaDB backend with real models."""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_model_loading(self, temp_dir):
        """Test that embedding models load correctly."""
        backend = ChromaDBBackend(persist_directory=temp_dir)
        
        assert backend.model is not None
        
        # Test embedding generation
        test_image = Image.new("RGB", (100, 100), "red")
        embedding = backend._generate_embedding(test_image)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_end_to_end_workflow(self, temp_dir):
        """Test complete workflow from adding to searching."""
        backend = ChromaDBBackend(
            collection_name="e2e_test",
            persist_directory=temp_dir
        )
        
        # Create test images
        test_images = []
        for color in ["red", "blue", "green"]:
            img = Image.new("RGB", (100, 100), color)
            img_path = Path(temp_dir) / f"{color}_test.jpg"
            img.save(img_path)
            test_images.append(str(img_path))
        
        # Add images
        backend.add_images_from_directory(str(temp_dir))
        
        # Test different search methods
        text_results = backend.find_similar(query_text="red color", top_k=2)
        image_results = backend.find_similar(query_image_path=test_images[0], top_k=2)
        
        assert len(text_results) > 0
        assert len(image_results) > 0
        
        # Verify we get reasonable results
        for result in text_results:
            assert 'filename' in result
            assert 'similarity' in result
            assert 0.0 <= result['similarity'] <= 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])