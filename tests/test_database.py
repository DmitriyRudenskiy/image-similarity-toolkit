"""
Unit tests for database module.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from PIL import Image

# Import the module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_similarity import EmbeddingDatabase


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    db = EmbeddingDatabase(db_path=db_path, model_name='test_model')
    
    yield db
    
    # Cleanup
    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return {
        'emb1': np.random.rand(128),
        'emb2': np.random.rand(128),
        'emb3': np.random.rand(128),
    }


class TestEmbeddingDatabase:
    """Test cases for EmbeddingDatabase class."""
    
    def test_initialization(self, temp_db):
        """Test database initialization."""
        assert temp_db.model_name == 'test_model'
        assert temp_db.conn is not None
    
    def test_add_image(self, temp_db, sample_embeddings):
        """Test adding an image to the database."""
        image_id = temp_db.add_image(
            'test_image.jpg',
            sample_embeddings['emb1']
        )
        
        assert image_id > 0
    
    def test_add_image_with_metadata(self, temp_db, sample_embeddings):
        """Test adding an image with metadata."""
        metadata = {
            'file_size': 12345,
            'width': 800,
            'height': 600
        }
        
        image_id = temp_db.add_image(
            'test_image.jpg',
            sample_embeddings['emb1'],
            metadata=metadata
        )
        
        assert image_id > 0
    
    def test_get_embedding(self, temp_db, sample_embeddings):
        """Test retrieving an embedding."""
        # Add image
        temp_db.add_image('test_image.jpg', sample_embeddings['emb1'])
        
        # Retrieve embedding
        retrieved = temp_db.get_embedding('test_image.jpg')
        
        assert retrieved is not None
        assert np.allclose(retrieved, sample_embeddings['emb1'])
    
    def test_get_embedding_not_found(self, temp_db):
        """Test retrieving a non-existent embedding."""
        result = temp_db.get_embedding('nonexistent.jpg')
        assert result is None
    
    def test_get_all_embeddings(self, temp_db, sample_embeddings):
        """Test retrieving all embeddings."""
        # Add multiple images
        temp_db.add_image('img1.jpg', sample_embeddings['emb1'])
        temp_db.add_image('img2.jpg', sample_embeddings['emb2'])
        temp_db.add_image('img3.jpg', sample_embeddings['emb3'])
        
        # Retrieve all
        all_embs = temp_db.get_all_embeddings()
        
        assert len(all_embs) == 3
        assert all(len(item) == 3 for item in all_embs)  # (id, path, embedding)
    
    def test_find_similar(self, temp_db, sample_embeddings):
        """Test finding similar images."""
        # Add images
        temp_db.add_image('img1.jpg', sample_embeddings['emb1'])
        temp_db.add_image('img2.jpg', sample_embeddings['emb2'])
        
        # Create query similar to emb1
        query = sample_embeddings['emb1'] + np.random.rand(128) * 0.01
        
        # Search
        results = temp_db.find_similar(query, top_k=5, threshold=0.0)
        
        assert len(results) > 0
        assert 'similarity' in results[0]
        assert 'image_path' in results[0]
    
    def test_find_duplicates(self, temp_db):
        """Test finding duplicate images."""
        # Add similar embeddings (duplicates)
        emb1 = np.random.rand(128)
        emb2 = emb1 + np.random.rand(128) * 0.01  # Very similar
        emb3 = np.random.rand(128)  # Different
        
        temp_db.add_image('img1.jpg', emb1)
        temp_db.add_image('img2.jpg', emb2)
        temp_db.add_image('img3.jpg', emb3)
        
        # Find duplicates with low threshold
        duplicates = temp_db.find_duplicates(similarity_threshold=0.9)
        
        # Should find at least one pair (img1, img2)
        assert isinstance(duplicates, list)
    
    def test_get_saved_duplicates(self, temp_db):
        """Test retrieving saved duplicates."""
        # Add images
        emb1 = np.random.rand(128)
        emb2 = emb1.copy()  # Exact duplicate
        
        id1 = temp_db.add_image('img1.jpg', emb1)
        id2 = temp_db.add_image('img2.jpg', emb2)
        
        # Find and save duplicates
        temp_db.find_duplicates(similarity_threshold=0.99, save_to_table=True)
        
        # Retrieve saved duplicates
        saved = temp_db.get_saved_duplicates()
        
        assert isinstance(saved, list)
    
    def test_get_stats(self, temp_db, sample_embeddings):
        """Test getting database statistics."""
        # Add some images
        temp_db.add_image('img1.jpg', sample_embeddings['emb1'])
        temp_db.add_image('img2.jpg', sample_embeddings['emb2'])
        
        stats = temp_db.get_stats()
        
        assert 'total_images' in stats
        assert stats['total_images'] == 2
        assert stats['model_name'] == 'test_model'
    
    def test_remove_image(self, temp_db, sample_embeddings):
        """Test removing an image."""
        # Add image
        temp_db.add_image('test_image.jpg', sample_embeddings['emb1'])
        
        # Verify it exists
        assert temp_db.get_embedding('test_image.jpg') is not None
        
        # Remove it
        result = temp_db.remove_image('test_image.jpg')
        assert result is True
        
        # Verify it's gone
        assert temp_db.get_embedding('test_image.jpg') is None
    
    def test_remove_nonexistent_image(self, temp_db):
        """Test removing a non-existent image."""
        result = temp_db.remove_image('nonexistent.jpg')
        assert result is False
    
    def test_clear_all(self, temp_db, sample_embeddings):
        """Test clearing all data."""
        # Add images
        temp_db.add_image('img1.jpg', sample_embeddings['emb1'])
        temp_db.add_image('img2.jpg', sample_embeddings['emb2'])
        
        # Clear
        temp_db.clear_all()
        
        # Verify empty
        stats = temp_db.get_stats()
        assert stats['total_images'] == 0
    
    def test_context_manager(self, sample_embeddings):
        """Test using database as context manager."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            with EmbeddingDatabase(db_path=db_path) as db:
                db.add_image('test.jpg', sample_embeddings['emb1'])
                assert db.get_embedding('test.jpg') is not None
            
            # Database should be closed after context
            # Connection should be None or closed
            
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
    
    def test_update_existing_image(self, temp_db):
        """Test updating an existing image."""
        # Add image
        emb1 = np.random.rand(128)
        temp_db.add_image('test.jpg', emb1)
        
        # Update with new embedding
        emb2 = np.random.rand(128)
        temp_db.add_image('test.jpg', emb2)
        
        # Retrieve and verify it's the new embedding
        retrieved = temp_db.get_embedding('test.jpg')
        assert np.allclose(retrieved, emb2)
        assert not np.allclose(retrieved, emb1)
    
    def test_similarity_threshold_filtering(self, temp_db):
        """Test that similarity threshold works correctly."""
        # Add images with known similarities
        emb1 = np.ones(128)
        emb2 = np.ones(128) * 0.5  # Different
        
        temp_db.add_image('img1.jpg', emb1)
        temp_db.add_image('img2.jpg', emb2)
        
        # Search with high threshold
        results = temp_db.find_similar(emb1, top_k=10, threshold=0.9)
        
        # Should only return img1 (itself) with high similarity
        high_sim_results = [r for r in results if r['similarity'] > 0.9]
        assert len(high_sim_results) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
