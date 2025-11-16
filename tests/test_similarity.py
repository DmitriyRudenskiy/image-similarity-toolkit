"""
Unit tests for Image Similarity Toolkit.
"""

import pytest
import numpy as np
import tempfile
import os
from PIL import Image


# Import the module - adjust path as needed
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_similarity import ImageSimilarity
from image_similarity.models import get_embedding_size


@pytest.fixture
def temp_images():
    """
    Create temporary test images.
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create two similar test images
    img1 = Image.new('RGB', (224, 224), color='red')
    img2 = Image.new('RGB', (224, 224), color='red')
    img3 = Image.new('RGB', (224, 224), color='blue')
    
    img1_path = os.path.join(temp_dir, 'image1.jpg')
    img2_path = os.path.join(temp_dir, 'image2.jpg')
    img3_path = os.path.join(temp_dir, 'image3.jpg')
    
    img1.save(img1_path)
    img2.save(img2_path)
    img3.save(img3_path)
    
    yield {
        'similar1': img1_path,
        'similar2': img2_path,
        'different': img3_path
    }
    
    # Cleanup
    for path in [img1_path, img2_path, img3_path]:
        if os.path.exists(path):
            os.remove(path)
    os.rmdir(temp_dir)


class TestImageSimilarity:
    """Test cases for ImageSimilarity class."""
    
    def test_initialization_efficientnet(self):
        """Test initialization with EfficientNet."""
        checker = ImageSimilarity(model_name='efficientnet')
        assert checker.model_name == 'efficientnet'
        assert checker.device in ['cuda', 'cpu']
    
    def test_initialization_resnet(self):
        """Test initialization with ResNet."""
        checker = ImageSimilarity(model_name='resnet')
        assert checker.model_name == 'resnet'
    
    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            ImageSimilarity(model_name='invalid_model')
    
    def test_get_embedding(self, temp_images):
        """Test embedding extraction."""
        checker = ImageSimilarity(model_name='efficientnet')
        embedding = checker.get_embedding(temp_images['similar1'])
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == get_embedding_size('efficientnet')
    
    def test_compare_similar_images(self, temp_images):
        """Test comparison of similar images."""
        checker = ImageSimilarity(model_name='efficientnet')
        results = checker.compare_images(
            temp_images['similar1'],
            temp_images['similar2']
        )
        
        assert 'cosine_similarity' in results
        assert 'euclidean_distance' in results
        assert 'normalized_similarity' in results
        
        # Similar images should have high cosine similarity
        assert results['cosine_similarity'] > 0.8
    
    def test_compare_different_images(self, temp_images):
        """Test comparison of different images."""
        checker = ImageSimilarity(model_name='efficientnet')
        results = checker.compare_images(
            temp_images['similar1'],
            temp_images['different']
        )
        
        # Different images should have lower similarity than similar ones
        results_similar = checker.compare_images(
            temp_images['similar1'],
            temp_images['similar2']
        )
        
        assert results['cosine_similarity'] < results_similar['cosine_similarity']
    
    def test_interpret_similarity(self):
        """Test similarity interpretation."""
        checker = ImageSimilarity(model_name='efficientnet')
        
        assert "Very similar" in checker.interpret_similarity(0.9)
        assert "Moderately similar" in checker.interpret_similarity(0.75)
        assert "Weakly similar" in checker.interpret_similarity(0.6)
        assert "Different" in checker.interpret_similarity(0.3)
    
    def test_visualize_comparison(self, temp_images):
        """Test visualization creation."""
        checker = ImageSimilarity(model_name='efficientnet')
        results = checker.compare_images(
            temp_images['similar1'],
            temp_images['similar2']
        )
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            checker.visualize_comparison(
                temp_images['similar1'],
                temp_images['similar2'],
                results,
                save_path=output_path
            )
            
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestModels:
    """Test cases for model utilities."""
    
    def test_get_embedding_size(self):
        """Test embedding size retrieval."""
        assert get_embedding_size('resnet') == 2048
        assert get_embedding_size('efficientnet') == 1280
        assert get_embedding_size('clip') == 512
        assert get_embedding_size('unknown') == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
