"""
Core module for image similarity comparison.
"""

import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple, Optional

from .models import load_model
from .visualization import visualize_comparison


class ImageSimilarity:
    """
    Main class for comparing image similarity using deep learning models.
    
    Attributes:
        model_name (str): Name of the model to use ('resnet', 'efficientnet', 'clip')
        device (str): Device to run the model on ('cuda' or 'cpu')
        model: The loaded neural network model
        preprocess: Image preprocessing transform
    
    Example:
        >>> checker = ImageSimilarity(model_name='efficientnet')
        >>> results = checker.compare_images('image1.jpg', 'image2.jpg')
        >>> print(f"Cosine similarity: {results['cosine_similarity']:.4f}")
    """
    
    def __init__(self, model_name: str = 'efficientnet'):
        """
        Initialize the ImageSimilarity class.
        
        Args:
            model_name: Model to use ('resnet', 'efficientnet', or 'clip')
            
        Raises:
            ValueError: If an unsupported model name is provided
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model, self.preprocess = load_model(model_name, self.device)
        
    def get_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract embedding vector from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array containing the image embedding
            
        Raises:
            Exception: If there's an error processing the image
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.model_name == 'clip':
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model.encode_image(image_input)
                    features /= features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy()[0]
            else:
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model(image_tensor)
                    features = features.squeeze()
                return features.cpu().numpy()
                
        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {e}")
    
    def compare_images(
        self, 
        image_path1: str, 
        image_path2: str
    ) -> Dict[str, float]:
        """
        Compare two images and return similarity metrics.
        
        Args:
            image_path1: Path to the first image
            image_path2: Path to the second image
            
        Returns:
            Dictionary containing similarity metrics:
                - cosine_similarity: Cosine similarity (0-1)
                - euclidean_distance: Euclidean distance
                - normalized_similarity: Normalized similarity (0-1)
                - embedding1: First image embedding
                - embedding2: Second image embedding
                
        Example:
            >>> checker = ImageSimilarity(model_name='efficientnet')
            >>> results = checker.compare_images('cat1.jpg', 'cat2.jpg')
            >>> if results['cosine_similarity'] > 0.85:
            ...     print("Images are very similar!")
        """
        embedding1 = self.get_embedding(image_path1)
        embedding2 = self.get_embedding(image_path2)
        
        # Cosine similarity
        cosine_sim = cosine_similarity([embedding1], [embedding2])[0][0]
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        
        # Normalized distance (0-1)
        normalized_dist = 1 / (1 + euclidean_dist)
        
        return {
            'cosine_similarity': float(cosine_sim),
            'euclidean_distance': float(euclidean_dist),
            'normalized_similarity': float(normalized_dist),
            'embedding1': embedding1,
            'embedding2': embedding2
        }
    
    def visualize_comparison(
        self, 
        image_path1: str, 
        image_path2: str, 
        results: Dict[str, float], 
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the comparison results.
        
        Args:
            image_path1: Path to the first image
            image_path2: Path to the second image
            results: Dictionary with comparison results
            save_path: Optional path to save the visualization
            
        Example:
            >>> checker = ImageSimilarity()
            >>> results = checker.compare_images('img1.jpg', 'img2.jpg')
            >>> checker.visualize_comparison('img1.jpg', 'img2.jpg', results, 'output.png')
        """
        visualize_comparison(image_path1, image_path2, results, save_path)
    
    def interpret_similarity(self, cosine_similarity: float) -> str:
        """
        Interpret cosine similarity value into a human-readable description.
        
        Args:
            cosine_similarity: Cosine similarity value (0-1)
            
        Returns:
            String description of similarity level
        """
        if cosine_similarity > 0.85:
            return "Very similar images"
        elif cosine_similarity > 0.7:
            return "Moderately similar images"
        elif cosine_similarity > 0.5:
            return "Weakly similar images"
        else:
            return "Different images"
