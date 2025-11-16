"""
Similarity Calculator Service
============================

Domain service for calculating similarity between embeddings and images.
"""

import numpy as np
from typing import List, Protocol, Optional, Tuple
from abc import ABC, abstractmethod

from ..vector_storage.vector_embedding import VectorEmbedding
from .similarity_query import SearchMode, SearchType


class SimilarityCalculator(Protocol):
    """
    Protocol for similarity calculation operations.
    
    Defines the interface that all similarity calculation implementations must provide.
    """
    
    def calculate_cosine_similarity(
        self, 
        embedding1: VectorEmbedding, 
        embedding2: VectorEmbedding
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        ...
    
    def calculate_euclidean_distance(
        self, 
        embedding1: VectorEmbedding, 
        embedding2: VectorEmbedding
    ) -> float:
        """
        Calculate Euclidean distance between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Euclidean distance (0.0 to infinity)
        """
        ...
    
    def find_most_similar(
        self,
        query_embedding: VectorEmbedding,
        candidates: List[VectorEmbedding],
        mode: SearchMode = SearchMode.COSINE_SIMILARITY,
        limit: Optional[int] = None
    ) -> List[Tuple[VectorEmbedding, float]]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding
            candidates: List of candidate embeddings
            mode: Similarity calculation mode
            limit: Maximum number of results
            
        Returns:
            List of (embedding, similarity_score) tuples
        """
        ...


class SimilarityCalculatorBase(ABC):
    """
    Abstract base class for similarity calculator implementations.
    
    Provides common functionality for similarity calculations.
    """
    
    def calculate_cosine_similarity(
        self, 
        embedding1: VectorEmbedding, 
        embedding2: VectorEmbedding
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        if embedding1.dimension != embedding2.dimension:
            raise ValueError("Embeddings must have same dimension")
        
        if embedding1.model_name != embedding2.model_name:
            raise ValueError("Embeddings must use same model")
        
        # Normalize vectors
        vec1_norm = self._normalize_vector(embedding1.vector)
        vec2_norm = self._normalize_vector(embedding2.vector)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Ensure result is in valid range
        return float(np.clip(similarity, 0.0, 1.0))
    
    def calculate_euclidean_distance(
        self, 
        embedding1: VectorEmbedding, 
        embedding2: VectorEmbedding
    ) -> float:
        """Calculate Euclidean distance between two embeddings."""
        if embedding1.dimension != embedding2.dimension:
            raise ValueError("Embeddings must have same dimension")
        
        if embedding1.model_name != embedding2.model_name:
            raise ValueError("Embeddings must use same model")
        
        distance = np.linalg.norm(embedding1.vector - embedding2.vector)
        return float(distance)
    
    def calculate_manhattan_distance(
        self, 
        embedding1: VectorEmbedding, 
        embedding2: VectorEmbedding
    ) -> float:
        """Calculate Manhattan (L1) distance between two embeddings."""
        if embedding1.dimension != embedding2.dimension:
            raise ValueError("Embeddings must have same dimension")
        
        if embedding1.model_name != embedding2.model_name:
            raise ValueError("Embeddings must use same model")
        
        distance = np.sum(np.abs(embedding1.vector - embedding2.vector))
        return float(distance)
    
    def find_most_similar(
        self,
        query_embedding: VectorEmbedding,
        candidates: List[VectorEmbedding],
        mode: SearchMode = SearchMode.COSINE_SIMILARITY,
        limit: Optional[int] = None
    ) -> List[Tuple[VectorEmbedding, float]]:
        """Find most similar embeddings to query."""
        if not candidates:
            return []
        
        similarities = []
        
        for candidate in candidates:
            if mode == SearchMode.COSINE_SIMILARITY:
                score = self.calculate_cosine_similarity(query_embedding, candidate)
            elif mode == SearchMode.EUCLIDEAN_DISTANCE:
                # Convert distance to similarity score (inverse relationship)
                distance = self.calculate_euclidean_distance(query_embedding, candidate)
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity
            elif mode == SearchMode.MANHATTAN_DISTANCE:
                # Convert distance to similarity score (inverse relationship)
                distance = self.calculate_manhattan_distance(query_embedding, candidate)
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity
            else:
                raise ValueError(f"Unsupported search mode: {mode}")
            
            similarities.append((candidate, score))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply limit if specified
        if limit is not None:
            similarities = similarities[:limit]
        
        return similarities
    
    def calculate_batch_similarities(
        self,
        query_embedding: VectorEmbedding,
        candidates: List[VectorEmbedding],
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> np.ndarray:
        """
        Calculate similarities for multiple candidates efficiently.
        
        Args:
            query_embedding: Query embedding
            candidates: List of candidate embeddings
            mode: Similarity calculation mode
            
        Returns:
            Array of similarity scores
        """
        if not candidates:
            return np.array([])
        
        # Stack candidate vectors
        candidate_matrix = np.stack([c.vector for c in candidates])
        
        if mode == SearchMode.COSINE_SIMILARITY:
            # Normalize vectors
            query_norm = self._normalize_vector(query_embedding.vector)
            candidates_norm = np.array([self._normalize_vector(c.vector) for c in candidates])
            
            # Calculate cosine similarities
            similarities = np.dot(candidates_norm, query_norm)
            
        elif mode == SearchMode.EUCLIDEAN_DISTANCE:
            # Calculate Euclidean distances
            query_vec = query_embedding.vector.reshape(1, -1)
            distances = np.linalg.norm(candidates_matrix - query_vec, axis=1)
            # Convert to similarity scores
            similarities = 1.0 / (1.0 + distances)
            
        elif mode == SearchMode.MANHATTAN_DISTANCE:
            # Calculate Manhattan distances
            query_vec = query_embedding.vector.reshape(1, -1)
            distances = np.sum(np.abs(candidates_matrix - query_vec), axis=1)
            # Convert to similarity scores
            similarities = 1.0 / (1.0 + distances)
            
        else:
            raise ValueError(f"Unsupported search mode: {mode}")
        
        return similarities
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector.copy()
    
    def is_similar(
        self,
        embedding1: VectorEmbedding,
        embedding2: VectorEmbedding,
        threshold: float = 0.8,
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> bool:
        """
        Check if two embeddings are similar.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            threshold: Similarity threshold
            mode: Similarity calculation mode
            
        Returns:
            True if embeddings are similar, False otherwise
        """
        if mode == SearchMode.COSINE_SIMILARITY:
            similarity = self.calculate_cosine_similarity(embedding1, embedding2)
            return similarity >= threshold
        else:
            # For distance metrics, lower distance = higher similarity
            if mode == SearchMode.EUCLIDEAN_DISTANCE:
                distance = self.calculate_euclidean_distance(embedding1, embedding2)
            elif mode == SearchMode.MANHATTAN_DISTANCE:
                distance = self.calculate_manhattan_distance(embedding1, embedding2)
            else:
                raise ValueError(f"Unsupported search mode: {mode}")
            
            # Convert distance threshold to similarity threshold
            similarity_threshold = 1.0 / (1.0 + threshold)
            similarity = 1.0 / (1.0 + distance)
            return similarity >= similarity_threshold
    
    def get_similarity_statistics(
        self,
        query_embedding: VectorEmbedding,
        candidates: List[VectorEmbedding],
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> dict:
        """
        Get statistics about similarities.
        
        Args:
            query_embedding: Query embedding
            candidates: List of candidate embeddings
            mode: Similarity calculation mode
            
        Returns:
            Dictionary with similarity statistics
        """
        if not candidates:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0
            }
        
        similarities = self.calculate_batch_similarities(query_embedding, candidates, mode)
        
        return {
            "count": len(candidates),
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "median": float(np.median(similarities)),
            "q25": float(np.percentile(similarities, 25)),
            "q75": float(np.percentile(similarities, 75))
        }


class DefaultSimilarityCalculator(SimilarityCalculatorBase):
    """
    Default implementation of similarity calculator.
    
    Uses numpy for efficient vector operations.
    """
    
    def __init__(self):
        """Initialize the similarity calculator."""
        pass
    
    def __repr__(self) -> str:
        return "DefaultSimilarityCalculator()"