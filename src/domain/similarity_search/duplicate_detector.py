"""
Duplicate Detector Service
=========================

Domain service for detecting duplicate and near-duplicate images.
"""

from typing import List, Protocol, Set
from abc import ABC, abstractmethod
from collections import defaultdict

from ..vector_storage.vector_embedding import VectorEmbedding
from ..similarity_search.similarity_calculator import SimilarityCalculatorBase
from ..similarity_search.similarity_query import SearchMode


class DuplicateDetector(Protocol):
    """
    Protocol for duplicate detection operations.
    
    Defines the interface that all duplicate detection implementations must provide.
    """
    
    def find_duplicates(
        self,
        embeddings: List[VectorEmbedding],
        threshold: float = 0.95,
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> List[List[VectorEmbedding]]:
        """
        Find duplicate embeddings in a collection.
        
        Args:
            embeddings: List of embeddings to check
            threshold: Similarity threshold for duplicates
            mode: Similarity calculation mode
            
        Returns:
            List of duplicate groups, each containing similar embeddings
        """
        ...
    
    def is_duplicate(
        self,
        embedding: VectorEmbedding,
        existing_embeddings: List[VectorEmbedding],
        threshold: float = 0.95,
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> bool:
        """
        Check if an embedding is a duplicate of existing ones.
        
        Args:
            embedding: Embedding to check
            existing_embeddings: List of existing embeddings
            threshold: Similarity threshold
            mode: Similarity calculation mode
            
        Returns:
            True if embedding is a duplicate, False otherwise
        """
        ...
    
    def get_duplicate_groups(
        self,
        embeddings: List[VectorEmbedding],
        threshold: float = 0.95,
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> List[List[VectorEmbedding]]:
        """
        Group embeddings into duplicate sets.
        
        Args:
            embeddings: List of embeddings
            threshold: Similarity threshold
            mode: Similarity calculation mode
            
        Returns:
            List of duplicate groups (each group contains similar embeddings)
        """
        ...


class DuplicateDetectorBase(ABC, DuplicateDetector):
    """
    Abstract base class for duplicate detector implementations.
    
    Provides common functionality for duplicate detection.
    """
    
    def __init__(self, similarity_calculator: SimilarityCalculatorBase):
        """
        Initialize duplicate detector.
        
        Args:
            similarity_calculator: Calculator for similarity operations
        """
        self.similarity_calculator = similarity_calculator
    
    def find_duplicates(
        self,
        embeddings: List[VectorEmbedding],
        threshold: float = 0.95,
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> List[List[VectorEmbedding]]:
        """Find duplicate embeddings in a collection."""
        return self.get_duplicate_groups(embeddings, threshold, mode)
    
    def is_duplicate(
        self,
        embedding: VectorEmbedding,
        existing_embeddings: List[VectorEmbedding],
        threshold: float = 0.95,
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> bool:
        """Check if an embedding is a duplicate of existing ones."""
        for existing in existing_embeddings:
            if self.similarity_calculator.is_similar(
                embedding, existing, threshold, mode
            ):
                return True
        return False
    
    def get_duplicate_groups(
        self,
        embeddings: List[VectorEmbedding],
        threshold: float = 0.95,
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> List[List[VectorEmbedding]]:
        """Group embeddings into duplicate sets."""
        if len(embeddings) < 2:
            return []
        
        # Use Union-Find algorithm to find connected components
        parent = {i: i for i in range(len(embeddings))}
        rank = {i: 0 for i in range(len(embeddings))}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                if rank[root_x] < rank[root_y]:
                    parent[root_x] = root_y
                elif rank[root_x] > rank[root_y]:
                    parent[root_y] = root_x
                else:
                    parent[root_y] = root_x
                    rank[root_x] += 1
        
        # Compare all pairs of embeddings
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if self.similarity_calculator.is_similar(
                    embeddings[i], embeddings[j], threshold, mode
                ):
                    union(i, j)
        
        # Group embeddings by root
        groups = defaultdict(list)
        for i, embedding in enumerate(embeddings):
            root = find(i)
            groups[root].append(embedding)
        
        # Return only groups with more than one embedding (actual duplicates)
        return [group for group in groups.values() if len(group) > 1]
    
    def get_duplicate_statistics(
        self,
        embeddings: List[VectorEmbedding],
        threshold: float = 0.95,
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> dict:
        """
        Get statistics about duplicates in the collection.
        
        Args:
            embeddings: List of embeddings
            threshold: Similarity threshold
            mode: Similarity calculation mode
            
        Returns:
            Dictionary with duplicate statistics
        """
        duplicate_groups = self.get_duplicate_groups(embeddings, threshold, mode)
        
        total_embeddings = len(embeddings)
        duplicate_embeddings = sum(len(group) for group in duplicate_groups)
        unique_embeddings = total_embeddings - duplicate_embeddings
        duplicate_count = len(duplicate_groups)
        
        # Calculate average group size
        if duplicate_groups:
            avg_group_size = sum(len(group) for group in duplicate_groups) / len(duplicate_groups)
            max_group_size = max(len(group) for group in duplicate_groups)
            min_group_size = min(len(group) for group in duplicate_groups)
        else:
            avg_group_size = 0
            max_group_size = 0
            min_group_size = 0
        
        return {
            "total_embeddings": total_embeddings,
            "unique_embeddings": unique_embeddings,
            "duplicate_embeddings": duplicate_embeddings,
            "duplicate_groups": duplicate_count,
            "duplicate_percentage": (duplicate_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0,
            "average_group_size": avg_group_size,
            "max_group_size": max_group_size,
            "min_group_size": min_group_size,
            "threshold": threshold,
            "mode": mode.value
        }
    
    def find_near_duplicates(
        self,
        embeddings: List[VectorEmbedding],
        threshold: float = 0.8,
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> List[List[VectorEmbedding]]:
        """
        Find near-duplicate embeddings with lower threshold.
        
        Args:
            embeddings: List of embeddings
            threshold: Similarity threshold for near-duplicates
            mode: Similarity calculation mode
            
        Returns:
            List of near-duplicate groups
        """
        return self.get_duplicate_groups(embeddings, threshold, mode)
    
    def filter_duplicates(
        self,
        embeddings: List[VectorEmbedding],
        keep_first: bool = True,
        threshold: float = 0.95,
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> List[VectorEmbedding]:
        """
        Filter out duplicates, keeping only unique embeddings.
        
        Args:
            embeddings: List of embeddings to filter
            keep_first: Whether to keep the first or last in each duplicate group
            threshold: Similarity threshold
            mode: Similarity calculation mode
            
        Returns:
            List of unique embeddings
        """
        duplicate_groups = self.get_duplicate_groups(embeddings, threshold, mode)
        
        if not duplicate_groups:
            return embeddings.copy()
        
        # Create set of embeddings to remove
        to_remove = set()
        for group in duplicate_groups:
            if keep_first:
                # Remove all except first
                to_remove.update(group[1:])
            else:
                # Remove all except last
                to_remove.update(group[:-1])
        
        # Return embeddings not in to_remove set
        result = [e for e in embeddings if e not in to_remove]
        return result
    
    def get_most_diverse_subset(
        self,
        embeddings: List[VectorEmbedding],
        target_size: int,
        threshold: float = 0.95,
        mode: SearchMode = SearchMode.COSINE_SIMILARITY
    ) -> List[VectorEmbedding]:
        """
        Get a diverse subset by removing similar embeddings.
        
        Args:
            embeddings: List of embeddings
            target_size: Target size of the subset
            threshold: Similarity threshold for considering embeddings similar
            mode: Similarity calculation mode
            
        Returns:
            Diverse subset of embeddings
        """
        if len(embeddings) <= target_size:
            return embeddings.copy()
        
        # Start with first embedding
        subset = [embeddings[0]]
        remaining = embeddings[1:]
        
        while len(subset) < target_size and remaining:
            # Find embedding most different from current subset
            best_candidate = None
            best_min_similarity = float('inf')
            
            for candidate in remaining:
                # Calculate minimum similarity to all embeddings in subset
                min_similarity = float('inf')
                for existing in subset:
                    similarity = self.similarity_calculator.calculate_cosine_similarity(
                        candidate, existing
                    )
                    min_similarity = min(min_similarity, similarity)
                
                # If this candidate is more diverse, select it
                if min_similarity < best_min_similarity:
                    best_min_similarity = min_similarity
                    best_candidate = candidate
            
            if best_candidate is not None:
                subset.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return subset


class DefaultDuplicateDetector(DuplicateDetectorBase):
    """
    Default implementation of duplicate detector.
    
    Uses Union-Find algorithm for efficient duplicate detection.
    """
    
    def __init__(self, similarity_calculator: SimilarityCalculatorBase):
        """
        Initialize the duplicate detector.
        
        Args:
            similarity_calculator: Calculator for similarity operations
        """
        super().__init__(similarity_calculator)
    
    def __repr__(self) -> str:
        return f"DefaultDuplicateDetector(calculator={self.similarity_calculator})"