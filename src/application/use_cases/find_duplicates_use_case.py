"""
Find Duplicates Use Case
=======================

Use case for finding duplicate and near-duplicate images.
"""

import time
from typing import Dict, Any, List, Optional
from collections import defaultdict

from ...domain.vector_storage import VectorEmbedding
from ...domain.similarity_search import SimilarityResult
from ...domain.configuration import ModelConfiguration


class FindDuplicatesRequest:
    """
    Request object for finding duplicates.
    
    Attributes:
        embedding_ids: Optional list of specific embeddings to check
        threshold: Similarity threshold for duplicates
        group_similar: Whether to group similar (not just identical) images
        include_metadata: Whether to include metadata in results
        min_group_size: Minimum group size to include
    """
    
    def __init__(
        self,
        embedding_ids: Optional[List[str]] = None,
        threshold: float = 0.95,
        group_similar: bool = True,
        include_metadata: bool = True,
        min_group_size: int = 2
    ):
        self.embedding_ids = embedding_ids
        self.threshold = threshold
        self.group_similar = group_similar
        self.include_metadata = include_metadata
        self.min_group_size = min_group_size


class FindDuplicatesResponse:
    """
    Response object for finding duplicates.
    
    Attributes:
        duplicate_groups: List of duplicate groups
        total_duplicates: Total number of duplicate images
        unique_images: Number of unique images
        processing_time: Time taken to find duplicates
        statistics: Duplicate detection statistics
    """
    
    def __init__(
        self,
        duplicate_groups: List[List[Dict[str, Any]]],
        total_duplicates: int,
        unique_images: int,
        processing_time: float,
        statistics: Dict[str, Any]
    ):
        self.duplicate_groups = duplicate_groups
        self.total_duplicates = total_duplicates
        self.unique_images = unique_images
        self.processing_time = processing_time
        self.statistics = statistics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "duplicate_groups": self.duplicate_groups,
            "total_duplicates": self.total_duplicates,
            "unique_images": self.unique_images,
            "duplicate_percentage": self.total_duplicates / (self.total_duplicates + self.unique_images) * 100,
            "processing_time": self.processing_time,
            "statistics": self.statistics
        }


class FindDuplicatesUseCase:
    """
    Use case for finding duplicate and near-duplicate images.
    
    Uses the duplicate detection service to find similar images.
    """
    
    def __init__(
        self,
        vector_store: "VectorStore",
        duplicate_detector: "DuplicateDetector"
    ):
        """
        Initialize the use case.
        
        Args:
            vector_store: Vector storage aggregate root
            duplicate_detector: Service for duplicate detection
        """
        self.vector_store = vector_store
        self.duplicate_detector = duplicate_detector
    
    def execute(self, request: FindDuplicatesRequest) -> FindDuplicatesResponse:
        """
        Execute the find duplicates use case.
        
        Args:
            request: Find duplicates request
            
        Returns:
            Find duplicates response
        """
        start_time = time.time()
        
        # Step 1: Get embeddings to check
        embeddings = self._get_embeddings_to_check(request.embedding_ids)
        
        if len(embeddings) < 2:
            return FindDuplicatesResponse(
                duplicate_groups=[],
                total_duplicates=0,
                unique_images=len(embeddings),
                processing_time=time.time() - start_time,
                statistics={
                    "total_embeddings_checked": len(embeddings),
                    "threshold": request.threshold,
                    "min_group_size": request.min_group_size
                }
            )
        
        # Step 2: Find duplicate groups
        try:
            duplicate_groups = self.duplicate_detector.get_duplicate_groups(
                embeddings, request.threshold
            )
        except Exception as e:
            raise RuntimeError(f"Duplicate detection failed: {str(e)}")
        
        # Step 3: Filter groups by minimum size
        filtered_groups = [
            group for group in duplicate_groups 
            if len(group) >= request.min_group_size
        ]
        
        # Step 4: Convert to response format
        response_groups = []
        total_duplicates = 0
        
        for group in filtered_groups:
            group_info = []
            for embedding in group:
                # Get associated image info
                image_info = self._get_image_info(embedding, request.include_metadata)
                group_info.append(image_info)
            
            response_groups.append(group_info)
            total_duplicates += len(group)
        
        # Step 5: Calculate statistics
        unique_images = len(embeddings) - total_duplicates
        
        processing_time = time.time() - start_time
        
        statistics = self._generate_statistics(
            embeddings, response_groups, request
        )
        
        return FindDuplicatesResponse(
            duplicate_groups=response_groups,
            total_duplicates=total_duplicates,
            unique_images=unique_images,
            processing_time=processing_time,
            statistics=statistics
        )
    
    def _get_embeddings_to_check(self, embedding_ids: Optional[List[str]]) -> List[VectorEmbedding]:
        """
        Get embeddings to check for duplicates.
        
        Args:
            embedding_ids: Optional list of specific embedding IDs
            
        Returns:
            List of embeddings to check
        """
        if embedding_ids:
            # Get specific embeddings
            embeddings = []
            for emb_id in embedding_ids:
                # This would need implementation in vector_store
                # For now, return empty list
                pass
            return embeddings
        else:
            # Get all embeddings
            all_images = self.vector_store.get_all_images()
            return [item["embedding"] for item in all_images]
    
    def _get_image_info(
        self, 
        embedding: VectorEmbedding, 
        include_metadata: bool
    ) -> Dict[str, Any]:
        """
        Get image information for an embedding.
        
        Args:
            embedding: Vector embedding
            include_metadata: Whether to include metadata
            
        Returns:
            Image information dictionary
        """
        # Get associated image from vector store
        image = self._get_image_for_embedding(embedding)
        
        info = {
            "image_path": str(image.path) if image else "unknown",
            "image_filename": image.filename if image else "unknown",
            "image_hash": image.file_hash if image else "unknown",
            "embedding_id": str(embedding.created_at),  # Placeholder
            "model_name": embedding.model_name,
            "embedding_dimension": embedding.dimension,
            "created_at": embedding.created_at.isoformat(),
            "norm": embedding.norm
        }
        
        if include_metadata and image and image.metadata:
            info["image_metadata"] = image.metadata
        
        return info
    
    def _get_image_for_embedding(self, embedding: VectorEmbedding) -> Optional[Image]:
        """
        Get associated image for an embedding.
        
        Args:
            embedding: Vector embedding
            
        Returns:
            Associated image or None
        """
        # This would need implementation in vector_store
        # For now, return None
        return None
    
    def _generate_statistics(
        self,
        embeddings: List[VectorEmbedding],
        duplicate_groups: List[List[Dict[str, Any]]],
        request: FindDuplicatesRequest
    ) -> Dict[str, Any]:
        """
        Generate duplicate detection statistics.
        
        Args:
            embeddings: All embeddings checked
            duplicate_groups: Found duplicate groups
            request: Original request
            
        Returns:
            Statistics dictionary
        """
        total_embeddings = len(embeddings)
        duplicate_embeddings = sum(len(group) for group in duplicate_groups)
        unique_embeddings = total_embeddings - duplicate_embeddings
        
        if duplicate_groups:
            group_sizes = [len(group) for group in duplicate_groups]
            avg_group_size = sum(group_sizes) / len(group_sizes)
            max_group_size = max(group_sizes)
            min_group_size = min(group_sizes)
        else:
            avg_group_size = 0
            max_group_size = 0
            min_group_size = 0
        
        # Model distribution
        model_counts = defaultdict(int)
        for embedding in embeddings:
            model_counts[embedding.model_name] += 1
        
        return {
            "total_embeddings_checked": total_embeddings,
            "duplicate_groups_found": len(duplicate_groups),
            "duplicate_embeddings": duplicate_embeddings,
            "unique_embeddings": unique_embeddings,
            "duplicate_percentage": (duplicate_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0,
            "threshold_used": request.threshold,
            "min_group_size": request.min_group_size,
            "average_group_size": avg_group_size,
            "max_group_size": max_group_size,
            "min_group_size": min_group_size,
            "model_distribution": dict(model_counts),
            "group_size_distribution": self._get_group_size_distribution(duplicate_groups)
        }
    
    def _get_group_size_distribution(self, duplicate_groups: List[List[Dict[str, Any]]]) -> Dict[int, int]:
        """
        Get distribution of group sizes.
        
        Args:
            duplicate_groups: List of duplicate groups
            
        Returns:
            Dictionary mapping group size to count
        """
        distribution = defaultdict(int)
        for group in duplicate_groups:
            distribution[len(group)] += 1
        return dict(distribution)
    
    def get_recommendations(
        self, 
        duplicate_groups: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for handling duplicates.
        
        Args:
            duplicate_groups: List of duplicate groups
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        for i, group in enumerate(duplicate_groups):
            if len(group) <= 1:
                continue
            
            group_info = {
                "group_id": i + 1,
                "group_size": len(group),
                "recommendations": []
            }
            
            # Analyze the group
            if len(group) == 2:
                group_info["recommendations"].append({
                    "action": "keep_one",
                    "description": "Keep one image and remove the other",
                    "images": group
                })
            elif len(group) <= 5:
                group_info["recommendations"].append({
                    "action": "manual_review",
                    "description": "Manual review needed to determine which images to keep",
                    "images": group
                })
            else:
                group_info["recommendations"].append({
                    "action": "bulk_remove",
                    "description": f"Consider removing {len(group) - 1} images from this group",
                    "images": group[:1]  # Keep only first
                })
            
            recommendations.append(group_info)
        
        return recommendations