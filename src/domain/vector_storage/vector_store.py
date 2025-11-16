"""
Vector Store Aggregate Root
===========================

Main aggregate root for vector storage operations.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Protocol
from uuid import UUID

from ..vector_storage.vector_embedding import VectorEmbedding
from ..image_processing import Image
from .vector_repository import VectorRepository


class VectorStore:
    """
    Aggregate Root for vector storage operations.
    
    Coordinates vector embeddings and their storage across different repositories.
    Provides a unified interface for all vector operations.
    
    Attributes:
        _repository: Repository for persistence
        _cache: Optional cache for frequently accessed embeddings
    """
    
    def __init__(self, repository: VectorRepository):
        """
        Initialize vector store.
        
        Args:
            repository: Repository implementation for persistence
        """
        self._repository = repository
        self._cache: Dict[UUID, tuple[VectorEmbedding, Image]] = {}
    
    def add_image(
        self, 
        image: Image, 
        embedding: VectorEmbedding
    ) -> UUID:
        """
        Add image with its embedding to the store.
        
        Args:
            image: Image to add
            embedding: Associated embedding vector
            
        Returns:
            Unique identifier of the added embedding
            
        Raises:
            ValueError: If embedding model doesn't match expected model
            ValueError: If image already exists in store
        """
        # Validate embedding model
        if embedding.model_name not in self.get_supported_models():
            raise ValueError(
                f"Unsupported embedding model: {embedding.model_name}. "
                f"Supported models: {self.get_supported_models()}"
            )
        
        # Check for duplicate image
        existing = self._repository.find_by_image_hash(image.file_hash)
        if existing is not None:
            raise ValueError(
                f"Image {image.filename} (hash: {image.file_hash}) "
                "already exists in the store"
            )
        
        # Save to repository
        embedding_id = self._repository.save(embedding, image)
        
        # Cache the embedding
        self._cache[embedding_id] = (embedding, image)
        
        return embedding_id
    
    def add_images_batch(
        self, 
        items: List[tuple[Image, VectorEmbedding]]
    ) -> List[UUID]:
        """
        Add multiple images with their embeddings.
        
        Args:
            items: List of (image, embedding) tuples
            
        Returns:
            List of unique identifiers for each added embedding
            
        Raises:
            ValueError: If any embedding model is unsupported
            ValueError: If any image already exists
        """
        if not items:
            return []
        
        # Validate all items first
        for i, (image, embedding) in enumerate(items):
            if embedding.model_name not in self.get_supported_models():
                raise ValueError(
                    f"Unsupported embedding model at index {i}: {embedding.model_name}"
                )
        
        # Check for duplicates
        for i, (image, _) in enumerate(items):
            existing = self._repository.find_by_image_hash(image.file_hash)
            if existing is not None:
                raise ValueError(
                    f"Image at index {i} ({image.filename}) already exists"
                )
        
        # Add all items
        result_ids = []
        for image, embedding in items:
            embedding_id = self.add_image(image, embedding)
            result_ids.append(embedding_id)
        
        return result_ids
    
    def find_similar_images(
        self, 
        query_embedding: VectorEmbedding, 
        limit: int = 10,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Find images similar to the query embedding.
        
        Args:
            query_embedding: Query embedding to search with
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of dictionaries with image info and similarity scores
        """
        # Use repository to find similar embeddings
        similar_embeddings = self._repository.find_similar(
            query_embedding, limit, threshold
        )
        
        results = []
        for embedding, similarity in similar_embeddings:
            # Get associated image from cache or repository
            image = self._get_image_for_embedding(embedding)
            
            results.append({
                "image": image,
                "embedding": embedding,
                "similarity_score": similarity
            })
        
        return results
    
    def find_duplicates(
        self, 
        threshold: float = 0.95
    ) -> List[List[Dict]]:
        """
        Find duplicate or near-duplicate images.
        
        Args:
            threshold: Similarity threshold for duplicates
            
        Returns:
            List of duplicate groups, each containing image info
        """
        # Use repository to find duplicate embeddings
        duplicate_groups = self._repository.find_duplicates(threshold)
        
        results = []
        for group in duplicate_groups:
            group_info = []
            for embedding in group:
                image = self._get_image_for_embedding(embedding)
                group_info.append({
                    "image": image,
                    "embedding": embedding
                })
            results.append(group_info)
        
        return results
    
    def remove_image(self, embedding_id: UUID) -> bool:
        """
        Remove image from the store.
        
        Args:
            embedding_id: Unique identifier of the embedding to remove
            
        Returns:
            True if removed, False if not found
        """
        # Remove from cache
        if embedding_id in self._cache:
            del self._cache[embedding_id]
        
        # Remove from repository
        return self._repository.delete_by_id(embedding_id)
    
    def get_image_by_id(self, embedding_id: UUID) -> Optional[Dict]:
        """
        Get image and embedding by ID.
        
        Args:
            embedding_id: Unique identifier
            
        Returns:
            Dictionary with image and embedding info, or None if not found
        """
        # Check cache first
        if embedding_id in self._cache:
            embedding, image = self._cache[embedding_id]
            return {
                "image": image,
                "embedding": embedding
            }
        
        # Fall back to repository
        # Note: This would require repository to provide both embedding and image
        # For now, return None if not in cache
        return None
    
    def get_all_images(self) -> List[Dict]:
        """
        Get all images with their embeddings.
        
        Returns:
            List of all images with their embeddings
        """
        all_embeddings = self._repository.get_all()
        results = []
        
        for embedding in all_embeddings:
            image = self._get_image_for_embedding(embedding)
            results.append({
                "image": image,
                "embedding": embedding
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Get store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        repo_stats = self._repository.get_stats()
        
        return {
            "total_images": repo_stats.get("total_embeddings", 0),
            "cache_size": len(self._cache),
            "supported_models": self.get_supported_models(),
            "repository_type": repo_stats.get("repository_type", "unknown"),
            "storage_size_mb": repo_stats.get("storage_size_mb", 0)
        }
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
    
    def _get_image_for_embedding(self, embedding: VectorEmbedding) -> Optional[Image]:
        """
        Get associated image for an embedding.
        
        This is a placeholder - in a real implementation, the repository
        would need to provide a method to retrieve both embedding and image.
        
        Args:
            embedding: Vector embedding
            
        Returns:
            Associated image or None if not found
        """
        # This would need to be implemented based on repository capabilities
        # For now, return None as placeholder
        return None
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported embedding models.
        
        Returns:
            List of supported model names
        """
        # This could be configurable or determined from repository capabilities
        return ["resnet", "efficientnet", "clip"]
    
    def export_data(self, format: str = "json") -> str:
        """
        Export all store data.
        
        Args:
            format: Export format ("json", "csv")
            
        Returns:
            Exported data as string
        """
        all_images = self.get_all_images()
        
        if format.lower() == "json":
            import json
            export_data = []
            for item in all_images:
                export_data.append({
                    "image_path": str(item["image"].path),
                    "image_hash": item["image"].file_hash,
                    "embedding": item["embedding"].to_dict()
                })
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["image_path", "image_hash", "model_name", "dimension", "created_at"])
            
            for item in all_images:
                writer.writerow([
                    str(item["image"].path),
                    item["image"].file_hash,
                    item["embedding"].model_name,
                    item["embedding"].dimension,
                    item["embedding"].created_at.isoformat()
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def __len__(self) -> int:
        """Get total number of images in store."""
        return len(self._repository.get_all())
    
    def __contains__(self, image: Image) -> bool:
        """Check if image is in store."""
        existing = self._repository.find_by_image_hash(image.file_hash)
        return existing is not None