"""
Search Similar Images Use Case
=============================

Use case for searching similar images in the system.
"""

import time
from typing import Dict, Any, List, Optional
from uuid import UUID
from pathlib import Path

from ...domain.similarity_search import SimilarityQuery, SimilaritySearchResults
from ...domain.vector_storage import VectorEmbedding
from ...domain.image_processing import Image
from ...domain.configuration import ModelConfiguration


class SearchSimilarImagesRequest:
    """
    Request object for searching similar images.
    
    Attributes:
        query_embedding: Embedding to search with
        query_image: Image to search with
        query_text: Text to search with
        limit: Maximum number of results
        threshold: Minimum similarity threshold
        model_config: Expected model configuration
        filters: Additional filters to apply
    """
    
    def __init__(
        self,
        query_embedding: Optional[VectorEmbedding] = None,
        query_image: Optional[Path] = None,
        query_text: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.0,
        model_config: Optional[ModelConfiguration] = None,
        filters: Optional[Dict[str, Any]] = None
    ):
        self.query_embedding = query_embedding
        self.query_image = query_image
        self.query_text = query_text
        self.limit = limit
        self.threshold = threshold
        self.model_config = model_config
        self.filters = filters or {}
    
    @classmethod
    def from_image(
        cls,
        image_path: Path,
        limit: int = 10,
        threshold: float = 0.0,
        model_config: Optional[ModelConfiguration] = None
    ) -> "SearchSimilarImagesRequest":
        """Create search request from image file."""
        return cls(
            query_image=image_path,
            limit=limit,
            threshold=threshold,
            model_config=model_config
        )
    
    @classmethod
    def from_embedding(
        cls,
        embedding: VectorEmbedding,
        limit: int = 10,
        threshold: float = 0.0
    ) -> "SearchSimilarImagesRequest":
        """Create search request from embedding."""
        return cls(
            query_embedding=embedding,
            limit=limit,
            threshold=threshold
        )
    
    @classmethod
    def from_text(
        cls,
        text: str,
        limit: int = 10,
        threshold: float = 0.0,
        model_config: Optional[ModelConfiguration] = None
    ) -> "SearchSimilarImagesRequest":
        """Create search request from text."""
        return cls(
            query_text=text,
            limit=limit,
            threshold=threshold,
            model_config=model_config
        )


class SearchSimilarImagesResponse:
    """
    Response object for searching similar images.
    
    Attributes:
        results: Search results
        total_found: Total number of matches found
        search_time: Time taken to perform search
        query_info: Information about the query
        statistics: Search statistics
    """
    
    def __init__(
        self,
        results: SimilaritySearchResults,
        search_time: float,
        query_info: Dict[str, Any],
        statistics: Dict[str, Any]
    ):
        self.results = results
        self.total_found = results.total_found
        self.search_time = search_time
        self.query_info = query_info
        self.statistics = statistics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "total_found": self.total_found,
            "search_time": self.search_time,
            "query_info": self.query_info,
            "statistics": self.statistics,
            "results": self.results.to_dict()
        }


class SearchSimilarImagesUseCase:
    """
    Use case for searching similar images.
    
    Coordinates image processing, embedding generation, and similarity search.
    """
    
    def __init__(
        self,
        vector_store: "VectorStore",
        image_processor: "ImageProcessor",
        embedding_generator: "EmbeddingGenerator",
        text_encoder: Optional["TextEncoder"] = None
    ):
        """
        Initialize the use case.
        
        Args:
            vector_store: Vector storage aggregate root
            image_processor: Image processing service
            embedding_generator: Embedding generation service
            text_encoder: Optional text encoding service (for text queries)
        """
        self.vector_store = vector_store
        self.image_processor = image_processor
        self.embedding_generator = embedding_generator
        self.text_encoder = text_encoder
    
    def execute(self, request: SearchSimilarImagesRequest) -> SearchSimilarImagesResponse:
        """
        Execute the search similar images use case.
        
        Args:
            request: Search request
            
        Returns:
            Search response
            
        Raises:
            ValueError: If no query is provided or query is invalid
        """
        start_time = time.time()
        
        # Step 1: Validate request and get query embedding
        query_embedding, query_info = self._prepare_query(request)
        
        # Step 2: Perform similarity search
        try:
            similar_images = self.vector_store.find_similar_images(
                query_embedding,
                limit=request.limit,
                threshold=request.threshold
            )
        except Exception as e:
            raise RuntimeError(f"Similarity search failed: {str(e)}")
        
        # Step 3: Create search results
        results = self._create_search_results(similar_images, request)
        
        # Step 4: Calculate search time
        search_time = time.time() - start_time
        
        # Step 5: Generate statistics
        statistics = self._generate_statistics(results, request)
        
        return SearchSimilarImagesResponse(
            results=results,
            search_time=search_time,
            query_info=query_info,
            statistics=statistics
        )
    
    def _prepare_query(
        self, 
        request: SearchSimilarImagesRequest
    ) -> tuple[VectorEmbedding, Dict[str, Any]]:
        """
        Prepare query embedding from request.
        
        Args:
            request: Search request
            
        Returns:
            Tuple of (query_embedding, query_info)
        """
        query_info = {
            "query_type": None,
            "source": None,
            "model_config": request.model_config.to_dict() if request.model_config else None
        }
        
        # Handle different query types
        if request.query_embedding:
            # Direct embedding provided
            query_info["query_type"] = "embedding"
            query_info["source"] = "direct"
            return request.query_embedding, query_info
        
        elif request.query_image:
            # Image file provided
            query_info["query_type"] = "image"
            query_info["source"] = str(request.query_image)
            
            # Load and process image
            image = self.image_processor.load_image(request.query_image)
            
            # Generate embedding
            if request.model_config:
                embedding = self.embedding_generator.generate_embedding(image, request.model_config)
            else:
                embedding = self.embedding_generator.generate_embedding(image)
            
            return embedding, query_info
        
        elif request.query_text:
            # Text query provided
            query_info["query_type"] = "text"
            query_info["source"] = request.query_text
            
            if not self.text_encoder:
                raise ValueError("Text encoder required for text queries")
            
            # Encode text to embedding
            embedding = self.text_encoder.encode_text(request.query_text)
            
            return embedding, query_info
        
        else:
            raise ValueError("At least one query source must be provided")
    
    def _create_search_results(
        self, 
        similar_images: List[Dict], 
        request: SearchSimilarImagesRequest
    ) -> SimilaritySearchResults:
        """
        Create search results from similar images.
        
        Args:
            similar_images: List of similar image results
            request: Original search request
            
        Returns:
            SimilaritySearchResults object
        """
        from ...domain.similarity_search import SimilarityResult
        
        # Convert results to SimilarityResult objects
        similarity_results = []
        for i, result in enumerate(similar_images):
            image = result["image"]
            embedding = result["embedding"]
            similarity_score = result["similarity_score"]
            
            similarity_result = SimilarityResult(
                image=image,
                embedding=embedding,
                similarity_score=similarity_score,
                rank=i + 1,
                confidence=similarity_score  # Use similarity as confidence
            )
            similarity_results.append(similarity_result)
        
        # Create search results object
        from ...domain.similarity_search import SimilaritySearchResults as ResultsClass
        from ...domain.similarity_search import SimilarityQuery as QueryClass
        
        # Create domain query
        query = QueryClass(
            query_embedding=similarity_results[0].embedding if similarity_results else None,
            limit=request.limit,
            threshold=request.threshold,
            model_name=request.model_config.model_name if request.model_config else None
        )
        
        return ResultsClass(
            query=query,
            results=tuple(similarity_results),
            total_found=len(similarity_results),
            search_time_ms=0.0  # Will be set by caller
        )
    
    def _generate_statistics(
        self, 
        results: SimilaritySearchResults, 
        request: SearchSimilarImagesRequest
    ) -> Dict[str, Any]:
        """
        Generate search statistics.
        
        Args:
            results: Search results
            request: Original search request
            
        Returns:
            Statistics dictionary
        """
        return {
            "total_results": len(results.results),
            "high_similarity_count": len(results.get_high_similarity_results()),
            "medium_similarity_count": len(results.get_medium_similarity_results()),
            "low_similarity_count": len(results.get_low_similarity_results()),
            "average_similarity": results.average_similarity,
            "highest_similarity": results.highest_similarity,
            "lowest_similarity": results.lowest_similarity,
            "threshold_used": request.threshold,
            "limit_used": request.limit,
            "query_filters": request.filters
        }


# Additional placeholder interfaces
class TextEncoder:
    """Placeholder for text encoding service."""
    
    def encode_text(self, text: str) -> VectorEmbedding:
        """Encode text to embedding."""
        raise NotImplementedError()