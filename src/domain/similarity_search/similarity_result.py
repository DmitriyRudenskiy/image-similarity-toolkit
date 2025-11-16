"""
Similarity Result Value Object
=============================

Immutable result of similarity search operations.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from ..image_processing import Image
from ..vector_storage.vector_embedding import VectorEmbedding


@dataclass(frozen=True)
class SimilarityResult:
    """
    Value Object representing a similarity search result.
    
    Immutable by design - once created, cannot be modified.
    
    Attributes:
        image: The matched image
        embedding: The embedding vector of the matched image
        similarity_score: Similarity score (higher = more similar)
        distance_score: Distance score (lower = more similar)
        search_mode: Mode used for this result
        query_info: Information about the query that produced this result
        rank: Position in search results (1-based)
        confidence: Confidence score for the match
    """
    image: Image
    embedding: VectorEmbedding
    similarity_score: float
    distance_score: Optional[float] = None
    search_mode: str = "cosine"
    query_info: Optional[dict] = None
    rank: int = 1
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate result after initialization."""
        if self.rank < 1:
            raise ValueError("Rank must be positive (1-based)")
        
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError("Similarity score must be between 0.0 and 1.0")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if self.distance_score is not None and self.distance_score < 0:
            raise ValueError("Distance score must be non-negative")
    
    @property
    def is_high_similarity(self) -> bool:
        """Check if similarity score indicates high similarity."""
        return self.similarity_score >= 0.8
    
    @property
    def is_medium_similarity(self) -> bool:
        """Check if similarity score indicates medium similarity."""
        return 0.5 <= self.similarity_score < 0.8
    
    @property
    def is_low_similarity(self) -> bool:
        """Check if similarity score indicates low similarity."""
        return self.similarity_score < 0.5
    
    @property
    def similarity_label(self) -> str:
        """Get human-readable similarity label."""
        if self.is_high_similarity:
            return "High"
        elif self.is_medium_similarity:
            return "Medium"
        else:
            return "Low"
    
    @property
    def file_path(self) -> str:
        """Get string representation of image path."""
        return str(self.image.path)
    
    @property
    def image_hash(self) -> str:
        """Get image file hash."""
        return self.image.file_hash
    
    @property
    def model_name(self) -> str:
        """Get embedding model name."""
        return self.embedding.model_name
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding.dimension
    
    def to_dict(self) -> dict:
        """Convert result to dictionary representation."""
        return {
            "rank": self.rank,
            "similarity_score": self.similarity_score,
            "distance_score": self.distance_score,
            "search_mode": self.search_mode,
            "confidence": self.confidence,
            "similarity_label": self.similarity_label,
            "image_info": {
                "path": str(self.image.path),
                "filename": self.image.filename,
                "dimensions": self.image.dimensions,
                "format": self.image.format,
                "hash": self.image.file_hash,
                "size_mb": self.image.size_mb,
                "aspect_ratio": self.image.aspect_ratio
            },
            "embedding_info": {
                "model_name": self.embedding.model_name,
                "dimension": self.embedding.dimension,
                "created_at": self.embedding.created_at.isoformat(),
                "norm": self.embedding.norm
            },
            "query_info": self.query_info
        }
    
    def __str__(self) -> str:
        return (
            f"SimilarityResult(rank={self.rank}, "
            f"score={self.similarity_score:.4f}, "
            f"label={self.similarity_label}, "
            f"image={self.image.filename})"
        )
    
    def __repr__(self) -> str:
        return (
            f"SimilarityResult("
            f"image={self.image.path}, "
            f"similarity={self.similarity_score:.4f}, "
            f"rank={self.rank}, "
            f"confidence={self.confidence:.4f}"
            f")"
        )


@dataclass(frozen=True)
class SimilaritySearchResults:
    """
    Value Object representing complete similarity search results.
    
    Immutable by design - once created, cannot be modified.
    
    Attributes:
        query: Original query that produced these results
        results: List of similarity results
        total_found: Total number of matches found
        search_time_ms: Search execution time in milliseconds
        average_similarity: Average similarity score of results
        highest_similarity: Highest similarity score
        lowest_similarity: Lowest similarity score
        model_info: Information about the model used
        search_metadata: Additional search metadata
    """
    query: 'SimilarityQuery'
    results: tuple[SimilarityResult, ...]
    total_found: int
    search_time_ms: float
    average_similarity: float = 0.0
    highest_similarity: float = 0.0
    lowest_similarity: float = 0.0
    model_info: Optional[dict] = None
    search_metadata: Optional[dict] = None
    
    def __post_init__(self):
        """Validate results after initialization."""
        if self.total_found < 0:
            raise ValueError("Total found must be non-negative")
        
        if self.search_time_ms < 0:
            raise ValueError("Search time must be non-negative")
        
        if self.results:
            similarities = [r.similarity_score for r in self.results]
            self._object.__setattr__(self, 'average_similarity', sum(similarities) / len(similarities))
            self._object.__setattr__(self, 'highest_similarity', max(similarities))
            self._object.__setattr__(self, 'lowest_similarity', min(similarities))
    
    @property
    def has_results(self) -> bool:
        """Check if there are any results."""
        return len(self.results) > 0
    
    @property
    def is_empty(self) -> bool:
        """Check if results are empty."""
        return len(self.results) == 0
    
    @property
    def result_count(self) -> int:
        """Get number of results returned."""
        return len(self.results)
    
    @property
    def top_result(self) -> Optional[SimilarityResult]:
        """Get the top result (highest similarity)."""
        return self.results[0] if self.results else None
    
    def get_results_above_threshold(self, threshold: float) -> tuple[SimilarityResult, ...]:
        """Get results above a similarity threshold."""
        return tuple(r for r in self.results if r.similarity_score >= threshold)
    
    def get_results_by_similarity_label(self, label: str) -> tuple[SimilarityResult, ...]:
        """Get results by similarity label."""
        return tuple(r for r in self.results if r.similarity_label == label)
    
    def get_high_similarity_results(self) -> tuple[SimilarityResult, ...]:
        """Get results with high similarity."""
        return self.get_results_by_similarity_label("High")
    
    def get_medium_similarity_results(self) -> tuple[SimilarityResult, ...]:
        """Get results with medium similarity."""
        return self.get_results_by_similarity_label("Medium")
    
    def get_low_similarity_results(self) -> tuple[SimilarityResult, ...]:
        """Get results with low similarity."""
        return self.get_results_by_similarity_label("Low")
    
    def get_top_n(self, n: int) -> tuple[SimilarityResult, ...]:
        """Get top N results."""
        return self.results[:n]
    
    def to_dict(self) -> dict:
        """Convert results to dictionary representation."""
        return {
            "query": self.query.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "total_found": self.total_found,
            "result_count": self.result_count,
            "search_time_ms": self.search_time_ms,
            "average_similarity": self.average_similarity,
            "highest_similarity": self.highest_similarity,
            "lowest_similarity": self.lowest_similarity,
            "model_info": self.model_info,
            "search_metadata": self.search_metadata,
            "statistics": {
                "high_similarity_count": len(self.get_high_similarity_results()),
                "medium_similarity_count": len(self.get_medium_similarity_results()),
                "low_similarity_count": len(self.get_low_similarity_results()),
                "above_threshold_count": len(self.get_results_above_threshold(self.query.threshold))
            }
        }
    
    def __str__(self) -> str:
        return (
            f"SimilaritySearchResults("
            f"query={self.query.search_type.value}, "
            f"results={self.result_count}, "
            f"avg_similarity={self.average_similarity:.4f}, "
            f"time={self.search_time_ms:.2f}ms"
            f")"
        )
    
    def __repr__(self) -> str:
        return (
            f"SimilaritySearchResults("
            f"query_type={self.query.search_type}, "
            f"result_count={self.result_count}, "
            f"total_found={self.total_found}, "
            f"search_time={self.search_time_ms:.2f}ms"
            f")"
        )
    
    def __len__(self) -> int:
        """Get number of results."""
        return len(self.results)
    
    def __iter__(self):
        """Iterate over results."""
        return iter(self.results)
    
    def __getitem__(self, index: int) -> SimilarityResult:
        """Get result by index."""
        return self.results[index]