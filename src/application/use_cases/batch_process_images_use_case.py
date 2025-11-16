"""
Batch Process Images Use Case
============================

Use case for batch processing multiple images.
"""

import time
from typing import Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...domain.image_processing import Image
from ...domain.vector_storage import VectorEmbedding
from ...domain.configuration import ModelConfiguration
from .add_image_use_case import AddImageRequest, AddImageResponse, AddImageUseCase


class BatchProcessImagesRequest:
    """
    Request object for batch processing images.
    
    Attributes:
        image_paths: List of paths to images
        model_config: Configuration for the model to use
        max_workers: Maximum number of worker threads
        fail_fast: Whether to stop on first failure
        skip_duplicates: Whether to skip duplicate images
        duplicate_threshold: Threshold for considering images duplicates
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        model_config: ModelConfiguration,
        max_workers: int = 4,
        fail_fast: bool = False,
        skip_duplicates: bool = True,
        duplicate_threshold: float = 0.95
    ):
        self.image_paths = image_paths
        self.model_config = model_config
        self.max_workers = max_workers
        self.fail_fast = fail_fast
        self.skip_duplicates = skip_duplicates
        self.duplicate_threshold = duplicate_threshold


class BatchProcessImagesResponse:
    """
    Response object for batch processing images.
    
    Attributes:
        successful_count: Number of successfully processed images
        failed_count: Number of failed images
        total_count: Total number of images
        results: List of processing results
        errors: List of processing errors
        processing_time: Total processing time
        statistics: Processing statistics
    """
    
    def __init__(
        self,
        successful_count: int,
        failed_count: int,
        total_count: int,
        results: List[AddImageResponse],
        errors: List[Dict[str, Any]],
        processing_time: float,
        statistics: Dict[str, Any]
    ):
        self.successful_count = successful_count
        self.failed_count = failed_count
        self.total_count = total_count
        self.results = results
        self.errors = errors
        self.processing_time = processing_time
        self.statistics = statistics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "total_count": self.total_count,
            "success_rate": self.successful_count / self.total_count * 100 if self.total_count > 0 else 0,
            "processing_time": self.processing_time,
            "average_time_per_image": self.processing_time / self.total_count if self.total_count > 0 else 0,
            "statistics": self.statistics,
            "errors": self.errors,
            "results": [r.to_dict() for r in self.results]
        }


class BatchProcessImagesUseCase:
    """
    Use case for batch processing multiple images.
    
    Coordinates parallel processing of multiple images.
    """
    
    def __init__(
        self,
        add_image_use_case: AddImageUseCase,
        duplicate_detector: "DuplicateDetector"
    ):
        """
        Initialize the use case.
        
        Args:
            add_image_use_case: Single image addition use case
            duplicate_detector: Service for duplicate detection
        """
        self.add_image_use_case = add_image_use_case
        self.duplicate_detector = duplicate_detector
    
    def execute(self, request: BatchProcessImagesRequest) -> BatchProcessImagesResponse:
        """
        Execute the batch processing use case.
        
        Args:
            request: Batch processing request
            
        Returns:
            Batch processing response
        """
        start_time = time.time()
        
        # Validate inputs
        if not request.image_paths:
            raise ValueError("No image paths provided")
        
        if request.max_workers <= 0:
            raise ValueError("Max workers must be positive")
        
        # Step 1: Validate all image paths
        valid_paths = []
        invalid_paths = []
        
        for path in request.image_paths:
            if not path.exists():
                invalid_paths.append({"path": str(path), "error": "File not found"})
            elif not path.is_file():
                invalid_paths.append({"path": str(path), "error": "Not a file"})
            else:
                valid_paths.append(path)
        
        if invalid_paths:
            # Report invalid paths as errors
            errors = invalid_paths
        else:
            errors = []
        
        # Step 2: Check for duplicates (if requested)
        if request.skip_duplicates and valid_paths:
            valid_paths = self._filter_duplicates(valid_paths, request.duplicate_threshold)
        
        # Step 3: Process images in parallel
        results = []
        processing_errors = []
        
        if valid_paths:
            if request.max_workers == 1:
                # Sequential processing
                for path in valid_paths:
                    try:
                        result = self._process_single_image(path, request)
                        results.append(result)
                    except Exception as e:
                        if request.fail_fast:
                            raise
                        processing_errors.append({
                            "path": str(path),
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
            else:
                # Parallel processing
                results, processing_errors = self._process_images_parallel(
                    valid_paths, request
                )
        
        # Step 4: Calculate statistics
        successful_count = len(results)
        failed_count = len(processing_errors) + len(invalid_paths)
        total_count = len(request.image_paths)
        
        processing_time = time.time() - start_time
        
        statistics = self._generate_statistics(results, processing_errors, request)
        
        return BatchProcessImagesResponse(
            successful_count=successful_count,
            failed_count=failed_count,
            total_count=total_count,
            results=results,
            errors=errors + processing_errors,
            processing_time=processing_time,
            statistics=statistics
        )
    
    def _filter_duplicates(self, image_paths: List[Path], threshold: float) -> List[Path]:
        """
        Filter out duplicate images.
        
        Args:
            image_paths: List of image paths
            threshold: Duplicate detection threshold
            
        Returns:
            List of unique image paths
        """
        # This would require loading images and comparing them
        # For now, return all paths (placeholder implementation)
        return image_paths
    
    def _process_single_image(
        self, 
        image_path: Path, 
        request: BatchProcessImagesRequest
    ) -> AddImageResponse:
        """
        Process a single image.
        
        Args:
            image_path: Path to image
            request: Batch request
            
        Returns:
            Add image response
        """
        add_request = AddImageRequest(
            image_path=image_path,
            model_config=request.model_config,
            force_add=not request.skip_duplicates,
            similarity_threshold=request.duplicate_threshold
        )
        
        return self.add_image_use_case.execute(add_request)
    
    def _process_images_parallel(
        self,
        image_paths: List[Path],
        request: BatchProcessImagesRequest
    ) -> tuple[List[AddImageResponse], List[Dict[str, Any]]]:
        """
        Process images in parallel.
        
        Args:
            image_paths: List of image paths
            request: Batch request
            
        Returns:
            Tuple of (results, errors)
        """
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=request.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_image, path, request): path
                for path in image_paths
            }
            
            # Collect results
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    if request.fail_fast:
                        # Cancel remaining futures and re-raise
                        for f in future_to_path:
                            f.cancel()
                        raise
                    
                    errors.append({
                        "path": str(path),
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
        
        return results, errors
    
    def _generate_statistics(
        self,
        results: List[AddImageResponse],
        errors: List[Dict[str, Any]],
        request: BatchProcessImagesRequest
    ) -> Dict[str, Any]:
        """
        Generate processing statistics.
        
        Args:
            results: List of successful results
            errors: List of errors
            request: Original request
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {
                "processing_mode": "parallel" if request.max_workers > 1 else "sequential",
                "max_workers": request.max_workers,
                "total_processing_time": 0,
                "average_processing_time": 0,
                "throughput_images_per_second": 0
            }
        
        processing_times = [r.processing_time for r in results]
        
        return {
            "processing_mode": "parallel" if request.max_workers > 1 else "sequential",
            "max_workers": request.max_workers,
            "total_processing_time": sum(processing_times),
            "average_processing_time": sum(processing_times) / len(processing_times),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "throughput_images_per_second": len(results) / sum(processing_times) if sum(processing_times) > 0 else 0,
            "images_with_warnings": len([r for r in results if r.warnings]),
            "total_warnings": sum(len(r.warnings) for r in results)
        }