"""
ChromaDB backend for image similarity toolkit.

This module provides ChromaDB integration for storing and searching image embeddings,
offering modern vector database capabilities with excellent Python integration.

Features:
- Embedded vector database (no server required)
- Advanced similarity search
- Metadata filtering
- Integration with modern embedding models
- Production-ready performance
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime

import numpy as np
import chromadb
from chromadb.config import Settings
from PIL import Image


class ChromaDBBackend:
    """
    ChromaDB backend for image similarity toolkit.
    
    Provides a modern alternative to SQLite with advanced vector search capabilities.
    Best for: prototyping, production with moderate scale (< 1M vectors).
    
    Example:
        >>> backend = ChromaDBBackend('image_embeddings')
        >>> backend.add_image('cat.jpg', embedding_vector, metadata={'width': 800, 'height': 600})
        >>> results = backend.find_similar('query.jpg', query_embedding, top_k=5)
        >>> for result in results:
        ...     print(f"{result['path']}: {result['similarity']:.3f}")
    """
    
    def __init__(
        self, 
        collection_name: str = "image_embeddings",
        persist_directory: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize ChromaDB backend.
        
        Args:
            collection_name: Name of the collection to store embeddings
            persist_directory: Directory to persist the database (None for in-memory)
            embedding_model: Model name for generating embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "./chroma_db"
        self.embedding_model = embedding_model
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",  # Use cosine similarity
                "description": f"Image embeddings collection using {embedding_model}"
            }
        )
        
        # Initialize embedding model
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model for generating image/text embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.embedding_model)
            print(f"✅ Loaded embedding model: {self.embedding_model}")
        except ImportError:
            print("❌ sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            print(f"❌ Failed to load model {self.embedding_model}: {e}")
            self.model = None
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a unique hash for a file."""
        return hashlib.md5(file_path.encode('utf-8')).hexdigest()
    
    def _generate_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """Generate embedding for an image using the loaded model."""
        if self.model is None:
            print("❌ No embedding model available")
            return None
        
        try:
            # For CLIP model, we can use encode_image method
            if hasattr(self.model, 'encode_image'):
                return self.model.encode_image(image)
            else:
                # Fallback for other models - convert to numpy array
                return self.model.encode(np.array(image))
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
    
    def add_image(
        self, 
        image_path: str, 
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add an image to the collection.
        
        Args:
            image_path: Path to the image file
            embedding: Pre-computed embedding vector (optional)
            metadata: Additional metadata (file_size, width, height, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return False
            
            # Load image
            image = Image.open(image_path)
            
            # Generate embedding if not provided
            if embedding is None:
                embedding = self._generate_embedding(image)
                if embedding is None:
                    return False
            
            # Prepare metadata
            metadata = metadata or {}
            metadata.update({
                "path": image_path,
                "filename": os.path.basename(image_path),
                "format": image.format,
                "width": image.size[0] if image.size else None,
                "height": image.size[1] if image.size else None,
                "file_size": os.path.getsize(image_path) if os.path.exists(image_path) else None,
                "added_at": datetime.now().isoformat(),
                "model": self.embedding_model
            })
            
            # Create unique ID
            doc_id = self._get_file_hash(image_path)
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            print(f"✅ Added image: {os.path.basename(image_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding image {image_path}: {e}")
            return False
    
    def add_images_from_directory(
        self, 
        directory_path: str,
        max_images: Optional[int] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> int:
        """
        Add multiple images from a directory.
        
        Args:
            directory_path: Directory containing images
            max_images: Maximum number of images to add (None for all)
            recursive: Whether to search recursively
            extensions: List of image extensions to include
            
        Returns:
            Number of successfully added images
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        
        # Find image files
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        image_files = []
        for ext in extensions:
            image_files.extend(Path(directory_path).rglob(f"*{ext}" if recursive else f"{ext}"))
        
        # Apply limit if specified
        if max_images:
            image_files = image_files[:max_images]
        
        if not image_files:
            print(f"❌ No images found in directory: {directory_path}")
            return 0
        
        print(f"📁 Found {len(image_files)} images to process")
        
        # Process images
        successful = 0
        for i, image_path in enumerate(image_files):
            if self.add_image(str(image_path)):
                successful += 1
            
            if max_images and i >= max_images - 1:
                break
        
        print(f"✅ Successfully added {successful}/{len(image_files)} images")
        return successful
    
    def get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific image from the collection.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Embedding vector or None if not found
        """
        try:
            doc_id = self._get_file_hash(image_path)
            result = self.collection.get(
                ids=[doc_id],
                include=['embeddings']
            )
            
            if result['embeddings']:
                return np.array(result['embeddings'][0])
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving embedding: {e}")
            return None
    
    def find_similar(
        self,
        query_embedding: Optional[np.ndarray] = None,
        query_image_path: Optional[str] = None,
        query_text: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Find similar images using vector similarity search.
        
        Args:
            query_embedding: Pre-computed query embedding
            query_image_path: Path to query image (generates embedding automatically)
            query_text: Text query (converted to embedding)
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filter_metadata: Metadata filters to apply
            
        Returns:
            List of similar images with metadata and similarity scores
        """
        try:
            # Generate query embedding
            if query_embedding is None:
                if query_image_path and os.path.exists(query_image_path):
                    image = Image.open(query_image_path)
                    query_embedding = self._generate_embedding(image)
                elif query_text and self.model:
                    query_embedding = self.model.encode(query_text)
                else:
                    print("❌ No query provided")
                    return []
            
            if query_embedding is None:
                return []
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['metadatas', 'distances', 'ids'],
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= threshold:
                        formatted_results.append({
                            'id': results['ids'][0][i],
                            'path': results['metadatas'][0][i].get('path', ''),
                            'filename': results['metadatas'][0][i].get('filename', ''),
                            'similarity': similarity,
                            'distance': distance,
                            'metadata': results['metadatas'][0][i]
                        })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error during similarity search: {e}")
            return []
    
    def find_duplicates(
        self,
        similarity_threshold: float = 0.95,
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Find duplicate or near-duplicate images in the collection.
        
        Args:
            similarity_threshold: Minimum similarity to consider as duplicate
            batch_size: Batch size for processing large collections
            
        Returns:
            List of duplicate pairs with similarity scores
        """
        try:
            # Get all embeddings
            all_data = self.collection.get(
                include=['embeddings', 'metadatas', 'ids']
            )
            
            if not all_data['ids']:
                print("❌ No images in collection")
                return []
            
            # Convert to numpy array for efficient computation
            embeddings = np.array(all_data['embeddings'])
            metadatas = all_data['metadatas']
            ids = all_data['ids']
            
            # Calculate pairwise similarities
            duplicates = []
            n_images = len(ids)
            
            for i in range(0, n_images, batch_size):
                batch_end = min(i + batch_size, n_images)
                
                # Compare current batch with all others
                for j in range(i, batch_end):
                    for k in range(j + 1, n_images):
                        # Calculate cosine similarity
                        similarity = np.dot(embeddings[j], embeddings[k]) / (
                            np.linalg.norm(embeddings[j]) * np.linalg.norm(embeddings[k])
                        )
                        
                        if similarity >= similarity_threshold:
                            duplicates.append({
                                'image1_id': ids[j],
                                'image1_path': metadatas[j].get('path', ''),
                                'image1_filename': metadatas[j].get('filename', ''),
                                'image2_id': ids[k],
                                'image2_path': metadatas[k].get('path', ''),
                                'image2_filename': metadatas[k].get('filename', ''),
                                'similarity': float(similarity)
                            })
            
            # Sort by similarity (highest first)
            duplicates.sort(key=lambda x: x['similarity'], reverse=True)
            
            print(f"🔍 Found {len(duplicates)} duplicate pairs")
            return duplicates
            
        except Exception as e:
            print(f"❌ Error finding duplicates: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample metadata to analyze
            sample_data = self.collection.get(limit=10, include=['metadatas'])
            
            stats = {
                'total_images': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'embedding_model': self.embedding_model,
                'last_updated': datetime.now().isoformat()
            }
            
            if sample_data['metadatas']:
                # Calculate averages from sample
                widths = [m.get('width') for m in sample_data['metadatas'] if m.get('width')]
                heights = [m.get('height') for m in sample_data['metadatas'] if m.get('height')]
                sizes = [m.get('file_size') for m in sample_data['metadatas'] if m.get('file_size')]
                
                if widths:
                    stats['avg_width'] = np.mean(widths)
                if heights:
                    stats['avg_height'] = np.mean(heights)
                if sizes:
                    stats['avg_file_size'] = np.mean(sizes)
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {}
    
    def remove_image(self, image_path: str) -> bool:
        """
        Remove an image from the collection.
        
        Args:
            image_path: Path to the image to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc_id = self._get_file_hash(image_path)
            self.collection.delete(ids=[doc_id])
            print(f"✅ Removed image: {os.path.basename(image_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Error removing image {image_path}: {e}")
            return False
    
    def clear_collection(self):
        """Clear all data from the collection."""
        try:
            self.collection.delete(where={})
            print("✅ Collection cleared")
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
    
    def export_collection(self, output_file: str) -> bool:
        """
        Export collection data to JSON file.
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all data
            all_data = self.collection.get(
                include=['embeddings', 'metadatas', 'ids']
            )
            
            # Prepare export data
            export_data = {
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model,
                'exported_at': datetime.now().isoformat(),
                'data': [
                    {
                        'id': id_val,
                        'embedding': embedding,
                        'metadata': metadata
                    }
                    for id_val, embedding, metadata in zip(
                        all_data['ids'],
                        all_data['embeddings'],
                        all_data['metadatas']
                    )
                ]
            }
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"✅ Collection exported to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting collection: {e}")
            return False
    
    def search_by_metadata(
        self,
        filter_metadata: Dict,
        limit: int = 100
    ) -> List[Dict]:
        """
        Search images by metadata filters.
        
        Args:
            filter_metadata: Dictionary of metadata filters
            limit: Maximum number of results
            
        Returns:
            List of matching images with metadata
        """
        try:
            results = self.collection.get(
                where=filter_metadata,
                limit=limit,
                include=['metadatas', 'ids']
            )
            
            formatted_results = []
            if results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    formatted_results.append({
                        'id': doc_id,
                        'metadata': results['metadatas'][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching by metadata: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # ChromaDB doesn't require explicit connection cleanup
        pass