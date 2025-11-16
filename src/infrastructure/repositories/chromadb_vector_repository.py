"""
ChromaDB Vector Repository
==========================

ChromaDB implementation of VectorRepository interface.
"""

import os
import json
import hashlib
from uuid import uuid4
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
import chromadb
from chromadb.config import Settings

from ...domain.vector_storage.vector_embedding import VectorEmbedding
from ...domain.image_processing.image_processing import Image
from ...domain.vector_storage.vector_repository import VectorRepositoryBase


class ChromaDBVectorRepository(VectorRepositoryBase):
    """
    ChromaDB implementation of VectorRepository.
    
    Provides modern vector database capabilities with excellent performance
    for similarity search and scalability.
    
    Example:
        >>> repository = ChromaDBVectorRepository('image_embeddings', './chroma_db')
        >>> embedding = repository.save(vector_embedding, image)
        >>> similar = repository.find_similar(query_embedding, limit=5)
    """
    
    def __init__(
        self,
        collection_name: str = "image_embeddings",
        persist_directory: Optional[str] = None,
        model_name: str = "efficientnet"
    ):
        """
        Initialize ChromaDB vector repository.
        
        Args:
            collection_name: Name of the collection to store embeddings
            persist_directory: Directory to persist the database (None for in-memory)
            model_name: Name of the embedding model
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "./chroma_db"
        self.model_name = model_name
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",  # Use cosine similarity
                "description": f"Image embeddings collection for {model_name}",
                "model_name": model_name
            }
        )
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a unique hash for a file."""
        return hashlib.md5(file_path.encode('utf-8')).hexdigest()
    
    def save(self, embedding: VectorEmbedding, image: Image) -> str:
        """
        Save an embedding with associated image.
        
        Args:
            embedding: The vector embedding to save
            image: The associated image
            
        Returns:
            Unique identifier of the saved embedding
        """
        try:
            # Create unique ID
            embedding_id = str(uuid4())
            
            # Prepare metadata
            metadata = {
                "image_path": str(image.path),
                "image_hash": image.file_hash,
                "image_name": image.path.name,
                "model_name": embedding.model_name,
                "vector_size": len(embedding.vector),
                "image_width": image.dimensions[0],
                "image_height": image.dimensions[1],
                "file_size": image.metadata.get('file_size') if image.metadata else None,
                "format": image.format,
                "created_at": embedding.created_at.isoformat() if embedding.created_at else datetime.now().isoformat(),
                **embedding.metadata
            }
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding.vector.tolist()],
                metadatas=[metadata],
                ids=[embedding_id]
            )
            
            return embedding_id
            
        except Exception as e:
            # Image might already exist, try to update
            try:
                doc_id = self._get_file_hash(str(image.path))
                
                # Update existing entry
                self.collection.update(
                    embeddings=[embedding.vector.tolist()],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                
                return doc_id
            except Exception as update_error:
                raise RuntimeError(f"Failed to save embedding: {e}")
    
    def find_by_id(self, embedding_id: str) -> Optional[VectorEmbedding]:
        """
        Find embedding by ID.
        
        Args:
            embedding_id: Unique identifier
            
        Returns:
            Embedding if found, None otherwise
        """
        try:
            result = self.collection.get(
                ids=[embedding_id],
                include=['embeddings', 'metadatas']
            )
            
            if result['embeddings'] and len(result['embeddings']) > 0:
                vector = np.array(result['embeddings'][0])
                metadata = result['metadatas'][0] if result['metadatas'] else {}
                
                # Extract creation time from metadata
                created_at = None
                if 'created_at' in metadata:
                    try:
                        created_at = datetime.fromisoformat(metadata['created_at'])
                    except ValueError:
                        created_at = None
                
                return VectorEmbedding(
                    vector=vector,
                    model_name=metadata.get('model_name', self.model_name),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ['image_path', 'image_hash', 'image_name', 'model_name', 'vector_size', 'image_width', 'image_height', 'file_size', 'format', 'created_at']},
                    created_at=created_at
                )
            return None
            
        except Exception:
            return None
    
    def find_by_image_hash(self, image_hash: str) -> Optional[VectorEmbedding]:
        """
        Find embedding by image file hash.
        
        Args:
            image_hash: MD5 hash of the image file
            
        Returns:
            Embedding if found, None otherwise
        """
        try:
            # Search by image hash in metadata
            results = self.collection.get(
                where={"image_hash": image_hash},
                include=['embeddings', 'metadatas']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                # Get the first match
                vector = np.array(results['embeddings'][0])
                metadata = results['metadatas'][0]
                
                # Extract creation time from metadata
                created_at = None
                if 'created_at' in metadata:
                    try:
                        created_at = datetime.fromisoformat(metadata['created_at'])
                    except ValueError:
                        created_at = None
                
                return VectorEmbedding(
                    vector=vector,
                    model_name=metadata.get('model_name', self.model_name),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ['image_path', 'image_hash', 'image_name', 'model_name', 'vector_size', 'image_width', 'image_height', 'file_size', 'format', 'created_at']},
                    created_at=created_at
                )
            return None
            
        except Exception:
            return None
    
    def find_similar(
        self, 
        query_embedding: VectorEmbedding, 
        limit: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[VectorEmbedding, float]]:
        """
        Find similar embeddings using vector similarity search.
        
        Args:
            query_embedding: Query embedding to search with
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (embedding, similarity_score) tuples
        """
        try:
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding.vector.tolist()],
                n_results=limit,
                include=['metadatas', 'distances', 'ids']
            )
            
            embeddings = []
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= threshold:
                        metadata = results['metadatas'][0][i]
                        
                        # Extract creation time from metadata
                        created_at = None
                        if 'created_at' in metadata:
                            try:
                                created_at = datetime.fromisoformat(metadata['created_at'])
                            except ValueError:
                                created_at = None
                        
                        # Create vector embedding from stored data
                        embedding = VectorEmbedding(
                            vector=query_embedding.vector,  # We don't have the stored vector here
                            model_name=metadata.get('model_name', self.model_name),
                            metadata={k: v for k, v in metadata.items() 
                                     if k not in ['image_path', 'image_hash', 'image_name', 'model_name', 'vector_size', 'image_width', 'image_height', 'file_size', 'format', 'created_at']},
                            created_at=created_at
                        )
                        
                        embeddings.append((embedding, float(similarity)))
            
            return embeddings
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
    
    def find_duplicates(self, threshold: float = 0.95) -> List[List[VectorEmbedding]]:
        """
        Find duplicate or near-duplicate embeddings.
        
        Args:
            threshold: Similarity threshold for duplicates
            
        Returns:
            List of duplicate groups (list of embeddings)
        """
        try:
            # Get all embeddings for comparison
            all_data = self.collection.get(
                include=['embeddings', 'metadatas', 'ids']
            )
            
            if not all_data['ids']:
                return []
            
            # Convert to numpy array for efficient computation
            embeddings = np.array(all_data['embeddings'])
            metadatas = all_data['metadatas']
            ids = all_data['ids']
            
            # Find duplicates
            duplicate_groups = []
            processed = set()
            
            for i in range(len(embeddings)):
                if i in processed:
                    continue
                
                group = []
                for j in range(i + 1, len(embeddings)):
                    if j in processed:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    
                    if similarity >= threshold:
                        # Create embedding objects
                        metadata_j = metadatas[j]
                        created_at_j = None
                        if 'created_at' in metadata_j:
                            try:
                                created_at_j = datetime.fromisoformat(metadata_j['created_at'])
                            except ValueError:
                                created_at_j = None
                        
                        embedding = VectorEmbedding(
                            vector=embeddings[j],
                            model_name=metadata_j.get('model_name', self.model_name),
                            metadata={k: v for k, v in metadata_j.items() 
                                     if k not in ['image_path', 'image_hash', 'image_name', 'model_name', 'vector_size', 'image_width', 'image_height', 'file_size', 'format', 'created_at']},
                            created_at=created_at_j
                        )
                        
                        group.append(embedding)
                        processed.add(j)
                
                if group:
                    # Add the first embedding to the group
                    metadata_i = metadatas[i]
                    created_at_i = None
                    if 'created_at' in metadata_i:
                        try:
                            created_at_i = datetime.fromisoformat(metadata_i['created_at'])
                        except ValueError:
                            created_at_i = None
                    
                    embedding = VectorEmbedding(
                        vector=embeddings[i],
                        model_name=metadata_i.get('model_name', self.model_name),
                        metadata={k: v for k, v in metadata_i.items() 
                                 if k not in ['image_path', 'image_hash', 'image_name', 'model_name', 'vector_size', 'image_width', 'image_height', 'file_size', 'format', 'created_at']},
                        created_at=created_at_i
                    )
                    
                    group.insert(0, embedding)
                    duplicate_groups.append(group)
            
            return duplicate_groups
            
        except Exception as e:
            print(f"Error finding duplicates: {e}")
            return []
    
    def delete_by_id(self, embedding_id: str) -> bool:
        """
        Delete embedding by ID.
        
        Args:
            embedding_id: Unique identifier
            
        Returns:
            True if deleted, False if not found
        """
        try:
            self.collection.delete(ids=[embedding_id])
            return True
        except Exception:
            return False
    
    def get_all(self) -> List[VectorEmbedding]:
        """
        Get all embeddings.
        
        Returns:
            List of all embeddings
        """
        try:
            all_data = self.collection.get(
                include=['embeddings', 'metadatas']
            )
            
            embeddings = []
            for i, metadata in enumerate(all_data['metadatas']):
                # Extract creation time from metadata
                created_at = None
                if 'created_at' in metadata:
                    try:
                        created_at = datetime.fromisoformat(metadata['created_at'])
                    except ValueError:
                        created_at = None
                
                embedding = VectorEmbedding(
                    vector=np.array(all_data['embeddings'][i]),
                    model_name=metadata.get('model_name', self.model_name),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ['image_path', 'image_hash', 'image_name', 'model_name', 'vector_size', 'image_width', 'image_height', 'file_size', 'format', 'created_at']},
                    created_at=created_at
                )
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            print(f"Error getting all embeddings: {e}")
            return []
    
    def get_stats(self) -> dict:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample metadata to analyze
            sample_data = self.collection.get(limit=10, include=['metadatas'])
            
            stats = {
                'total_embeddings': count,
                'model_name': self.model_name,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'repository_type': 'chromadb'
            }
            
            if sample_data['metadatas']:
                # Calculate averages from sample
                widths = [m.get('image_width') for m in sample_data['metadatas'] if m.get('image_width')]
                heights = [m.get('image_height') for m in sample_data['metadatas'] if m.get('image_height')]
                sizes = [m.get('file_size') for m in sample_data['metadatas'] if m.get('file_size')]
                
                if widths:
                    stats['avg_width'] = np.mean(widths)
                if heights:
                    stats['avg_height'] = np.mean(heights)
                if sizes:
                    stats['avg_file_size'] = np.mean(sizes)
            
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                'total_embeddings': 0,
                'model_name': self.model_name,
                'collection_name': self.collection_name,
                'repository_type': 'chromadb'
            }
    
    def clear(self) -> None:
        """Clear all embeddings from repository."""
        try:
            self.collection.delete(where={})
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def close(self) -> None:
        """Close repository connections."""
        # ChromaDB doesn't require explicit connection cleanup
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()