"""
SQLite Vector Repository
========================

SQLite implementation of VectorRepository interface.
"""

import sqlite3
import numpy as np
import json
import os
from pathlib import Path
from uuid import uuid4
from typing import List, Optional, Tuple
from datetime import datetime

from ...domain.vector_storage.vector_embedding import VectorEmbedding
from ...domain.image_processing.image_processing import Image
from ...domain.vector_storage.vector_repository import VectorRepositoryBase


class SQLiteVectorRepository(VectorRepositoryBase):
    """
    SQLite implementation of VectorRepository.
    
    Provides persistent storage and search capabilities for vector embeddings
    using SQLite database for simple deployment and management.
    
    Example:
        >>> repository = SQLiteVectorRepository('embeddings.db', 'efficientnet')
        >>> embedding = repository.save(vector_embedding, image)
        >>> similar = repository.find_similar(query_embedding, limit=5)
    """
    
    def __init__(self, db_path: str = 'embeddings.db', model_name: str = 'efficientnet'):
        """
        Initialize SQLite vector repository.
        
        Args:
            db_path: Path to SQLite database file
            model_name: Name of the embedding model
        """
        self.db_path = db_path
        self.model_name = model_name
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self) -> None:
        """Establish connection to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
    
    def _create_tables(self) -> None:
        """Create necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                image_path TEXT UNIQUE NOT NULL,
                image_hash TEXT NOT NULL,
                image_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                embedding_vector TEXT NOT NULL,
                embedding_size INTEGER NOT NULL,
                file_size INTEGER,
                image_width INTEGER,
                image_height INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indexes for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_image_path 
            ON embeddings(image_path)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_image_hash 
            ON embeddings(image_hash)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_model_name 
            ON embeddings(model_name)
        ''')
        
        self.conn.commit()
    
    def save(self, embedding: VectorEmbedding, image: Image) -> str:
        """
        Save an embedding with associated image.
        
        Args:
            embedding: The vector embedding to save
            image: The associated image
            
        Returns:
            Unique identifier of the saved embedding
        """
        cursor = self.conn.cursor()
        
        # Generate unique ID if not provided
        embedding_id = str(uuid4())
        
        # Convert embedding to JSON for storage
        embedding_json = json.dumps(embedding.vector.tolist())
        
        # Prepare metadata
        metadata = {
            'model_name': embedding.model_name,
            'vector_size': len(embedding.vector),
            **embedding.metadata
        }
        
        try:
            cursor.execute('''
                INSERT INTO embeddings 
                (id, image_path, image_hash, image_name, model_name, embedding_vector,
                 embedding_size, file_size, image_width, image_height, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                embedding_id,
                str(image.path),
                image.file_hash,
                image.path.name,
                self.model_name,
                embedding_json,
                len(embedding.vector),
                image.metadata.get('file_size') if image.metadata else None,
                image.dimensions[0],
                image.dimensions[1],
                json.dumps(metadata)
            ))
            
            self.conn.commit()
            return embedding_id
            
        except sqlite3.IntegrityError:
            # Image already exists, update it
            cursor.execute('''
                UPDATE embeddings 
                SET embedding_vector = ?,
                    embedding_size = ?,
                    metadata = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE image_path = ? AND model_name = ?
            ''', (
                embedding_json,
                len(embedding.vector),
                json.dumps(metadata),
                str(image.path),
                self.model_name
            ))
            
            self.conn.commit()
            
            cursor.execute(
                'SELECT id FROM embeddings WHERE image_path = ? AND model_name = ?',
                (str(image.path), self.model_name)
            )
            row = cursor.fetchone()
            return row['id'] if row else embedding_id
    
    def find_by_id(self, embedding_id: str) -> Optional[VectorEmbedding]:
        """
        Find embedding by ID.
        
        Args:
            embedding_id: Unique identifier
            
        Returns:
            Embedding if found, None otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT embedding_vector, model_name, metadata, created_at
            FROM embeddings 
            WHERE id = ? AND model_name = ?
        ''', (embedding_id, self.model_name))
        
        row = cursor.fetchone()
        if row:
            vector = np.array(json.loads(row['embedding_vector']))
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            
            return VectorEmbedding(
                vector=vector,
                model_name=row['model_name'],
                metadata=metadata,
                created_at=row['created_at']
            )
        return None
    
    def find_by_image_hash(self, image_hash: str) -> Optional[VectorEmbedding]:
        """
        Find embedding by image file hash.
        
        Args:
            image_hash: MD5 hash of the image file
            
        Returns:
            Embedding if found, None otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, embedding_vector, model_name, metadata, created_at
            FROM embeddings 
            WHERE image_hash = ? AND model_name = ?
        ''', (image_hash, self.model_name))
        
        row = cursor.fetchone()
        if row:
            vector = np.array(json.loads(row['embedding_vector']))
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            
            return VectorEmbedding(
                vector=vector,
                model_name=row['model_name'],
                metadata=metadata,
                created_at=row['created_at']
            )
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
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, embedding_vector, model_name, metadata, created_at
            FROM embeddings 
            WHERE model_name = ?
        ''', (self.model_name,))
        
        results = []
        query_vector = query_embedding.vector
        
        for row in cursor.fetchall():
            vector = np.array(json.loads(row['embedding_vector']))
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            
            if similarity >= threshold:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                
                embedding = VectorEmbedding(
                    vector=vector,
                    model_name=row['model_name'],
                    metadata=metadata,
                    created_at=row['created_at']
                )
                
                results.append((embedding, float(similarity)))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def find_duplicates(self, threshold: float = 0.95) -> List[List[VectorEmbedding]]:
        """
        Find duplicate or near-duplicate embeddings.
        
        Args:
            threshold: Similarity threshold for duplicates
            
        Returns:
            List of duplicate groups (list of embeddings)
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, embedding_vector, model_name, metadata, created_at
            FROM embeddings 
            WHERE model_name = ?
        ''', (self.model_name,))
        
        rows = cursor.fetchall()
        embeddings = []
        
        for row in rows:
            vector = np.array(json.loads(row['embedding_vector']))
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            
            embedding = VectorEmbedding(
                vector=vector,
                model_name=row['model_name'],
                metadata=metadata,
                created_at=row['created_at']
            )
            embeddings.append(embedding)
        
        # Find duplicates
        duplicate_groups = []
        processed = set()
        
        for i, emb1 in enumerate(embeddings):
            if i in processed:
                continue
                
            group = [emb1]
            processed.add(i)
            
            for j, emb2 in enumerate(embeddings[i+1:], start=i+1):
                if j in processed:
                    continue
                    
                similarity = np.dot(emb1.vector, emb2.vector) / (
                    np.linalg.norm(emb1.vector) * np.linalg.norm(emb2.vector)
                )
                
                if similarity >= threshold:
                    group.append(emb2)
                    processed.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    def delete_by_id(self, embedding_id: str) -> bool:
        """
        Delete embedding by ID.
        
        Args:
            embedding_id: Unique identifier
            
        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            DELETE FROM embeddings 
            WHERE id = ? AND model_name = ?
        ''', (embedding_id, self.model_name))
        
        self.conn.commit()
        return cursor.rowcount > 0
    
    def get_all(self) -> List[VectorEmbedding]:
        """
        Get all embeddings.
        
        Returns:
            List of all embeddings
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT embedding_vector, model_name, metadata, created_at
            FROM embeddings 
            WHERE model_name = ?
        ''', (self.model_name,))
        
        embeddings = []
        for row in cursor.fetchall():
            vector = np.array(json.loads(row['embedding_vector']))
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            
            embedding = VectorEmbedding(
                vector=vector,
                model_name=row['model_name'],
                metadata=metadata,
                created_at=row['created_at']
            )
            embeddings.append(embedding)
        
        return embeddings
    
    def get_stats(self) -> dict:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with statistics
        """
        cursor = self.conn.cursor()
        
        cursor.execute(
            'SELECT COUNT(*) as count FROM embeddings WHERE model_name = ?',
            (self.model_name,)
        )
        total_count = cursor.fetchone()['count']
        
        cursor.execute('''
            SELECT AVG(file_size) as avg_size, 
                   AVG(image_width) as avg_width,
                   AVG(image_height) as avg_height
            FROM embeddings WHERE model_name = ?
        ''', (self.model_name,))
        
        stats_row = cursor.fetchone()
        
        return {
            'total_embeddings': total_count,
            'model_name': self.model_name,
            'database_path': self.db_path,
            'avg_file_size': stats_row['avg_size'],
            'avg_width': stats_row['avg_width'],
            'avg_height': stats_row['avg_height'],
            'repository_type': 'sqlite'
        }
    
    def clear(self) -> None:
        """Clear all embeddings from repository."""
        cursor = self.conn.cursor()
        cursor.execute('''
            DELETE FROM embeddings 
            WHERE model_name = ?
        ''', (self.model_name,))
        
        self.conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()