"""
Database module for storing and searching image embeddings.
"""

import sqlite3
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingDatabase:
    """
    Database for storing and searching image embeddings.
    
    Uses SQLite for storage and provides efficient similarity search.
    
    Attributes:
        db_path (str): Path to the SQLite database file
        model_name (str): Name of the model used for embeddings
        
    Example:
        >>> db = EmbeddingDatabase('embeddings.db', model_name='efficientnet')
        >>> db.add_image('cat.jpg', embedding_vector)
        >>> similar = db.find_similar('query.jpg', query_embedding, top_k=5)
    """
    
    def __init__(self, db_path: str = 'embeddings.db', model_name: str = 'efficientnet'):
        """
        Initialize the embedding database.
        
        Args:
            db_path: Path to the SQLite database file
            model_name: Name of the model used for embeddings
        """
        self.db_path = db_path
        self.model_name = model_name
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish connection to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE NOT NULL,
                image_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                embedding_vector TEXT NOT NULL,
                embedding_size INTEGER NOT NULL,
                file_size INTEGER,
                image_width INTEGER,
                image_height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_image_path 
            ON embeddings(image_path)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_model_name 
            ON embeddings(model_name)
        ''')
        
        # Duplicates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS duplicates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image1_id INTEGER NOT NULL,
                image2_id INTEGER NOT NULL,
                similarity REAL NOT NULL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image1_id) REFERENCES embeddings(id),
                FOREIGN KEY (image2_id) REFERENCES embeddings(id),
                UNIQUE(image1_id, image2_id)
            )
        ''')
        
        self.conn.commit()
    
    def add_image(
        self,
        image_path: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add an image embedding to the database.
        
        Args:
            image_path: Path to the image file
            embedding: Embedding vector as numpy array
            metadata: Optional metadata (file_size, width, height)
            
        Returns:
            Database ID of the inserted/updated record
            
        Example:
            >>> db.add_image('cat.jpg', embedding_vector, 
            ...              {'file_size': 12345, 'width': 800, 'height': 600})
        """
        cursor = self.conn.cursor()
        
        image_name = os.path.basename(image_path)
        embedding_json = json.dumps(embedding.tolist())
        embedding_size = len(embedding)
        
        metadata = metadata or {}
        file_size = metadata.get('file_size')
        width = metadata.get('width')
        height = metadata.get('height')
        
        try:
            cursor.execute('''
                INSERT INTO embeddings 
                (image_path, image_name, model_name, embedding_vector, 
                 embedding_size, file_size, image_width, image_height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (image_path, image_name, self.model_name, embedding_json,
                  embedding_size, file_size, width, height))
            
            self.conn.commit()
            return cursor.lastrowid
            
        except sqlite3.IntegrityError:
            # Image already exists, update it
            cursor.execute('''
                UPDATE embeddings 
                SET embedding_vector = ?,
                    embedding_size = ?,
                    file_size = ?,
                    image_width = ?,
                    image_height = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE image_path = ? AND model_name = ?
            ''', (embedding_json, embedding_size, file_size, 
                  width, height, image_path, self.model_name))
            
            self.conn.commit()
            
            cursor.execute(
                'SELECT id FROM embeddings WHERE image_path = ? AND model_name = ?',
                (image_path, self.model_name)
            )
            return cursor.fetchone()[0]
    
    def get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding for an image from the database.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Embedding vector as numpy array, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT embedding_vector FROM embeddings 
            WHERE image_path = ? AND model_name = ?
        ''', (image_path, self.model_name))
        
        row = cursor.fetchone()
        if row:
            return np.array(json.loads(row['embedding_vector']))
        return None
    
    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """
        Retrieve all embeddings from the database.
        
        Returns:
            List of tuples (id, image_path, embedding)
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, image_path, embedding_vector 
            FROM embeddings 
            WHERE model_name = ?
        ''', (self.model_name,))
        
        results = []
        for row in cursor.fetchall():
            embedding = np.array(json.loads(row['embedding_vector']))
            results.append((row['id'], row['image_path'], embedding))
        
        return results
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Find similar images based on embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of dictionaries with similarity results
            
        Example:
            >>> results = db.find_similar(query_embedding, top_k=5, threshold=0.7)
            >>> for result in results:
            ...     print(f"{result['image_path']}: {result['similarity']:.4f}")
        """
        all_embeddings = self.get_all_embeddings()
        
        if not all_embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for db_id, image_path, embedding in all_embeddings:
            sim = cosine_similarity([query_embedding], [embedding])[0][0]
            
            if sim >= threshold:
                similarities.append({
                    'id': db_id,
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'similarity': float(sim)
                })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def find_duplicates(
        self,
        similarity_threshold: float = 0.95,
        save_to_table: bool = True
    ) -> List[Dict]:
        """
        Find duplicate or near-duplicate images in the database.
        
        Args:
            similarity_threshold: Minimum similarity to consider as duplicate
            save_to_table: Whether to save duplicates to the duplicates table
            
        Returns:
            List of duplicate pairs with similarity scores
            
        Example:
            >>> duplicates = db.find_duplicates(similarity_threshold=0.95)
            >>> print(f"Found {len(duplicates)} duplicate pairs")
        """
        all_embeddings = self.get_all_embeddings()
        duplicates = []
        
        # Compare all pairs
        for i, (id1, path1, emb1) in enumerate(all_embeddings):
            for id2, path2, emb2 in all_embeddings[i+1:]:
                sim = cosine_similarity([emb1], [emb2])[0][0]
                
                if sim >= similarity_threshold:
                    duplicates.append({
                        'image1_id': id1,
                        'image1_path': path1,
                        'image1_name': os.path.basename(path1),
                        'image2_id': id2,
                        'image2_path': path2,
                        'image2_name': os.path.basename(path2),
                        'similarity': float(sim)
                    })
                    
                    if save_to_table:
                        self._save_duplicate(id1, id2, sim)
        
        return duplicates
    
    def _save_duplicate(self, image1_id: int, image2_id: int, similarity: float):
        """Save a duplicate pair to the duplicates table."""
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO duplicates (image1_id, image2_id, similarity)
                VALUES (?, ?, ?)
            ''', (image1_id, image2_id, similarity))
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Duplicate already recorded
            pass
    
    def get_saved_duplicates(self) -> List[Dict]:
        """
        Retrieve saved duplicates from the database.
        
        Returns:
            List of duplicate pairs
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                d.image1_id,
                e1.image_path as image1_path,
                e1.image_name as image1_name,
                d.image2_id,
                e2.image_path as image2_path,
                e2.image_name as image2_name,
                d.similarity,
                d.detected_at
            FROM duplicates d
            JOIN embeddings e1 ON d.image1_id = e1.id
            JOIN embeddings e2 ON d.image2_id = e2.id
            ORDER BY d.similarity DESC
        ''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'image1_id': row['image1_id'],
                'image1_path': row['image1_path'],
                'image1_name': row['image1_name'],
                'image2_id': row['image2_id'],
                'image2_path': row['image2_path'],
                'image2_name': row['image2_name'],
                'similarity': row['similarity'],
                'detected_at': row['detected_at']
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        cursor = self.conn.cursor()
        
        cursor.execute(
            'SELECT COUNT(*) as count FROM embeddings WHERE model_name = ?',
            (self.model_name,)
        )
        total_images = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM duplicates')
        total_duplicates = cursor.fetchone()['count']
        
        cursor.execute('''
            SELECT AVG(file_size) as avg_size, 
                   AVG(image_width) as avg_width,
                   AVG(image_height) as avg_height
            FROM embeddings WHERE model_name = ?
        ''', (self.model_name,))
        
        stats_row = cursor.fetchone()
        
        return {
            'total_images': total_images,
            'total_duplicates': total_duplicates,
            'model_name': self.model_name,
            'avg_file_size': stats_row['avg_size'],
            'avg_width': stats_row['avg_width'],
            'avg_height': stats_row['avg_height'],
            'database_path': self.db_path
        }
    
    def remove_image(self, image_path: str) -> bool:
        """
        Remove an image from the database.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if removed, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            DELETE FROM embeddings 
            WHERE image_path = ? AND model_name = ?
        ''', (image_path, self.model_name))
        
        self.conn.commit()
        return cursor.rowcount > 0
    
    def clear_all(self):
        """Clear all data from the database."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM duplicates')
        cursor.execute('DELETE FROM embeddings')
        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
