"""
In-memory database for storing face embeddings and metadata.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class FaceRecord:
    """Data class for storing face information."""
    id: int
    image_path: str
    embedding: np.ndarray
    face_bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'image_path': self.image_path,
            'embedding_shape': self.embedding.shape,
            'face_bbox': self.face_bbox,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }

class FaceDatabase:
    """In-memory database for face embeddings and metadata."""
    
    def __init__(self):
        """Initialize the database."""
        self.records: List[FaceRecord] = []
        self.next_id = 1
        self._embedding_matrix: Optional[np.ndarray] = None
        self._needs_rebuild = True
        
        logger.info("Initialized face database")
    
    def add_face(self, image_path: str, embedding: np.ndarray, 
                 face_bbox: Tuple[int, int, int, int], confidence: float) -> int:
        """
        Add a face record to the database.
        
        Args:
            image_path: Path to the source image
            embedding: Face embedding vector
            face_bbox: Bounding box coordinates (x, y, width, height)
            confidence: Detection confidence score
            
        Returns:
            Record ID
        """
        record = FaceRecord(
            id=self.next_id,
            image_path=image_path,
            embedding=embedding.copy(),
            face_bbox=face_bbox,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        self.records.append(record)
        self.next_id += 1
        self._needs_rebuild = True
        
        logger.debug(f"Added face record {record.id} for {image_path}")
        return record.id
    
    def get_record(self, record_id: int) -> Optional[FaceRecord]:
        """
        Get a record by ID.
        
        Args:
            record_id: Record ID to retrieve
            
        Returns:
            FaceRecord or None if not found
        """
        for record in self.records:
            if record.id == record_id:
                return record
        return None
    
    def get_all_embeddings(self) -> np.ndarray:
        """
        Get all embeddings as a matrix.
        
        Returns:
            Matrix of embeddings (n_records x embedding_dim)
        """
        if not self.records:
            return np.array([]).reshape(0, 512)  # Empty array with correct shape
        
        if self._needs_rebuild or self._embedding_matrix is None:
            embeddings = [record.embedding for record in self.records]
            self._embedding_matrix = np.vstack(embeddings)
            self._needs_rebuild = False
            logger.debug(f"Rebuilt embedding matrix: {self._embedding_matrix.shape}")
        
        return self._embedding_matrix
    
    def get_image_paths(self) -> List[str]:
        """
        Get all image paths in order.
        
        Returns:
            List of image paths
        """
        return [record.image_path for record in self.records]
    
    def clear(self):
        """Clear all records from the database."""
        self.records.clear()
        self.next_id = 1
        self._embedding_matrix = None
        self._needs_rebuild = True
        logger.info("Cleared face database")
    
    def size(self) -> int:
        """Get the number of records in the database."""
        return len(self.records)
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.records:
            return {
                'total_records': 0,
                'unique_images': 0,
                'avg_confidence': 0.0,
                'embedding_dimension': 0
            }
        
        unique_images = len(set(record.image_path for record in self.records))
        avg_confidence = np.mean([record.confidence for record in self.records])
        embedding_dim = self.records[0].embedding.shape[0] if self.records else 0
        
        return {
            'total_records': len(self.records),
            'unique_images': unique_images,
            'avg_confidence': float(avg_confidence),
            'embedding_dimension': embedding_dim
        }
    
    def print_summary(self):
        """Print a summary of the database contents."""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("FACE DATABASE SUMMARY")
        print("="*50)
        print(f"Total Records: {stats['total_records']}")
        print(f"Unique Images: {stats['unique_images']}")
        print(f"Average Confidence: {stats['avg_confidence']:.3f}")
        print(f"Embedding Dimension: {stats['embedding_dimension']}")
        
        if self.records:
            print(f"\nRecent Records:")
            for record in self.records[-3:]:  # Show last 3 records
                filename = record.image_path.split('/')[-1]
                print(f"  ID {record.id}: {filename} (conf: {record.confidence:.3f})")
        
        print("="*50)
