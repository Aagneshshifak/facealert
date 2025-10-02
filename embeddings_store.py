"""
Storage system for face embeddings.
"""
import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class EmbeddingsStore:
    """Store and retrieve face embeddings."""
    
    def __init__(self, storage_path: str = 'embeddings_data'):
        """
        Initialize the embeddings store.
        
        Args:
            storage_path: Directory to store embeddings
        """
        self.storage_path = storage_path
        self.embeddings_file = os.path.join(storage_path, 'embeddings.json')
        self.embeddings_npy = os.path.join(storage_path, 'embeddings.npy')
        
        os.makedirs(storage_path, exist_ok=True)
        
        self.persons = []
        self.embeddings = None
        self._load_data()
    
    def _load_data(self):
        """Load embeddings from disk."""
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'r') as f:
                    self.persons = json.load(f)
                
                if os.path.exists(self.embeddings_npy):
                    self.embeddings = np.load(self.embeddings_npy)
                    print(f"Loaded {len(self.persons)} person records")
                else:
                    self.embeddings = np.array([])
            else:
                print("No existing embeddings found, starting fresh")
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            self.persons = []
            self.embeddings = np.array([])
    
    def _save_data(self):
        """Save embeddings to disk."""
        try:
            with open(self.embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(self.persons, f, indent=2)
            
            if self.embeddings is not None and len(self.embeddings) > 0:
                np.save(self.embeddings_npy, self.embeddings)
            
            print(f"Saved {len(self.persons)} person records")
        except Exception as e:
            print(f"Error saving embeddings: {str(e)}")
    
    def add_person(self, name: str, embedding: np.ndarray, photo_url: Optional[str] = None) -> int:
        """
        Add a new person with their face embedding.
        
        Args:
            name: Person's name or identifier
            embedding: Face embedding vector
            photo_url: URL or path to the photo
            
        Returns:
            Person ID
        """
        person_id = len(self.persons) + 1
        
        person_data = {
            'id': person_id,
            'name': name,
            'photo_url': photo_url,
            'added_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.persons.append(person_data)
        
        # Add embedding to array
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        self._save_data()
        print(f"Added person: {name} (ID: {person_id})")
        return person_id
    
    def find_match(self, query_embedding: np.ndarray, threshold: float = 0.6) -> Tuple[bool, float, Optional[Dict]]:
        """
        Find a matching person for the query embedding.
        
        Args:
            query_embedding: Query face embedding
            threshold: Similarity threshold (0-1, higher = more similar)
            
        Returns:
            Tuple of (found, similarity_score, person_data)
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return False, 0.0, None
        
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        
        if best_score >= threshold:
            return True, best_score, self.persons[best_idx]
        else:
            return False, best_score, None
    
    def get_all_persons(self) -> List[Dict]:
        """Get all stored persons."""
        return self.persons
    
    def get_person_count(self) -> int:
        """Get the number of stored persons."""
        return len(self.persons)
    
    def delete_person(self, person_id: int) -> bool:
        """
        Delete a person by ID.
        
        Args:
            person_id: Person ID to delete
            
        Returns:
            True if successful
        """
        try:
            idx = None
            for i, person in enumerate(self.persons):
                if person['id'] == person_id:
                    idx = i
                    break
            
            if idx is not None:
                self.persons.pop(idx)
                if self.embeddings is not None and len(self.embeddings) > 0:
                    self.embeddings = np.delete(self.embeddings, idx, axis=0)
                self._save_data()
                print(f"Deleted person ID: {person_id}")
                return True
            return False
        except Exception as e:
            print(f"Error deleting person: {str(e)}")
            return False
