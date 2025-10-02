"""
Similarity matching using FAISS for efficient vector search.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from config import SIMILARITY_THRESHOLD, FORCE_CPU, GPU_DEVICE_ID
from utils import check_gpu_availability

logger = logging.getLogger(__name__)

class SimilarityMatcher:
    """FAISS-based similarity matcher for face embeddings."""
    
    def __init__(self, embedding_dimension: int = 512):
        """
        Initialize the similarity matcher.
        
        Args:
            embedding_dimension: Dimension of the embedding vectors
        """
        self.embedding_dimension = embedding_dimension
        self.index = None
        self.using_gpu = False
        self.gpu_resources = None
        self._initialize_faiss()
    
    def _initialize_faiss(self):
        """Initialize FAISS index with GPU support if available."""
        try:
            import faiss
            
            # Check GPU availability and FAISS GPU support
            gpu_available, gpu_info = check_gpu_availability()
            has_gpu_support = hasattr(faiss, 'StandardGpuResources')
            
            # Determine which device to use
            if FORCE_CPU or not gpu_available or not has_gpu_support:
                self._initialize_cpu_index(faiss)
                if FORCE_CPU:
                    device_info = "CPU (forced)"
                elif not gpu_available:
                    device_info = f"CPU (no GPU: {gpu_info})"
                else:
                    device_info = "CPU (FAISS-GPU not available)"
            else:
                try:
                    self._initialize_gpu_index(faiss)
                    device_info = f"GPU {GPU_DEVICE_ID}"
                except Exception as e:
                    logger.warning(f"GPU initialization failed, falling back to CPU: {str(e)}")
                    self._initialize_cpu_index(faiss)
                    device_info = f"CPU (GPU fallback: {str(e)})"
            
            logger.info(f"Initialized FAISS index on {device_info}")
            
        except ImportError as e:
            logger.error(f"FAISS not available: {str(e)}")
            raise RuntimeError("FAISS library is required but not installed")
    
    def _initialize_cpu_index(self, faiss):
        """Initialize CPU-based FAISS index."""
        # Use inner product for cosine similarity (normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.using_gpu = False
        logger.debug("Initialized CPU FAISS index (IndexFlatIP)")
    
    def _initialize_gpu_index(self, faiss):
        """Initialize GPU-based FAISS index."""
        # Create GPU resources
        self.gpu_resources = faiss.StandardGpuResources()
        
        # Create CPU index first
        cpu_index = faiss.IndexFlatIP(self.embedding_dimension)
        
        # Move to GPU
        self.index = faiss.index_cpu_to_gpu(
            self.gpu_resources, 
            GPU_DEVICE_ID, 
            cpu_index
        )
        
        self.using_gpu = True
        logger.debug(f"Initialized GPU FAISS index on device {GPU_DEVICE_ID}")
    
    def add_embeddings(self, embeddings: np.ndarray):
        """
        Add embeddings to the FAISS index.
        
        Args:
            embeddings: Matrix of embeddings (n_vectors x embedding_dim)
        """
        if embeddings.size == 0:
            logger.warning("Attempted to add empty embeddings array")
            return
        
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {embeddings.shape[1]}")
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self._normalize_embeddings(embeddings)
        
        # Add to index
        self.index.add(normalized_embeddings.astype(np.float32))
        
        logger.debug(f"Added {embeddings.shape[0]} embeddings to FAISS index")
    
    def search(self, query_embedding: np.ndarray, k: int = 10, 
               threshold: float = None) -> Tuple[List[float], List[int]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to return
            threshold: Similarity threshold (uses config default if None)
            
        Returns:
            Tuple of (similarities, indices)
        """
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")
        
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return [], []
        
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD
        
        # Normalize query embedding
        query_normalized = self._normalize_embeddings(query_embedding.reshape(1, -1))
        
        # Search
        similarities, indices = self.index.search(
            query_normalized.astype(np.float32), 
            min(k, self.index.ntotal)
        )
        
        # Filter by threshold and convert to lists
        valid_similarities = []
        valid_indices = []
        
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= threshold and idx != -1:  # -1 indicates invalid result
                valid_similarities.append(float(sim))
                valid_indices.append(int(idx))
        
        logger.debug(f"Found {len(valid_similarities)} matches above threshold {threshold}")
        return valid_similarities, valid_indices
    
    def find_matches(self, query_embedding: np.ndarray, image_paths: List[str],
                    k: int = 10, threshold: float = None) -> List[Dict]:
        """
        Find matching images for a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            image_paths: List of image paths corresponding to stored embeddings
            k: Maximum number of matches to return
            threshold: Similarity threshold
            
        Returns:
            List of match dictionaries with similarity scores and paths
        """
        similarities, indices = self.search(query_embedding, k, threshold)
        
        matches = []
        for sim, idx in zip(similarities, indices):
            if idx < len(image_paths):
                matches.append({
                    'image_path': image_paths[idx],
                    'similarity': sim,
                    'index': idx
                })
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms
    
    def get_index_info(self) -> Dict:
        """
        Get information about the FAISS index.
        
        Returns:
            Dictionary with index information
        """
        return {
            'embedding_dimension': self.embedding_dimension,
            'using_gpu': self.using_gpu,
            'gpu_device_id': GPU_DEVICE_ID if self.using_gpu else None,
            'total_vectors': self.index.ntotal if self.index else 0,
            'index_type': type(self.index).__name__ if self.index else None
        }
    
    def clear(self):
        """Clear the FAISS index."""
        if self.index is not None:
            self.index.reset()
            logger.debug("Cleared FAISS index")
    
    def cleanup(self):
        """Clean up GPU resources."""
        if self.using_gpu and self.gpu_resources is not None:
            # FAISS GPU resources are automatically managed
            self.gpu_resources = None
            logger.debug("Cleaned up GPU resources")
