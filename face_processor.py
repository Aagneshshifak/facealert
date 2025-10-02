"""
Face processing module using InsightFace and ArcFace embeddings.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import logging
import os
from config import (
    ARCFACE_MODEL_NAME, FORCE_CPU, GPU_DEVICE_ID, 
    MIN_FACE_SIZE, BATCH_SIZE
)
from utils import check_gpu_availability

logger = logging.getLogger(__name__)

class FaceProcessor:
    """Face detection and embedding extraction using InsightFace."""
    
    def __init__(self):
        """Initialize the face processor."""
        self.app = None
        self.model_loaded = False
        self.using_gpu = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the InsightFace model."""
        try:
            import insightface
            
            # Check GPU availability
            gpu_available, gpu_info = check_gpu_availability()
            
            # Determine which context to use
            if FORCE_CPU or not gpu_available:
                ctx_id = -1  # CPU
                device_info = "CPU (forced)" if FORCE_CPU else f"CPU (fallback: {gpu_info})"
                self.using_gpu = False
            else:
                ctx_id = GPU_DEVICE_ID  # GPU
                device_info = f"GPU {GPU_DEVICE_ID}"
                self.using_gpu = True
            
            logger.info(f"Initializing InsightFace model on {device_info}")
            
            # Initialize the face analysis app
            self.app = insightface.app.FaceAnalysis(
                name=ARCFACE_MODEL_NAME,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.using_gpu else ['CPUExecutionProvider']
            )
            
            # Prepare the model with context
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            self.model_loaded = True
            logger.info(f"Successfully initialized {ARCFACE_MODEL_NAME} model on {device_info}")
            
        except ImportError as e:
            logger.error(f"InsightFace not available: {str(e)}")
            raise RuntimeError("InsightFace library is required but not installed")
        except Exception as e:
            logger.error(f"Failed to initialize face model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image in RGB format
            
        Returns:
            List of face detection results
        """
        if not self.model_loaded:
            raise RuntimeError("Face model not loaded")
        
        try:
            # Detect faces
            faces = self.app.get(image)
            
            # Filter faces by minimum size
            valid_faces = []
            for face in faces:
                bbox = face.bbox.astype(int)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                if width >= MIN_FACE_SIZE and height >= MIN_FACE_SIZE:
                    face_info = {
                        'bbox': (bbox[0], bbox[1], width, height),
                        'confidence': float(face.det_score),
                        'embedding': face.embedding,
                        'landmarks': face.kps if hasattr(face, 'kps') else None
                    }
                    valid_faces.append(face_info)
                else:
                    logger.debug(f"Filtered out small face: {width}x{height}")
            
            return valid_faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []
    
    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image containing a single face.
        
        Args:
            image: Input image in RGB format
            
        Returns:
            Face embedding vector or None if no face found
        """
        # Resize image for faster processing (max 800px width)
        h, w = image.shape[:2]
        if w > 800:
            scale = 800 / w
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected, using the largest one")
            # Select the face with largest area
            faces.sort(key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)
        
        return faces[0]['embedding']
    
    def extract_all_embeddings(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract ALL face embeddings from an image (for group photos).
        
        Args:
            image: Input image in RGB format
            
        Returns:
            List of face embedding vectors
        """
        # Resize image for faster processing (max 800px width)
        h, w = image.shape[:2]
        if w > 800:
            scale = 800 / w
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        faces = self.detect_faces(image)
        
        if not faces:
            return []
        
        # Return ALL face embeddings, not just the largest
        embeddings = []
        for face in faces:
            embeddings.append(face['embedding'])
        
        logger.info(f"Extracted {len(embeddings)} face embeddings from image")
        return embeddings
    
    def process_image(self, image_path: str) -> List[Dict]:
        """
        Process a single image and extract face information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face information dictionaries
        """
        from utils import load_and_preprocess_image
        
        # Load and preprocess image
        image = load_and_preprocess_image(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return []
        
        # Detect faces
        faces = self.detect_faces(image)
        
        # Add image path to each face info
        for face in faces:
            face['image_path'] = image_path
        
        logger.debug(f"Processed {image_path}: found {len(faces)} faces")
        return faces
    
    def process_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Process a batch of images efficiently.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of all face information dictionaries
        """
        all_faces = []
        
        logger.info(f"Processing batch of {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:  # Log progress every 10 images
                logger.info(f"Processing image {i+1}/{len(image_paths)}")
            
            faces = self.process_image(image_path)
            all_faces.extend(faces)
        
        logger.info(f"Batch processing complete: {len(all_faces)} faces found in {len(image_paths)} images")
        return all_faces
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': ARCFACE_MODEL_NAME,
            'model_loaded': self.model_loaded,
            'using_gpu': self.using_gpu,
            'gpu_device_id': GPU_DEVICE_ID if self.using_gpu else None,
            'embedding_dimension': 512  # buffalo_l produces 512-dim embeddings
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.app is not None:
            # InsightFace doesn't have explicit cleanup, but we can clear the reference
            self.app = None
            self.model_loaded = False
            logger.info("Face processor cleaned up")
