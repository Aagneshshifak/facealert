"""
Utility functions for the face recognition system.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from config import SUPPORTED_IMAGE_FORMATS, MAX_IMAGE_SIZE, LOG_LEVEL

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_image_files(folder_path: str) -> List[str]:
    """
    Get all supported image files from a folder.
    
    Args:
        folder_path: Path to the folder containing images
        
    Returns:
        List of image file paths
    """
    if not os.path.exists(folder_path):
        logger.error(f"Folder does not exist: {folder_path}")
        return []
    
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS):
                image_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(image_files)} image files in {folder_path}")
    return image_files

def load_and_preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load and preprocess an image for face detection.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image array or None if failed
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if too large
        height, width = image.shape[:2]
        if max(height, width) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image {image_path} from {width}x{height} to {new_width}x{new_height}")
        
        return image
    
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def check_gpu_availability() -> Tuple[bool, str]:
    """
    Check if GPU is available for processing.
    
    Returns:
        Tuple of (is_available, device_info)
    """
    try:
        import onnxruntime as ort
        
        # Check available providers
        providers = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in providers:
            return True, "CUDA GPU detected and available"
        else:
            return False, "CUDA provider not available"
    
    except ImportError:
        return False, "ONNXRuntime not available"
    except Exception as e:
        return False, f"GPU check failed: {str(e)}"

def print_system_info():
    """Print system information for debugging."""
    gpu_available, gpu_info = check_gpu_availability()
    
    print("="*60)
    print("FACE RECOGNITION SYSTEM - SYSTEM INFO")
    print("="*60)
    print(f"GPU Available: {gpu_available}")
    print(f"GPU Info: {gpu_info}")
    
    try:
        import faiss
        print(f"FAISS Version: {faiss.__version__}")
        print(f"FAISS GPU Support: {hasattr(faiss, 'StandardGpuResources')}")
    except ImportError:
        print("FAISS not available")
    
    try:
        import insightface
        print(f"InsightFace Available: True")
    except ImportError:
        print("InsightFace not available")
    
    print("="*60)

def create_progress_bar(current: int, total: int, prefix: str = "Progress") -> str:
    """
    Create a simple text progress bar.
    
    Args:
        current: Current progress value
        total: Total progress value
        prefix: Prefix text
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return f"{prefix}: 0/0 (100%)"
    
    percentage = (current / total) * 100
    filled_length = int(50 * current // total)
    bar = '█' * filled_length + '-' * (50 - filled_length)
    return f"{prefix}: |{bar}| {current}/{total} ({percentage:.1f}%)"
