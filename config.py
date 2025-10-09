"""
Configuration settings for the face recognition system.
"""

import os

# Model configuration
ARCFACE_MODEL_NAME = 'buffalo_l'
EMBEDDING_DIMENSION = 512  # buffalo_l produces 512-dimensional embeddings
SIMILARITY_THRESHOLD = 0.4  # Minimum similarity score for matches

# GPU configuration
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"
GPU_DEVICE_ID = int(os.getenv("GPU_DEVICE_ID", "0"))

# Processing configuration
BATCH_SIZE = 32  # Images to process in a single batch
MAX_IMAGE_SIZE = 1024  # Maximum image dimension for processing
MIN_FACE_SIZE = 12  # Minimum face size for detection (supports group photos with tiny faces)

# File handling
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
MAX_IMAGES_TO_PROCESS = 1000

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SHOW_PROGRESS = True
