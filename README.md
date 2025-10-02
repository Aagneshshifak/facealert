# GPU-Accelerated Face Recognition System

A high-performance Python face recognition system using ArcFace embeddings and FAISS for efficient similarity matching. The system supports both GPU acceleration and CPU fallback, making it suitable for various deployment scenarios.

## Features

- **ArcFace Embeddings**: Uses InsightFace's buffalo_l model for 512-dimensional face embeddings
- **GPU Acceleration**: Automatic GPU detection with CPU fallback
- **FAISS Integration**: Efficient similarity search for large-scale face matching
- **Batch Processing**: Optimized for processing up to 1000+ images
- **Modular Architecture**: Clean, testable code structure
- **Progress Tracking**: Real-time progress indicators and detailed logging

## System Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- InsightFace
- FAISS (CPU or GPU version)
- ONNXRuntime

## Installation

The required dependencies are already installed in this environment:
- `insightface` - Face detection and embedding extraction
- `faiss-cpu` - Similarity search (CPU version)
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `onnxruntime` - Model inference

## Quick Start

### 1. Check System Information

```bash
python main.py --system-info
```

This displays GPU availability, FAISS support, and InsightFace status.

### 2. Basic Usage

```bash
python main.py --folder /path/to/images --selfie /path/to/selfie.jpg
```

### 3. Advanced Options

```bash
python main.py --folder ./photos --selfie ./query.jpg --max-matches 20 --threshold 0.7 --max-images 500
```

## Command Line Arguments

- `--folder`: Directory containing images to process
- `--selfie`: Path to the query image for matching
- `--max-images`: Maximum number of images to process (default: unlimited)
- `--max-matches`: Maximum number of matches to return (default: 10)
- `--threshold`: Similarity threshold for matches (default: 0.6)
- `--system-info`: Display system information and exit

## How It Works

1. **Image Loading**: Processes all supported image formats (.jpg, .jpeg, .png, .bmp, .tiff, .webp)
2. **Face Detection**: Uses InsightFace to detect faces in each image
3. **Embedding Extraction**: Generates 512-dimensional ArcFace embeddings
4. **Database Storage**: Stores embeddings with metadata in memory
5. **Index Building**: Creates FAISS index for efficient similarity search
6. **Query Processing**: Extracts embedding from selfie image
7. **Similarity Matching**: Finds closest matches using cosine similarity
8. **Results Display**: Shows matching images with similarity scores

## Architecture

The system follows a modular design:

- **config.py**: Configuration management
- **utils.py**: Utility functions and system checks
- **database.py**: In-memory face database
- **face_processor.py**: Face detection and embedding extraction
- **similarity_matcher.py**: FAISS-based similarity matching
- **main.py**: Main orchestration and CLI interface

## Performance

- **CPU Processing**: ~2-5 images/second (depending on image size)
- **GPU Processing**: ~10-20 images/second (with CUDA support)
- **Memory Usage**: ~1-2 GB for 1000 face embeddings
- **Search Speed**: Sub-second for queries against 10,000+ faces

## Configuration

Key settings in `config.py`:

```python
ARCFACE_MODEL_NAME = 'buffalo_l'           # ArcFace model
EMBEDDING_DIMENSION = 512                   # Embedding size
SIMILARITY_THRESHOLD = 0.6                  # Match threshold
MAX_IMAGES_TO_PROCESS = 1000               # Processing limit
BATCH_SIZE = 32                            # Batch processing size
```

## GPU Support

The system automatically detects and uses GPU acceleration when available:

- **CUDA**: For InsightFace model inference
- **FAISS-GPU**: For similarity search acceleration

If GPU is not available, it gracefully falls back to CPU processing.

## Error Handling

The system includes comprehensive error handling:

- Invalid image formats are skipped with warnings
- Face detection failures are logged but don't stop processing
- GPU initialization failures trigger CPU fallback
- Memory constraints are monitored and reported

## Limitations

- Requires at least one face per image for processing
- Multiple faces per image: uses the largest detected face
- Works best with frontal face views
- Minimum face size: 50x50 pixels for reliable detection

## Examples

### Process a folder of event photos
```bash
python main.py --folder ./event_photos --selfie ./my_selfie.jpg
```

### Find matches with custom threshold
```bash
python main.py --folder ./images --selfie ./query.jpg --threshold 0.8
```

### Limit processing for testing
```bash
python main.py --folder ./large_dataset --selfie ./test.jpg --max-images 100
```

## Troubleshooting

1. **"No faces detected"**: Ensure images contain clear, frontal faces
2. **Slow processing**: Check if GPU acceleration is working with `--system-info`
3. **Memory issues**: Reduce `MAX_IMAGES_TO_PROCESS` or `BATCH_SIZE`
4. **Import errors**: Verify all dependencies are installed correctly

## License

This project demonstrates face recognition capabilities using open-source libraries. Ensure compliance with your local data protection and privacy regulations.