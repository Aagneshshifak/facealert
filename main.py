#!/usr/bin/env python3
"""
GPU-accelerated Python face recognition script using ArcFace embeddings and FAISS-GPU.

This script provides batch processing of face images, efficient similarity matching,
and GPU acceleration with automatic CPU fallback.
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Optional
import logging

# Import our modules
from config import (
    SIMILARITY_THRESHOLD, MAX_IMAGES_TO_PROCESS, SHOW_PROGRESS,
    EMBEDDING_DIMENSION
)
from utils import (
    setup_logging, get_image_files, load_and_preprocess_image,
    print_system_info, create_progress_bar
)
from database import FaceDatabase
from face_processor import FaceProcessor
from similarity_matcher import SimilarityMatcher

logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    """Main face recognition system orchestrating all components."""
    
    def __init__(self):
        """Initialize the face recognition system."""
        self.database = FaceDatabase()
        self.face_processor = None
        self.similarity_matcher = None
        self._initialized = False
    
    def initialize(self):
        """Initialize all system components."""
        if self._initialized:
            return
        
        logger.info("Initializing Face Recognition System...")
        
        try:
            # Initialize face processor
            print("🔄 Initializing face detection and embedding extraction...")
            self.face_processor = FaceProcessor()
            
            # Initialize similarity matcher
            print("🔄 Initializing similarity matching system...")
            self.similarity_matcher = SimilarityMatcher(EMBEDDING_DIMENSION)
            
            self._initialized = True
            logger.info("System initialization complete")
            
            # Print system information
            self._print_initialization_summary()
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize system: {str(e)}")
    
    def _print_initialization_summary(self):
        """Print system initialization summary."""
        face_info = self.face_processor.get_model_info()
        matcher_info = self.similarity_matcher.get_index_info()
        
        print("\n" + "="*60)
        print("🚀 SYSTEM INITIALIZATION COMPLETE")
        print("="*60)
        print(f"Face Model: {face_info['model_name']}")
        print(f"Face Processing: {'GPU' if face_info['using_gpu'] else 'CPU'}")
        print(f"Similarity Matching: {'GPU' if matcher_info['using_gpu'] else 'CPU'}")
        print(f"Embedding Dimension: {face_info['embedding_dimension']}")
        print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
        print("="*60)
    
    def load_images_from_folder(self, folder_path: str, max_images: int = None) -> int:
        """
        Load and process all images from a folder.
        
        Args:
            folder_path: Path to folder containing images
            max_images: Maximum number of images to process
            
        Returns:
            Number of faces added to database
        """
        if not self._initialized:
            self.initialize()
        
        # Get image files
        image_files = get_image_files(folder_path)
        if not image_files:
            print(f"❌ No supported image files found in {folder_path}")
            return 0
        
        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]
        
        # Limit to maximum allowed
        if len(image_files) > MAX_IMAGES_TO_PROCESS:
            logger.warning(f"Limiting to {MAX_IMAGES_TO_PROCESS} images (found {len(image_files)})")
            image_files = image_files[:MAX_IMAGES_TO_PROCESS]
        
        print(f"\n📁 Processing {len(image_files)} images from {folder_path}")
        
        # Process images
        total_faces = 0
        processed_images = 0
        start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            try:
                # Show progress
                if SHOW_PROGRESS and (i % 10 == 0 or i == len(image_files) - 1):
                    progress = create_progress_bar(i + 1, len(image_files), "Processing")
                    print(f"\r{progress}", end="", flush=True)
                
                # Process image
                faces = self.face_processor.process_image(image_path)
                
                # Add faces to database
                for face in faces:
                    self.database.add_face(
                        image_path=face['image_path'],
                        embedding=face['embedding'],
                        face_bbox=face['bbox'],
                        confidence=face['confidence']
                    )
                    total_faces += 1
                
                processed_images += 1
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                continue
        
        if SHOW_PROGRESS:
            print()  # New line after progress bar
        
        # Update similarity matcher with all embeddings
        if total_faces > 0:
            print("🔄 Building similarity search index...")
            embeddings = self.database.get_all_embeddings()
            self.similarity_matcher.add_embeddings(embeddings)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Processing complete!")
        print(f"   📊 Processed: {processed_images}/{len(image_files)} images")
        print(f"   👥 Found: {total_faces} faces")
        print(f"   ⏱️  Time: {elapsed_time:.2f} seconds")
        print(f"   🚀 Speed: {processed_images/elapsed_time:.1f} images/sec")
        
        return total_faces
    
    def find_matches(self, selfie_path: str, max_matches: int = 10, 
                    threshold: float = None) -> List[Dict]:
        """
        Find matching faces for a selfie image.
        
        Args:
            selfie_path: Path to the selfie image
            max_matches: Maximum number of matches to return
            threshold: Similarity threshold (uses config default if None)
            
        Returns:
            List of matching results
        """
        if not self._initialized:
            raise RuntimeError("System not initialized")
        
        if self.database.size() == 0:
            print("❌ No faces in database. Load images first.")
            return []
        
        print(f"\n🔍 Searching for matches for: {selfie_path}")
        
        # Load and process selfie
        image = load_and_preprocess_image(selfie_path)
        if image is None:
            print(f"❌ Could not load selfie image: {selfie_path}")
            return []
        
        # Extract embedding
        embedding = self.face_processor.extract_embedding(image)
        if embedding is None:
            print(f"❌ No face detected in selfie: {selfie_path}")
            return []
        
        print(f"✅ Face detected in selfie")
        
        # Find matches
        image_paths = self.database.get_image_paths()
        matches = self.similarity_matcher.find_matches(
            query_embedding=embedding,
            image_paths=image_paths,
            k=max_matches,
            threshold=threshold or SIMILARITY_THRESHOLD
        )
        
        # Print results
        if matches:
            print(f"\n🎯 Found {len(matches)} matches:")
            print("-" * 80)
            for i, match in enumerate(matches, 1):
                filename = os.path.basename(match['image_path'])
                print(f"{i:2d}. {filename:<40} (similarity: {match['similarity']:.3f})")
            print("-" * 80)
        else:
            print(f"❌ No matches found above threshold {threshold or SIMILARITY_THRESHOLD}")
        
        return matches
    
    def print_database_summary(self):
        """Print database summary."""
        self.database.print_summary()
    
    def cleanup(self):
        """Clean up system resources."""
        if self.face_processor:
            self.face_processor.cleanup()
        if self.similarity_matcher:
            self.similarity_matcher.cleanup()
        logger.info("System cleanup complete")

def main():
    """Main function."""
    # Setup logging
    setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="GPU-accelerated face recognition using ArcFace and FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --folder ./photos --selfie ./my_selfie.jpg
  python main.py --folder ./event_photos --selfie ./query.jpg --max-matches 20
  python main.py --folder ./images --selfie ./face.jpg --threshold 0.7
        """
    )
    
    parser.add_argument('--folder', type=str,
                       help='Folder containing images to process')
    parser.add_argument('--selfie', type=str,
                       help='Path to selfie image for matching')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process from folder')
    parser.add_argument('--max-matches', type=int, default=10,
                       help='Maximum number of matches to return')
    parser.add_argument('--threshold', type=float, default=None,
                       help=f'Similarity threshold (default: {SIMILARITY_THRESHOLD})')
    parser.add_argument('--system-info', action='store_true',
                       help='Print system information and exit')
    
    args = parser.parse_args()
    
    # Print system info if requested
    if args.system_info:
        print_system_info()
        return
    
    # Validate required inputs for normal operation
    if not args.folder or not args.selfie:
        parser.error("--folder and --selfie are required unless using --system-info")
    
    # Validate inputs exist
    if not os.path.exists(args.folder):
        print(f"❌ Folder does not exist: {args.folder}")
        sys.exit(1)
    
    if not os.path.exists(args.selfie):
        print(f"❌ Selfie file does not exist: {args.selfie}")
        sys.exit(1)
    
    # Initialize system
    system = FaceRecognitionSystem()
    
    try:
        # Print welcome message
        print("🎭 GPU-Accelerated Face Recognition System")
        print("   Using ArcFace embeddings and FAISS similarity search")
        print()
        
        # Load and process images
        faces_found = system.load_images_from_folder(
            args.folder, 
            max_images=args.max_images
        )
        
        if faces_found == 0:
            print("❌ No faces found in the provided images")
            sys.exit(1)
        
        # Print database summary
        system.print_database_summary()
        
        # Find matches for selfie
        matches = system.find_matches(
            args.selfie,
            max_matches=args.max_matches,
            threshold=args.threshold
        )
        
        # Print final summary
        print(f"\n🏁 Search complete: {len(matches)} matches found")
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        print(f"❌ Error: {str(e)}")
        sys.exit(1)
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()
