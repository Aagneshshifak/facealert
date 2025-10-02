#!/usr/bin/env python3
"""
Simple example demonstrating the face recognition system.
"""

import os
import sys
import cv2
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import FaceRecognitionSystem

def create_realistic_face_pattern(filename, variation=0):
    """Create a simple pattern that might be detected as a face."""
    # Create a 300x300 image
    img = np.ones((300, 300, 3), dtype=np.uint8) * 240
    
    # Add noise for more realistic texture
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Create face-like pattern with gradients
    center_x, center_y = 150, 150
    
    # Face outline (larger oval)
    cv2.ellipse(img, (center_x, center_y), (80, 100), 0, 0, 360, (200+variation, 180, 160), -1)
    
    # Eyes (dark circles)
    cv2.circle(img, (center_x-25, center_y-20), 8, (50, 50, 50), -1)
    cv2.circle(img, (center_x+25, center_y-20), 8, (50, 50, 50), -1)
    
    # Eye highlights
    cv2.circle(img, (center_x-23, center_y-22), 3, (255, 255, 255), -1)
    cv2.circle(img, (center_x+23, center_y-22), 3, (255, 255, 255), -1)
    
    # Nose
    pts = np.array([[center_x, center_y-5], [center_x-5, center_y+10], [center_x+5, center_y+10]], np.int32)
    cv2.fillPoly(img, [pts], (150, 120, 100))
    
    # Mouth
    cv2.ellipse(img, (center_x, center_y+25), (15, 8), 0, 0, 180, (100, 80, 80), 2)
    
    # Add some shading for depth
    overlay = img.copy()
    cv2.ellipse(overlay, (center_x-15, center_y-15), (60, 80), 0, 0, 360, (180, 150, 130), -1)
    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    cv2.imwrite(filename, img)
    print(f"Created face pattern: {filename}")

def run_example():
    """Run the face recognition example."""
    print("Creating Face Recognition System Example")
    print("=" * 50)
    
    # Create test images
    os.makedirs('demo_images', exist_ok=True)
    
    create_realistic_face_pattern('demo_images/person1.jpg', 0)
    create_realistic_face_pattern('demo_images/person2.jpg', 20)
    create_realistic_face_pattern('demo_images/person3.jpg', -15)
    create_realistic_face_pattern('query_face.jpg', 5)  # Similar to person1
    
    print(f"\nCreated demo images:")
    print(f"- demo_images/person1.jpg")
    print(f"- demo_images/person2.jpg") 
    print(f"- demo_images/person3.jpg")
    print(f"- query_face.jpg (for matching)")
    
    print(f"\nInitializing Face Recognition System...")
    
    try:
        # Initialize system
        system = FaceRecognitionSystem()
        system.initialize()
        
        # Load images from folder
        print(f"\nLoading images from demo_images folder...")
        faces_found = system.load_images_from_folder('demo_images', max_images=10)
        
        if faces_found > 0:
            # Show database summary
            system.print_database_summary()
            
            # Find matches for query
            print(f"\nSearching for matches...")
            matches = system.find_matches('query_face.jpg', max_matches=5, threshold=0.3)
            
            print(f"\nExample completed successfully!")
            print(f"Found {len(matches)} potential matches")
            
        else:
            print("No faces detected in the demo images")
            print("Note: The system uses real face detection - simple patterns may not be detected")
        
        # Cleanup
        system.cleanup()
        
    except Exception as e:
        print(f"Error during example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_example()