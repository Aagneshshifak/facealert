#!/usr/bin/env python3
"""
Demo script for the face recognition system.
This creates sample images for testing the system.
"""

import cv2
import numpy as np
import os
from main import FaceRecognitionSystem

def create_sample_face_image(filename, color=(100, 150, 200)):
    """Create a simple sample face image for testing."""
    # Create a simple face-like image with basic features
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    # Add a simple face shape (oval)
    center = (100, 100)
    axes = (80, 100)
    cv2.ellipse(img, center, axes, 0, 0, 360, color, -1)
    
    # Add eyes
    cv2.circle(img, (80, 80), 10, (0, 0, 0), -1)
    cv2.circle(img, (120, 80), 10, (0, 0, 0), -1)
    
    # Add nose
    cv2.circle(img, (100, 100), 5, (50, 50, 50), -1)
    
    # Add mouth
    cv2.ellipse(img, (100, 130), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    
    # Save the image
    cv2.imwrite(filename, img)
    print(f"Created sample face image: {filename}")

def demo_face_recognition():
    """Demonstrate the face recognition system."""
    # Create test images directory
    os.makedirs('test_images', exist_ok=True)
    
    # Create sample face images with different colors
    create_sample_face_image('test_images/face1.jpg', (120, 160, 200))
    create_sample_face_image('test_images/face2.jpg', (100, 140, 180))
    create_sample_face_image('test_images/face3.jpg', (140, 170, 220))
    create_sample_face_image('selfie.jpg', (120, 160, 200))  # Similar to face1
    
    print("\n" + "="*60)
    print("FACE RECOGNITION DEMO")
    print("="*60)
    print("Created sample images for testing:")
    print("- test_images/face1.jpg")
    print("- test_images/face2.jpg") 
    print("- test_images/face3.jpg")
    print("- selfie.jpg (similar to face1)")
    print("\nNow running face recognition system...")
    print("="*60)

if __name__ == "__main__":
    demo_face_recognition()