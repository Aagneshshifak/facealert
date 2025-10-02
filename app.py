"""
Flask web application for FaceAlert Admin Dashboard.
"""
import os
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from face_processor import FaceProcessor
from embeddings_store import EmbeddingsStore
from utils import get_image_files

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

# Local folder path for face images
LOCAL_IMAGES_FOLDER = '/Users/aagneshshifak/Downloads/facealertapi'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'photos'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'videos'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'temp'), exist_ok=True)

# Initialize face recognition system
face_processor = None
embeddings_store = EmbeddingsStore()

def get_face_processor():
    """Lazy load face processor to avoid startup delays."""
    global face_processor
    if face_processor is None:
        face_processor = FaceProcessor()
    return face_processor

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm'}


def allowed_file(filename, file_type='image'):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return False

@app.route('/')
def index():
    return render_template('report.html')

@app.route('/report')
def report_page():
    return render_template('report.html')

@app.route('/report/submit', methods=['POST'])
def submit_report():
    temp_path = None
    
    try:
        # Get photo from form
        if 'photo' not in request.files:
            return jsonify({'error': 'No photo provided'}), 400
        
        photo = request.files['photo']
        
        if not photo.filename or photo.filename == '':
            return jsonify({'error': 'No photo selected'}), 400
        
        if not allowed_file(photo.filename, 'image'):
            return jsonify({'error': 'Invalid photo format'}), 400
        
        print("Step 1: Saving uploaded photo...")
        photo_filename = secure_filename(photo.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        photo_filename = f"{timestamp}_{photo_filename}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', photo_filename)
        photo.save(temp_path)
        
        # Step 2: Extract face embedding using InsightFace
        print("Step 2: Extracting face embedding...")
        processor = get_face_processor()
        
        # Load image
        image = cv2.imread(temp_path)
        if image is None:
            raise Exception("Failed to load image")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract ALL face embeddings (for group photos)
        uploaded_embeddings = processor.extract_all_embeddings(image_rgb)
        
        if not uploaded_embeddings:
            return jsonify({
                'found': False,
                'score': 0.0,
                'message': 'No face detected in the photo'
            })
        
        print(f"Found {len(uploaded_embeddings)} faces in uploaded image")
        
        # Step 3: Compare with stored embeddings (using first uploaded face)
        print("Step 3: Comparing with stored embeddings...")
        found, score, person_data = embeddings_store.find_match(uploaded_embeddings[0], threshold=0.2)
        
        # Step 4: Compare ALL uploaded faces with ALL database faces
        print("Step 4: Comparing ALL faces in uploaded image with database images...")
        matched_local_images = []
        MAX_FILES_TO_PROCESS = 75  # Increased to focus on group photos
        
        try:
            local_image_files = get_image_files(LOCAL_IMAGES_FOLDER)
            print(f"Found {len(local_image_files)} images in local folder (processing max {MAX_FILES_TO_PROCESS})")
            
            # Sort images to prioritize group photos (images with multiple faces)
            def get_group_photo_priority(image_path):
                try:
                    # Quick face count check to prioritize group photos
                    temp_image = cv2.imread(image_path)
                    if temp_image is not None:
                        temp_rgb = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
                        # Resize for quick processing
                        h, w = temp_rgb.shape[:2]
                        if w > 400:
                            scale = 400 / w
                            new_h, new_w = int(h * scale), int(w * scale)
                            temp_rgb = cv2.resize(temp_rgb, (new_w, new_h))
                        
                        faces = processor.detect_faces(temp_rgb)
                        return len(faces) if faces else 0
                except:
                    return 0
                return 0
            
            # Sort by group photo priority (more faces = higher priority)
            local_image_files.sort(key=get_group_photo_priority, reverse=True)
            
            for idx, image_path in enumerate(local_image_files[:MAX_FILES_TO_PROCESS]):
                try:
                    local_image = cv2.imread(image_path)
                    
                    if local_image is not None:
                        local_image_rgb = cv2.cvtColor(local_image, cv2.COLOR_BGR2RGB)
                        # Extract ALL faces from database image
                        local_embeddings = processor.extract_all_embeddings(local_image_rgb)
                        
                        if local_embeddings:
                            # Compare each uploaded face with each database face
                            best_similarity = 0.0
                            face_count = len(local_embeddings)
                            
                            for uploaded_embedding in uploaded_embeddings:
                                for local_embedding in local_embeddings:
                                    similarity = np.dot(uploaded_embedding, local_embedding) / (
                                        np.linalg.norm(uploaded_embedding) * np.linalg.norm(local_embedding)
                                    )
                                    best_similarity = max(best_similarity, similarity)
                            
                            # Boost similarity for group photos (more faces = higher priority)
                            group_boost = 1.0 + (face_count - 1) * 0.1  # 10% boost per additional face
                            adjusted_similarity = best_similarity * group_boost
                            
                            # Lower threshold for group photos to catch more matches
                            threshold = 0.3 if face_count > 1 else 0.4
                            
                            if adjusted_similarity > threshold:
                                matched_local_images.append({
                                    'path': image_path,
                                    'name': os.path.basename(image_path),
                                    'similarity': float(adjusted_similarity),
                                    'face_count': face_count,
                                    'is_group_photo': face_count > 1
                                })
                                print(f"Match found: {os.path.basename(image_path)} with similarity {adjusted_similarity:.2f} (faces: {face_count})")
                except Exception as e:
                    print(f"Error processing local file {image_path}: {str(e)}")
                    continue
            
            matched_local_images.sort(key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            print(f"Error comparing with local images: {str(e)}")
        
        # Save photo
        final_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], 'photos', photo_filename)
        os.rename(temp_path, final_photo_path)
        temp_path = None
        
        # Step 5: Return JSON response
        local_top_score = matched_local_images[0]['similarity'] if matched_local_images else 0.0
        display_score = max(score, local_top_score)
        
        response_data = {
            'found': found or len(matched_local_images) > 0,
            'score': round(display_score, 2),
            'matches': matched_local_images[:5]
        }
        
        if found and person_data:
            response_data['person'] = {
                'name': person_data['name'],
                'id': person_data['id']
            }
        
        print(f"Result: Found={found}, Score={score}, Local Top Score={local_top_score}, Local Matches={len(matched_local_images)}")
        print(f"Response data: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in submit_report: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Cleanup temporary files
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/uploads/<folder>/<filename>')
def serve_upload(folder, filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], folder, filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return "File not found", 404

@app.route('/facealertapi/<filename>')
def serve_facealertapi(filename):
    """Serve images from the facealertapi directory."""
    filepath = os.path.join(LOCAL_IMAGES_FOLDER, filename)
    print(f"Trying to serve: {filepath}")
    print(f"File exists: {os.path.exists(filepath)}")
    if os.path.exists(filepath):
        return send_file(filepath)
    return "File not found", 404


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum upload size is 100MB.'}), 413

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
