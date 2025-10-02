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
        
        # Extract embedding
        embedding = processor.extract_embedding(image_rgb)
        
        if embedding is None:
            return jsonify({
                'found': False,
                'score': 0.0,
                'message': 'No face detected in the photo'
            })
        
        # Step 3: Compare with stored embeddings
        print("Step 3: Comparing with stored embeddings...")
        found, score, person_data = embeddings_store.find_match(embedding, threshold=0.3)
        
        # Step 4: Compare with images from local folder
        print("Step 4: Comparing with local folder images...")
        matched_local_images = []
        MAX_FILES_TO_PROCESS = 100 
        
        try:
            local_image_files = get_image_files(LOCAL_IMAGES_FOLDER)
            print(f"Found {len(local_image_files)} images in local folder (processing max {MAX_FILES_TO_PROCESS})")
            
            for idx, image_path in enumerate(local_image_files[:MAX_FILES_TO_PROCESS]):
                try:
                    local_image = cv2.imread(image_path)
                    
                    if local_image is not None:
                        local_image_rgb = cv2.cvtColor(local_image, cv2.COLOR_BGR2RGB)
                        local_embedding = processor.extract_embedding(local_image_rgb)
                        
                        if local_embedding is not None:
                            similarity = np.dot(embedding, local_embedding) / (
                                np.linalg.norm(embedding) * np.linalg.norm(local_embedding)
                            )
                            
                            if similarity > 0.3:
                                matched_local_images.append({
                                    'path': image_path,
                                    'name': os.path.basename(image_path),
                                    'similarity': float(similarity)
                                })
                                print(f"Match found: {os.path.basename(image_path)} with similarity {similarity:.2f}")
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
