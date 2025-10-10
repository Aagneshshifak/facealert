"""
Flask web application for FaceAlert Admin Dashboard - INDIVIDUAL PHOTOS ENHANCED VERSION.
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
            print("❌ No faces detected in uploaded image")
            return jsonify({
                'found': False,
                'score': 0.0,
                'message': 'No face detected in the photo'
            })
        
        print(f"✅ Found {len(uploaded_embeddings)} faces in uploaded image")
        
        # Step 3: Compare with stored embeddings (using first uploaded face)
        print("Step 3: Comparing with stored embeddings...")
        found, score, person_data = embeddings_store.find_match(uploaded_embeddings[0], threshold=0.4)
        
        # Step 4: Compare ALL uploaded faces with ALL database faces
        print("Step 4: Comparing ALL faces in uploaded image with database images...")
        matched_local_images = []
        MAX_FILES_TO_PROCESS = 200  # 🔍 MAXIMUM: Process all available images to find your group photo
        
        try:
            local_image_files = get_image_files(LOCAL_IMAGES_FOLDER)
            print(f"Found {len(local_image_files)} images in local folder (processing max {MAX_FILES_TO_PROCESS})")
            
            # 🎯 ENHANCED: Smart sorting - prioritize individual photos AND group photos
            def smart_priority_sort(image_path):
                try:
                    # Get file size for quality
                    file_size = os.path.getsize(image_path) if os.path.exists(image_path) else 0
                    
                    # Quick face count check for prioritization
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
                        face_count = len(faces) if faces else 0
                        
                        # 🎯 PRIORITY: Individual photos (1 face) get highest priority
                        if face_count == 1:
                            return (1000, file_size)  # Individual photos first
                        elif face_count > 1:
                            return (500 + face_count, file_size)  # Group photos second, by face count
                        else:
                            return (0, file_size)  # No faces last
                    return (0, file_size)
                except:
                    return (0, 0)
            
            # Sort by smart priority (individual photos first, then group photos)
            local_image_files.sort(key=smart_priority_sort, reverse=True)
            
            individual_photos_found = 0
            group_photos_found = 0
            excellent_match_count = 0  # ⚡ OPTIMIZED: Early termination counter
            
            for idx, image_path in enumerate(local_image_files[:MAX_FILES_TO_PROCESS]):
                # ⚡ OPTIMIZED: Early termination if we have enough high-quality matches
                if excellent_match_count >= 5:  # Reduced to focus on quality over quantity
                    print(f"⚡ Early termination: Found {excellent_match_count} high-quality matches, stopping...")
                    break
                
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
                            
                            # 🎯 BALANCED: Reasonable thresholds to prevent false positives while allowing matches
                            # Use raw similarity without artificial boosts
                            raw_similarity = best_similarity
                            
                            # Set thresholds to show pictures above 40% similarity
                            if face_count == 1:
                                # Individual photos: 40% minimum
                                threshold = 0.4
                            else:
                                # Group photos: 40% minimum (even for 12px faces)
                                threshold = 0.4
                            
                            # Use raw similarity without boosts
                            adjusted_similarity = raw_similarity
                            
                            # 🎯 BALANCED: Face quality validation with reasonable thresholds
                            # Only consider matches if similarity meets the threshold
                            print(f"🔍 Similarity check: {adjusted_similarity:.3f} vs threshold {threshold:.3f} for {os.path.basename(image_path)}")
                            # Special debug for very low similarities to catch your group photo
                            if adjusted_similarity > 0.01 and adjusted_similarity < threshold:
                                print(f"🔍 LOW SIMILARITY: {adjusted_similarity:.3f} for {os.path.basename(image_path)} - might be your group photo!")
                            if adjusted_similarity > threshold:
                                is_individual = face_count == 1
                                
                                matched_local_images.append({
                                    'path': image_path,
                                    'name': os.path.basename(image_path),
                                    'similarity': float(adjusted_similarity),
                                    'face_count': face_count,
                                    'is_group_photo': face_count > 1,
                                    'is_individual_photo': is_individual
                                })
                                
                                # Track photo types
                                if is_individual:
                                    individual_photos_found += 1
                                else:
                                    group_photos_found += 1
                                
                                print(f"Match found: {os.path.basename(image_path)} with similarity {adjusted_similarity:.2f} (faces: {face_count}) {'[INDIVIDUAL]' if is_individual else '[GROUP]'}")
                                
                                # ⚡ OPTIMIZED: Track excellent matches for early termination
                                # Only count as excellent if similarity is very high (0.8+)
                                if adjusted_similarity > 0.8:
                                    excellent_match_count += 1
                except Exception as e:
                    print(f"Error processing local file {image_path}: {str(e)}")
                    continue
            
            # 🎯 ENHANCED: Sort by similarity and filter low-quality matches
            def result_priority(item):
                # Sort purely by similarity (no artificial boosts)
                return item['similarity']
            
            matched_local_images.sort(key=result_priority, reverse=True)
            
            # 🎯 FINAL FILTER: Show pictures above 40% similarity (including 12px faces in group photos)
            high_confidence_matches = [match for match in matched_local_images if match['similarity'] >= 0.4]
            print(f"🔍 DEBUG: Found {len(high_confidence_matches)} matches before deduplication")
            
            # 🎯 SIMPLE DEDUPLICATION: Remove duplicates by exact filename matching
            seen_filenames = set()
            unique_matches = []
            
            for match in high_confidence_matches:
                filename = match['name']
                # Simple approach: if we've seen this exact filename before, skip it
                if filename not in seen_filenames:
                    seen_filenames.add(filename)
                    unique_matches.append(match)
                    print(f"✅ Added: {filename}")
                else:
                    print(f"🚫 Skipped duplicate: {filename}")
            
            # 🎯 SECOND PASS: Remove similar filenames (like "file.jpeg" and "file (1).jpeg")
            final_matches = []
            seen_base_names = set()
            
            for match in unique_matches:
                filename = match['name']
                # Extract base name without variations
                base_name = filename.replace(' (1)', '').replace(' (2)', '').replace(' (3)', '')
                base_name = base_name.replace(' (1)', '').replace(' (2)', '').replace(' (3)', '')
                
                if base_name not in seen_base_names:
                    seen_base_names.add(base_name)
                    final_matches.append(match)
                    print(f"✅ Final: {filename}")
                else:
                    print(f"🚫 Similar: {filename} (base: {base_name})")
            
            matched_local_images = final_matches
            print(f"📊 Final deduplication: {len(high_confidence_matches)} -> {len(unique_matches)} -> {len(final_matches)} matches")
            
            print(f"📊 Results: {individual_photos_found} individual photos, {group_photos_found} group photos found")
            
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
            'matches': matched_local_images[:15],  # Show maximum results to find your group photo
            'individual_photos_found': individual_photos_found,
            'group_photos_found': group_photos_found,
            'quality_filter': '40% similarity threshold to show high-confidence matches (including 12px faces in group photos)',
            'total_high_confidence_matches': len(matched_local_images)
        }
        
        if found and person_data:
            response_data['person'] = {
                'name': person_data['name'],
                'id': person_data['id']
            }
        
        print(f"Result: Found={found}, Score={score}, Local Top Score={local_top_score}, High-Confidence Matches={len(matched_local_images)}")
        print(f"Individual photos: {individual_photos_found}, Group photos: {group_photos_found}")
        print(f"🎯 QUALITY FILTER: 40% similarity threshold to show high-confidence matches (including 12px faces in group photos)")
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
    app.run(host='0.0.0.0', port=5005, debug=debug_mode)
