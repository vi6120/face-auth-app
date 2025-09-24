import streamlit as st
import cv2
import numpy as np
import json
import bcrypt
import os
import hashlib
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
from insightface.app import FaceAnalysis

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'show_logs' not in st.session_state:
    st.session_state.show_logs = False
if 'last_camera_raw_sig' not in st.session_state:
    st.session_state.last_camera_raw_sig = None

DATA_FILE = 'users.json'
PEOPLE_FILE = 'people.json'

def add_log(message, log_type="INFO"):
    """Add a log message to the session state logs"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {log_type}: {message}"
    
    # Add to session state logs
    if len(st.session_state.logs) >= 50:  # Keep only last 50 logs
        st.session_state.logs.pop(0)
    st.session_state.logs.append(log_entry)
    
    # Also print to console for backup
    print(log_entry)

def display_logs_sidebar():
    """Display logs in the sidebar"""
    with st.sidebar:
        st.markdown("---")
        
        # Toggle button for showing/hiding logs
        if st.button("Toggle Debug Logs"):
            st.session_state.show_logs = not st.session_state.show_logs
        
        if st.session_state.show_logs:
            st.subheader("Debug Logs")
            
            # Clear logs button
            if st.button("Clear Logs"):
                st.session_state.logs = []
                st.rerun()
            
            # Display logs in a scrollable container
            if st.session_state.logs:
                # Show logs in reverse order (newest first)
                logs_text = "\n".join(reversed(st.session_state.logs[-20:]))  # Show last 20 logs
                st.text_area(
                    "Recent Activity:",
                    value=logs_text,
                    height=300,
                    disabled=True,
                    key="logs_display"
                )
            else:
                st.info("No logs yet. Perform some actions to see debug information.")

def load_users():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    with open(DATA_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_people():
    if os.path.exists(PEOPLE_FILE):
        try:
            with open(PEOPLE_FILE, 'r') as f:
                data = json.load(f)
                add_log(f"Loaded {len(data)} people from {PEOPLE_FILE}")
                return data
        except Exception as e:
            add_log(f"Error loading people: {e}", "ERROR")
            # Try to restore from backup
            backup_file = PEOPLE_FILE + '.backup'
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, 'r') as f:
                        data = json.load(f)
                        add_log(f"Restored {len(data)} people from backup", "WARNING")
                        return data
                except:
                    pass
            return {}
    else:
        add_log(f"{PEOPLE_FILE} does not exist, returning empty dict")
    return {}

def save_people(people):
    try:
        # Create backup before saving
        if os.path.exists(PEOPLE_FILE):
            import shutil
            shutil.copy2(PEOPLE_FILE, PEOPLE_FILE + '.backup')
        
        with open(PEOPLE_FILE, 'w') as f:
            json.dump(people, f, indent=2)
        
        add_log(f"Saved {len(people)} people to {PEOPLE_FILE}")
        
    except Exception as e:
        add_log(f"Error saving people: {e}", "ERROR")
        st.error(f"Error saving people data: {e}")

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

@st.cache_resource
def get_face_analyzer():
    """Initialize and cache InsightFace FaceAnalysis (ArcFace embeddings)."""
    try:
        add_log("Initializing InsightFace FaceAnalysis (ArcFace embeddings)")
        # Use a lighter model pack and smaller detector size for faster CPU inference
        app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(240, 240))  # Smaller size for faster processing
        add_log("InsightFace model loaded successfully")
        return app
    except Exception as e:
        add_log(f"Error initializing InsightFace: {e}", "ERROR")
        st.error(f"Failed to initialize face recognition model: {e}")
        return None

def extract_face_features(image):
    """Extract robust 512D identity embedding using InsightFace (ArcFace)."""
    if image is None:
        add_log("No image provided for feature extraction", "WARNING")
        return None
    
    try:
        add_log("Starting ArcFace embedding extraction")
        analyzer = get_face_analyzer()
        
        if analyzer is None:
            add_log("Face analyzer not initialized", "ERROR")
            return None
        
        # Ensure image is in correct format
        if len(image.shape) != 3 or image.shape[2] != 3:
            add_log(f"Invalid image shape: {image.shape}", "ERROR")
            return None
        
        # InsightFace expects BGR image; we pass as-is
        faces = analyzer.get(image)
        
        if not faces:
            add_log("No face detected by InsightFace", "WARNING")
            return None
        
        add_log(f"Detected {len(faces)} face(s)")
        
        # Select the largest face
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        face = faces[0]
        
        if face.normed_embedding is None and face.embedding is None:
            add_log("Embedding not available from model", "ERROR")
            return None
        
        # Prefer normalized embedding
        emb = face.normed_embedding if face.normed_embedding is not None else face.embedding
        emb = np.asarray(emb, dtype=np.float32)
        
        # Ensure normalized
        norm = np.linalg.norm(emb) + 1e-7
        emb = emb / norm
        
        add_log(f"Extracted embedding of length {emb.shape[0]}")
        return [float(x) for x in emb.tolist()]
        
    except Exception as e:
        add_log(f"ArcFace embedding error: {e}", "ERROR")
        st.error(f"Error extracting face embedding: {str(e)}")
        return None

def compare_faces(features1, features2, threshold=0.35, person_name="Unknown"):
    """Compare ArcFace embeddings using cosine similarity (dot product)."""
    if not features1 or not features2:
        return False, 0
    try:
        v1 = np.asarray(features1, dtype=np.float32)
        v2 = np.asarray(features2, dtype=np.float32)
        # Normalize to be safe
        v1 = v1 / (np.linalg.norm(v1) + 1e-7)
        v2 = v2 / (np.linalg.norm(v2) + 1e-7)
        cosine_sim = float(np.dot(v1, v2))
        is_match = cosine_sim >= threshold
        add_log(f"ArcFace compare with {person_name}: Cosine={cosine_sim:.3f}, Match={is_match}")
        return is_match, max(0.0, min(1.0, cosine_sim))
    except Exception as e:
        add_log(f"Comparison error: {e}", "ERROR")
        st.error(f"Error comparing faces: {str(e)}")
        return False, 0

# Helper functions for multi-sample enrollment and matching

def best_similarity(query_emb, embeddings_list, person_name="Unknown"):
    """Return best match result and highest confidence across multiple embeddings."""
    best_conf = 0.0
    best_match = False
    for emb in embeddings_list:
        match, conf = compare_faces(query_emb, emb, person_name=person_name)
        if conf > best_conf:
            best_conf = conf
            best_match = match
    return best_match, best_conf


def embeddings_are_similar(e1, e2, threshold=0.98):
    """Check if two embeddings are essentially the same (to avoid duplicates)."""
    v1 = np.asarray(e1, dtype=np.float32)
    v2 = np.asarray(e2, dtype=np.float32)
    v1 = v1 / (np.linalg.norm(v1) + 1e-7)
    v2 = v2 / (np.linalg.norm(v2) + 1e-7)
    sim = float(np.dot(v1, v2))
    return sim >= threshold


def average_embedding(embs):
    """Compute normalized centroid of multiple embeddings."""
    arr = np.asarray(embs, dtype=np.float32)
    centroid = np.mean(arr, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-7)
    return centroid.tolist()

def capture_face():
    """Capture face using Streamlit camera input with improved error handling"""
    # Add comprehensive CSS to flip the camera preview horizontally
    st.markdown("""
    <style>
    /* Target all possible camera input elements */
    [data-testid="stCameraInput"] video {
        transform: scaleX(-1) !important;
        -webkit-transform: scaleX(-1) !important;
        -moz-transform: scaleX(-1) !important;
        -ms-transform: scaleX(-1) !important;
        -o-transform: scaleX(-1) !important;
    }
    
    [data-testid="stCameraInput"] img {
        transform: scaleX(-1) !important;
        -webkit-transform: scaleX(-1) !important;
        -moz-transform: scaleX(-1) !important;
        -ms-transform: scaleX(-1) !important;
        -o-transform: scaleX(-1) !important;
    }
    
    /* Alternative selectors in case the above don't work */
    .stCameraInput video {
        transform: scaleX(-1) !important;
    }
    
    .stCameraInput img {
        transform: scaleX(-1) !important;
    }
    
    /* More specific targeting */
    div[data-testid="stCameraInput"] > div > div > div > video {
        transform: scaleX(-1) !important;
    }
    
    div[data-testid="stCameraInput"] > div > div > div > img {
        transform: scaleX(-1) !important;
    }
    
    /* Catch-all for any video/img elements within camera input */
    [data-testid="stCameraInput"] * video {
        transform: scaleX(-1) !important;
    }
    
    [data-testid="stCameraInput"] * img {
        transform: scaleX(-1) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add instructions for better capture
    st.info("🀱Position your face clearly in the camera frame and click the capture button")
    
    try:
        camera_input = st.camera_input("📷Take a photo", key="face_camera")
        
        if camera_input is not None:
            add_log("Camera input received, processing image")
            
            try:
                bytes_data = camera_input.getvalue()
                
                if not bytes_data:
                    add_log("Empty camera data received", "WARNING")
                    st.warning("Empty image captured. Please try again.")
                    return None
                
                # Skip duplicate processing of the same captured image across reruns
                raw_sig = hashlib.md5(bytes_data).hexdigest()
                if st.session_state.get('last_camera_raw_sig') == raw_sig:
                    st.caption("⏳ Processing the same image... waiting for a new capture")
                    return None
                
                st.session_state.last_camera_raw_sig = raw_sig
                add_log(f"Processing new image with signature: {raw_sig[:8]}...")
                
                # Decode image
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                if cv2_img is None:
                    add_log("Failed to decode camera image", "ERROR")
                    st.error("Failed to process the captured image. Please try again.")
                    return None
                
                add_log(f"Image decoded successfully, shape: {cv2_img.shape}")
                
                # Since we're flipping the preview, we need to flip the captured image back
                # to match what the user saw in the preview
                cv2_img = cv2.flip(cv2_img, 1)
                
                # Show a progress indicator
                with st.spinner("🔍 Analyzing face..."):
                    features = extract_face_features(cv2_img)
                
                if features:
                    st.success("👍 Face captured and processed successfully!")
                    add_log("Face features extracted successfully")
                    return features
                else:
                    st.error("😑 No face detected in the image. Please ensure your face is clearly visible and try again.")
                    add_log("No face features extracted", "WARNING")
                    
            except Exception as e:
                add_log(f"Error processing camera input: {e}", "ERROR")
                st.error(f"Error processing image: {str(e)}")
                return None
        
    except Exception as e:
        add_log(f"Camera input error: {e}", "ERROR")
        st.error(f"Camera error: {str(e)}")
        st.info("If camera issues persist, try refreshing the page or using a different browser.")
    
    return None

def home_page():
    st.title("Welcome to Face Recognition System")
    st.write("Create your account to get started")
    
    # Show existing users and people for debugging
    users = load_users()
    people = load_people()
    st.sidebar.write(f"Debug: {len(users)} users registered")
    st.sidebar.write(f"Debug: {len(people)} people registered")
    
    with st.form("signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        name = st.text_input("Your Name")
        submitted = st.form_submit_button("Sign Up")
    
    if submitted:
        st.write(f"Form submitted with: {email}, {name}, password: {'Yes' if password else 'No'}")
        
        if not email or not password or not name:
            st.error("Please fill in all fields")
            return
            
        users = load_users()
        if email in users:
            st.error("Email already exists")
            return
        
        try:
            users[email] = {
                'password_hash': hash_password(password),
                'name': name,
                'created': datetime.now().isoformat()
            }
            save_users(users)
            st.success("Account created successfully!")
            st.info("Go to Login page to continue")
        except Exception as e:
            st.error(f"Error creating account: {e}")
    
    if st.button("Already have an account? Login"):
        st.session_state.page = 'login'
        st.rerun()

def login_page():
    st.title("Login")
    
    # Initialize session state for login step
    if 'login_step' not in st.session_state:
        st.session_state.login_step = 1
    
    if st.session_state.login_step == 1:
        # Step 1: Email/Password verification
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Verify Account")
        
        if submitted and email and password:
            users = load_users()
            if email not in users:
                st.error("Account not found")
                return
            
            if not verify_password(password, users[email]['password_hash']):
                st.error("Invalid password")
                return
            
            st.session_state.user_email = email
            st.session_state.login_step = 2
            st.rerun()
    
    elif st.session_state.login_step == 2:
        # Step 2: Face recognition
        st.success("Account verified! Now show your face to login:")
        
        face_features = capture_face()
        if face_features:
            people = load_people()
            user_people = {k: v for k, v in people.items() if v.get('owner') == st.session_state.user_email}
            
            best_match = None
            best_score = 0
            
            for person_id, person_data in user_people.items():
                enc_list = person_data.get('encodings')
                if isinstance(enc_list, list) and enc_list and isinstance(enc_list[0], list):
                    match_result, confidence = best_similarity(face_features, enc_list, person_name=person_data['name'])
                else:
                    # Fallback to single centroid/embedding
                    enc = person_data.get('encoding')
                    if not isinstance(enc, list) or len(enc) < 256:
                        add_log(f"Skipping {person_data.get('name','?')} due to incompatible embedding length", "WARNING")
                        continue
                    match_result, confidence = compare_faces(face_features, enc, person_name=person_data['name'])
                
                if match_result and confidence > best_score:
                    best_score = confidence
                    best_match = person_data['name']
            
            if best_match:
                st.session_state.authenticated = True
                st.session_state.current_user = best_match
                st.session_state.page = 'welcome'
                st.session_state.login_step = 1
                st.success(f"Welcome {best_match}! (Confidence: {best_score:.2f})")
                st.rerun()
            else:
                st.session_state.authenticated = True
                st.session_state.current_user = "Newbie"
                st.session_state.page = 'add_people'
                st.session_state.login_step = 1
                st.info("Face not recognized. You'll be redirected to add your face.")
                st.rerun()
    
    if st.button("Don't have an account? Sign Up"):
        st.session_state.page = 'home'
        st.session_state.login_step = 1
        st.rerun()

def welcome_page():
    if not st.session_state.authenticated:
        st.session_state.page = 'login'
        st.rerun()
        return
    
    current_user = st.session_state.get('current_user', 'User')
    
    st.title(f"Welcome, {current_user}!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Add People", use_container_width=True):
            st.session_state.page = 'add_people'
            st.rerun()
    
    with col2:
        if st.button("Recognize Person", use_container_width=True):
            st.session_state.page = 'recognize'
            st.rerun()
    
    # Show registered people with remove option
    people = load_people()
    user_people = {k: v for k, v in people.items() if v.get('owner') == st.session_state.user_email}
    
    if user_people:
        st.subheader("Your Registered People:")
        for person_id, person_data in user_people.items():
            col_name, col_remove = st.columns([3, 1])
            with col_name:
                st.write(f"• {person_data['name']}")
            with col_remove:
                if st.button("Remove", key=f"remove_{person_id}", help="Remove person"):
                    people = load_people()
                    del people[person_id]
                    save_people(people)
                    st.success(f"Removed {person_data['name']}")
                    st.rerun()
    else:
        st.info("No people registered yet. Click 'Add People' to get started!")

def add_people_page():
    if not st.session_state.authenticated:
        st.session_state.page = 'login'
        st.rerun()
        return
    
    st.title("Add People")
    
    # Initialize enrollment session state
    if 'person_added' not in st.session_state:
        st.session_state.person_added = False
    if 'current_person_name' not in st.session_state:
        st.session_state.current_person_name = ""
    if 'enroll_embeddings' not in st.session_state:
        st.session_state.enroll_embeddings = []
    if 'last_capture_sig' not in st.session_state:
        st.session_state.last_capture_sig = None
    
    person_name = st.text_input("Person's Name", key="person_name_input")
    
    # Reset if name changed
    if person_name != st.session_state.current_person_name:
        st.session_state.person_added = False
        st.session_state.current_person_name = person_name
        st.session_state.enroll_embeddings = []
        st.session_state.last_capture_sig = None
    
    target_samples = 5
    captured = len(st.session_state.enroll_embeddings)
    st.info(f"Capture progress: {captured}/{target_samples} samples")
    
    if person_name and not st.session_state.person_added:
        st.write(f"Please capture {person_name}'s face (take {target_samples} distinct photos):")
        emb = capture_face()
        
        if emb:
            # Prevent re-adding the same capture repeatedly on reruns
            sig = hash(tuple(np.round(emb, 4)))
            if st.session_state.last_capture_sig != sig:
                st.session_state.last_capture_sig = sig
                # Check similarity to avoid near-duplicate samples
                if any(embeddings_are_similar(emb, e) for e in st.session_state.enroll_embeddings):
                    st.warning("This photo looks too similar to a previous one. Try a different angle, expression, or lighting.")
                else:
                    st.session_state.enroll_embeddings.append(emb)
                    st.success(f"Sample {len(st.session_state.enroll_embeddings)}/{target_samples} captured.")
            else:
                st.caption("Waiting for a new capture...")
        
        # Once we have enough samples, save the person
        if len(st.session_state.enroll_embeddings) >= target_samples:
            centroid = average_embedding(st.session_state.enroll_embeddings)
            people = load_people()
            person_id = f"{st.session_state.user_email}_{person_name}_{len(people)}"
            people[person_id] = {
                'name': person_name,
                'encodings': st.session_state.enroll_embeddings,
                'encoding': centroid,  # centroid for backward compatibility
                'centroid': centroid,
                'samples': len(st.session_state.enroll_embeddings),
                'owner': st.session_state.user_email,
                'created': datetime.now().isoformat(),
                'model': 'insightface_arcface_r100',
                'embedding_size': len(centroid)
            }
            save_people(people)
            st.session_state.person_added = True
            st.success(f"{person_name} enrolled with {target_samples} samples!")
            # Reset enrollment state for next person
            st.session_state.enroll_embeddings = []
            st.session_state.last_capture_sig = None
    
    elif st.session_state.person_added:
        st.success(f"{st.session_state.current_person_name} has been added!")
        colA, colB = st.columns(2)
        with colA:
            if st.button("Add Another Person"):
                st.session_state.person_added = False
                st.session_state.current_person_name = ""
                st.session_state.enroll_embeddings = []
                st.session_state.last_capture_sig = None
                st.rerun()
        with colB:
            if st.button("Reset Enrollment"):
                st.session_state.person_added = False
                st.session_state.enroll_embeddings = []
                st.session_state.last_capture_sig = None
                st.rerun()
    
    if st.button("← Back to Welcome"):
        st.session_state.person_added = False
        st.session_state.current_person_name = ""
        st.session_state.enroll_embeddings = []
        st.session_state.last_capture_sig = None
        st.session_state.page = 'welcome'
        st.rerun()

def recognize_page():
    if not st.session_state.authenticated:
        st.session_state.page = 'login'
        st.rerun()
        return
    
    st.title("Recognize Person")
    st.write("Take a photo to identify the person")
    
    face_features = capture_face()
    if face_features:
        people = load_people()
        user_people = {k: v for k, v in people.items() if v.get('owner') == st.session_state.user_email}
        
        if not user_people:
            st.error("No people registered yet. Please add people first.")
            return
        
        best_match = None
        best_score = 0
        all_scores = []
        
        st.info("Comparing with registered faces...")
        
        for person_id, person_data in user_people.items():
            enc_list = person_data.get('encodings')
            if isinstance(enc_list, list) and enc_list and isinstance(enc_list[0], list):
                match_result, confidence = best_similarity(face_features, enc_list, person_name=person_data['name'])
            else:
                enc = person_data.get('encoding')
                if not isinstance(enc, list) or len(enc) < 256:  # Skip incompatible old encodings
                    add_log(f"Skipping {person_data.get('name','?')} due to incompatible embedding length", "WARNING")
                    continue
                match_result, confidence = compare_faces(face_features, enc, person_name=person_data['name'])
            all_scores.append((person_data['name'], confidence, match_result))
            
            if match_result and confidence > best_score:
                best_score = confidence
                best_match = person_data['name']
        
        # Show all comparison scores for debugging
        st.subheader("Recognition Scores:")
        for name, score, is_match in sorted(all_scores, key=lambda x: x[1], reverse=True):
            status = "MATCH!" if is_match else " No match!!"
            st.write(f"**{name}**: {score:.3f} {status}")
        
        if best_match:
            st.success(f" **{best_match}** recognized! (Confidence: {best_score:.3f})")
        else:
            st.error("Person not recognized. Highest score was too low for a match.")
            st.info("Try taking another photo with better lighting or a clearer view of your face.")
    
    if st.button("← Back to Welcome"):
        st.session_state.page = 'welcome'
        st.rerun()

def main():
    st.set_page_config(page_title="Face Recognition System", layout="wide")
    
    # Add global CSS to flip camera preview
    st.markdown("""
    <style>
    /* Global CSS to flip all camera inputs */
    video {
        transform: scaleX(-1) !important;
    }
    
    /* More specific targeting for Streamlit camera */
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] img,
    .stCameraInput video,
    .stCameraInput img {
        transform: scaleX(-1) !important;
        -webkit-transform: scaleX(-1) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display logs in sidebar
    display_logs_sidebar()
    
    # Logout button in sidebar if authenticated
    if st.session_state.authenticated:
        with st.sidebar:
            if st.button("Logout"):
                add_log(f"User {st.session_state.user_email} logged out")
                st.session_state.authenticated = False
                st.session_state.user_email = None
                st.session_state.page = 'home'
                st.rerun()
    
    # Page routing
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'login':
        login_page()
    elif st.session_state.page == 'welcome':
        welcome_page()
    elif st.session_state.page == 'add_people':
        add_people_page()
    elif st.session_state.page == 'recognize':
        recognize_page()

if __name__ == "__main__":
    main()
