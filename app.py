import streamlit as st
import cv2
import numpy as np
import json
import bcrypt
import os
from datetime import datetime

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'page' not in st.session_state:
    st.session_state.page = 'home'

DATA_FILE = 'users.json'
PEOPLE_FILE = 'people.json'

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
                return json.load(f)
        except:
            return {}
    return {}

def save_people(people):
    with open(PEOPLE_FILE, 'w') as f:
        json.dump(people, f, indent=2)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def extract_face_features(image):
    """Extract simple face features using OpenCV"""
    if image is None:
        return None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (64, 64))
        return face_resized.flatten().tolist()
    return None

def compare_faces(features1, features2, threshold=0.7):
    """Simple face comparison using correlation"""
    if not features1 or not features2:
        return False
    
    arr1 = np.array(features1)
    arr2 = np.array(features2)
    
    # Normalize arrays
    arr1 = arr1 / np.linalg.norm(arr1)
    arr2 = arr2 / np.linalg.norm(arr2)
    
    # Calculate correlation
    correlation = np.corrcoef(arr1, arr2)[0, 1]
    return correlation > threshold

def capture_face():
    """Capture face using Streamlit camera input"""
    camera_input = st.camera_input("📸 Take a photo")
    
    if camera_input is not None:
        bytes_data = camera_input.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        features = extract_face_features(cv2_img)
        if features:
            st.success("Face captured successfully!")
            return features
        else:
            st.error("No face detected. Please try again.")
    
    return None

def home_page():
    st.title("👋 Welcome to Face Recognition System")
    st.write("Create your account to get started")
    
    # Show existing users for debugging
    users = load_users()
    st.sidebar.write(f"Debug: {len(users)} users registered")
    
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
            st.success("✅ Account created successfully!")
            st.info("Go to Login page to continue")
        except Exception as e:
            st.error(f"Error creating account: {e}")
    
    if st.button("Already have an account? Login"):
        st.session_state.page = 'login'
        st.rerun()

def login_page():
    st.title("🚪 Login")
    
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
        st.success("✅ Account verified! Now show your face to login:")
        
        face_features = capture_face()
        if face_features:
            people = load_people()
            user_people = {k: v for k, v in people.items() if v.get('owner') == st.session_state.user_email}
            
            best_match = None
            best_score = 0
            
            for person_id, person_data in user_people.items():
                if compare_faces(face_features, person_data['encoding']):
                    arr1 = np.array(face_features)
                    arr2 = np.array(person_data['encoding'])
                    arr1 = arr1 / np.linalg.norm(arr1)
                    arr2 = arr2 / np.linalg.norm(arr2)
                    score = np.corrcoef(arr1, arr2)[0, 1]
                    
                    if score > best_score:
                        best_score = score
                        best_match = person_data['name']
            
            if best_match:
                st.session_state.authenticated = True
                st.session_state.current_user = best_match
                st.session_state.page = 'welcome'
                st.session_state.login_step = 1
                st.rerun()
            else:
                st.session_state.authenticated = True
                st.session_state.current_user = "Newbie"
                st.session_state.page = 'add_people'
                st.session_state.login_step = 1
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
    
    st.title(f"👋 Welcome, {current_user}!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ Add People", use_container_width=True):
            st.session_state.page = 'add_people'
            st.rerun()
    
    with col2:
        if st.button("🔍 Recognize Person", use_container_width=True):
            st.session_state.page = 'recognize'
            st.rerun()
    
    # Show registered people
    people = load_people()
    user_people = {k: v for k, v in people.items() if v.get('owner') == st.session_state.user_email}
    
    if user_people:
        st.subheader("Your Registered People:")
        for person_id, person_data in user_people.items():
            st.write(f"• {person_data['name']}")
    else:
        st.info("No people registered yet. Click 'Add People' to get started!")

def add_people_page():
    if not st.session_state.authenticated:
        st.session_state.page = 'login'
        st.rerun()
        return
    
    st.title("➕ Add People")
    
    # Initialize session state for preventing duplicates
    if 'person_added' not in st.session_state:
        st.session_state.person_added = False
    if 'current_person_name' not in st.session_state:
        st.session_state.current_person_name = ""
    
    person_name = st.text_input("Person's Name", key="person_name_input")
    
    # Reset if name changed
    if person_name != st.session_state.current_person_name:
        st.session_state.person_added = False
        st.session_state.current_person_name = person_name
    
    if person_name and not st.session_state.person_added:
        st.write(f"Please capture {person_name}'s face:")
        face_features = capture_face()
        
        if face_features:
            people = load_people()
            person_id = f"{st.session_state.user_email}_{person_name}_{len(people)}"
            
            people[person_id] = {
                'name': person_name,
                'encoding': face_features,
                'owner': st.session_state.user_email,
                'created': datetime.now().isoformat()
            }
            save_people(people)
            st.session_state.person_added = True
            st.success(f"✅ {person_name} added successfully!")
    
    elif st.session_state.person_added:
        st.success(f"✅ {st.session_state.current_person_name} has been added!")
        if st.button("Add Another Person"):
            st.session_state.person_added = False
            st.session_state.current_person_name = ""
            st.rerun()
    
    if st.button("← Back to Welcome"):
        st.session_state.person_added = False
        st.session_state.current_person_name = ""
        st.session_state.page = 'welcome'
        st.rerun()

def recognize_page():
    if not st.session_state.authenticated:
        st.session_state.page = 'login'
        st.rerun()
        return
    
    st.title("🔍 Recognize Person")
    st.write("Take a photo to identify the person")
    
    face_features = capture_face()
    if face_features:
        people = load_people()
        user_people = {k: v for k, v in people.items() if v.get('owner') == st.session_state.user_email}
        
        best_match = None
        best_score = 0
        
        for person_id, person_data in user_people.items():
            if compare_faces(face_features, person_data['encoding']):
                # Calculate similarity score
                arr1 = np.array(face_features)
                arr2 = np.array(person_data['encoding'])
                arr1 = arr1 / np.linalg.norm(arr1)
                arr2 = arr2 / np.linalg.norm(arr2)
                score = np.corrcoef(arr1, arr2)[0, 1]
                
                if score > best_score:
                    best_score = score
                    best_match = person_data['name']
        
        if best_match:
            st.success(f"✅ **{best_match}** recognized! (Confidence: {best_score:.2f})")
        else:
            st.error("❌ Person not recognized. Please add them first.")
    
    if st.button("← Back to Welcome"):
        st.session_state.page = 'welcome'
        st.rerun()

def main():
    st.set_page_config(page_title="Face Recognition System", page_icon="🔍", layout="wide")
    
    # Logout button in sidebar if authenticated
    if st.session_state.authenticated:
        with st.sidebar:
            if st.button("🚪 Logout"):
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