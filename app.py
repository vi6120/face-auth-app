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
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_member' not in st.session_state:
    st.session_state.current_member = None

DATA_FILE = 'users.json'

def load_users():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(DATA_FILE, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def extract_face_features(image):
    """Extract simple face features using OpenCV"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (64, 64))
        return face_resized.flatten().tolist()
    return None

def compare_faces(features1, features2, threshold=0.8):
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
        # Convert to OpenCV format
        bytes_data = camera_input.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        features = extract_face_features(cv2_img)
        if features:
            st.success("Face captured successfully!")
            return features
        else:
            st.error("No face detected. Please try again.")
    
    return None

def signup():
    st.header("🔐 Create Account")
    
    with st.form("signup_form"):
        username = st.text_input("Account Name")
        password = st.text_input("Password", type="password")
        owner_name = st.text_input("Your Name (Account Owner)")
        submitted = st.form_submit_button("Create Account")
    
    if submitted and username and password and owner_name:
        users = load_users()
        if username in users:
            st.error("Account already exists")
            return
        
        st.write(f"Please capture {owner_name}'s face:")
        face_features = capture_face()
        
        if face_features:
            users[username] = {
                'password_hash': hash_password(password),
                'members': [{
                    'name': owner_name,
                    'encoding': face_features,
                    'is_owner': True
                }],
                'created': datetime.now().isoformat()
            }
            save_users(users)
            st.success(f"✅ Account created! {owner_name} registered as owner.")

def login():
    st.header("🚪 Account Login")
    
    with st.form("login_form"):
        username = st.text_input("Account Name")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock Account")
    
    if submitted and username and password:
        users = load_users()
        if username not in users:
            st.error("Account not found")
            return
        
        if not verify_password(password, users[username]['password_hash']):
            st.error("Invalid password")
            return
        
        st.session_state.authenticated = True
        st.session_state.username = username
        st.success("✅ Account unlocked! Show your face to identify yourself.")
        st.rerun()

def identify_member():
    if not st.session_state.authenticated:
        st.error("Please login first")
        return
    
    st.header("👤 Who Are You?")
    st.write("Show your face to identify yourself from the registered members")
    
    face_features = capture_face()
    if face_features:
        users = load_users()
        account = users[st.session_state.username]
        
        for member in account['members']:
            if compare_faces(face_features, member['encoding']):
                st.session_state.current_member = member['name']
                st.success(f"✅ Welcome, **{member['name']}** (under {st.session_state.username}'s account)!")
                st.rerun()
                return
        
        st.error("❌ Face not recognized. Please ask the account owner to add you.")

def manage_members():
    if not st.session_state.authenticated:
        return
    
    st.header("👥 Manage Family/Friends")
    users = load_users()
    account = users[st.session_state.username]
    
    # Show current members
    st.subheader("Current Members:")
    for member in account['members']:
        owner_tag = " (Owner)" if member.get('is_owner') else ""
        st.write(f"• {member['name']}{owner_tag}")
    
    # Add new member
    st.subheader("Add New Member:")
    with st.form("add_member"):
        member_name = st.text_input("Member Name (e.g., Dad, Mom, Sister)")
        submitted = st.form_submit_button("Add Member")
    
    if submitted and member_name:
        # Check if name already exists
        existing_names = [m['name'].lower() for m in account['members']]
        if member_name.lower() in existing_names:
            st.error("Member name already exists")
            return
        
        st.write(f"Please capture {member_name}'s face:")
        face_features = capture_face()
        
        if face_features:
            account['members'].append({
                'name': member_name,
                'encoding': face_features,
                'is_owner': False
            })
            users[st.session_state.username] = account
            save_users(users)
            st.success(f"✅ {member_name} added successfully!")
            st.rerun()

def main():
    st.set_page_config(page_title="Family Face Auth", page_icon="👨👩👧👦", layout="wide")
    st.title("👨👩👧👦 Family Face Recognition")
    
    if st.session_state.authenticated:
        # Account is unlocked
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚪 Logout"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.current_member = None
                st.rerun()
        
        if st.session_state.current_member:
            # Member identified
            st.success(f"Logged in as: **{st.session_state.current_member}** (Account: {st.session_state.username})")
            st.write("🎉 Access granted! You can now use the application.")
        else:
            # Account unlocked but member not identified
            st.info(f"Account **{st.session_state.username}** is unlocked")
            
            tab1, tab2 = st.tabs(["👤 Identify Yourself", "👥 Manage Members"])
            
            with tab1:
                identify_member()
            
            with tab2:
                manage_members()
        
        return
    
    # Not authenticated
    tab1, tab2 = st.tabs(["📝 Create Account", "🚪 Login"])
    
    with tab1:
        signup()
    
    with tab2:
        login()

if __name__ == "__main__":
    main()