# Face recognition mimicing FaceID

Multi-user face authentication system where each account can have multiple Users.

## Features
- **Account Creation**: Create family accounts with owner registration
- **Member Management**: Add family or friends to accounts
- **Two-Step Login**: Password unlocks account and the face identifies specific member
- **Secure Storage**: Hashed passwords, face embeddings only

## Data Structure
```json
{
  "family_account": {
    "password_hash": "xxx",
    "members": [
      {"name": "XX1", "encoding": [...], "is_owner": true},
      {"name": "XX2", "encoding": [...], "is_owner": false},
      {"name": "XX3", "encoding": [...], "is_owner": false}
    ]
  }
}
```

## Usage
1. **Create Account**: Account name + password + owner face
2. **Add Members**: Login → Manage Members → Add family or friends
3. **Login**: Enter password → show face → get identified as specific member

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
1. Copy this to your Github
2. Sign into your Streamlit
3. Deploy the app.py from the github repo on you Streamlit

## Disclaimer
This is for Educational and Learning purpose only
