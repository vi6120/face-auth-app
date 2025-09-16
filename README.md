# Family Face Recognition App

Multi-user face authentication system where each account can have multiple family/friend members.

## Features
- **Account Creation**: Create family accounts with owner registration
- **Member Management**: Add family/friends to accounts
- **Two-Step Login**: Password unlocks account → face identifies specific member
- **Secure Storage**: Hashed passwords, face embeddings only

## Data Structure
```json
{
  "family_account": {
    "password_hash": "xxx",
    "members": [
      {"name": "Dad", "encoding": [...], "is_owner": true},
      {"name": "Mom", "encoding": [...], "is_owner": false},
      {"name": "Sister", "encoding": [...], "is_owner": false}
    ]
  }
}
```

## Usage
1. **Create Account**: Account name + password + owner face
2. **Add Members**: Login → Manage Members → Add family/friends
3. **Login**: Enter password → show face → get identified as specific member

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Push to GitHub → Connect Streamlit Community Cloud → Deploy