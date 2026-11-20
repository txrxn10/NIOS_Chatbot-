import os
from typing import Optional, Dict
from fastapi import Header, HTTPException, status
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

MOCK_IDENTITY_MAP = {
    "taran@example.com": "270041123456",
    "student@nios.ac.in": "270041987654",
}


def get_current_student(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, str]]:
    """
    Validates Authorization: Bearer <token>.
    Supports:
      1. Real Google OAuth2 ID Token verification via Google's public certs.
      2. Development token prefix ('Bearer dev_student_<enrollment_no>') for offline/CI testing.
    Returns:
      Dict with student identity metadata, or None for unauthenticated Phase 1 public users.
    """
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'",
        )

    token = authorization.split(" ", 1)[1].strip()

    # 1. Local / CI testing bypass token
    if token.startswith("dev_student_"):
        enrollment_no = token.replace("dev_student_", "")
        return {
            "enrollment_no": enrollment_no,
            "email": f"{enrollment_no}@student.nios.ac.in",
            "auth_method": "dev_token",
        }

    # 2. Production Google SSO ID Token Verification
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google SSO is not configured on this server",
        )

    try:
        id_info = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
        email = id_info.get("email")
        enrollment_no = MOCK_IDENTITY_MAP.get(email, "270041123456")
        return {
            "enrollment_no": enrollment_no,
            "email": email,
            "name": id_info.get("name", "Student"),
            "auth_method": "google_sso",
        }
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Google SSO token",
        )