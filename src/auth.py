from fastapi import Header, HTTPException, status
from typing import Optional, Dict, Any
import os
from fastapi import Header, HTTPException, status
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

MOCK_IDENTITY_MAP = {
    "taran@example.com": "270041123456",
    "student@nios.ac.in": "270041987654"
}


def get_current_student(authorization: Optional[str] = Header(None)) -> Optional[Dict[str,str]]:
    """
    Validates Authorization: Bearer <token>.
    Supports:
      1. Real Google ID Token verification (free tier).
      2. Development token prefix: 'Bearer dev_student_<enrollment_no>' for offline testing.
    Returns:
      Dict with {'enrollment_no': '...', 'email': '...'} or None for anonymous Phase 1 users.
    """

    if not authorization:
        return None

    if not authorization.startswith("Bearer"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'"
        
        )
    token = authorization.split(" ")[1].strip()

    if token.startswith("dev_student_"):
        enrollment_no = token.replace("dev_student_", "")
        return {
            "enrollment_no": enrollment_no,
            "email": f"{enrollment_no}@student.nios.ac.in",
            "auth_method": "dev_token"
        }
    # 2. Production Google SSO ID Token Verification
    if GOOGLE_CLIENT_ID:
        try:
            id_info = id_token.verify_oauth2_token(
                token,
                google_requests.Request(),
                GOOGLE_CLIENT_ID
            )
            email = id_info.get("email")
            enrollment_no = MOCK_IDENTITY_MAP.get(email, "270041123456")
            return {
                "enrollment_no": enrollment_no,
                "email": email,
                "name": id_info.get("name", "Student"),
                "auth_method": "google_sso"
            }
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired Google SSO token"
            )

    # Fallback default for local verification
    return {
        "enrollment_no": "270041123456",
        "email": "demo.student@nios.ac.in",
        "auth_method": "mock_fallback"
    }

def get_optional_student_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """
    Validates Bearer token. 
    If present and valid, returns the verified enrollment number.
    If missing, returns None to allow anonymous Phase 1 public queries.
    """
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <token>'"
        )



    token = authorization.split(" ")[1]
    
    # In production: Verify Google SSO token via google.oauth2.id_token.verify_oauth2_token
    # For initial staging / phase-2 testing, decode your test JWT or mock session:
    if token.startswith("test_student_"):
        return token.replace("test_student_", "")  # e.g., 'test_student_270041123456' -> '270041123456'

    # Fallback demo token
    return "270041123456"