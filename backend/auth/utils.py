from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os

load_dotenv()

secretkey = os.getenv("SECRET_KEY")

ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Hash password
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# Verify password
def verify_password(currpassword: str, hashedpassword: str) -> bool:
    return pwd_context.verify(currpassword, hashedpassword)

# Create JWT token
def create_access_token(data: dict, expires_minutes: int = 60):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)  # ✅ timezone-aware
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secretkey, algorithm=ALGORITHM)
