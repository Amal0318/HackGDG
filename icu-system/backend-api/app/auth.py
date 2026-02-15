"""
Authentication module with JWT tokens
Integrated with Requestly API for request monitoring
"""
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel
from .config import settings

# Security scheme
security = HTTPBearer()

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class User(BaseModel):
    username: str
    role: str
    full_name: str

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate user with username and password"""
    user = settings.DEMO_USERS.get(username)
    if not user:
        return None
    if user["password"] != password:  # In production, use password hashing
        return None
    return user

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        
        if username is None:
            raise credentials_exception
            
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception
    
    user = settings.DEMO_USERS.get(token_data.username)
    if user is None:
        raise credentials_exception
    
    return User(
        username=user["username"],
        role=user["role"],
        full_name=user["full_name"]
    )

def require_role(required_roles: list):
    """Decorator to require specific roles"""
    async def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {required_roles}"
            )
        return current_user
    return role_checker
