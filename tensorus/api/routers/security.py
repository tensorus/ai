"""
Security router for Tensorus API.

This router handles authentication and authorization.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel

# Create router
router = APIRouter()

# Initialize password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize HTTP bearer scheme
security = HTTPBearer()

# Secret key for JWT
# In production, this should be an environment variable
SECRET_KEY = "tensorus_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_at: datetime


class User(BaseModel):
    """User model."""
    username: str
    password: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    """User model stored in database."""
    hashed_password: str


# Simulated user database - in a real application, this would be a database
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@tensorus.ai",
        "hashed_password": pwd_context.hash("password"),
        "disabled": False,
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.
    
    Args:
        plain_password: Plain password.
        hashed_password: Hashed password.
        
    Returns:
        True if the password matches the hash.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Get the hash of a password.
    
    Args:
        password: Password to hash.
        
    Returns:
        Hashed password.
    """
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get a user from the database.
    
    Args:
        username: Username of the user.
        
    Returns:
        User if found, None otherwise.
    """
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user.
    
    Args:
        username: Username of the user.
        password: Password of the user.
        
    Returns:
        User if authentication succeeds, None otherwise.
    """
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: Dict[str, str], expires_delta: Optional[timedelta] = None) -> str:
    """Create an access token.
    
    Args:
        data: Data to encode in the token.
        expires_delta: Expiration time.
        
    Returns:
        Access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserInDB:
    """Get the current user.
    
    Args:
        credentials: HTTP authorization credentials.
        
    Returns:
        Current user.
        
    Raises:
        HTTPException: If the token is invalid.
    """
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = {"sub": username}
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user(username=username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user),
) -> UserInDB:
    """Get the current active user.
    
    Args:
        current_user: Current user.
        
    Returns:
        Current active user.
        
    Raises:
        HTTPException: If the user is disabled.
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


class LoginCredentials(BaseModel):
    """Login credentials."""
    username: str
    password: str


@router.post("/token", response_model=Token)
async def login_for_access_token(credentials: LoginCredentials) -> Token:
    """Login for access token.
    
    Args:
        credentials: Login credentials.
        
    Returns:
        Access token.
        
    Raises:
        HTTPException: If the credentials are invalid.
    """
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    expires_at = datetime.utcnow() + access_token_expires
    return Token(access_token=access_token, token_type="bearer", expires_at=expires_at)


@router.get("/users/me", response_model=User)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)) -> UserInDB:
    """Get the current user.
    
    Args:
        current_user: Current user.
        
    Returns:
        Current user.
    """
    return current_user 