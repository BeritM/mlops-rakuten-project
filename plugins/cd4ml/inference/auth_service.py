### --- auth_api.py ---
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from hashlib import sha256
from datetime import datetime, timedelta
from typing import Dict

# --- FastAPI Setup ---
auth_app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- JWT Configuration ---
SECRET_KEY = "your_secret_key_here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Simple In-Memory User Database ---
users_db: Dict[str, Dict[str, str]] = {
    "admin": {"username": "admin", "password": sha256("admin123".encode()).hexdigest(), "role": "admin"},
    "user": {"username": "user", "password": sha256("user123".encode()).hexdigest(), "role": "user"},
}

# --- JWT Utilities ---
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in users_db:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return users_db[username]
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# --- Models ---
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

# --- Auth Endpoints ---
@auth_app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    hashed_pw = sha256(form_data.password.encode()).hexdigest()
    if not user or user["password"] != hashed_pw:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- Admin-only Endpoints ---
def admin_required(user=Depends(verify_token)):
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admins only")
    return user

@auth_app.post("/users")
def create_user(user_data: UserCreate, user=Depends(admin_required)):
    if user_data.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    users_db[user_data.username] = {
        "username": user_data.username,
        "password": sha256(user_data.password.encode()).hexdigest(),
        "role": user_data.role
    }
    return {"msg": "User created successfully."}

@auth_app.delete("/users/{username}")
def delete_user(username: str, user=Depends(admin_required)):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del users_db[username]
    return {"msg": "User deleted successfully."}

# --- Health Check Endpoint ---
@auth_app.get("/health")
def health_check():
    return {"status": "ok"}