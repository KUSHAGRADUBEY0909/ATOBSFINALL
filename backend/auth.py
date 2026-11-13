"""Password hashing, JWT session cookies and role guards."""
import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path

import jwt
from fastapi import Depends, HTTPException, Request

from . import db

COOKIE = "atobs_session"
TOKEN_TTL = 60 * 60 * 24 * 7

_secret_file = Path(__file__).resolve().parent.parent / "data" / ".secret_key"


def _secret() -> str:
    env = os.environ.get("ATOBS_SECRET")
    if env:
        return env
    _secret_file.parent.mkdir(parents=True, exist_ok=True)
    if not _secret_file.exists():
        _secret_file.write_text(secrets.token_hex(32))
    return _secret_file.read_text().strip()


SECRET = _secret()


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)
    return f"pbkdf2_sha256$200000${salt.hex()}${dk.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, iters, salt, digest = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(dk.hex(), digest)
    except Exception:
        return False


def make_token(user: dict) -> str:
    now = int(time.time())
    return jwt.encode({"sub": str(user["id"]), "role": user["role"], "iat": now, "exp": now + TOKEN_TTL},
                      SECRET, algorithm="HS256")


def current_user_optional(request: Request) -> dict | None:
    token = request.cookies.get(COOKIE)
    if not token:
        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else None
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    with db.get_db() as conn:
        return db.row(conn.execute("SELECT id,name,email,phone,role,created_at FROM users WHERE id=?",
                                   (int(payload["sub"]),)))


def current_user(user: dict | None = Depends(current_user_optional)) -> dict:
    if not user:
        raise HTTPException(401, "Please log in")
    return user


def require_role(*roles):
    def dep(user: dict = Depends(current_user)):
        if user["role"] not in roles:
            raise HTTPException(403, "You don't have access to this")
        return user
    return dep
