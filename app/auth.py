"""Users with argon2 hashes. HTTP Basic for the JSON API; a signed session cookie for pages."""

import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import User

_hasher = PasswordHasher()
_DUMMY_HASH = _hasher.hash("timing-equaliser")
_basic = HTTPBasic(auto_error=False)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def authenticate(s: Session, username: str, password: str) -> User | None:
    user = s.scalar(select(User).where(User.username == username))
    try:
        _hasher.verify(user.password_hash if user else _DUMMY_HASH, password)
    except (VerificationError, InvalidHashError):
        return None
    return user


def create_user(s: Session, username: str, password: str, is_admin: bool = False) -> User:
    username = username.strip()
    if not username or len(password) < 8:
        raise ValueError("username required and password must be at least 8 characters")
    if s.scalar(select(User).where(User.username == username)):
        raise ValueError(f"user '{username}' already exists")
    user = User(username=username, password_hash=hash_password(password), is_admin=is_admin)
    s.add(user)
    s.commit()
    return user


def bootstrap_admin(s: Session, username: str | None, password: str | None) -> bool:
    """Create the first admin from APP_USERNAME / APP_PASSWORD if there are no users yet."""
    if not username or not password or s.scalar(select(func.count()).select_from(User)):
        return False
    try:
        create_user(s, username, password, is_admin=True)
    except (IntegrityError, ValueError):  # another web worker created it first
        s.rollback()
        return False
    return True


# ---------- API: HTTP Basic ----------


def api_user(
    creds: HTTPBasicCredentials | None = Depends(_basic), s: Session = Depends(get_session)
) -> User:
    user = authenticate(s, creds.username, creds.password) if creds else None
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user


# ---------- pages: session cookie ----------


class LoginRequired(Exception):
    pass


def page_user(request: Request, s: Session = Depends(get_session)) -> User:
    uid = request.session.get("uid")
    user = s.get(User, uid) if uid else None
    if user is None:
        raise LoginRequired
    return user


def admin_user(user: User = Depends(page_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return user


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = request.session["csrf"] = secrets.token_urlsafe(32)
    return token


def check_csrf(request: Request, token: str) -> None:
    expected = request.session.get("csrf")
    if not expected or not secrets.compare_digest(expected, token or ""):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "bad CSRF token")
