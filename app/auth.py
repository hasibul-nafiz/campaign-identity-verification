from __future__ import annotations

import hmac
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings

JWT_ALGORITHM = "HS256"

bearer_scheme = HTTPBearer(auto_error=False)


class AuthError(Exception):
    pass


def authenticate(username: str, password: str, cfg: Settings) -> bool:
    return hmac.compare_digest(username, cfg.auth_username) and hmac.compare_digest(
        password, cfg.auth_password
    )


def create_access_token(username: str, cfg: Settings) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + timedelta(minutes=cfg.jwt_expire_minutes),
    }
    return jwt.encode(payload, cfg.jwt_secret, algorithm=JWT_ALGORITHM)


def decode_token(token: str, cfg: Settings) -> str:
    try:
        payload = jwt.decode(token, cfg.jwt_secret, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise AuthError("invalid or expired token") from exc
    return payload["sub"]


def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    cfg: Settings = request.app.state.settings
    if credentials is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        return decode_token(credentials.credentials, cfg)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
