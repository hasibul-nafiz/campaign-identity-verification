from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.auth import AuthError, authenticate, create_access_token, decode_token
from app.badge import BadgeMatcher
from app.config import Settings
from app.main import create_app
from scripts.make_badge_ref import make_badge
from tests.conftest import auth_header
from tests.test_api import FakeEngine


@pytest.fixture
def cfg(monkeypatch, tmp_path) -> Settings:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "auth.db"))
    monkeypatch.setenv("BADGE_REF_PATH", str(tmp_path / "missing.png"))
    monkeypatch.setenv("CAMPAIGN_ASSETS_DIR", str(tmp_path / "campaigns"))
    monkeypatch.setenv("AUTH_USERNAME", "admin")
    monkeypatch.setenv("AUTH_PASSWORD", "s3cret")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-thats-long-enough")
    monkeypatch.setenv("JWT_EXPIRE_MINUTES", "60")
    return Settings.from_env()


@pytest.fixture
def client(cfg: Settings):
    app = create_app(cfg, engine=FakeEngine(), badge=BadgeMatcher(make_badge(), cfg))
    with TestClient(app) as c:
        yield c


class TestAuthenticate:
    def test_correct_credentials(self, cfg):
        assert authenticate("admin", "s3cret", cfg) is True

    def test_wrong_password(self, cfg):
        assert authenticate("admin", "nope", cfg) is False

    def test_wrong_username(self, cfg):
        assert authenticate("nope", "s3cret", cfg) is False


class TestToken:
    def test_round_trips_the_subject(self, cfg):
        token = create_access_token("admin", cfg)
        assert decode_token(token, cfg) == "admin"

    def test_expired_token_is_rejected(self, cfg, monkeypatch):
        monkeypatch.setenv("JWT_EXPIRE_MINUTES", "0")
        expiring_cfg = Settings.from_env()
        token = create_access_token("admin", expiring_cfg)
        time.sleep(1)
        with pytest.raises(AuthError):
            decode_token(token, expiring_cfg)

    def test_wrong_secret_is_rejected(self, cfg):
        token = create_access_token("admin", cfg)
        other = Settings.from_env()  # same env, but tamper the secret directly
        tampered = other.__class__(
            **{**other.__dict__, "jwt_secret": "a-completely-different-secret-value"}
        )
        with pytest.raises(AuthError):
            decode_token(token, tampered)


class TestLoginEndpoint:
    def test_correct_credentials_return_a_token(self, client):
        res = client.post("/login", data={"username": "admin", "password": "s3cret"})
        assert res.status_code == 200
        body = res.json()
        assert body["token_type"] == "bearer"
        assert body["access_token"]

    def test_wrong_password_is_401(self, client):
        res = client.post("/login", data={"username": "admin", "password": "wrong"})
        assert res.status_code == 401

    def test_token_grants_access_to_a_protected_route(self, client):
        token = client.post("/login", data={"username": "admin", "password": "s3cret"}).json()[
            "access_token"
        ]
        res = client.get("/people", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200


class TestProtectedRoutes:
    def test_missing_token_is_401(self, client):
        assert client.get("/people").status_code == 401

    def test_bad_token_is_401(self, client):
        res = client.get("/people", headers={"Authorization": "Bearer garbage"})
        assert res.status_code == 401

    def test_valid_token_is_allowed(self, client, cfg):
        res = client.get("/people", headers=auth_header(cfg))
        assert res.status_code == 200

    def test_health_needs_no_token(self, client):
        assert client.get("/health").status_code == 200
