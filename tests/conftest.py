from __future__ import annotations

import numpy as np
import pytest

from app.auth import create_access_token
from app.config import Settings
from app.face import Face


@pytest.fixture
def settings() -> Settings:
    return Settings.from_env()


def auth_header(cfg: Settings) -> dict:
    return {"Authorization": f"Bearer {create_access_token(cfg.auth_username, cfg)}"}


@pytest.fixture
def models_available(settings: Settings) -> bool:
    return settings.yunet_path.exists() and settings.sface_path.exists()


def make_face(
    nose: tuple = (60.0, 77.5),
    right_eye: tuple = (40.0, 50.0),
    left_eye: tuple = (80.0, 50.0),
    right_mouth: tuple = (48.0, 100.0),
    left_mouth: tuple = (72.0, 100.0),
    box: tuple = (20, 20, 80, 100),
    score: float = 0.99,
) -> Face:
    landmarks = np.array(
        [right_eye, left_eye, nose, right_mouth, left_mouth], dtype=np.float32
    )
    return Face(x=box[0], y=box[1], w=box[2], h=box[3], score=score, landmarks=landmarks)


def rotate(face: Face, degrees: float) -> Face:
    theta = np.deg2rad(degrees)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    pivot = face.landmarks.mean(axis=0)
    rotated = (face.landmarks - pivot) @ rot.T + pivot
    return Face(
        x=face.x, y=face.y, w=face.w, h=face.h, score=face.score,
        landmarks=rotated.astype(np.float32),
    )


def unit_vector(index: int, dim: int = 128) -> np.ndarray:
    """One-hot, so dot products between distinct indices are exactly 0.0."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[index] = 1.0
    return vec


def solid(color_rgb: tuple, size: int = 128) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = color_rgb[::-1]
    return img
