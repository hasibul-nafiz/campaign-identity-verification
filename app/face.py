from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import Settings, get_settings


class FaceError(ValueError):
    """Base for user-correctable face problems (maps to 4xx)."""


class NoFaceError(FaceError):
    pass


class MultipleFacesError(FaceError):
    pass


@dataclass(frozen=True)
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float
    landmarks: np.ndarray

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    @property
    def right_eye(self) -> np.ndarray:
        return self.landmarks[0]

    @property
    def left_eye(self) -> np.ndarray:
        return self.landmarks[1]

    @property
    def nose(self) -> np.ndarray:
        return self.landmarks[2]

    @property
    def right_mouth(self) -> np.ndarray:
        return self.landmarks[3]

    @property
    def left_mouth(self) -> np.ndarray:
        return self.landmarks[4]

    def to_row(self) -> np.ndarray:
        row = np.empty(15, dtype=np.float32)
        row[:4] = (self.x, self.y, self.w, self.h)
        row[4:14] = self.landmarks.reshape(-1)
        row[14] = self.score
        return row.reshape(1, 15)


@dataclass(frozen=True)
class PoseResult:
    label: str
    yaw: float
    pitch: float


def downscale(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(
        img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
    )
    return resized, scale


def pose_label(face: Face, settings: Optional[Settings] = None) -> PoseResult:
    cfg = settings or get_settings()
    right_eye = face.right_eye.astype(np.float64)
    left_eye = face.left_eye.astype(np.float64)
    nose = face.nose.astype(np.float64)
    mouth_mid = (face.right_mouth.astype(np.float64) + face.left_mouth.astype(np.float64)) / 2.0

    eye_mid = (right_eye + left_eye) / 2.0
    eye_axis = left_eye - right_eye
    eye_dist = float(np.linalg.norm(eye_axis))
    vertical = mouth_mid - eye_mid
    vertical_dist = float(np.linalg.norm(vertical))
    if eye_dist < 1e-6 or vertical_dist < 1e-6:
        return PoseResult("unknown", 0.0, 0.0)

    offset = nose - eye_mid
    # Projecting onto the eye/mouth axes rather than raw x/y keeps this stable under head roll.
    yaw = float(offset @ (eye_axis / eye_dist)) / eye_dist
    t = float(offset @ (vertical / vertical_dist)) / vertical_dist
    # A frontal nose tip sits partway down the eye-to-mouth span, not at its midpoint.
    pitch = t - cfg.pose_pitch_neutral

    yaw_ratio = abs(yaw) / cfg.pose_yaw_thresh if cfg.pose_yaw_thresh > 0 else 0.0
    pitch_ratio = abs(pitch) / cfg.pose_pitch_thresh if cfg.pose_pitch_thresh > 0 else 0.0

    if yaw_ratio <= 1.0 and pitch_ratio <= 1.0:
        label = "front"
    elif yaw_ratio >= pitch_ratio:
        label = "left" if yaw > 0 else "right"
    else:
        label = "up" if pitch > 0 else "down"
    return PoseResult(label, yaw, pitch)


def torso_roi(
    face: Face, shape: Tuple[int, ...], settings: Optional[Settings] = None
) -> Tuple[int, int, int, int]:
    cfg = settings or get_settings()
    img_h, img_w = shape[:2]
    cx, _ = face.center

    width = face.w * cfg.torso_width_ratio
    height = face.h * cfg.torso_height_ratio
    x0 = int(round(cx - width / 2.0))
    y0 = int(round(face.y + face.h * cfg.torso_top_offset_ratio))

    x0 = max(0, min(x0, img_w))
    y0 = max(0, min(y0, img_h))
    x1 = max(0, min(int(round(x0 + width)), img_w))
    y1 = max(0, min(int(round(y0 + height)), img_h))
    if x1 <= x0 or y1 <= y0:
        raise FaceError("torso region falls outside the image")
    return x0, y0, x1 - x0, y1 - y0


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1), b.reshape(-1)))


class FaceEngine:
    """Loads YuNet + SFace once. Safe to share across threadpool workers."""

    def __init__(self, settings: Optional[Settings] = None):
        cfg = settings or get_settings()
        for path in (cfg.yunet_path, cfg.sface_path):
            if not path.exists():
                raise FileNotFoundError(f"model not found: {path}")
        self.settings = cfg
        self._detector = cv2.FaceDetectorYN.create(
            str(cfg.yunet_path),
            "",
            (320, 320),
            cfg.detect_score_thresh,
            cfg.detect_nms_thresh,
            cfg.detect_top_k,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(cfg.sface_path), "")
        self._detector_lock = threading.Lock()
        self._recognizer_lock = threading.Lock()

    def detect(self, img: np.ndarray) -> List[Face]:
        if img is None or img.size == 0:
            raise FaceError("empty image")
        small, scale = downscale(img, self.settings.max_side)
        h, w = small.shape[:2]
        with self._detector_lock:
            self._detector.setInputSize((w, h))
            _, raw = self._detector.detect(small)
        if raw is None:
            return []

        inv = 1.0 / scale
        faces = []
        for row in raw:
            box = row[:4].astype(np.float64) * inv
            landmarks = (row[4:14].astype(np.float64) * inv).reshape(5, 2).astype(np.float32)
            faces.append(
                Face(
                    x=int(round(box[0])),
                    y=int(round(box[1])),
                    w=int(round(box[2])),
                    h=int(round(box[3])),
                    score=float(row[14]),
                    landmarks=landmarks,
                )
            )
        return faces

    def detect_single(self, img: np.ndarray) -> Face:
        faces = self.detect(img)
        if not faces:
            raise NoFaceError("no face detected")
        if len(faces) > 1:
            raise MultipleFacesError(f"expected one face, found {len(faces)}")
        return faces[0]

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        with self._recognizer_lock:
            aligned = self._recognizer.alignCrop(img, face.to_row())
            feature = self._recognizer.feature(aligned)
        vec = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            raise FaceError("degenerate face embedding")
        return (vec / norm).astype(np.float32)
