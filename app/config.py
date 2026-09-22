from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).resolve().parent.parent


def _str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _path(key: str, default: Path) -> Path:
    return Path(os.getenv(key, str(default))).expanduser()


@dataclass(frozen=True)
class Settings:
    yunet_path: Path
    sface_path: Path
    badge_ref_path: Path
    campaign_assets_dir: Path
    db_path: Path

    max_side: int
    detect_score_thresh: float
    detect_nms_thresh: float
    detect_top_k: int
    face_match_thresh: float

    pose_yaw_thresh: float
    pose_pitch_neutral: float
    pose_pitch_thresh: float

    torso_width_ratio: float
    torso_height_ratio: float
    torso_top_offset_ratio: float

    badge_search_region: str
    badge_min_ref_keypoints: int
    badge_ref_min_fill: float
    badge_ref_max_side: int
    badge_ref_scales: Tuple[int, ...]
    badge_nfeatures: int
    badge_query_nfeatures: int
    badge_contrast_thresh: float
    badge_rootsift: bool
    badge_ratio: float
    badge_min_inliers: int
    badge_colour_min_inliers: int
    badge_colour_agreement: float
    badge_colour_rescue: bool
    badge_ransac_reproj: float
    badge_min_area_px: int
    badge_max_area_ratio: float
    badge_max_edge_ratio: float
    badge_max_aspect_skew: float

    shirt_k: int
    shirt_size: int
    shirt_delta_e_max: float
    shirt_expected_hex: str
    shirt_lightness_weight: float
    shirt_sample_inset_x: float
    shirt_sample_top: float
    shirt_sample_bottom: float
    shirt_min_cluster: float
    shirt_cluster_ratio: float
    shirt_merge_delta: float

    model_version: str
    cors_origins: Tuple[str, ...]
    debug: bool
    max_register_images: int
    max_badge_refs: int

    @classmethod
    def from_env(cls) -> "Settings":
        models = _path("MODEL_DIR", BASE_DIR / "models")
        return cls(
            yunet_path=_path("YUNET_PATH", models / "face_detection_yunet_2023mar.onnx"),
            sface_path=_path("SFACE_PATH", models / "face_recognition_sface_2021dec.onnx"),
            badge_ref_path=_path("BADGE_REF_PATH", BASE_DIR / "assets" / "badge_ref.png"),
            campaign_assets_dir=_path("CAMPAIGN_ASSETS_DIR", BASE_DIR / "assets" / "campaigns"),
            db_path=_path("DB_PATH", BASE_DIR / "data" / "faces.db"),
            max_side=_int("MAX_SIDE", 960),
            detect_score_thresh=_float("DETECT_SCORE_THRESH", 0.9),
            detect_nms_thresh=_float("DETECT_NMS_THRESH", 0.3),
            detect_top_k=_int("DETECT_TOP_K", 5000),
            face_match_thresh=_float("FACE_MATCH_THRESH", 0.363),
            pose_yaw_thresh=_float("POSE_YAW_THRESH", 0.16),
            pose_pitch_neutral=_float("POSE_PITCH_NEUTRAL", 0.55),
            pose_pitch_thresh=_float("POSE_PITCH_THRESH", 0.18),
            torso_width_ratio=_float("TORSO_WIDTH_RATIO", 2.2),
            torso_height_ratio=_float("TORSO_HEIGHT_RATIO", 2.0),
            torso_top_offset_ratio=_float("TORSO_TOP_OFFSET_RATIO", 1.15),
            badge_search_region=_str("BADGE_SEARCH_REGION", "frame"),
            badge_min_ref_keypoints=_int("BADGE_MIN_REF_KEYPOINTS", 60),
            badge_ref_min_fill=_float("BADGE_REF_MIN_FILL", 0.25),
            badge_ref_max_side=_int("BADGE_REF_MAX_SIDE", 240),
            badge_ref_scales=tuple(
                int(v) for v in _str("BADGE_REF_SCALES", "640,420,280,180,120,90,70").split(",") if v.strip()
            ),
            badge_nfeatures=_int("BADGE_NFEATURES", 1500),
            badge_query_nfeatures=_int("BADGE_QUERY_NFEATURES", 0),
            badge_contrast_thresh=_float("BADGE_CONTRAST_THRESH", 0.015),
            badge_rootsift=_str("BADGE_ROOTSIFT", "1") not in ("0", "", "false", "False"),
            badge_ratio=_float("BADGE_RATIO", 0.75),
            badge_min_inliers=_int("BADGE_MIN_INLIERS", 7),
            badge_colour_min_inliers=_int("BADGE_COLOUR_MIN_INLIERS", 4),
            badge_colour_agreement=_float("BADGE_COLOUR_AGREEMENT", 0.55),
            badge_colour_rescue=_str("BADGE_COLOUR_RESCUE", "1") not in ("0", "", "false", "False"),
            badge_ransac_reproj=_float("BADGE_RANSAC_REPROJ", 5.0),
            badge_min_area_px=_int("BADGE_MIN_AREA_PX", 1200),
            badge_max_area_ratio=_float("BADGE_MAX_AREA_RATIO", 1.5),
            badge_max_edge_ratio=_float("BADGE_MAX_EDGE_RATIO", 8.0),
            badge_max_aspect_skew=_float("BADGE_MAX_ASPECT_SKEW", 2.5),
            shirt_k=_int("SHIRT_K", 4),
            shirt_size=_int("SHIRT_SIZE", 64),
            shirt_delta_e_max=_float("SHIRT_DELTA_E_MAX", 14.0),
            shirt_expected_hex=_str("SHIRT_EXPECTED_HEX", "#FFFFFF"),
            shirt_lightness_weight=_float("SHIRT_LIGHTNESS_WEIGHT", 0.5),
            shirt_sample_inset_x=_float("SHIRT_SAMPLE_INSET_X", 0.20),
            shirt_sample_top=_float("SHIRT_SAMPLE_TOP", 0.15),
            shirt_sample_bottom=_float("SHIRT_SAMPLE_BOTTOM", 0.85),
            shirt_min_cluster=_float("SHIRT_MIN_CLUSTER", 0.25),
            shirt_cluster_ratio=_float("SHIRT_CLUSTER_RATIO", 0.5),
            shirt_merge_delta=_float("SHIRT_MERGE_DELTA", 8.0),
            model_version=_str("MODEL_VERSION", "sface_2021dec"),
            cors_origins=tuple(
                o.strip()
                for o in _str("CORS_ORIGINS", "http://localhost:5173").split(",")
                if o.strip()
            ),
            debug=_str("DEBUG", "0") not in ("0", "", "false", "False"),
            max_register_images=_int("MAX_REGISTER_IMAGES", 5),
            max_badge_refs=_int("MAX_BADGE_REFS", 8),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
