from __future__ import annotations

from pathlib import Path

from app.config import BASE_DIR, Settings


def test_defaults():
    cfg = Settings.from_env()
    assert cfg.max_side == 960
    assert cfg.badge_nfeatures == 1500
    assert cfg.shirt_k == 4
    assert cfg.shirt_size == 64
    assert cfg.yunet_path == BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MAX_SIDE", "480")
    monkeypatch.setenv("SHIRT_DELTA_E_MAX", "5.5")
    monkeypatch.setenv("BADGE_MIN_INLIERS", "40")
    cfg = Settings.from_env()
    assert cfg.max_side == 480
    assert cfg.shirt_delta_e_max == 5.5
    assert cfg.badge_min_inliers == 40


def test_model_dir_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    cfg = Settings.from_env()
    assert cfg.yunet_path.parent == tmp_path
    assert cfg.sface_path.parent == tmp_path


def test_settings_is_frozen():
    cfg = Settings.from_env()
    try:
        cfg.max_side = 10  # type: ignore[misc]
    except Exception as exc:
        assert "frozen" in str(type(exc)).lower() or "cannot assign" in str(exc).lower()
    else:
        raise AssertionError("Settings should be immutable")
