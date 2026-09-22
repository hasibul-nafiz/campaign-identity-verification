from __future__ import annotations

import numpy as np
import pytest

from app import face as face_mod
from app.config import Settings
from app.face import Face, FaceEngine, MultipleFacesError, NoFaceError, downscale, pose_label, torso_roi
from tests.conftest import make_face, rotate

needs_models = pytest.mark.skipif(
    not (Settings.from_env().yunet_path.exists() and Settings.from_env().sface_path.exists()),
    reason="ONNX models not present",
)


class TestDownscale:
    def test_large_image_is_shrunk_to_max_side(self):
        out, scale = downscale(np.zeros((2000, 1000, 3), np.uint8), 960)
        assert max(out.shape[:2]) == 960
        assert scale == pytest.approx(0.48)

    def test_small_image_is_untouched(self):
        img = np.zeros((300, 200, 3), np.uint8)
        out, scale = downscale(img, 960)
        assert scale == 1.0
        assert out is img

    def test_aspect_ratio_is_preserved(self):
        out, _ = downscale(np.zeros((1600, 800, 3), np.uint8), 960)
        assert out.shape[0] / out.shape[1] == pytest.approx(2.0, rel=0.01)


class TestPose:
    def test_frontal_face(self, settings: Settings):
        assert pose_label(make_face(), settings).label == "front"

    def test_turned_to_subject_left(self, settings: Settings):
        result = pose_label(make_face(nose=(70.0, 77.5)), settings)
        assert result.label == "left"
        assert result.yaw > 0

    def test_turned_to_subject_right(self, settings: Settings):
        result = pose_label(make_face(nose=(50.0, 77.5)), settings)
        assert result.label == "right"
        assert result.yaw < 0

    def test_looking_up(self, settings: Settings):
        result = pose_label(make_face(nose=(60.0, 92.5)), settings)
        assert result.label == "up"
        assert result.pitch > 0

    def test_looking_down(self, settings: Settings):
        result = pose_label(make_face(nose=(60.0, 62.5)), settings)
        assert result.label == "down"
        assert result.pitch < 0

    @pytest.mark.parametrize("degrees", [-30, -15, 15, 30, 45])
    def test_roll_does_not_change_a_frontal_label(self, settings: Settings, degrees):
        assert pose_label(rotate(make_face(), degrees), settings).label == "front"

    @pytest.mark.parametrize("degrees", [-25, 25])
    def test_roll_does_not_change_a_yawed_label(self, settings: Settings, degrees):
        turned = make_face(nose=(72.0, 77.5))
        assert pose_label(rotate(turned, degrees), settings).label == "left"

    def test_degenerate_landmarks_report_unknown(self, settings: Settings):
        flat = make_face(
            nose=(5.0, 5.0), right_eye=(5.0, 5.0), left_eye=(5.0, 5.0),
            right_mouth=(5.0, 5.0), left_mouth=(5.0, 5.0),
        )
        assert pose_label(flat, settings).label == "unknown"

    def test_thresholds_come_from_settings(self, monkeypatch):
        nudged = make_face(nose=(64.0, 77.5))
        assert pose_label(nudged, Settings.from_env()).label == "front"
        monkeypatch.setenv("POSE_YAW_THRESH", "0.05")
        assert pose_label(nudged, Settings.from_env()).label == "left"


class TestTorsoRoi:
    def test_box_sits_below_the_face_and_is_wider(self, settings: Settings):
        face = make_face(box=(100, 100, 80, 100))
        x, y, w, h = torso_roi(face, (600, 500, 3), settings)
        assert y >= face.y + face.h
        assert w > face.w
        assert h > 0

    def test_box_is_clipped_to_the_image(self, settings: Settings):
        face = make_face(box=(10, 10, 80, 100))
        x, y, w, h = torso_roi(face, (200, 120, 3), settings)
        assert x >= 0 and y >= 0
        assert x + w <= 120 and y + h <= 200

    def test_face_at_the_bottom_edge_raises(self, settings: Settings):
        face = make_face(box=(10, 190, 80, 100))
        with pytest.raises(face_mod.FaceError):
            torso_roi(face, (200, 300, 3), settings)


class TestFaceRow:
    def test_row_layout_matches_yunet(self):
        face = make_face(box=(1, 2, 3, 4), score=0.75)
        row = face.to_row()
        assert row.shape == (1, 15)
        assert row.dtype == np.float32
        assert list(row[0, :4]) == [1, 2, 3, 4]
        assert row[0, 14] == pytest.approx(0.75)
        np.testing.assert_allclose(row[0, 4:14].reshape(5, 2), face.landmarks)


@needs_models
class TestEngine:
    @pytest.fixture(scope="class")
    def engine(self) -> FaceEngine:
        return FaceEngine(Settings.from_env())

    def test_blank_image_has_no_face(self, engine: FaceEngine):
        with pytest.raises(NoFaceError):
            engine.detect_single(np.full((480, 640, 3), 127, np.uint8))

    def test_empty_image_raises(self, engine: FaceEngine):
        with pytest.raises(face_mod.FaceError):
            engine.detect(np.zeros((0, 0, 3), np.uint8))

    def test_multiple_faces_raises(self, engine: FaceEngine, monkeypatch):
        monkeypatch.setattr(engine, "detect", lambda img: [make_face(), make_face()])
        with pytest.raises(MultipleFacesError):
            engine.detect_single(np.zeros((100, 100, 3), np.uint8))

    def test_embedding_is_normalized_float32(self, engine: FaceEngine):
        rng = np.random.default_rng(5)
        img = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
        face = make_face(
            box=(100, 100, 80, 100),
            right_eye=(120.0, 130.0), left_eye=(160.0, 130.0), nose=(140.0, 157.0),
            right_mouth=(128.0, 180.0), left_mouth=(152.0, 180.0),
        )
        emb = engine.embed(img, face)
        assert emb.dtype == np.float32
        assert emb.ndim == 1 and emb.shape[0] == 128
        assert float(np.linalg.norm(emb)) == pytest.approx(1.0, abs=1e-5)

    def test_self_similarity_is_one(self, engine: FaceEngine):
        rng = np.random.default_rng(6)
        img = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
        face = make_face(
            box=(100, 100, 80, 100),
            right_eye=(120.0, 130.0), left_eye=(160.0, 130.0), nose=(140.0, 157.0),
            right_mouth=(128.0, 180.0), left_mouth=(152.0, 180.0),
        )
        emb = engine.embed(img, face)
        assert face_mod.similarity(emb, emb) == pytest.approx(1.0, abs=1e-5)

    def test_missing_model_path_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            FaceEngine(Settings.from_env())
