from __future__ import annotations

import base64
import io
from typing import List, Optional

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.badge import BadgeMatcher
from app.config import Settings
from app.face import MultipleFacesError, NoFaceError
from app.main import create_app
from scripts.make_badge_ref import make_badge
from tests.conftest import auth_header, make_face, unit_vector

POSE_FACES = {
    "front": make_face(nose=(60.0, 77.5)),
    "left": make_face(nose=(74.0, 77.5)),
    "right": make_face(nose=(46.0, 77.5)),
}


class FakeEngine:
    """Scripted stand-in: one plan consumed per detect_single call."""

    def __init__(self, plans: Optional[List] = None):
        self.plans = list(plans or [])
        self.default = ("front", unit_vector(0))
        self._embedding = None
        self.calls = 0

    def detect_single(self, img):
        self.calls += 1
        plan = self.plans.pop(0) if self.plans else self.default
        if isinstance(plan, Exception):
            raise plan
        pose, embedding = plan
        self._embedding = embedding
        return POSE_FACES[pose]

    def embed(self, img, face):
        return self._embedding


def scene_with_badge(width: int = 400, height: int = 500) -> bytes:
    """White shirt with the badge inside the torso ROI of POSE_FACES boxes."""
    img = np.full((height, width, 3), 252, np.uint8)
    badge = cv2.resize(make_badge(), (120, 150))
    img[170:320, 40:160] = badge
    return cv2.imencode(".png", img)[1].tobytes()


def plain_scene(color=(252, 252, 252), width: int = 400, height: int = 500) -> bytes:
    return cv2.imencode(".png", np.full((height, width, 3), color, np.uint8))[1].tobytes()


def upload(name: str = "a.png", data: Optional[bytes] = None):
    return (name, io.BytesIO(data if data is not None else plain_scene()), "image/png")


@pytest.fixture
def cfg(monkeypatch, tmp_path) -> Settings:
    # Seeding the Default campaign reads BADGE_REF_PATH, so point it at the test badge.
    reference = tmp_path / "badge_ref.png"
    cv2.imwrite(str(reference), make_badge())
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("BADGE_REF_PATH", str(reference))
    monkeypatch.setenv("CAMPAIGN_ASSETS_DIR", str(tmp_path / "campaigns"))
    monkeypatch.delenv("DEBUG", raising=False)
    return Settings.from_env()


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(cfg: Settings, engine: FakeEngine):
    app = create_app(cfg, engine=engine, badge=BadgeMatcher(make_badge(), cfg))
    with TestClient(app) as c:
        c.headers.update(auth_header(cfg))
        c.engine = engine
        yield c


def register(client, name="Ana", poses=("front",), data=None):
    client.engine.plans = [(p, unit_vector(i)) for i, p in enumerate(poses)]
    files = [("images", upload(f"{i}.png", data)) for i in range(len(poses))]
    return client.post("/register", data={"name": name}, files=files)


class TestHealth:
    def test_reports_state(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["people"] == 0 and body["templates"] == 0
        assert body["badge_reference_keypoints"] > 100
        assert body["debug"] is False

    def test_counts_track_registrations(self, client):
        register(client, poses=("front", "left"))
        body = client.get("/health").json()
        assert body["people"] == 1 and body["templates"] == 2


class TestCors:
    def test_allows_the_vite_dev_origin(self, client):
        res = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert res.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_preflight_is_answered(self, client):
        res = client.options(
            "/detect",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert res.status_code == 200

    def test_other_origins_are_not_allowed(self, client):
        res = client.get("/health", headers={"Origin": "http://evil.example"})
        assert res.headers.get("access-control-allow-origin") != "http://evil.example"


class TestRegister:
    def test_happy_path(self, client):
        res = register(client, poses=("front",))
        assert res.status_code == 201
        body = res.json()
        assert body["person"]["name"] == "Ana"
        assert body["templates_written"] == 1
        assert body["poses"] == ["front"]
        assert body["images"][0] == {
            "index": 0, "filename": "0.png", "ok": True, "pose": "front", "error": None,
        }

    def test_multiple_poses(self, client):
        body = register(client, poses=("front", "left", "right")).json()
        assert body["templates_written"] == 3
        assert body["poses"] == ["front", "left", "right"]

    def test_requires_a_front_pose(self, client):
        res = register(client, poses=("left", "right"))
        assert res.status_code == 400
        assert "front" in res.json()["detail"]

    def test_nothing_is_persisted_when_front_is_missing(self, client):
        register(client, poses=("left",))
        assert client.get("/people").json()["people"] == []
        assert client.get("/health").json()["templates"] == 0

    def test_zero_images_rejected(self, client):
        res = client.post("/register", data={"name": "Ana"}, files=[])
        assert res.status_code == 422

    def test_too_many_images_rejected(self, client, cfg):
        client.engine.plans = [("front", unit_vector(0))] * 6
        files = [("images", upload(f"{i}.png")) for i in range(6)]
        res = client.post("/register", data={"name": "Ana"}, files=files)
        assert res.status_code == 400
        assert f"1-{cfg.max_register_images}" in res.json()["detail"]

    def test_blank_name_rejected(self, client):
        res = register(client, name="   ")
        assert res.status_code == 400

    def test_same_pose_twice_keeps_one_template(self, client):
        body = register(client, poses=("front", "front")).json()
        assert body["templates_written"] == 1
        assert body["images"][1].get("replaced_earlier_image") is True
        assert client.get("/health").json()["templates"] == 1

    def test_reregistering_replaces_not_appends(self, client):
        register(client, poses=("front",))
        register(client, poses=("front",))
        assert client.get("/health").json()["templates"] == 1
        assert client.get("/people").json()["people"][0]["template_count"] == 1

    def test_undecodable_image_is_reported_per_row(self, client):
        client.engine.plans = [("front", unit_vector(0))]
        files = [
            ("images", ("good.png", io.BytesIO(plain_scene()), "image/png")),
            ("images", ("bad.png", io.BytesIO(b"not an image"), "image/png")),
        ]
        res = client.post("/register", data={"name": "Ana"}, files=files)
        assert res.status_code == 201
        rows = res.json()["images"]
        assert rows[0]["ok"] is True
        assert rows[1]["ok"] is False and "decodable" in rows[1]["error"]

    def test_no_face_image_is_reported_per_row(self, client):
        client.engine.plans = [("front", unit_vector(0)), NoFaceError("no face detected")]
        files = [("images", upload(f"{i}.png")) for i in range(2)]
        res = client.post("/register", data={"name": "Ana"}, files=files)
        assert res.status_code == 201
        assert res.json()["images"][1]["error"] == "no face detected"


class TestPeople:
    def test_empty(self, client):
        assert client.get("/people").json() == {"people": []}

    def test_lists_poses_and_counts(self, client):
        register(client, name="Ana", poses=("front", "left"))
        person = client.get("/people").json()["people"][0]
        assert person["name"] == "Ana"
        assert sorted(person["poses"]) == ["front", "left"]
        assert person["template_count"] == 2

    def test_delete_removes_person_and_templates(self, client):
        person_id = register(client, poses=("front", "left")).json()["person"]["id"]
        res = client.delete(f"/people/{person_id}")
        assert res.status_code == 200 and res.json() == {"deleted": person_id}
        assert client.get("/people").json()["people"] == []
        assert client.get("/health").json()["templates"] == 0

    def test_delete_unknown_is_404(self, client):
        res = client.delete("/people/4242")
        assert res.status_code == 404
        assert "4242" in res.json()["detail"]


class TestDetect:
    def test_no_templates_enrolled(self, client):
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload()}).json()
        assert body["passed"] is False
        assert body["person"] is None
        assert body["reason"] == "no templates enrolled"
        assert body["badge"] is None and body["shirt"] is None

    def test_unknown_face_skips_badge_and_shirt(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(99))]
        body = client.post("/detect", files={"image": upload()}).json()
        assert body["passed"] is False
        assert body["person"] is None
        assert body["reason"] == "face not recognised"
        assert body["badge"] is None and body["shirt"] is None
        assert "badge" not in body["timings_ms"]
        assert "shirt" not in body["timings_ms"]

    def test_known_face_runs_badge_and_shirt(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        res = client.post("/detect", files={"image": upload(data=scene_with_badge())})
        body = res.json()
        assert body["person"]["name"] == "Ana"
        assert body["person"]["score"] == pytest.approx(1.0)
        assert body["badge"]["ok"] is True
        assert body["badge"]["inliers"] >= 12
        assert body["shirt"]["ok"] is True
        assert body["passed"] is True

    def test_face_block_carries_box_and_pose(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        face = client.post("/detect", files={"image": upload()}).json()["face"]
        assert face["box"] == [20, 20, 80, 100]
        assert face["pose"] == "front"

    def test_missing_badge_fails_the_check(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=plain_scene())}).json()
        assert body["badge"]["ok"] is False
        assert body["shirt"]["ok"] is True
        assert body["passed"] is False

    def test_wrong_shirt_colour_fails_the_check(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post(
            "/detect",
            files={"image": upload(data=scene_with_badge())},
            data={"expected_color": "#C81414"},
        ).json()
        assert body["badge"]["ok"] is True
        assert body["shirt"]["ok"] is False
        assert body["passed"] is False

    def test_expected_colour_defaults_to_white(self, client, cfg):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload()}).json()
        assert body["shirt"]["expected_hex"] == cfg.shirt_expected_hex == "#FFFFFF"
        assert body["shirt"]["expected_rgb"] == [255, 255, 255]

    def test_no_face_is_400(self, client):
        client.engine.plans = [NoFaceError("no face detected")]
        res = client.post("/detect", files={"image": upload()})
        assert res.status_code == 400
        assert res.json() == {"detail": "no face detected"}

    def test_multiple_faces_is_400(self, client):
        client.engine.plans = [MultipleFacesError("expected one face, found 2")]
        res = client.post("/detect", files={"image": upload()})
        assert res.status_code == 400
        assert "found 2" in res.json()["detail"]

    def test_undecodable_image_is_400(self, client):
        res = client.post("/detect", files={"image": ("x.png", io.BytesIO(b"junk"), "image/png")})
        assert res.status_code == 400
        assert "decodable" in res.json()["detail"]

    def test_empty_upload_is_400(self, client):
        res = client.post("/detect", files={"image": ("x.png", io.BytesIO(b""), "image/png")})
        assert res.status_code == 400

    def test_timings_cover_every_stage(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        timings = client.post(
            "/detect", files={"image": upload(data=scene_with_badge())}
        ).json()["timings_ms"]
        assert set(timings) == {"decode", "detect", "embed", "identify", "badge", "shirt", "total"}
        assert all(v >= 0 for v in timings.values())
        assert timings["total"] >= timings["badge"]


class TestDebugFlag:
    def test_torso_absent_by_default(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=scene_with_badge())}).json()
        assert "torso_b64" not in body

    def test_torso_returned_when_debug_is_on(self, monkeypatch, tmp_path):
        reference = tmp_path / "badge_ref.png"
        cv2.imwrite(str(reference), make_badge())
        monkeypatch.setenv("DB_PATH", str(tmp_path / "debug.db"))
        monkeypatch.setenv("BADGE_REF_PATH", str(reference))
        monkeypatch.setenv("CAMPAIGN_ASSETS_DIR", str(tmp_path / "campaigns"))
        monkeypatch.setenv("DEBUG", "1")
        cfg = Settings.from_env()
        assert cfg.debug is True
        engine = FakeEngine()
        app = create_app(cfg, engine=engine, badge=BadgeMatcher(make_badge(), cfg))
        with TestClient(app) as c:
            c.headers.update(auth_header(cfg))
            c.engine = engine
            register(c, poses=("front",))
            engine.plans = [("front", unit_vector(0))]
            body = c.post("/detect", files={"image": upload(data=scene_with_badge())}).json()

        assert "torso_b64" in body
        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(body["torso_b64"]), np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        x0, y0, x1, y1 = body["torso_box"]
        assert (decoded.shape[1], decoded.shape[0]) == (x1 - x0, y1 - y0)


class TestGeometry:
    """Every coordinate is reported against the ORIGINAL upload, not the 960px working copy."""

    def test_image_size_is_always_present(self, client):
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=plain_scene())}).json()
        assert body["image_size"] == [400, 500]

    def test_unknown_face_still_reports_face_box_and_size(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(99))]
        body = client.post("/detect", files={"image": upload()}).json()
        assert body["face"]["box"] == [20, 20, 80, 100]
        assert body["image_size"] == [400, 500]
        assert body["torso_box"] is None
        assert body["badge_crop_b64"] is None

    def test_torso_box_is_x1y1x2y2(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=scene_with_badge())}).json()
        x0, y0, x1, y1 = body["torso_box"]
        assert (x0, y0, x1, y1) == (0, 135, 176, 335)
        assert x1 > x0 and y1 > y0

    def test_badge_box_is_a_four_point_polygon(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=scene_with_badge())}).json()
        polygon = body["badge"]["box"]
        assert polygon is not None
        assert len(polygon) == 4
        assert all(len(pt) == 2 for pt in polygon)

    def test_badge_box_lands_where_the_badge_was_pasted(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=scene_with_badge())}).json()
        xs = [p[0] for p in body["badge"]["box"]]
        ys = [p[1] for p in body["badge"]["box"]]
        # scene_with_badge() pastes at x 40..160, y 170..320 of the full image.
        assert min(xs) == pytest.approx(40, abs=12)
        assert max(xs) == pytest.approx(160, abs=12)
        assert min(ys) == pytest.approx(170, abs=12)
        assert max(ys) == pytest.approx(320, abs=12)

    def test_badge_box_is_null_when_absent(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=plain_scene())}).json()
        assert body["badge"]["ok"] is False
        assert body["badge"]["box"] is None
        assert body["badge_crop_b64"] is None

    def test_badge_crop_is_decodable_jpeg(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=scene_with_badge())}).json()
        raw = base64.b64decode(body["badge_crop_b64"])
        assert raw[:2] == b"\xff\xd8"  # JPEG SOI marker
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.size > 0

    def test_coordinates_are_scaled_back_to_a_large_original(self, client, cfg):
        """A 1600x2000 upload is processed at 768x960; output must be in 1600x2000 space."""
        big = np.full((2000, 1600, 3), 252, np.uint8)
        badge = cv2.resize(make_badge(), (480, 600))
        big[680:1280, 160:640] = badge
        data = cv2.imencode(".png", big)[1].tobytes()

        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=data)}).json()

        assert body["image_size"] == [1600, 2000]
        scale = cfg.max_side / 2000.0
        # FakeEngine returns a fixed box in processed space; it must come back scaled up.
        assert body["face"]["box"] == [round(v / scale) for v in (20, 20, 80, 100)]
        x0, y0, x1, y1 = body["torso_box"]
        assert x1 <= 1600 and y1 <= 2000
        assert y0 > body["face"]["box"][1]

    def test_polygon_stays_inside_the_original_image(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=scene_with_badge())}).json()
        width, height = body["image_size"]
        for x, y in body["badge"]["box"]:
            assert -5 <= x <= width + 5
            assert -5 <= y <= height + 5

    def test_badge_geometry_survives_a_downscale(self, client):
        """1600x2000 upload: badge must be located in ORIGINAL pixels, not 960px ones."""
        big = np.full((2000, 1600, 3), 252, np.uint8)
        big[330:642, 60:310] = cv2.resize(make_badge(), (250, 312))
        data = cv2.imencode(".png", big)[1].tobytes()

        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=data)}).json()

        assert body["image_size"] == [1600, 2000]
        assert body["badge"]["ok"] is True
        xs = [p[0] for p in body["badge"]["box"]]
        ys = [p[1] for p in body["badge"]["box"]]
        assert min(xs) == pytest.approx(60, abs=30)
        assert max(xs) == pytest.approx(310, abs=30)
        assert min(ys) == pytest.approx(330, abs=30)
        assert max(ys) == pytest.approx(642, abs=30)

        raw = base64.b64decode(body["badge_crop_b64"])
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        # Cut from the full-resolution original, so it is far bigger than the 960px copy.
        assert decoded.shape[1] > 200 and decoded.shape[0] > 250


class TestBadgeRef:
    def test_serves_the_reference_image(self, client):
        res = client.get("/badge-ref")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        decoded = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.size > 0

    def test_404_when_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DB_PATH", str(tmp_path / "br.db"))
        monkeypatch.setenv("CAMPAIGN_ASSETS_DIR", str(tmp_path / "campaigns"))
        monkeypatch.setenv("BADGE_REF_PATH", str(tmp_path / "gone.png"))
        cfg = Settings.from_env()
        app = create_app(cfg, engine=FakeEngine(), badge=BadgeMatcher(make_badge(), cfg))
        with TestClient(app) as c:
            c.headers.update(auth_header(cfg))
            res = c.get("/badge-ref")
        assert res.status_code == 404
        assert "badge reference" in res.json()["detail"]
