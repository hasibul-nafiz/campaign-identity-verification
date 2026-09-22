from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.badge import BadgeMatcher
from app.campaigns import CampaignError, CampaignRegistry
from app.config import Settings
from app.db import Database
from app.main import create_app
from scripts.make_badge_ref import make_badge
from tests.test_api import FakeEngine, plain_scene, register, scene_with_badge, upload


def png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def badge_upload(name="ref.png", img=None):
    return (name, io.BytesIO(png(img if img is not None else make_badge())), "image/png")


@pytest.fixture
def cfg(monkeypatch, tmp_path) -> Settings:
    reference = tmp_path / "badge_ref.png"
    cv2.imwrite(str(reference), make_badge())
    monkeypatch.setenv("DB_PATH", str(tmp_path / "c.db"))
    monkeypatch.setenv("BADGE_REF_PATH", str(reference))
    monkeypatch.setenv("CAMPAIGN_ASSETS_DIR", str(tmp_path / "campaigns"))
    monkeypatch.delenv("DEBUG", raising=False)
    return Settings.from_env()


@pytest.fixture
def client(cfg: Settings):
    engine = FakeEngine()
    app = create_app(cfg, engine=engine, badge=BadgeMatcher(make_badge(), cfg))
    with TestClient(app) as c:
        c.engine = engine
        yield c


def create(client, name="Spring", colors=("#FFFFFF:home",), images=1):
    files = [("images", badge_upload(f"{i}.png")) for i in range(images)]
    return client.post("/campaigns", data={"name": name, "colors": list(colors)}, files=files)


class TestSeeding:
    def test_default_campaign_is_created_from_the_legacy_reference(self, client, cfg):
        body = client.get("/campaigns").json()["campaigns"]
        assert [c["name"] for c in body] == ["Default"]
        assert body[0]["badge_ref_count"] == 1
        assert [c["hex"] for c in body[0]["colors"]] == [cfg.shirt_expected_hex]

    def test_seeding_is_not_repeated(self, cfg):
        engine = FakeEngine()
        for _ in range(2):
            with TestClient(create_app(cfg, engine=engine, badge=BadgeMatcher(make_badge(), cfg))) as c:
                names = [x["name"] for x in c.get("/campaigns").json()["campaigns"]]
        assert names == ["Default"]


class TestCrud:
    def test_create_with_colours_and_reference(self, client):
        res = create(client, colors=("#FFFFFF:home", "#1E3A8A:away"))
        assert res.status_code == 201
        body = res.json()
        assert body["name"] == "Spring"
        assert [c["hex"] for c in body["colors"]] == ["#FFFFFF", "#1E3A8A"]
        assert [c["label"] for c in body["colors"]] == ["home", "away"]
        assert body["badge_ref_count"] == 1
        assert body["uploads"][0]["ok"] is True

    def test_duplicate_name_is_409(self, client):
        create(client, name="Spring")
        assert create(client, name="Spring").status_code == 409

    def test_bad_colour_is_rejected(self, client):
        res = client.post("/campaigns", data={"name": "X", "colors": ["nothex"]}, files=[])
        assert res.status_code == 400

    def test_list_and_get(self, client):
        cid = create(client).json()["id"]
        assert len(client.get("/campaigns").json()["campaigns"]) == 2  # Default + Spring
        body = client.get(f"/campaigns/{cid}").json()
        assert body["id"] == cid
        assert "badge_refs" in body

    def test_get_unknown_is_400(self, client):
        assert client.get("/campaigns/999").status_code == 400

    def test_patch_name_colours_and_overrides(self, client):
        cid = create(client).json()["id"]
        res = client.patch(
            f"/campaigns/{cid}",
            data={"name": "Renamed", "colors": ["#000000:night"],
                  "badge_min_inliers": "40", "shirt_delta_e_max": "6.5", "active": "false"},
        )
        body = res.json()
        assert body["name"] == "Renamed"
        assert [c["hex"] for c in body["colors"]] == ["#000000"]
        assert body["active"] is False
        assert body["overrides"]["badge_min_inliers"] == 40
        assert body["overrides"]["shirt_delta_e_max"] == 6.5
        assert body["effective"]["badge_min_inliers"] == 40

    def test_no_badge_type_setting_is_required(self, client):
        """Glossy and matte are handled by trying both pipelines, not by configuration."""
        cid = create(client).json()["id"]
        overrides = client.get(f"/campaigns/{cid}").json()["overrides"]
        assert "glare_suppression" not in overrides
        assert "despecular" not in overrides

    def test_overrides_fall_back_to_config(self, client, cfg):
        cid = create(client).json()["id"]
        body = client.get(f"/campaigns/{cid}").json()
        assert body["overrides"]["badge_min_inliers"] is None
        assert body["effective"]["badge_min_inliers"] == cfg.badge_min_inliers

    def test_delete_removes_campaign_and_files(self, client, cfg):
        cid = create(client).json()["id"]
        stored = list((cfg.campaign_assets_dir / str(cid)).glob("*.png"))
        assert len(stored) == 1
        assert client.delete(f"/campaigns/{cid}").status_code == 200
        assert client.get(f"/campaigns/{cid}").status_code == 400
        assert not stored[0].exists()

    def test_delete_unknown_is_404(self, client):
        assert client.delete("/campaigns/999").status_code == 404


class TestBadgeReferences:
    def test_add_and_serve(self, client):
        cid = create(client).json()["id"]
        res = client.post(f"/campaigns/{cid}/badges", files=[("images", badge_upload())])
        assert res.status_code == 201
        assert res.json()["badge_ref_count"] == 2
        url = res.json()["badge_refs"][-1]["url"]
        image = client.get(url)
        assert image.status_code == 200
        assert cv2.imdecode(np.frombuffer(image.content, np.uint8), cv2.IMREAD_COLOR) is not None

    def test_sparse_first_reference_is_rejected(self, client):
        # With no existing reference there is nothing to locate with, so the
        # keypoint floor is what catches a featureless image.
        cid = create(client, name="Empty", images=0).json()["id"]
        blank = np.full((300, 300, 3), 250, np.uint8)
        res = client.post(f"/campaigns/{cid}/badges", files=[("images", badge_upload(img=blank))])
        assert res.status_code == 400
        assert "too few features" in res.json()["uploads"][0]["error"]
        assert client.get(f"/campaigns/{cid}").json()["badge_ref_count"] == 0

    def test_foreign_badge_is_rejected(self, client):
        # A textured image that is not this campaign's badge must not silently join it.
        rng = np.random.default_rng(3)
        other = rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)
        cid = create(client).json()["id"]
        res = client.post(f"/campaigns/{cid}/badges", files=[("images", badge_upload(img=other))])
        assert res.status_code == 400
        assert "does not look like the same badge" in res.json()["uploads"][0]["error"]
        assert client.get(f"/campaigns/{cid}").json()["badge_ref_count"] == 1

    def test_wide_frame_is_rejected(self, client):
        """A whole frame is not a reference. Auto-locating the badge inside one was
        removed: background out-textures a glossy badge, so the homography fitted the
        scene and the full frame got stored as the badge."""
        cid = create(client).json()["id"]
        scene = np.full((720, 1280, 3), 248, np.uint8)
        scene[300:450, 560:680] = cv2.resize(make_badge(), (120, 150))
        res = client.post(f"/campaigns/{cid}/badges", files=[("images", badge_upload(img=scene))])
        assert res.status_code == 400
        assert "fills only" in res.json()["uploads"][0]["error"]
        # Nothing was stored, so the reference set is still just the one from create().
        assert client.get(f"/campaigns/{cid}").json()["badge_ref_count"] == 1

    def test_undecodable_upload_is_reported(self, client):
        cid = create(client).json()["id"]
        res = client.post(
            f"/campaigns/{cid}/badges",
            files=[("images", ("x.png", io.BytesIO(b"junk"), "image/png"))],
        )
        assert res.status_code == 400
        assert "decodable" in res.json()["uploads"][0]["error"]

    def test_reference_cap_is_enforced(self, client, cfg):
        cid = create(client).json()["id"]
        files = [("images", badge_upload(f"{i}.png")) for i in range(cfg.max_badge_refs)]
        res = client.post(f"/campaigns/{cid}/badges", files=files)
        assert res.status_code == 400
        assert str(cfg.max_badge_refs) in res.json()["detail"]

    def test_delete_reference(self, client):
        cid = create(client).json()["id"]
        ref_id = client.get(f"/campaigns/{cid}").json()["badge_refs"][0]["id"]
        assert client.delete(f"/campaigns/{cid}/badges/{ref_id}").status_code == 200
        assert client.get(f"/campaigns/{cid}").json()["badge_ref_count"] == 0
        assert client.delete(f"/campaigns/{cid}/badges/{ref_id}").status_code == 404

    def test_reference_of_another_campaign_is_404(self, client):
        first = create(client, name="A").json()
        second = create(client, name="B").json()["id"]
        ref_id = first["badge_refs"][0]["id"]
        assert client.get(f"/campaigns/{second}/badges/{ref_id}/image").status_code == 404


class TestDetectWithCampaigns:
    def test_uses_the_single_active_campaign_by_default(self, client):
        register(client, poses=("front",))
        client.engine.plans = [("front", __import__("tests.conftest", fromlist=["unit_vector"]).unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=scene_with_badge())}).json()
        assert body["campaign"]["name"] == "Default"
        assert body["badge"]["ok"] is True
        assert body["badge"]["matched_ref_id"] is not None
        assert body["passed"] is True

    def test_explicit_campaign_id(self, client):
        from tests.conftest import unit_vector

        cid = create(client, name="Spring").json()["id"]
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post(
            "/detect",
            files={"image": upload(data=scene_with_badge())},
            data={"campaign_id": str(cid)},
        ).json()
        assert body["campaign"]["id"] == cid
        assert body["badge"]["ok"] is True

    def test_auto_identifies_which_campaign_the_badge_belongs_to(self, client):
        """With no campaign_id the badge itself decides which campaign it is."""
        from tests.test_badge import distractor
        from tests.conftest import unit_vector

        # Default already holds make_badge(); add a campaign with a different badge.
        other = client.post(
            "/campaigns",
            data={"name": "Visitor", "colors": ["#FFFFFF"]},
            files=[("images", badge_upload(img=distractor()))],
        ).json()
        assert other["badge_ref_count"] == 1

        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=scene_with_badge())}).json()

        assert body["campaign"]["name"] == "Default"
        assert body["campaign"]["auto"] is True
        assert body["badge"]["ok"] is True

    def test_auto_reports_the_runners_up(self, client):
        from tests.test_badge import distractor
        from tests.conftest import unit_vector

        client.post(
            "/campaigns",
            data={"name": "Visitor", "colors": ["#FFFFFF"]},
            files=[("images", badge_upload(img=distractor()))],
        )
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post("/detect", files={"image": upload(data=scene_with_badge())}).json()

        names = [c["name"] for c in body["campaign_candidates"]]
        assert names[0] == "Default"
        assert "Visitor" in names
        assert body["campaign_candidates"][0]["ok"] is True
        assert all(c["ok"] is False for c in body["campaign_candidates"][1:])

    def test_explicit_campaign_id_is_not_auto(self, client):
        from tests.conftest import unit_vector

        cid = create(client, name="Pinned").json()["id"]
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post(
            "/detect",
            files={"image": upload(data=scene_with_badge())},
            data={"campaign_id": str(cid)},
        ).json()
        assert body["campaign"]["id"] == cid
        assert body["campaign"]["auto"] is False
        assert "campaign_candidates" not in body

    def test_auto_with_no_references_anywhere_errors(self, monkeypatch, tmp_path):
        from tests.conftest import unit_vector

        monkeypatch.setenv("DB_PATH", str(tmp_path / "n.db"))
        monkeypatch.setenv("BADGE_REF_PATH", str(tmp_path / "missing.png"))
        monkeypatch.setenv("CAMPAIGN_ASSETS_DIR", str(tmp_path / "campaigns"))
        cfg = Settings.from_env()
        engine = FakeEngine()
        app = create_app(cfg, engine=engine, badge=BadgeMatcher(make_badge(), cfg))
        with TestClient(app) as c:
            c.engine = engine
            register(c, poses=("front",))
            engine.plans = [("front", unit_vector(0))]
            res = c.post("/detect", files={"image": upload(data=scene_with_badge())})
        assert res.status_code == 400
        assert "no active campaign" in res.json()["detail"]

    def test_unknown_campaign_is_rejected(self, client):
        from tests.conftest import unit_vector

        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        res = client.post(
            "/detect", files={"image": upload(data=scene_with_badge())},
            data={"campaign_id": "999"},
        )
        assert res.status_code == 400

    def test_campaign_without_references_errors_clearly(self, client):
        from tests.conftest import unit_vector

        cid = create(client, name="Empty", images=0).json()["id"]
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        res = client.post(
            "/detect", files={"image": upload(data=scene_with_badge())},
            data={"campaign_id": str(cid)},
        )
        assert res.status_code == 400
        assert "no badge reference images" in res.json()["detail"]

    def test_colour_list_matches_any_member(self, client):
        from tests.conftest import unit_vector

        cid = create(client, name="Kit", colors=("#1E3A8A:away", "#FFFFFF:home")).json()["id"]
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post(
            "/detect",
            files={"image": upload(data=scene_with_badge())},
            data={"campaign_id": str(cid)},
        ).json()
        assert body["shirt"]["ok"] is True
        assert body["shirt"]["matched_color"] == "#FFFFFF"

    def test_per_campaign_threshold_override_applies(self, client):
        from tests.conftest import unit_vector

        cid = create(client, name="Strict").json()["id"]
        client.patch(f"/campaigns/{cid}", data={"badge_min_inliers": "100000"})
        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post(
            "/detect",
            files={"image": upload(data=scene_with_badge())},
            data={"campaign_id": str(cid)},
        ).json()
        assert body["badge"]["ok"] is False
        assert body["passed"] is False


class TestRegistry:
    def test_unknown_campaign_raises(self, cfg):
        registry = CampaignRegistry(Database(cfg.db_path), cfg)
        with pytest.raises(CampaignError):
            registry.get(42)

    def test_resolve_default_needs_exactly_one_active(self, cfg):
        db = Database(cfg.db_path)
        registry = CampaignRegistry(db, cfg)
        assert registry.resolve_default() is None
        first = db.add_campaign("A")
        registry.rebuild()
        assert registry.resolve_default().campaign.name == "A"
        db.add_campaign("B")
        registry.rebuild()
        assert registry.resolve_default() is None
        db.update_campaign(first.id, active=False)
        registry.rebuild()
        assert registry.resolve_default().campaign.name == "B"

    def test_missing_reference_file_is_skipped(self, cfg):
        db = Database(cfg.db_path)
        campaign = db.add_campaign("Gone")
        db.add_badge_ref(campaign.id, "/nonexistent/x.png", 10, 10, 200)
        registry = CampaignRegistry(db, cfg)
        assert registry.get(campaign.id).refs == []


class TestBadgeAnywhere:
    """The badge may be anywhere in frame, not only on the chest."""

    def frame_with_badge_at(self, x: int, y: int, size: int = 130) -> bytes:
        img = np.full((500, 400, 3), 252, np.uint8)
        badge = cv2.resize(make_badge(), (size, int(size * 1.25)))
        img[y : y + badge.shape[0], x : x + badge.shape[1]] = badge
        return cv2.imencode(".png", img)[1].tobytes()

    @pytest.mark.parametrize(
        "position,x,y",
        [("top-left corner", 5, 5), ("top-right corner", 260, 10), ("far left, low", 5, 330)],
    )
    def test_badge_outside_the_torso_box_is_found(self, client, position, x, y):
        from tests.conftest import unit_vector

        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post(
            "/detect", files={"image": upload(data=self.frame_with_badge_at(x, y))}
        ).json()
        assert body["badge"]["ok"] is True, f"{position}: {body['badge']}"
        assert body["badge"]["searched"] == "frame"

    def test_polygon_points_at_the_badge_not_the_torso(self, client):
        from tests.conftest import unit_vector

        register(client, poses=("front",))
        client.engine.plans = [("front", unit_vector(0))]
        body = client.post(
            "/detect", files={"image": upload(data=self.frame_with_badge_at(260, 10))}
        ).json()
        xs = [p[0] for p in body["badge"]["box"]]
        ys = [p[1] for p in body["badge"]["box"]]
        assert min(xs) == pytest.approx(260, abs=25)
        assert min(ys) == pytest.approx(10, abs=25)
        # Well above the torso box, which starts below the chin.
        assert min(ys) < body["torso_box"][1]

    def test_torso_only_mode_ignores_a_badge_elsewhere(self, monkeypatch, tmp_path):
        from tests.conftest import unit_vector

        reference = tmp_path / "badge_ref.png"
        cv2.imwrite(str(reference), make_badge())
        monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
        monkeypatch.setenv("BADGE_REF_PATH", str(reference))
        monkeypatch.setenv("CAMPAIGN_ASSETS_DIR", str(tmp_path / "campaigns"))
        monkeypatch.setenv("BADGE_SEARCH_REGION", "torso")
        cfg = Settings.from_env()
        engine = FakeEngine()
        app = create_app(cfg, engine=engine, badge=BadgeMatcher(make_badge(), cfg))
        with TestClient(app) as c:
            c.engine = engine
            register(c, poses=("front",))
            engine.plans = [("front", unit_vector(0))]
            body = c.post(
                "/detect", files={"image": upload(data=self.frame_with_badge_at(260, 10))}
            ).json()
        assert body["badge"]["searched"] == "torso"
        assert body["badge"]["ok"] is False


class TestUploadReporting:
    def test_rejected_upload_still_reports_its_keypoints(self, client):
        """A rejection must not report 0 keypoints as if the image were blank."""
        rng = np.random.default_rng(5)
        noise = rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)
        cid = create(client).json()["id"]
        row = client.post(
            f"/campaigns/{cid}/badges", files=[("images", badge_upload(img=noise))]
        ).json()["uploads"][0]
        assert row["ok"] is False
        assert row["keypoints"] > 100

    def test_accepted_upload_reports_its_keypoints(self, client):
        cid = create(client).json()["id"]
        row = client.post(
            f"/campaigns/{cid}/badges", files=[("images", badge_upload())]
        ).json()["uploads"][0]
        assert row["ok"] is True
        assert row["keypoints"] > 0

    def test_undecodable_upload_reports_zero(self, client):
        cid = create(client).json()["id"]
        row = client.post(
            f"/campaigns/{cid}/badges",
            files=[("images", ("x.png", io.BytesIO(b"junk"), "image/png"))],
        ).json()["uploads"][0]
        assert row["keypoints"] == 0
