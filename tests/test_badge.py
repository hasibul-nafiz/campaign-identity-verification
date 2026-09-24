from __future__ import annotations

import cv2
import numpy as np
import pytest

from app import badge
from app.config import BASE_DIR, Settings
from scripts.make_badge_ref import make_badge


@pytest.fixture(scope="module")
def reference() -> np.ndarray:
    return make_badge()


@pytest.fixture
def matcher(reference: np.ndarray, settings: Settings) -> badge.BadgeMatcher:
    return badge.BadgeMatcher(reference, settings)


def place_on_canvas(ref: np.ndarray, canvas=(700, 520), offset=(60, 50)) -> np.ndarray:
    scene = np.full((canvas[0], canvas[1], 3), 90, dtype=np.uint8)
    h, w = ref.shape[:2]
    scale = min((canvas[0] - offset[1] * 2) / h, (canvas[1] - offset[0] * 2) / w)
    small = cv2.resize(ref, (int(w * scale), int(h * scale)))
    scene[offset[1] : offset[1] + small.shape[0], offset[0] : offset[0] + small.shape[1]] = small
    return scene


def test_reference_has_rich_features(matcher: badge.BadgeMatcher):
    assert matcher.reference_keypoints > 200


def test_reference_is_built_at_several_scales(reference, settings: Settings):
    # A single reference scale only matches badges that happen to appear at about
    # that size, so descriptors are built across a pyramid instead.
    scales = badge.BadgeMatcher(reference, settings).scales
    assert len(scales) >= 3
    assert scales == sorted(scales, reverse=True)
    assert max(scales) <= max(reference.shape[:2])


@pytest.mark.parametrize("width", [60, 100, 180, 320, 520])
def test_matches_across_a_wide_size_range(matcher: badge.BadgeMatcher, reference, width):
    """The badge may be any size on screen, not one the reference happens to match."""
    height = int(width * reference.shape[0] / reference.shape[1])
    scene = np.full((max(400, height + 120), max(500, width + 160), 3), 244, np.uint8)
    scene[60 : 60 + height, 70 : 70 + width] = cv2.resize(
        reference, (width, height), interpolation=cv2.INTER_AREA
    )
    result = matcher.check(scene)
    assert result.ok, f"{width}px: {result.inliers} inliers, {result.reason}"


def test_identical_crop_matches(matcher: badge.BadgeMatcher, reference: np.ndarray):
    result = matcher.check(reference)
    assert result.ok
    assert result.inliers >= 50
    assert result.reason == "ok"


def test_badge_placed_in_a_scene_matches(matcher: badge.BadgeMatcher, reference: np.ndarray):
    result = matcher.check(place_on_canvas(reference))
    assert result.ok
    assert result.inliers >= matcher.settings.badge_min_inliers


def test_perspective_warp_still_matches(matcher: badge.BadgeMatcher, reference: np.ndarray):
    h, w = reference.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[40, 20], [w - 15, 55], [w - 45, h - 25], [20, h - 60]])
    warped = cv2.warpPerspective(reference, cv2.getPerspectiveTransform(src, dst), (w, h))
    result = matcher.check(warped)
    assert result.ok


def test_random_noise_is_rejected(matcher: badge.BadgeMatcher):
    rng = np.random.default_rng(11)
    noise = rng.integers(0, 256, (600, 420, 3), dtype=np.uint8)
    assert not matcher.check(noise).ok


def test_blank_crop_is_rejected(matcher: badge.BadgeMatcher):
    assert not matcher.check(np.full((400, 300, 3), 200, np.uint8)).ok


def distractor(seed: int = 4) -> np.ndarray:
    """Textured, but sharing no layout with the reference badge."""
    rng = np.random.default_rng(seed)
    img = np.full((560, 380, 3), 30, dtype=np.uint8)
    for _ in range(140):
        pt1 = (int(rng.integers(0, 380)), int(rng.integers(0, 560)))
        pt2 = (int(rng.integers(0, 380)), int(rng.integers(0, 560)))
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        cv2.line(img, pt1, pt2, color, int(rng.integers(1, 5)))
    cv2.putText(img, "VISITOR", (24, 420), cv2.FONT_HERSHEY_TRIPLEX, 1.6, (255, 255, 0), 3)
    return img


def test_structurally_different_card_is_rejected(matcher: badge.BadgeMatcher):
    assert not matcher.check(distractor()).ok


def test_same_template_different_seed_still_matches(matcher: badge.BadgeMatcher):
    # SIFT keys on the shared layout, so this answers "is this our badge template",
    # not "is this this person's badge".
    assert matcher.check(make_badge(seed=999)).ok


def test_empty_crop_raises(matcher: badge.BadgeMatcher):
    with pytest.raises(badge.BadgeError):
        matcher.check(np.zeros((0, 0, 3), np.uint8))


def test_untextured_reference_is_refused(settings: Settings):
    with pytest.raises(badge.BadgeError):
        badge.BadgeMatcher(np.full((300, 300, 3), 255, np.uint8), settings)


def test_missing_reference_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("BADGE_REF_PATH", str(tmp_path / "nope.png"))
    with pytest.raises(FileNotFoundError):
        badge.BadgeMatcher(settings=Settings.from_env())


class TestDegenerateHomography:
    def _quad(self, points):
        return np.array(points, dtype=np.float64)

    def test_collapsed_quad_rejected(self, settings: Settings):
        quad = self._quad([[10, 10], [10, 10], [10, 10], [10, 10]])
        ok, reason = badge._quad_is_sane(quad, (400, 300), settings)
        assert not ok and reason == "collapsed edge"

    def test_self_intersecting_quad_rejected(self, settings: Settings):
        quad = self._quad([[0, 0], [200, 200], [200, 0], [0, 200]])
        ok, reason = badge._quad_is_sane(quad, (400, 300), settings)
        assert not ok and reason == "non-convex quad"

    def test_sliver_quad_rejected(self, settings: Settings):
        quad = self._quad([[0, 0], [300, 0], [300, 1.5], [0, 1.5]])
        ok, reason = badge._quad_is_sane(quad, (400, 300), settings)
        assert not ok
        assert reason in {"extreme edge ratio", "projection too small"}

    def test_oversized_quad_rejected(self, settings: Settings):
        quad = self._quad([[0, 0], [5000, 0], [5000, 5000], [0, 5000]])
        ok, reason = badge._quad_is_sane(quad, (400, 300), settings)
        assert not ok and reason == "projection too large"

    def test_non_finite_quad_rejected(self, settings: Settings):
        quad = self._quad([[0, 0], [np.inf, 0], [100, 100], [0, 100]])
        ok, reason = badge._quad_is_sane(quad, (400, 300), settings)
        assert not ok and reason == "non-finite homography"

    def test_reasonable_quad_accepted(self, settings: Settings):
        quad = self._quad([[20, 20], [280, 30], [270, 370], [30, 360]])
        ok, reason = badge._quad_is_sane(quad, (400, 300), settings)
        assert ok and reason == "ok"


class TestBadgeFinishes:
    """Glossy and matte badges must both work with no per-badge configuration."""

    def glossy(self, base: np.ndarray, seed: int = 2) -> np.ndarray:
        """Add mirror-like highlights, the way a domed enamel badge reflects a room."""
        rng = np.random.default_rng(seed)
        out = base.astype(np.float32)
        h, w = out.shape[:2]
        for _ in range(3):
            glare = np.zeros((h, w), np.float32)
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.ellipse(glare, (cx, cy), (int(w * 0.22), int(h * 0.14)),
                        float(rng.integers(0, 180)), 0, 360, 1.0, -1)
            out += cv2.GaussianBlur(glare, (0, 0), w * 0.05)[..., None] * 150
        return np.clip(out, 0, 255).astype(np.uint8)

    def scene(self, badge: np.ndarray, width: int = 150) -> np.ndarray:
        height = int(width * badge.shape[0] / badge.shape[1])
        small = cv2.resize(badge, (width, height), interpolation=cv2.INTER_AREA)
        img = np.full((420, 560, 3), 245, np.uint8)
        img[90 : 90 + height, 140 : 140 + width] = small
        return img

    def test_both_pipelines_are_built(self, matcher: badge.BadgeMatcher):
        assert matcher.variants == ["deglare", "plain"]

    def test_matte_badge_matches(self, matcher: badge.BadgeMatcher, reference: np.ndarray):
        assert matcher.check(self.scene(reference)).ok

    def test_glossy_badge_matches(self, matcher: badge.BadgeMatcher, reference: np.ndarray):
        """Reflections the reference never saw must not break the match."""
        result = matcher.check(self.scene(self.glossy(reference)))
        assert result.ok, f"{result.inliers} inliers, {result.reason}"

    def test_glare_does_not_create_false_positives(self, matcher: badge.BadgeMatcher):
        rng = np.random.default_rng(9)
        noise = self.glossy(rng.integers(0, 256, (500, 400, 3), dtype=np.uint8))
        assert not matcher.check(noise).ok


class TestHardConditions:
    """Distance and a shirt the colour of the badge, both seen on real captures."""

    def worn_on(self, reference: np.ndarray, shirt_bgr, width: int, seed: int = 3) -> np.ndarray:
        """Paste only the badge -- not the fabric it was photographed on -- onto a shirt."""
        rng = np.random.default_rng(seed)
        scene = np.empty((420, 560, 3), np.float32)
        scene[:] = shirt_bgr
        scene += rng.normal(0, 5, scene.shape[:2])[..., None]
        mask = badge._foreground_mask(reference, Settings.from_env())
        height = int(width * reference.shape[0] / reference.shape[1])
        small = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
        keep = cv2.resize(mask.astype(np.float32), (width, height))[..., None]
        region = scene[100 : 100 + height, 200 : 200 + width]
        scene[100 : 100 + height, 200 : 200 + width] = small * keep + region * (1 - keep)
        return np.clip(scene, 0, 255).astype(np.uint8)

    def test_badge_is_separated_from_its_margin(self, reference, settings: Settings):
        mask = badge._foreground_mask(reference, settings)
        assert mask is not None
        assert 0.25 < mask.mean() < 0.9
        # The corners are fabric, the centre is badge.
        assert mask[0, 0] == 0 and mask[-1, -1] == 0
        assert mask[mask.shape[0] // 2, mask.shape[1] // 2] == 1

    def test_tight_crop_keeps_everything(self, reference, settings: Settings):
        """No uniform margin means no background to remove."""
        rng = np.random.default_rng(1)
        busy = rng.integers(0, 256, reference.shape, dtype=np.uint8)
        assert badge._foreground_mask(busy, settings) is None

    def test_colour_check_ignores_the_shirt(self, reference, settings: Settings):
        """The reference's fabric must not decide whether the colours agree."""
        mask = badge._foreground_mask(reference, settings)
        on_navy = reference.copy()
        on_navy[mask == 0] = (110, 40, 20)
        masked = float(np.dot(
            badge._chroma_signature(reference, mask=mask),
            badge._chroma_signature(on_navy, mask=mask),
        ))
        unmasked = float(np.dot(
            badge._chroma_signature(reference), badge._chroma_signature(on_navy)
        ))
        assert masked > 0.9
        assert masked - unmasked > 0.1

    @pytest.mark.parametrize("shirt,width", [
        ((200, 110, 20), 45), ((200, 110, 20), 60), ((90, 40, 20), 45), ((30, 30, 30), 50),
    ])
    def test_real_badge_on_a_shirt_of_its_own_colour(self, settings: Settings, shirt, width):
        """The real crest has a blue rim. On a blue shirt that rim -- its outline -- is
        gone, and the colour check used to be dragged down by the white fabric the
        reference photo was taken on."""
        real = cv2.imread(str(BASE_DIR / "assets" / "badge_ref.png"), cv2.IMREAD_COLOR)
        result = badge.BadgeMatcher(real, settings).check(self.worn_on(real, shirt, width, seed=1))
        assert result.ok, f"{shirt} {width}px: {result.inliers} inliers, {result.reason}"


class TestZoom:
    def test_small_region_is_enlarged(self, settings: Settings):
        crop = np.zeros((200, 148, 3), np.uint8)
        zoomed, factor = badge.zoom_for_search(crop, settings)
        assert factor == pytest.approx(settings.badge_zoom_target / 200)
        assert max(zoomed.shape[:2]) == settings.badge_zoom_target

    def test_zoom_is_capped(self, settings: Settings):
        zoomed, factor = badge.zoom_for_search(np.zeros((40, 30, 3), np.uint8), settings)
        assert factor == settings.badge_zoom_max
        assert zoomed.shape[:2] == (120, 90)

    def test_large_region_is_left_alone(self, settings: Settings):
        crop = np.zeros((720, 1280, 3), np.uint8)
        zoomed, factor = badge.zoom_for_search(crop, settings)
        assert factor == 1.0 and zoomed is crop

    def test_zoom_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("BADGE_ZOOM_TARGET", "0")
        crop = np.zeros((100, 100, 3), np.uint8)
        assert badge.zoom_for_search(crop, Settings.from_env()) == (crop, 1.0)

    def test_zoom_finds_a_badge_too_small_to_match(self, matcher, reference):
        scene = np.full((200, 148, 3), 252, np.uint8)
        scene[60:95, 50:78] = cv2.resize(reference, (28, 35), interpolation=cv2.INTER_AREA)
        assert not matcher.check(scene).ok
        zoomed, _ = badge.zoom_for_search(scene, matcher.settings)
        assert matcher.check(zoomed).ok
