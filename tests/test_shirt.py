from __future__ import annotations

import numpy as np
import pytest

from app import shirt
from app.config import Settings
from tests.conftest import solid


def test_parse_hex_forms():
    assert shirt.parse_hex("#1E3A8A") == (0x1E, 0x3A, 0x8A)
    assert shirt.parse_hex("1e3a8a") == (0x1E, 0x3A, 0x8A)
    assert shirt.parse_hex("#fff") == (255, 255, 255)


@pytest.mark.parametrize("bad", ["", "#12", "xyzxyz", "#12345", "nothex"])
def test_parse_hex_rejects_garbage(bad):
    with pytest.raises(shirt.ShirtError):
        shirt.parse_hex(bad)


def test_exact_match_has_near_zero_delta(settings: Settings):
    result = shirt.check(solid((30, 58, 138)), "#1E3A8A", settings)
    assert result.ok
    assert result.delta_e < 1.0
    assert result.rgb == (30, 58, 138)
    assert result.coverage == pytest.approx(1.0)


def test_wrong_colour_is_rejected(settings: Settings):
    result = shirt.check(solid((220, 40, 40)), "#1E3A8A", settings)
    assert not result.ok
    assert result.delta_e > settings.shirt_delta_e_max


def test_dominant_cluster_wins_over_minority(settings: Settings):
    crop = solid((30, 58, 138), size=128)
    crop[:, :24] = (40, 220, 40)[::-1]
    result = shirt.check(crop, "#1E3A8A", settings)
    assert result.ok
    assert result.coverage > 0.5


def test_noise_does_not_break_dominant_colour(settings: Settings):
    rng = np.random.default_rng(3)
    crop = solid((30, 58, 138), size=128).astype(np.int16)
    crop += rng.integers(-12, 13, crop.shape, dtype=np.int16)
    result = shirt.check(np.clip(crop, 0, 255).astype(np.uint8), "#1E3A8A", settings)
    assert result.ok
    assert result.delta_e < settings.shirt_delta_e_max


def test_delta_e_is_symmetric_in_magnitude(settings: Settings):
    a = shirt.check(solid((30, 58, 138)), "#C81414", settings)
    b = shirt.check(solid((200, 20, 20)), "#1E3A8A", settings)
    assert a.delta_e == pytest.approx(b.delta_e, rel=0.05)


def test_empty_and_malformed_crops_raise(settings: Settings):
    with pytest.raises(shirt.ShirtError):
        shirt.check(np.zeros((0, 0, 3), np.uint8), "#1E3A8A", settings)
    with pytest.raises(shirt.ShirtError):
        shirt.check(np.zeros((16, 16), np.uint8), "#1E3A8A", settings)


class TestWhiteShirt:
    def test_pure_white_matches(self, settings: Settings):
        result = shirt.check(solid((255, 255, 255)), "#FFFFFF", settings)
        assert result.ok
        assert result.delta_e < 1.0

    def test_expected_hex_defaults_to_config(self, settings: Settings):
        assert settings.shirt_expected_hex == "#FFFFFF"
        assert shirt.check(solid((255, 255, 255)), settings=settings).ok

    def test_slightly_off_white_still_matches(self, settings: Settings):
        assert shirt.check(solid((240, 240, 238)), "#FFFFFF", settings).ok

    def test_underexposed_white_is_accepted(self, settings: Settings):
        # A white shirt in a dim or warm room still reads as white to a person, so
        # lightness is down-weighted and this must pass on hue rather than brightness.
        assert shirt.check(solid((200, 200, 200)), "#FFFFFF", settings).ok
        assert shirt.check(solid((224, 215, 204)), "#FFFFFF", settings).ok

    def test_grey_and_beige_are_still_rejected(self, settings: Settings):
        # Down-weighting lightness must not turn the check into "anything pale".
        for rgb in [(170, 170, 170), (128, 128, 128), (210, 180, 140)]:
            assert not shirt.check(solid(rgb), "#FFFFFF", settings).ok, rgb

    def test_lightness_weight_is_configurable(self, monkeypatch, settings: Settings):
        dim = solid((200, 200, 200))
        weighted = shirt.check(dim, "#FFFFFF", settings).delta_e
        monkeypatch.setenv("SHIRT_LIGHTNESS_WEIGHT", "1.0")
        unweighted = shirt.check(dim, "#FFFFFF", Settings.from_env()).delta_e
        assert unweighted > weighted

    def test_coloured_shirt_is_rejected(self, settings: Settings):
        assert not shirt.check(solid((30, 58, 138)), "#FFFFFF", settings).ok
        assert not shirt.check(solid((40, 160, 70)), "#FFFFFF", settings).ok


def test_threshold_comes_from_settings(monkeypatch):
    monkeypatch.setenv("SHIRT_DELTA_E_MAX", "0.001")
    strict = Settings.from_env()
    assert not shirt.check(solid((35, 62, 142)), "#1E3A8A", strict).ok
