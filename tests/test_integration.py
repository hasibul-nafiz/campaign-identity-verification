"""Composes a synthetic scene so the torso geometry, badge and shirt stages are
exercised together without needing a real photograph."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app import badge as badge_mod
from app import shirt as shirt_mod
from app.config import Settings
from app.face import torso_roi
from scripts.make_badge_ref import make_badge
from tests.conftest import make_face

SHIRT_HEX = "#FFFFFF"
SHIRT_BGR = (255, 255, 255)


@pytest.fixture
def scene(settings: Settings):
    canvas = np.full((900, 700, 3), 118, np.uint8)
    face = make_face(box=(300, 80, 120, 150))

    x, y, w, h = torso_roi(face, canvas.shape, settings)
    canvas[y : y + h, x : x + w] = SHIRT_BGR

    badge_img = cv2.resize(make_badge(), (150, 210))
    bx, by = x + 40, y + 45
    canvas[by : by + 210, bx : bx + 150] = badge_img
    return canvas, face


def test_torso_roi_lands_on_the_shirt(scene, settings: Settings):
    canvas, face = scene
    x, y, w, h = torso_roi(face, canvas.shape, settings)
    assert (x, y, w, h) == (228, 252, 264, 300)
    assert y > face.y + face.h


def test_badge_is_found_inside_the_torso_crop(scene, settings: Settings):
    canvas, face = scene
    x, y, w, h = torso_roi(face, canvas.shape, settings)
    result = badge_mod.BadgeMatcher(make_badge(), settings).check(canvas[y : y + h, x : x + w])
    assert result.ok
    assert result.inliers >= settings.badge_min_inliers


def test_shirt_colour_is_read_from_the_torso_crop(scene, settings: Settings):
    canvas, face = scene
    x, y, w, h = torso_roi(face, canvas.shape, settings)
    result = shirt_mod.check(canvas[y : y + h, x : x + w], SHIRT_HEX, settings)
    assert result.ok
    assert result.delta_e < settings.shirt_delta_e_max
    assert result.coverage > 0.5


def test_wrong_expected_shirt_colour_fails(scene, settings: Settings):
    canvas, face = scene
    x, y, w, h = torso_roi(face, canvas.shape, settings)
    assert not shirt_mod.check(canvas[y : y + h, x : x + w], "#C81414", settings).ok


def test_badge_absent_from_a_plain_shirt(settings: Settings):
    plain = np.full((300, 360, 3), SHIRT_BGR, np.uint8)
    assert not badge_mod.BadgeMatcher(make_badge(), settings).check(plain).ok
