from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate import Case, _suggest, describe, find_images, verdict


def case(group: str, face=0.9, badge=100, delta=2.0, error=None) -> Case:
    return Case(
        path=Path(f"{group}.png"), group=group, error=error,
        face_score=face, badge_inliers=badge, shirt_delta_e=delta, pose="front",
    )


class TestSuggest:
    def test_separable_picks_the_midpoint(self):
        s = _suggest([0.8, 0.9, 1.0], [0.1, 0.2, 0.3])
        assert s.separable
        assert s.threshold == pytest.approx(0.55)
        assert s.margin == pytest.approx(0.5)

    def test_separable_margin_is_the_gap(self):
        s = _suggest([10.0, 20.0], [4.0])
        assert s.margin == pytest.approx(6.0)
        assert s.threshold == pytest.approx(7.0)

    def test_overlap_is_flagged(self):
        s = _suggest([0.4, 0.9], [0.1, 0.6])
        assert not s.separable
        assert s.margin < 0

    def test_overlap_still_picks_a_usable_split(self):
        s = _suggest([0.5, 0.8, 0.9, 1.0], [0.1, 0.2, 0.3, 0.7])
        # Everything at or above the threshold should be mostly positives.
        assert 0.3 < s.threshold <= 0.9

    def test_ranges_are_reported(self):
        s = _suggest([0.8, 1.0], [0.1, 0.3])
        assert s.pos == (0.8, 1.0)
        assert s.neg == (0.1, 0.3)

    def test_missing_side_returns_none(self):
        assert _suggest([], [0.1]) is None
        assert _suggest([0.9], []) is None

    def test_single_sample_each_side(self):
        s = _suggest([0.9], [0.1])
        assert s.separable and s.threshold == pytest.approx(0.5)

    def test_lower_is_better_via_negation(self):
        # shirt delta_e: correct shirts score low, wrong shirts high
        correct, wrong = [1.0, 3.0, 5.0], [30.0, 45.0]
        s = _suggest([-v for v in correct], [-v for v in wrong])
        threshold = -s.threshold
        assert s.separable
        assert 5.0 < threshold < 30.0

    def test_identical_distributions_are_not_separable(self):
        assert not _suggest([0.5, 0.5], [0.5, 0.5]).separable


class TestVerdict:
    def test_genuine_passes(self):
        assert verdict(case("genuine"), 0.4, 12, 14.0) == (True, "pass")

    def test_face_is_checked_first(self):
        c = case("other_person", face=0.1, badge=0, delta=99.0)
        assert verdict(c, 0.4, 12, 14.0) == (False, "face")

    def test_badge_failure_reported(self):
        assert verdict(case("no_badge", badge=3), 0.4, 12, 14.0) == (False, "badge")

    def test_shirt_failure_reported(self):
        assert verdict(case("wrong_shirt", delta=40.0), 0.4, 12, 14.0) == (False, "shirt")

    def test_errored_case_never_passes(self):
        c = case("genuine", error="no face detected")
        assert verdict(c, 0.4, 12, 14.0) == (False, "no face detected")

    def test_boundaries_are_inclusive(self):
        assert verdict(case("genuine", face=0.4), 0.4, 12, 14.0)[0] is True
        assert verdict(case("genuine", badge=12), 0.4, 12, 14.0)[0] is True
        assert verdict(case("genuine", delta=14.0), 0.4, 12, 14.0)[0] is True


class TestHelpers:
    def test_describe_handles_empty(self):
        assert describe([]) == "n/a"

    def test_describe_reports_spread(self):
        text = describe([1.0, 2.0, 3.0])
        assert "min=1.000" in text and "med=2.000" in text and "max=3.000" in text

    def test_find_images_filters_by_extension(self, tmp_path):
        for name in ("a.png", "b.JPG", "c.txt", "d.webp"):
            (tmp_path / name).touch()
        assert [p.name for p in find_images(tmp_path)] == ["a.png", "b.JPG", "d.webp"]

    def test_find_images_on_missing_folder(self, tmp_path):
        assert find_images(tmp_path / "nope") == []


class TestEndToEndReport:
    """Stubs only face detection; badge, shirt, store and reporting all run for real."""

    @pytest.fixture
    def dataset(self, tmp_path):
        import cv2
        import numpy as np
        from scripts.make_badge_ref import make_badge

        # Kept small: a badge covering most of the torso would become the
        # dominant colour cluster and mask the shirt colour entirely.
        badge = cv2.resize(make_badge(), (100, 125))
        plan = {
            "genuine": (True, (252, 252, 252), 3),
            "no_badge": (False, (252, 252, 252), 2),
            "wrong_shirt": (True, (40, 40, 200), 2),
            "other_person": (True, (252, 252, 252), 2),
        }
        for group, (has_badge, color, count) in plan.items():
            folder = tmp_path / group
            folder.mkdir()
            for i in range(count):
                img = np.full((500, 400, 3), color, np.uint8)
                if has_badge:
                    img[150:275, 40:140] = badge
                cv2.imwrite(str(folder / f"{group}_{i}.png"), img)
        return tmp_path

    @pytest.fixture
    def patched(self, monkeypatch, tmp_path):
        from scripts.make_badge_ref import make_badge
        import scripts.evaluate as ev
        from app.badge import BadgeMatcher
        from tests.conftest import make_face, unit_vector

        class FakeEngine:
            def __init__(self, cfg):
                pass

            def detect_single(self, img):
                return make_face()

            def embed(self, img, face):
                return unit_vector(0)

        monkeypatch.setattr(ev, "FaceEngine", FakeEngine)
        monkeypatch.setattr(ev, "BadgeMatcher", lambda settings=None: BadgeMatcher(make_badge(), settings))
        monkeypatch.setenv("DB_PATH", str(tmp_path / "eval.db"))
        return ev

    def test_report_covers_every_section(self, patched, dataset, capsys, monkeypatch):
        # other_person needs a low face score so the face stage has something to separate.
        original = patched.run_case

        def run_case(path, group, engine, store, matcher, cfg, expected_hex):
            case = original(path, group, engine, store, matcher, cfg, expected_hex)
            if group == "other_person":
                case.face_score = 0.05
            return case

        monkeypatch.setattr(patched, "run_case", run_case)
        monkeypatch.setattr("sys.argv", ["evaluate.py", str(dataset)])
        assert patched.main() == 0

        out = capsys.readouterr().out
        for section in ("genuine/", "no_badge/", "wrong_shirt/", "other_person/",
                        "score distributions", "suggested thresholds",
                        "FACE_MATCH_THRESH", "BADGE_MIN_INLIERS", "SHIRT_DELTA_E_MAX",
                        "errors at each threshold set", "current", "suggested",
                        "export FACE_MATCH_THRESH="):
            assert section in out, f"missing {section!r}"

    def test_suggested_thresholds_eliminate_errors(self, patched, dataset, capsys, monkeypatch):
        original = patched.run_case

        def run_case(path, group, engine, store, matcher, cfg, expected_hex):
            case = original(path, group, engine, store, matcher, cfg, expected_hex)
            if group == "other_person":
                case.face_score = 0.05
            return case

        monkeypatch.setattr(patched, "run_case", run_case)
        monkeypatch.setattr("sys.argv", ["evaluate.py", str(dataset)])
        patched.main()

        out = capsys.readouterr().out
        suggested = out.split("suggested: face")[1]
        assert "false accepts: none" in suggested
        assert "false rejects: none" in suggested
