from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.db import Database
from app.store import TemplateStore
from tests.conftest import unit_vector


@pytest.fixture
def cfg(monkeypatch, tmp_path) -> Settings:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "store.db"))
    return Settings.from_env()


@pytest.fixture
def store(cfg: Settings) -> TemplateStore:
    return TemplateStore(Database(cfg.db_path), cfg)


class TestEmpty:
    def test_identify_on_empty_store(self, store: TemplateStore):
        assert store.identify(unit_vector(0)) is None

    def test_length_is_zero(self, store: TemplateStore):
        assert len(store) == 0
        assert store.people() == []


class TestIdentify:
    def test_exact_match_scores_one(self, store: TemplateStore):
        store.register("Ana", "front", unit_vector(1))
        result = store.identify(unit_vector(1))
        assert result.name == "Ana"
        assert result.pose == "front"
        assert result.score == pytest.approx(1.0)
        assert result.matched

    def test_orthogonal_vector_scores_zero_and_is_unmatched(self, store: TemplateStore):
        store.register("Ana", "front", unit_vector(1))
        result = store.identify(unit_vector(9))
        assert result.score == pytest.approx(0.0)
        assert not result.matched

    def test_picks_the_best_of_many(self, store: TemplateStore):
        store.register("Ana", "front", unit_vector(1))
        store.register("Bo", "front", unit_vector(2))
        store.register("Cy", "front", unit_vector(3))
        assert store.identify(unit_vector(2)).name == "Bo"
        assert len(store) == 3

    def test_reports_which_pose_matched(self, store: TemplateStore):
        store.register("Ana", "front", unit_vector(1))
        store.register("Ana", "left", unit_vector(4))
        assert store.identify(unit_vector(4)).pose == "left"

    def test_just_below_threshold_is_unmatched(self, store: TemplateStore, cfg: Settings):
        base = unit_vector(1)
        other = unit_vector(2)
        angle = cfg.face_match_thresh * 0.9
        probe = (angle * base + np.sqrt(1 - angle**2) * other).astype(np.float32)
        store.register("Ana", "front", base)
        result = store.identify(probe)
        assert result.score == pytest.approx(angle, abs=1e-5)
        assert not result.matched

    def test_wrong_dimension_rejected(self, store: TemplateStore):
        store.register("Ana", "front", unit_vector(1))
        with pytest.raises(ValueError):
            store.identify(np.ones(64, np.float32))


class TestMutations:
    def test_same_pose_replaces(self, store: TemplateStore):
        store.register("Ana", "front", unit_vector(1))
        store.register("Ana", "front", unit_vector(5))
        assert len(store) == 1
        assert store.identify(unit_vector(5)).matched
        assert not store.identify(unit_vector(1)).matched

    def test_new_pose_appends(self, store: TemplateStore):
        store.register("Ana", "front", unit_vector(1))
        store.register("Ana", "left", unit_vector(2))
        assert len(store) == 2
        assert len(store.people()) == 1

    def test_delete_person_empties_the_matrix(self, store: TemplateStore):
        person = store.register("Ana", "front", unit_vector(1))
        store.register("Ana", "left", unit_vector(2))
        assert store.delete_person(person.id) is True
        assert len(store) == 0
        assert store.identify(unit_vector(1)) is None

    def test_delete_unknown_person_is_false(self, store: TemplateStore):
        assert store.delete_person(4242) is False

    def test_delete_template_shrinks_the_matrix(self, store: TemplateStore):
        person = store.register("Ana", "front", unit_vector(1))
        store.register("Ana", "left", unit_vector(2))
        assert store.delete_template(person.id, "left") is True
        assert len(store) == 1

    def test_people_summary(self, store: TemplateStore):
        store.register("Ana", "front", unit_vector(1))
        store.register("Ana", "left", unit_vector(2))
        store.register("Bo", "front", unit_vector(3))
        summary = {p.name: p for p in store.people()}
        assert sorted(summary["Ana"].poses) == ["front", "left"]
        assert summary["Ana"].template_count == 2
        assert summary["Bo"].template_count == 1


class TestModelVersioning:
    def test_stale_version_templates_are_ignored(self, cfg: Settings):
        db = Database(cfg.db_path)
        person = db.add_person("Ana")
        db.upsert_template(person.id, "front", unit_vector(1), "some_old_model")
        store = TemplateStore(db, cfg)
        assert len(store) == 0
        assert store.identify(unit_vector(1)) is None

    def test_current_version_templates_are_used(self, cfg: Settings):
        db = Database(cfg.db_path)
        person = db.add_person("Ana")
        db.upsert_template(person.id, "front", unit_vector(1), cfg.model_version)
        assert len(TemplateStore(db, cfg)) == 1


def test_rebuild_picks_up_out_of_band_writes(store: TemplateStore, cfg: Settings):
    person = store.db.add_person("Ana")
    store.db.upsert_template(person.id, "front", unit_vector(1), cfg.model_version)
    assert len(store) == 0
    store.rebuild()
    assert len(store) == 1


class TestRegisterMany:
    def test_writes_every_pose_in_one_rebuild(self, store: TemplateStore, monkeypatch):
        rebuilds = {"n": 0}
        original = store.rebuild

        def counting():
            rebuilds["n"] += 1
            original()

        monkeypatch.setattr(store, "rebuild", counting)
        store.register_many("Ana", {"front": unit_vector(1), "left": unit_vector(2)})
        assert rebuilds["n"] == 1
        assert len(store) == 2

    def test_empty_mapping_rejected(self, store: TemplateStore):
        with pytest.raises(ValueError):
            store.register_many("Ana", {})


def test_people_summary_ignores_stale_version_templates(cfg: Settings):
    db = Database(cfg.db_path)
    person = db.add_person("Ana")
    db.upsert_template(person.id, "front", unit_vector(1), cfg.model_version)
    db.upsert_template(person.id, "left", unit_vector(2), "OLD_MODEL")
    store = TemplateStore(db, cfg)
    summary = store.people()[0]
    # The count must agree with the matrix, or /people advertises unusable templates.
    assert summary.poses == ["front"]
    assert summary.template_count == len(store) == 1
