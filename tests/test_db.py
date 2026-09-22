from __future__ import annotations

import numpy as np
import pytest

from app.db import Database, DatabaseError, from_blob, to_blob
from tests.conftest import unit_vector

VERSION = "sface_2021dec"


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "test.db")


class TestBlobs:
    def test_roundtrip(self):
        vec = unit_vector(7)
        np.testing.assert_array_equal(from_blob(to_blob(vec)), vec)

    def test_rejects_wrong_dimension(self):
        with pytest.raises(DatabaseError):
            to_blob(np.ones(64, np.float32) / 8.0)

    def test_rejects_unnormalized(self):
        with pytest.raises(DatabaseError):
            to_blob(np.full(128, 0.5, np.float32))

    def test_rejects_corrupt_blob(self):
        with pytest.raises(DatabaseError):
            from_blob(b"\x00\x01\x02\x03")

    def test_stored_as_float32(self):
        assert len(to_blob(unit_vector(0))) == 128 * 4


class TestPragmas:
    def test_wal_enabled(self, db: Database):
        mode = db.connection().execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_foreign_keys_enabled(self, db: Database):
        assert db.connection().execute("PRAGMA foreign_keys").fetchone()[0] == 1


class TestPeople:
    def test_add_and_fetch(self, db: Database):
        person = db.add_person("Ana")
        assert person.id > 0
        assert db.get_person(person.id).name == "Ana"
        assert db.get_person_by_name("Ana").id == person.id

    def test_add_is_idempotent_on_name(self, db: Database):
        first = db.add_person("Ana")
        second = db.add_person("Ana")
        assert first.id == second.id
        assert len(db.list_people()) == 1

    def test_name_is_trimmed(self, db: Database):
        assert db.add_person("  Ana  ").name == "Ana"

    def test_empty_name_rejected(self, db: Database):
        with pytest.raises(DatabaseError):
            db.add_person("   ")

    def test_missing_person_is_none(self, db: Database):
        assert db.get_person(999) is None
        assert db.get_person_by_name("nobody") is None

    def test_delete_reports_whether_it_hit(self, db: Database):
        person = db.add_person("Ana")
        assert db.delete_person(person.id) is True
        assert db.delete_person(person.id) is False


class TestTemplates:
    def test_insert_and_read_back(self, db: Database):
        person = db.add_person("Ana")
        db.upsert_template(person.id, "front", unit_vector(1), VERSION)
        templates = db.templates_for(person.id)
        assert len(templates) == 1
        assert templates[0].pose == "front"
        np.testing.assert_array_equal(templates[0].embedding, unit_vector(1))

    def test_same_pose_replaces_rather_than_appends(self, db: Database):
        person = db.add_person("Ana")
        db.upsert_template(person.id, "front", unit_vector(1), VERSION)
        db.upsert_template(person.id, "front", unit_vector(2), VERSION)
        templates = db.templates_for(person.id)
        assert len(templates) == 1
        np.testing.assert_array_equal(templates[0].embedding, unit_vector(2))

    def test_distinct_poses_coexist(self, db: Database):
        person = db.add_person("Ana")
        for i, pose in enumerate(("front", "left", "right")):
            db.upsert_template(person.id, pose, unit_vector(i), VERSION)
        assert {t.pose for t in db.templates_for(person.id)} == {"front", "left", "right"}
        assert db.count_templates() == 3

    def test_same_pose_for_different_people_coexist(self, db: Database):
        ana, bo = db.add_person("Ana"), db.add_person("Bo")
        db.upsert_template(ana.id, "front", unit_vector(1), VERSION)
        db.upsert_template(bo.id, "front", unit_vector(2), VERSION)
        assert db.count_templates() == 2

    def test_delete_person_cascades(self, db: Database):
        person = db.add_person("Ana")
        db.upsert_template(person.id, "front", unit_vector(1), VERSION)
        db.upsert_template(person.id, "left", unit_vector(2), VERSION)
        assert db.count_templates() == 2
        db.delete_person(person.id)
        assert db.count_templates() == 0
        assert db.all_templates() == []

    def test_delete_single_template(self, db: Database):
        person = db.add_person("Ana")
        db.upsert_template(person.id, "front", unit_vector(1), VERSION)
        assert db.delete_template(person.id, "front") is True
        assert db.delete_template(person.id, "front") is False

    def test_template_for_unknown_person_is_rejected(self, db: Database):
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            db.upsert_template(4242, "front", unit_vector(1), VERSION)

    def test_unnormalized_embedding_never_reaches_disk(self, db: Database):
        person = db.add_person("Ana")
        with pytest.raises(DatabaseError):
            db.upsert_template(person.id, "front", np.full(128, 0.9, np.float32), VERSION)
        assert db.count_templates() == 0


def test_persists_across_connections(tmp_path):
    path = tmp_path / "persist.db"
    first = Database(path)
    person = first.add_person("Ana")
    first.upsert_template(person.id, "front", unit_vector(3), VERSION)
    first.close()

    second = Database(path)
    assert [p.name for p in second.list_people()] == ["Ana"]
    np.testing.assert_array_equal(second.all_templates()[0].embedding, unit_vector(3))


class TestConcurrency:
    def test_concurrent_add_person_does_not_race(self, db: Database):
        import threading

        errors: list = []

        def add():
            try:
                db.add_person("Ana")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=add) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(db.list_people()) == 1

    def test_created_at_matches_what_was_stored(self, db: Database):
        person = db.add_person("Ana")
        assert person.created_at == db.get_person(person.id).created_at

    def test_close_releases_every_thread_connection(self, db: Database):
        import threading

        def touch():
            db.list_people()

        threads = [threading.Thread(target=touch) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(db._open) >= 5
        db.close()
        assert db._open == []
        assert db.list_people() == []


class TestVersionFilter:
    def test_templates_for_can_filter_by_version(self, db: Database):
        person = db.add_person("Ana")
        db.upsert_template(person.id, "front", unit_vector(1), VERSION)
        db.upsert_template(person.id, "left", unit_vector(2), "OLD")
        assert len(db.templates_for(person.id)) == 2
        assert [t.pose for t in db.templates_for(person.id, VERSION)] == ["front"]
        assert [t.pose for t in db.templates_for(person.id, "OLD")] == ["left"]
