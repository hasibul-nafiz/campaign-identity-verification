from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

EMBEDDING_DIM = 128

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS templates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    pose          TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    model_version TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    UNIQUE(person_id, pose)
);
CREATE INDEX IF NOT EXISTS idx_templates_person ON templates(person_id);
CREATE TABLE IF NOT EXISTS campaigns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT NOT NULL UNIQUE,
    active            INTEGER NOT NULL DEFAULT 1,
    badge_min_inliers INTEGER,
    shirt_delta_e_max REAL,
    glare_suppression INTEGER,
    despecular        INTEGER,
    created_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS campaign_colors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    hex         TEXT NOT NULL,
    label       TEXT NOT NULL DEFAULT '',
    UNIQUE(campaign_id, hex)
);
CREATE TABLE IF NOT EXISTS badge_refs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    keypoints   INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_colors_campaign ON campaign_colors(campaign_id);
CREATE INDEX IF NOT EXISTS idx_badgerefs_campaign ON badge_refs(campaign_id);
"""


class DatabaseError(ValueError):
    pass


@dataclass(frozen=True)
class Person:
    id: int
    name: str
    created_at: str


@dataclass(frozen=True)
class CampaignColor:
    id: int
    hex: str
    label: str


@dataclass(frozen=True)
class BadgeRef:
    id: int
    campaign_id: int
    path: str
    width: int
    height: int
    keypoints: int


@dataclass(frozen=True)
class Campaign:
    id: int
    name: str
    active: bool
    badge_min_inliers: Optional[int]
    shirt_delta_e_max: Optional[float]
    glare_suppression: Optional[bool]
    despecular: Optional[bool]
    created_at: str


@dataclass(frozen=True)
class Template:
    id: int
    person_id: int
    pose: str
    embedding: np.ndarray
    model_version: str


def to_blob(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != EMBEDDING_DIM:
        raise DatabaseError(f"expected {EMBEDDING_DIM}-d embedding, got {arr.shape[0]}")
    norm = float(np.linalg.norm(arr))
    if abs(norm - 1.0) > 1e-3:
        raise DatabaseError(f"embedding must be L2-normalized (norm={norm:.6f})")
    return arr.tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.shape[0] != EMBEDDING_DIM:
        raise DatabaseError(f"corrupt embedding blob: {arr.shape[0]} floats")
    return arr


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thread-local connections: blocking routes run across threadpool workers."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._open: List[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        conn = self.connection()
        conn.executescript(SCHEMA)
        conn.commit()

    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            with self._conn_lock:
                self._open.append(conn)
        return conn

    def close(self) -> None:
        # Every threadpool worker opens its own connection, so closing only the
        # caller's would leak the rest on shutdown.
        with self._conn_lock:
            for conn in self._open:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            self._open.clear()
        self._local = threading.local()

    def add_person(self, name: str) -> Person:
        name = name.strip()
        if not name:
            raise DatabaseError("name must not be empty")
        conn = self.connection()
        # Single atomic statement: a check-then-insert races between threadpool workers.
        conn.execute(
            "INSERT INTO people (name, created_at) VALUES (?, ?) ON CONFLICT(name) DO NOTHING",
            (name, _now()),
        )
        conn.commit()
        person = self.get_person_by_name(name)
        if person is None:
            raise DatabaseError(f"could not create person {name!r}")
        return person

    def get_person(self, person_id: int) -> Optional[Person]:
        row = self.connection().execute(
            "SELECT id, name, created_at FROM people WHERE id = ?", (person_id,)
        ).fetchone()
        return Person(row["id"], row["name"], row["created_at"]) if row else None

    def get_person_by_name(self, name: str) -> Optional[Person]:
        row = self.connection().execute(
            "SELECT id, name, created_at FROM people WHERE name = ?", (name.strip(),)
        ).fetchone()
        return Person(row["id"], row["name"], row["created_at"]) if row else None

    def list_people(self) -> List[Person]:
        rows = self.connection().execute(
            "SELECT id, name, created_at FROM people ORDER BY id"
        ).fetchall()
        return [Person(r["id"], r["name"], r["created_at"]) for r in rows]

    def delete_person(self, person_id: int) -> bool:
        conn = self.connection()
        cur = conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        conn.commit()
        return cur.rowcount > 0

    def upsert_template(
        self, person_id: int, pose: str, embedding: np.ndarray, model_version: str
    ) -> int:
        blob = to_blob(embedding)
        conn = self.connection()
        cur = conn.execute(
            """
            INSERT INTO templates (person_id, pose, embedding, model_version, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(person_id, pose) DO UPDATE SET
                embedding = excluded.embedding,
                model_version = excluded.model_version,
                created_at = excluded.created_at
            """,
            (person_id, pose, blob, model_version, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)

    def delete_template(self, person_id: int, pose: str) -> bool:
        conn = self.connection()
        cur = conn.execute(
            "DELETE FROM templates WHERE person_id = ? AND pose = ?", (person_id, pose)
        )
        conn.commit()
        return cur.rowcount > 0

    def templates_for(self, person_id: int, model_version: Optional[str] = None) -> List[Template]:
        sql = (
            "SELECT id, person_id, pose, embedding, model_version FROM templates "
            "WHERE person_id = ?"
        )
        params: tuple = (person_id,)
        if model_version is not None:
            sql += " AND model_version = ?"
            params += (model_version,)
        rows = self.connection().execute(sql + " ORDER BY pose", params).fetchall()
        return [
            Template(r["id"], r["person_id"], r["pose"], from_blob(r["embedding"]), r["model_version"])
            for r in rows
        ]

    def all_templates(self) -> List[Template]:
        rows = self.connection().execute(
            "SELECT id, person_id, pose, embedding, model_version FROM templates ORDER BY id"
        ).fetchall()
        return [
            Template(r["id"], r["person_id"], r["pose"], from_blob(r["embedding"]), r["model_version"])
            for r in rows
        ]

    def count_templates(self) -> int:
        return int(self.connection().execute("SELECT COUNT(*) c FROM templates").fetchone()["c"])

    # ---- campaigns -------------------------------------------------------

    def _campaign(self, row: sqlite3.Row) -> Campaign:
        return Campaign(
            id=row["id"], name=row["name"], active=bool(row["active"]),
            badge_min_inliers=row["badge_min_inliers"],
            shirt_delta_e_max=row["shirt_delta_e_max"],
            glare_suppression=None if row["glare_suppression"] is None else bool(row["glare_suppression"]),
            despecular=None if row["despecular"] is None else bool(row["despecular"]),
            created_at=row["created_at"],
        )

    def add_campaign(self, name: str) -> Campaign:
        name = name.strip()
        if not name:
            raise DatabaseError("campaign name must not be empty")
        conn = self.connection()
        conn.execute(
            "INSERT INTO campaigns (name, active, created_at) VALUES (?, 1, ?) "
            "ON CONFLICT(name) DO NOTHING",
            (name, _now()),
        )
        conn.commit()
        campaign = self.get_campaign_by_name(name)
        if campaign is None:
            raise DatabaseError(f"could not create campaign {name!r}")
        return campaign

    def get_campaign(self, campaign_id: int) -> Optional[Campaign]:
        row = self.connection().execute(
            "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        return self._campaign(row) if row else None

    def get_campaign_by_name(self, name: str) -> Optional[Campaign]:
        row = self.connection().execute(
            "SELECT * FROM campaigns WHERE name = ?", (name.strip(),)
        ).fetchone()
        return self._campaign(row) if row else None

    def list_campaigns(self, active_only: bool = False) -> List[Campaign]:
        sql = "SELECT * FROM campaigns"
        if active_only:
            sql += " WHERE active = 1"
        rows = self.connection().execute(sql + " ORDER BY id").fetchall()
        return [self._campaign(r) for r in rows]

    def update_campaign(self, campaign_id: int, **fields) -> Optional[Campaign]:
        allowed = {"name", "active", "badge_min_inliers", "shirt_delta_e_max", "glare_suppression", "despecular"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if updates:
            if "name" in updates:
                updates["name"] = str(updates["name"]).strip()
                if not updates["name"]:
                    raise DatabaseError("campaign name must not be empty")
            if "active" in updates:
                updates["active"] = 1 if updates["active"] else 0
            for flag in ("glare_suppression", "despecular"):
                if updates.get(flag) is not None:
                    updates[flag] = 1 if updates[flag] else 0
            assignments = ", ".join(f"{k} = ?" for k in updates)
            conn = self.connection()
            try:
                conn.execute(
                    f"UPDATE campaigns SET {assignments} WHERE id = ?",
                    (*updates.values(), campaign_id),
                )
            except sqlite3.IntegrityError as exc:
                raise DatabaseError(f"campaign name already in use: {exc}") from None
            conn.commit()
        return self.get_campaign(campaign_id)

    def delete_campaign(self, campaign_id: int) -> bool:
        conn = self.connection()
        cur = conn.execute("DELETE FROM campaigns WHERE id = ?", (campaign_id,))
        conn.commit()
        return cur.rowcount > 0

    def set_colors(self, campaign_id: int, colors: List[tuple]) -> List[CampaignColor]:
        """Replaces the whole colour set; `colors` is [(hex, label), ...]."""
        conn = self.connection()
        conn.execute("DELETE FROM campaign_colors WHERE campaign_id = ?", (campaign_id,))
        for hex_value, label in colors:
            conn.execute(
                "INSERT INTO campaign_colors (campaign_id, hex, label) VALUES (?, ?, ?) "
                "ON CONFLICT(campaign_id, hex) DO UPDATE SET label = excluded.label",
                (campaign_id, hex_value, label or ""),
            )
        conn.commit()
        return self.colors_for(campaign_id)

    def colors_for(self, campaign_id: int) -> List[CampaignColor]:
        rows = self.connection().execute(
            "SELECT id, hex, label FROM campaign_colors WHERE campaign_id = ? ORDER BY id",
            (campaign_id,),
        ).fetchall()
        return [CampaignColor(r["id"], r["hex"], r["label"]) for r in rows]

    def add_badge_ref(
        self, campaign_id: int, path: str, width: int, height: int, keypoints: int
    ) -> BadgeRef:
        conn = self.connection()
        cur = conn.execute(
            "INSERT INTO badge_refs (campaign_id, path, width, height, keypoints, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (campaign_id, path, width, height, keypoints, _now()),
        )
        conn.commit()
        return BadgeRef(int(cur.lastrowid), campaign_id, path, width, height, keypoints)

    def badge_refs_for(self, campaign_id: int) -> List[BadgeRef]:
        rows = self.connection().execute(
            "SELECT id, campaign_id, path, width, height, keypoints FROM badge_refs "
            "WHERE campaign_id = ? ORDER BY id",
            (campaign_id,),
        ).fetchall()
        return [
            BadgeRef(r["id"], r["campaign_id"], r["path"], r["width"], r["height"], r["keypoints"])
            for r in rows
        ]

    def get_badge_ref(self, badge_id: int) -> Optional[BadgeRef]:
        row = self.connection().execute(
            "SELECT id, campaign_id, path, width, height, keypoints FROM badge_refs WHERE id = ?",
            (badge_id,),
        ).fetchone()
        return BadgeRef(
            row["id"], row["campaign_id"], row["path"], row["width"], row["height"], row["keypoints"]
        ) if row else None

    def delete_badge_ref(self, badge_id: int) -> Optional[BadgeRef]:
        ref = self.get_badge_ref(badge_id)
        if ref is None:
            return None
        conn = self.connection()
        conn.execute("DELETE FROM badge_refs WHERE id = ?", (badge_id,))
        conn.commit()
        return ref
