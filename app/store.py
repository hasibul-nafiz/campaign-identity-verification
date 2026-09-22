from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from app.config import Settings, get_settings
from app.db import EMBEDDING_DIM, Database, Person


@dataclass(frozen=True)
class IdentifyResult:
    person_id: int
    name: str
    pose: str
    score: float
    matched: bool


@dataclass(frozen=True)
class PersonSummary:
    id: int
    name: str
    created_at: str
    poses: List[str]

    @property
    def template_count(self) -> int:
        return len(self.poses)


class TemplateStore:
    """All templates in one matrix; identify is a single matmul."""

    def __init__(self, db: Database, settings: Optional[Settings] = None):
        self.db = db
        self.settings = settings or get_settings()
        self._lock = threading.RLock()
        self._matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._person_ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self._poses: List[str] = []
        self._names: Dict[int, str] = {}
        self.rebuild()

    def rebuild(self) -> None:
        templates = [
            t for t in self.db.all_templates()
            if t.model_version == self.settings.model_version
        ]
        names = {p.id: p.name for p in self.db.list_people()}
        if templates:
            matrix = np.vstack([t.embedding for t in templates]).astype(np.float32)
            person_ids = np.array([t.person_id for t in templates], dtype=np.int64)
            poses = [t.pose for t in templates]
        else:
            matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            person_ids = np.zeros(0, dtype=np.int64)
            poses = []
        with self._lock:
            # Replaced wholesale, never mutated, so identify() can read a snapshot
            # outside the lock without a writer tearing it mid-multiply.
            self._matrix = np.ascontiguousarray(matrix)
            self._person_ids = person_ids
            self._poses = poses
            self._names = names

    def __len__(self) -> int:
        with self._lock:
            return int(self._matrix.shape[0])

    def identify(self, embedding: np.ndarray) -> Optional[IdentifyResult]:
        with self._lock:
            matrix, person_ids, poses, names = (
                self._matrix, self._person_ids, self._poses, self._names,
            )
        if matrix.shape[0] == 0:
            return None

        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"expected {EMBEDDING_DIM}-d embedding, got {vec.shape[0]}")

        scores = matrix @ vec
        best = int(np.argmax(scores))
        score = float(scores[best])
        person_id = int(person_ids[best])
        return IdentifyResult(
            person_id=person_id,
            name=names.get(person_id, "?"),
            pose=poses[best],
            score=score,
            matched=score >= self.settings.face_match_thresh,
        )

    def register(self, name: str, pose: str, embedding: np.ndarray) -> Person:
        return self.register_many(name, {pose: embedding})

    def register_many(self, name: str, embeddings: Dict[str, np.ndarray]) -> Person:
        if not embeddings:
            raise ValueError("no embeddings to register")
        with self._lock:
            person = self.db.add_person(name)
            for pose, embedding in embeddings.items():
                self.db.upsert_template(person.id, pose, embedding, self.settings.model_version)
            self.rebuild()
        return person

    def delete_person(self, person_id: int) -> bool:
        with self._lock:
            deleted = self.db.delete_person(person_id)
            if deleted:
                self.rebuild()
        return deleted

    def delete_template(self, person_id: int, pose: str) -> bool:
        with self._lock:
            deleted = self.db.delete_template(person_id, pose)
            if deleted:
                self.rebuild()
        return deleted

    def people(self) -> List[PersonSummary]:
        return [
            PersonSummary(
                id=p.id,
                name=p.name,
                created_at=p.created_at,
                # Version-filtered so the count can't disagree with the match matrix.
                poses=[t.pose for t in self.db.templates_for(p.id, self.settings.model_version)],
            )
            for p in self.db.list_people()
        ]
