from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.auth import AuthError, authenticate, create_access_token, require_auth
from app.badge import BadgeError, BadgeMatcher
from app.campaigns import CampaignError, CampaignRegistry, reference_keypoints
from app.config import Settings, get_settings
from app.db import Database, DatabaseError
from app.face import FaceEngine, FaceError, downscale, pose_label, torso_roi
from app.shirt import ShirtError
from app import shirt as shirt_mod
from app.store import TemplateStore

FRONT_POSE = "front"
router = APIRouter(dependencies=[Depends(require_auth)])
public_router = APIRouter()


def domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def get_cfg(request: Request) -> Settings:
    return request.app.state.settings


def create_app(
    settings: Optional[Settings] = None,
    engine: Optional[FaceEngine] = None,
    badge: Optional[BadgeMatcher] = None,
) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(instance: FastAPI):
        instance.state.settings = cfg
        instance.state.engine = engine or FaceEngine(cfg)
        instance.state.badge = badge or BadgeMatcher(settings=cfg)
        instance.state.db = Database(cfg.db_path)
        instance.state.store = TemplateStore(instance.state.db, cfg)
        instance.state.campaigns = CampaignRegistry(instance.state.db, cfg)
        seed_default_campaign(instance.state.db, instance.state.campaigns, cfg)
        try:
            yield
        finally:
            instance.state.db.close()

    instance = FastAPI(title="face-verify", lifespan=lifespan)
    instance.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for exc_type in (FaceError, BadgeError, ShirtError, DatabaseError, CampaignError):
        instance.add_exception_handler(exc_type, domain_error_handler)
    instance.include_router(public_router)
    instance.include_router(router)
    return instance


class Timer:
    def __init__(self) -> None:
        self.marks: Dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def mark(self, name: str):
        begin = time.perf_counter()
        try:
            yield
        finally:
            self.marks[name] = round((time.perf_counter() - begin) * 1000, 2)

    def finish(self) -> Dict[str, float]:
        self.marks["total"] = round((time.perf_counter() - self._start) * 1000, 2)
        return self.marks


@dataclass
class Loaded:
    """Detection runs on `processed`; every coordinate is reported against `original`."""

    original: np.ndarray
    processed: np.ndarray
    scale: float

    @property
    def size(self) -> List[int]:
        return [int(self.original.shape[1]), int(self.original.shape[0])]

    def to_original(self, value: float) -> int:
        return int(round(value / self.scale))

    def point_to_original(self, x: float, y: float) -> List[int]:
        return [self.to_original(x), self.to_original(y)]


def read_image(upload: UploadFile, cfg: Settings) -> Loaded:
    raw = upload.file.read()
    if not raw:
        raise FaceError(f"{upload.filename or 'image'}: empty upload")
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FaceError(f"{upload.filename or 'image'}: not a decodable image")
    processed, scale = downscale(img, cfg.max_side)
    return Loaded(original=img, processed=processed, scale=scale)


def encode_b64(img: np.ndarray, ext: str = ".png") -> Optional[str]:
    ok, buf = cv2.imencode(ext, img)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def crop_polygon_b64(img: np.ndarray, polygon: List[List[int]]) -> Optional[str]:
    """Axis-aligned bounds of the polygon, cut from the full-resolution image."""
    height, width = img.shape[:2]
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, y0 = max(0, min(xs)), max(0, min(ys))
    x1, y1 = min(width, max(xs)), min(height, max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    return encode_b64(img[y0:y1, x0:x1], ".jpg")


def campaign_block(campaign, auto: bool) -> Optional[Dict[str, Any]]:
    if campaign is None:
        return None
    return {"id": campaign.campaign.id, "name": campaign.campaign.name, "auto": auto}


def seed_default_campaign(db: Database, registry: CampaignRegistry, cfg: Settings) -> None:
    """Carry the pre-campaign single badge/colour setup into a real campaign row."""
    if db.list_campaigns():
        return
    if not cfg.badge_ref_path.exists():
        return
    image = cv2.imread(str(cfg.badge_ref_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    campaign = db.add_campaign("Default")
    db.set_colors(campaign.id, [(cfg.shirt_expected_hex, "default")])
    registry.rebuild()
    path = registry.storage_dir(campaign.id) / "seed.png"
    cv2.imwrite(str(path), image)
    db.add_badge_ref(
        campaign.id, str(path), image.shape[1], image.shape[0], reference_keypoints(image, cfg)
    )
    registry.rebuild()


@public_router.post("/login")
def login(
    username: str = Form(...),
    password: str = Form(...),
    cfg: Settings = Depends(get_cfg),
) -> Dict[str, Any]:
    if not authenticate(username, password, cfg):
        raise HTTPException(status_code=401, detail="invalid username or password")
    token = create_access_token(username, cfg)
    return {"access_token": token, "token_type": "bearer"}


@public_router.get("/health")
def health(request: Request, cfg: Settings = Depends(get_cfg)) -> Dict[str, Any]:
    store: TemplateStore = request.app.state.store
    return {
        "status": "ok",
        "model_version": cfg.model_version,
        "models_loaded": True,
        "people": len(store.people()),
        "templates": len(store),
        "badge_reference_keypoints": request.app.state.badge.reference_keypoints,
        "campaigns": len(request.app.state.campaigns.all()),
        "debug": cfg.debug,
    }


@router.get("/badge-ref")
def badge_ref(cfg: Settings = Depends(get_cfg)) -> FileResponse:
    if not cfg.badge_ref_path.exists():
        return JSONResponse(  # type: ignore[return-value]
            status_code=404, content={"detail": "no badge reference configured"}
        )
    return FileResponse(cfg.badge_ref_path, media_type="image/png")


@router.get("/people")
def list_people(request: Request) -> Dict[str, Any]:
    store: TemplateStore = request.app.state.store
    return {
        "people": [
            {
                "id": p.id,
                "name": p.name,
                "created_at": p.created_at,
                "poses": p.poses,
                "template_count": p.template_count,
            }
            for p in store.people()
        ]
    }


@router.delete("/people/{person_id}")
def delete_person(person_id: int, request: Request) -> JSONResponse:
    store: TemplateStore = request.app.state.store
    if not store.delete_person(person_id):
        return JSONResponse(status_code=404, content={"detail": f"no person with id {person_id}"})
    return JSONResponse(status_code=200, content={"deleted": person_id})


@router.post("/register")
def register(
    request: Request,
    name: str = Form(...),
    images: List[UploadFile] = File(...),
    cfg: Settings = Depends(get_cfg),
) -> JSONResponse:
    if not name.strip():
        return JSONResponse(status_code=400, content={"detail": "name must not be empty"})
    if not 1 <= len(images) <= cfg.max_register_images:
        return JSONResponse(
            status_code=400,
            content={"detail": f"expected 1-{cfg.max_register_images} images, got {len(images)}"},
        )

    engine: FaceEngine = request.app.state.engine
    store: TemplateStore = request.app.state.store

    # Everything is processed before anything is written: a partial commit could
    # leave a person stored with no front template.
    statuses: List[Dict[str, Any]] = []
    accepted: Dict[str, np.ndarray] = {}
    for index, upload in enumerate(images):
        row: Dict[str, Any] = {"index": index, "filename": upload.filename, "ok": False,
                               "pose": None, "error": None}
        try:
            img = read_image(upload, cfg).processed
            face = engine.detect_single(img)
            pose = pose_label(face, cfg).label
            embedding = engine.embed(img, face)
        except (FaceError, ValueError) as exc:
            row["error"] = str(exc)
            statuses.append(row)
            continue
        row["ok"] = True
        row["pose"] = pose
        if pose in accepted:
            row["replaced_earlier_image"] = True
        accepted[pose] = embedding
        statuses.append(row)

    if FRONT_POSE not in accepted:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "at least one image with a 'front' pose is required",
                "images": statuses,
            },
        )

    person = store.register_many(name, accepted)

    return JSONResponse(
        status_code=201,
        content={
            "person": {"id": person.id, "name": person.name},
            "images": statuses,
            "templates_written": len(accepted),
            "poses": sorted(accepted),
        },
    )


@router.post("/detect")
def detect(
    request: Request,
    image: UploadFile = File(...),
    expected_color: Optional[str] = Form(None),
    campaign_id: Optional[int] = Form(None),
    cfg: Settings = Depends(get_cfg),
) -> Dict[str, Any]:
    engine: FaceEngine = request.app.state.engine
    store: TemplateStore = request.app.state.store
    registry: CampaignRegistry = request.app.state.campaigns

    # campaign_id pins the answer to one campaign ("is this the X badge?").
    # Without it the badge itself decides which campaign it belongs to.
    campaign = registry.get(campaign_id) if campaign_id is not None else None

    timer = Timer()
    with timer.mark("decode"):
        frame = read_image(image, cfg)
    img = frame.processed
    with timer.mark("detect"):
        face = engine.detect_single(img)
    pose = pose_label(face, cfg)
    with timer.mark("embed"):
        embedding = engine.embed(img, face)
    with timer.mark("identify"):
        match = store.identify(embedding)

    face_block = {
        "box": [
            frame.to_original(face.x),
            frame.to_original(face.y),
            frame.to_original(face.w),
            frame.to_original(face.h),
        ],
        "score": round(face.score, 4),
        "pose": pose.label,
        "yaw": round(pose.yaw, 4),
        "pitch": round(pose.pitch, 4),
    }
    person_block = (
        {
            "id": match.person_id,
            "name": match.name,
            "score": round(match.score, 4),
            "matched_pose": match.pose,
        }
        if match is not None and match.matched
        else None
    )

    response: Dict[str, Any] = {
        "passed": False,
        "person": person_block,
        "face": face_block,
        "badge": None,
        "shirt": None,
        "campaign": campaign_block(campaign, campaign_id is None),
        "image_size": frame.size,
        "torso_box": None,
        "badge_crop_b64": None,
        "best_score": round(match.score, 4) if match is not None else None,
    }

    # Badge and shirt are skipped entirely on an unknown face - no point paying
    # for SIFT when the identity gate already failed.
    if person_block is None:
        response["reason"] = "face not recognised" if match is not None else "no templates enrolled"
        response["timings_ms"] = timer.finish()
        return response

    tx, ty, tw, th = torso_roi(face, img.shape, cfg)
    # Cut the torso from the ORIGINAL frame: the badge is often only ~70px wide in the
    # downscaled copy, which is below where SIFT matching stays reliable.
    oh, ow = frame.original.shape[:2]
    ox0, oy0 = max(0, frame.to_original(tx)), max(0, frame.to_original(ty))
    ox1, oy1 = min(ow, frame.to_original(tx + tw)), min(oh, frame.to_original(ty + th))
    if ox1 <= ox0 or oy1 <= oy0:
        raise FaceError("torso region falls outside the image")
    torso = frame.original[oy0:oy1, ox0:ox1]
    response["torso_box"] = [ox0, oy0, ox1, oy1]

    # The badge is usually on the chest, but it may be on a sleeve, a cap or a lanyard.
    # The torso is searched first because it is roughly six times fewer pixels and it
    # excludes the face and the room -- both of which otherwise compete with a small
    # glossy badge for keypoints and for the match itself. The whole frame is only
    # searched when the torso comes up empty. The shirt check always uses the torso,
    # because that is what the shirt colour means.
    regions = {
        "torso": [(torso, (ox0, oy0))],
        "frame": [(frame.original, (0, 0))],
    }.get(
        cfg.badge_search_region,
        [(torso, (ox0, oy0)), (frame.original, (0, 0))],
    )

    candidates: List[Dict[str, Any]] = []
    searched = ""
    with timer.mark("badge"):
        badge_match = None
        for index, (badge_region, badge_origin) in enumerate(regions):
            searched = "torso" if badge_region is torso else "frame"
            if campaign is not None:
                badge_match = registry.check_badge(campaign.campaign.id, badge_region)
            else:
                ranked = registry.identify(badge_region)
                if not ranked:
                    raise CampaignError("no active campaign has a badge reference image")
                candidates = [
                    {
                        "id": c.campaign.campaign.id,
                        "name": c.campaign.campaign.name,
                        "inliers": c.inliers,
                        "ok": c.match.result.ok,
                    }
                    for c in ranked[:5]
                ]
                badge_match = ranked[0].match
                if badge_match.result.ok or index == len(regions) - 1:
                    campaign = ranked[0].campaign
            if badge_match.result.ok:
                break
    assert badge_match is not None
    badge_result = badge_match.result
    response["campaign"] = campaign_block(campaign, campaign_id is None)
    if candidates:
        response["campaign_candidates"] = candidates

    with timer.mark("shirt"):
        if expected_color:
            shirt_result, matched_hex = shirt_mod.check_any(
                torso, [expected_color], campaign.settings
            )
        else:
            shirt_result, matched_hex = registry.check_shirt(campaign.campaign.id, torso)

    badge_polygon = None
    if badge_result.quad is not None:
        # The crop is already full-resolution, so the quad only needs the torso origin.
        badge_polygon = [
            [int(round(px)) + badge_origin[0], int(round(py)) + badge_origin[1]]
            for px, py in badge_result.quad
        ]
        response["badge_crop_b64"] = crop_polygon_b64(frame.original, badge_polygon)

    response["badge"] = {
        "ok": badge_result.ok,
        "inliers": badge_result.inliers,
        "matches": badge_result.matches,
        "reason": badge_result.reason,
        "box": badge_polygon,
        "matched_ref_id": badge_match.ref_id,
        "searched": searched,
    }
    response["shirt"] = {
        "ok": shirt_result.ok,
        "delta_e": round(shirt_result.delta_e, 3),
        "rgb": list(shirt_result.rgb),
        "expected_rgb": list(shirt_result.expected_rgb),
        "expected_hex": matched_hex,
        "matched_color": matched_hex,
        "coverage": round(shirt_result.coverage, 3),
    }
    response["passed"] = bool(badge_result.ok and shirt_result.ok)
    if cfg.debug:
        response["torso_b64"] = encode_b64(torso, ".png")
        # Keeps the exact frame the matcher saw, so a failure can be reproduced
        # offline instead of guessed at from a screenshot.
        debug_dir = cfg.db_path.parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "last_detect.png"), frame.original)
        cv2.imwrite(str(debug_dir / "last_torso.png"), torso)
    response["timings_ms"] = timer.finish()
    return response


def parse_color_spec(spec: str) -> tuple:
    """"#FFFFFF" or "#FFFFFF:home" -> (normalised hex, label)."""
    text = spec.strip()
    label = ""
    if ":" in text:
        text, label = text.split(":", 1)
    rgb = shirt_mod.parse_hex(text)
    return "#{:02X}{:02X}{:02X}".format(*rgb), label.strip()


def campaign_payload(loaded, include_refs: bool = True) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "id": loaded.campaign.id,
        "name": loaded.campaign.name,
        "active": loaded.campaign.active,
        "created_at": loaded.campaign.created_at,
        "colors": [{"id": c.id, "hex": c.hex, "label": c.label} for c in loaded.colors],
        "badge_ref_count": len(loaded.refs),
        "overrides": {
            "badge_min_inliers": loaded.campaign.badge_min_inliers,
            "shirt_delta_e_max": loaded.campaign.shirt_delta_e_max,
        },
        "effective": {
            "badge_min_inliers": loaded.settings.badge_min_inliers,
            "shirt_delta_e_max": loaded.settings.shirt_delta_e_max,
        },
    }
    if include_refs:
        body["badge_refs"] = [
            {
                "id": entry.ref.id,
                "width": entry.ref.width,
                "height": entry.ref.height,
                "keypoints": entry.ref.keypoints,
                "url": f"/campaigns/{loaded.campaign.id}/badges/{entry.ref.id}/image",
            }
            for entry in loaded.refs
        ]
    return body


def store_badge_ref(
    request: Request, campaign_id: int, upload: UploadFile, cfg: Settings
) -> Dict[str, Any]:
    """Save one reference, gating on feature count and agreement with existing refs."""
    registry: CampaignRegistry = request.app.state.campaigns
    db: Database = request.app.state.db
    row: Dict[str, Any] = {"filename": upload.filename, "ok": False, "error": None,
                           "keypoints": 0, "id": None}

    raw = upload.file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
    if image is None:
        row["error"] = "not a decodable image"
        return row

    # Reported before anything else can fail, so a rejected upload still says how
    # much detail the image had rather than a meaningless zero.
    row["keypoints"] = reference_keypoints(image, cfg)

    # Every reference arrives already cropped to the badge by the UI. Auto-locating the
    # badge inside a wider photo was tried and removed: on a glossy badge the background
    # carries far more texture, so the homography that "found" it was usually fitted to
    # the table, and a full frame got stored as a reference.
    if registry.get(campaign_id).refs:
        existing = registry.cross_match(campaign_id, image)
        if existing is not None and not existing.ok:
            row["error"] = (
                f"this does not look like the same badge as the existing references "
                f"({existing.inliers} inliers, {existing.reason}). Crop tightly to the "
                "badge, or check you picked the right campaign."
            )
            return row
        # Matching the badge is not enough; it has to dominate the image being stored.
        # A badge filling a tenth of a photo spends nine tenths of its feature budget
        # describing the table it sat on, which is what every later match then compares.
        if existing is not None and existing.quad is not None:
            fill = float(abs(cv2.contourArea(existing.quad.astype(np.float32)))) / float(
                image.shape[0] * image.shape[1]
            )
            row["fill"] = round(fill, 3)
            if fill < cfg.badge_ref_min_fill:
                row["error"] = (
                    f"the badge fills only {fill * 100:.0f}% of this image "
                    f"(needs {cfg.badge_ref_min_fill * 100:.0f}%). Crop tightly to the "
                    "badge — background texture, not the badge, would be stored."
                )
                return row

    keypoints = reference_keypoints(image, cfg)
    row["keypoints"] = keypoints
    if keypoints < cfg.badge_min_ref_keypoints:
        row["error"] = (
            f"too few features ({keypoints} < {cfg.badge_min_ref_keypoints}); "
            "crop tightly to the badge and shoot it closer, flatter and better lit"
        )
        return row

    path = registry.storage_dir(campaign_id) / f"{uuid4().hex}.png"
    cv2.imwrite(str(path), image)
    ref = db.add_badge_ref(campaign_id, str(path), image.shape[1], image.shape[0], keypoints)
    row.update(ok=True, id=ref.id)
    return row


@router.get("/campaigns")
def list_campaigns(request: Request) -> Dict[str, Any]:
    registry: CampaignRegistry = request.app.state.campaigns
    return {"campaigns": [campaign_payload(c, include_refs=False) for c in registry.all()]}


@router.get("/campaigns/{campaign_id}")
def get_campaign(campaign_id: int, request: Request) -> Dict[str, Any]:
    registry: CampaignRegistry = request.app.state.campaigns
    return campaign_payload(registry.get(campaign_id))


@router.post("/campaigns")
def create_campaign(
    request: Request,
    name: str = Form(...),
    colors: List[str] = Form(...),
    images: List[UploadFile] = File(default=[]),
    cfg: Settings = Depends(get_cfg),
) -> JSONResponse:
    db: Database = request.app.state.db
    registry: CampaignRegistry = request.app.state.campaigns
    if db.get_campaign_by_name(name.strip()) is not None:
        return JSONResponse(status_code=409, content={"detail": f"campaign {name!r} already exists"})
    if len(images) > cfg.max_badge_refs:
        return JSONResponse(
            status_code=400,
            content={"detail": f"at most {cfg.max_badge_refs} badge references"},
        )

    campaign = db.add_campaign(name)
    db.set_colors(campaign.id, [parse_color_spec(c) for c in colors])
    registry.rebuild()

    rows = []
    for upload in images:
        rows.append(store_badge_ref(request, campaign.id, upload, cfg))
        registry.rebuild()  # so the next upload cross-matches against this one
    return JSONResponse(
        status_code=201,
        content={**campaign_payload(registry.get(campaign.id)), "uploads": rows},
    )


@router.patch("/campaigns/{campaign_id}")
def update_campaign(
    campaign_id: int,
    request: Request,
    name: Optional[str] = Form(None),
    active: Optional[bool] = Form(None),
    colors: Optional[List[str]] = Form(None),
    badge_min_inliers: Optional[int] = Form(None),
    shirt_delta_e_max: Optional[float] = Form(None),
) -> Dict[str, Any]:
    db: Database = request.app.state.db
    registry: CampaignRegistry = request.app.state.campaigns
    registry.get(campaign_id)

    fields: Dict[str, Any] = {}
    for key, value in (
        ("name", name), ("active", active),
        ("badge_min_inliers", badge_min_inliers), ("shirt_delta_e_max", shirt_delta_e_max),
    ):
        if value is not None:
            fields[key] = value
    if fields:
        db.update_campaign(campaign_id, **fields)
    if colors is not None:
        db.set_colors(campaign_id, [parse_color_spec(c) for c in colors])
    registry.rebuild()
    return campaign_payload(registry.get(campaign_id))


@router.delete("/campaigns/{campaign_id}")
def delete_campaign(campaign_id: int, request: Request) -> JSONResponse:
    db: Database = request.app.state.db
    registry: CampaignRegistry = request.app.state.campaigns
    paths = [r.path for r in db.badge_refs_for(campaign_id)]
    if not db.delete_campaign(campaign_id):
        return JSONResponse(status_code=404, content={"detail": f"no campaign with id {campaign_id}"})
    for path in paths:
        Path(path).unlink(missing_ok=True)
    registry.rebuild()
    return JSONResponse(status_code=200, content={"deleted": campaign_id})


@router.post("/campaigns/{campaign_id}/badges")
def add_badge_refs(
    campaign_id: int,
    request: Request,
    images: List[UploadFile] = File(...),
    cfg: Settings = Depends(get_cfg),
) -> JSONResponse:
    registry: CampaignRegistry = request.app.state.campaigns
    loaded = registry.get(campaign_id)
    if len(loaded.refs) + len(images) > cfg.max_badge_refs:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"campaign would exceed {cfg.max_badge_refs} badge references "
                          f"(has {len(loaded.refs)})"
            },
        )
    rows = []
    for upload in images:
        rows.append(store_badge_ref(request, campaign_id, upload, cfg))
        registry.rebuild()
    accepted = sum(1 for r in rows if r["ok"])
    return JSONResponse(
        status_code=201 if accepted else 400,
        content={**campaign_payload(registry.get(campaign_id)), "uploads": rows},
    )


@router.delete("/campaigns/{campaign_id}/badges/{badge_id}")
def delete_badge_ref(campaign_id: int, badge_id: int, request: Request) -> JSONResponse:
    db: Database = request.app.state.db
    registry: CampaignRegistry = request.app.state.campaigns
    ref = db.get_badge_ref(badge_id)
    if ref is None or ref.campaign_id != campaign_id:
        return JSONResponse(status_code=404, content={"detail": f"no badge reference {badge_id}"})
    db.delete_badge_ref(badge_id)
    Path(ref.path).unlink(missing_ok=True)
    registry.rebuild()
    return JSONResponse(status_code=200, content={"deleted": badge_id})


@router.get("/campaigns/{campaign_id}/badges/{badge_id}/image")
def badge_ref_image(campaign_id: int, badge_id: int, request: Request):
    db: Database = request.app.state.db
    ref = db.get_badge_ref(badge_id)
    if ref is None or ref.campaign_id != campaign_id or not Path(ref.path).exists():
        return JSONResponse(status_code=404, content={"detail": f"no badge reference {badge_id}"})
    return FileResponse(ref.path, media_type="image/png")


app = create_app()
