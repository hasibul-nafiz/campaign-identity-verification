from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app import shirt as shirt_mod
from app.badge import BadgeError, BadgeMatcher, BadgeResult
from app.config import Settings, get_settings
from app.db import BadgeRef, Campaign, CampaignColor, Database
from app.shirt import ShirtResult


class CampaignError(ValueError):
    pass


@dataclass
class LoadedRef:
    ref: BadgeRef
    matcher: BadgeMatcher


@dataclass
class LoadedCampaign:
    campaign: Campaign
    colors: List[CampaignColor]
    refs: List[LoadedRef]
    settings: Settings

    @property
    def hexes(self) -> List[str]:
        return [c.hex for c in self.colors]


@dataclass(frozen=True)
class BadgeMatch:
    result: BadgeResult
    ref_id: Optional[int]


@dataclass(frozen=True)
class CampaignMatch:
    campaign: "LoadedCampaign"
    match: BadgeMatch

    @property
    def inliers(self) -> int:
        return self.match.result.inliers


def reference_keypoints(image: np.ndarray, settings: Optional[Settings] = None) -> int:
    """Feature count a candidate reference would contribute, for the upload gate."""
    cfg = settings or get_settings()
    gray = image if image.ndim == 2 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    keypoints, _ = cv2.SIFT.create(nfeatures=cfg.badge_nfeatures).detectAndCompute(gray, None)
    return len(keypoints)


class CampaignRegistry:
    """Campaigns with their badge descriptors pre-extracted. Rebuilt on change."""

    def __init__(self, db: Database, settings: Optional[Settings] = None):
        self.db = db
        self.settings = settings or get_settings()
        self._lock = threading.RLock()
        self._loaded: Dict[int, LoadedCampaign] = {}
        # Trying the reference that matched last time first turns the common case
        # into a single match instead of a scan over every reference.
        self._last_matched: Dict[int, int] = {}
        self.rebuild()

    def rebuild(self) -> None:
        loaded: Dict[int, LoadedCampaign] = {}
        for campaign in self.db.list_campaigns():
            cfg = self.settings
            overrides = {}
            if campaign.badge_min_inliers is not None:
                overrides["badge_min_inliers"] = campaign.badge_min_inliers
                # Setting the inlier bar by hand means inliers are the authority for this
                # campaign, so the colour rescue must not quietly reach underneath it.
                overrides["badge_colour_rescue"] = False
            if campaign.shirt_delta_e_max is not None:
                overrides["shirt_delta_e_max"] = campaign.shirt_delta_e_max
            if overrides:
                cfg = replace(self.settings, **overrides)

            refs: List[LoadedRef] = []
            for ref in self.db.badge_refs_for(campaign.id):
                image = cv2.imread(ref.path, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                try:
                    refs.append(LoadedRef(ref, BadgeMatcher(image, cfg)))
                except BadgeError:
                    continue
            loaded[campaign.id] = LoadedCampaign(
                campaign=campaign,
                colors=self.db.colors_for(campaign.id),
                refs=refs,
                settings=cfg,
            )
        with self._lock:
            self._loaded = loaded

    def get(self, campaign_id: int) -> LoadedCampaign:
        with self._lock:
            loaded = self._loaded.get(campaign_id)
        if loaded is None:
            raise CampaignError(f"no campaign with id {campaign_id}")
        return loaded

    def all(self) -> List[LoadedCampaign]:
        with self._lock:
            return list(self._loaded.values())

    def resolve_default(self) -> Optional[LoadedCampaign]:
        """The single active campaign, if there is exactly one."""
        active = [c for c in self.all() if c.campaign.active]
        return active[0] if len(active) == 1 else None

    def check_badge(self, campaign_id: int, crop: np.ndarray) -> BadgeMatch:
        loaded = self.get(campaign_id)
        if not loaded.refs:
            raise CampaignError(
                f"campaign '{loaded.campaign.name}' has no badge reference images"
            )

        with self._lock:
            preferred = self._last_matched.get(campaign_id)
        order = sorted(loaded.refs, key=lambda r: r.ref.id != preferred)

        best: Optional[Tuple[BadgeResult, int]] = None
        for entry in order:
            result = entry.matcher.check(crop)
            if best is None or result.inliers > best[0].inliers:
                best = (result, entry.ref.id)
            if result.ok:
                with self._lock:
                    self._last_matched[campaign_id] = entry.ref.id
                return BadgeMatch(result, entry.ref.id)

        assert best is not None
        return BadgeMatch(best[0], best[1] if best[0].quad is not None else None)

    def identify(
        self, image: np.ndarray, active_only: bool = True
    ) -> List[CampaignMatch]:
        """Which campaign's badge is in this image? Best first.

        The query is described once and reused against every reference, so the cost of
        an extra campaign is a few milliseconds of matching rather than another SIFT pass.
        """
        candidates = [c for c in self.all() if c.refs and (c.campaign.active or not active_only)]
        if not candidates:
            return []

        prober = candidates[0].refs[0].matcher
        query = prober.describe(image)

        results: List[CampaignMatch] = []
        for loaded in candidates:
            best: Optional[Tuple[BadgeResult, int]] = None
            for entry in loaded.refs:
                result = entry.matcher.check_prepared(query)
                if best is None or result.inliers > best[0].inliers:
                    best = (result, entry.ref.id)
                if result.ok:
                    break
            if best is not None:
                results.append(CampaignMatch(loaded, BadgeMatch(best[0], best[1])))

        # A campaign that passed its own threshold always outranks one that did not.
        results.sort(key=lambda c: (c.match.result.ok, c.inliers), reverse=True)
        return results

    def check_shirt(self, campaign_id: int, crop: np.ndarray) -> Tuple[ShirtResult, str]:
        loaded = self.get(campaign_id)
        if not loaded.colors:
            raise CampaignError(
                f"campaign '{loaded.campaign.name}' has no expected colours"
            )
        return shirt_mod.check_any(crop, loaded.hexes, loaded.settings)

    def locate(self, campaign_id: int, image: np.ndarray, margin: float = 0.12):
        """Find this campaign's badge inside a wider photo and return just that region.

        Lets someone upload a plain webcam frame as an extra reference: an existing
        reference is used to find the badge, and the crop is what gets stored. A full
        frame stored as-is would be useless, since references are downscaled to
        `badge_ref_max_side` and the badge would shrink to a dozen pixels.
        """
        loaded = self.get(campaign_id)
        best_quad, best_inliers = None, 0
        for entry in loaded.refs:
            result = entry.matcher.check(image)
            if result.ok and result.quad is not None and result.inliers > best_inliers:
                best_quad, best_inliers = result.quad, result.inliers
        if best_quad is None:
            return None

        height, width = image.shape[:2]
        xs, ys = best_quad[:, 0], best_quad[:, 1]
        pad_x = (xs.max() - xs.min()) * margin
        pad_y = (ys.max() - ys.min()) * margin
        x0 = int(max(0, xs.min() - pad_x))
        y0 = int(max(0, ys.min() - pad_y))
        x1 = int(min(width, xs.max() + pad_x))
        y1 = int(min(height, ys.max() + pad_y))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        return image[y0:y1, x0:x1]

    def cross_match(self, campaign_id: int, image: np.ndarray) -> Optional[BadgeResult]:
        """Best score of a candidate reference against the campaign's existing ones.

        None when the campaign has no references yet. A candidate that matches nothing
        is probably a different badge or a bad photo, and should be rejected at upload.
        """
        loaded = self.get(campaign_id)
        if not loaded.refs:
            return None
        best: Optional[BadgeResult] = None
        for entry in loaded.refs:
            result = entry.matcher.check(image)
            if best is None or result.inliers > best.inliers:
                best = result
        return best

    def storage_dir(self, campaign_id: int) -> Path:
        path = self.settings.campaign_assets_dir / str(campaign_id)
        path.mkdir(parents=True, exist_ok=True)
        return path
