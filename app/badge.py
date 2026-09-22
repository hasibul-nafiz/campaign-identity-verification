from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from app.config import Settings, get_settings


class BadgeError(ValueError):
    pass


@dataclass
class QueryFeatures:
    """SIFT output for one query image under one preprocessing variant."""

    shape: Tuple[int, int]
    keypoints: tuple
    descriptors: Optional[np.ndarray]
    index: Optional[Any] = None
    """FLANN index over `descriptors`, so every reference searches one prebuilt tree."""


class Query:
    """Lazily describes a query image per variant, shared across every reference."""

    def __init__(self, image: np.ndarray, sift, lock, rootsift: bool = True):
        self._image = image
        self._sift = sift
        self._lock = lock
        self._rootsift = rootsift
        self._cache: dict = {}

    @property
    def image(self) -> np.ndarray:
        return self._image

    def features(self, variant: str) -> QueryFeatures:
        cached = self._cache.get(variant)
        if cached is None:
            gray = _prepare(self._image, variant)
            with self._lock:
                keypoints, descriptors = self._sift.detectAndCompute(gray, None)
            if self._rootsift:
                descriptors = _rootsift(descriptors)
            cached = QueryFeatures(
                gray.shape[:2], keypoints, descriptors, _build_index(descriptors)
            )
            self._cache[variant] = cached
        return cached


@dataclass
class _Shaped:
    """Carries just the query shape into the geometry sanity check."""

    shape: Tuple[int, int]


@dataclass
class _Level:
    shape: Tuple[int, int]
    points: np.ndarray
    descriptors: np.ndarray
    corners: np.ndarray
    variant: str


@dataclass(frozen=True)
class BadgeResult:
    ok: bool
    inliers: int
    matches: int
    reason: str
    quad: Optional[np.ndarray] = None
    """Reference corners projected into crop-local pixels; None when no sane match."""


def _as_gray(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        raise BadgeError("empty image")
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise BadgeError(f"unsupported image shape {img.shape}")


def _despecular(img: np.ndarray) -> np.ndarray:
    """Paint over mirror reflections so they stop being treated as badge detail.

    A blown-out, desaturated pixel on a glossy badge is the room reflected in it, and
    it moves with the wearer. Removing those before feature extraction leaves the
    printing underneath, which does not move.
    """
    if img.ndim != 3 or img.shape[2] < 3:
        return img
    bgr = img[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 60)).astype(np.uint8)
    if not mask.any():
        return bgr
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)


# Embroidery matches best raw; a glossy badge only matches once its reflections are
# painted out. Rather than asking which kind a badge is, both are built and whichever
# scores is used.
VARIANTS = ("plain", "deglare")


def _make_sift(cfg: Settings, nfeatures: Optional[int] = None):
    """A glossy badge is smooth next to wood grain or fabric weave, so its keypoints
    score low on contrast. The OpenCV default (0.04) discards most of them."""
    return cv2.SIFT.create(
        nfeatures=cfg.badge_nfeatures if nfeatures is None else nfeatures,
        contrastThreshold=cfg.badge_contrast_thresh,
    )


def _build_index(descriptors: Optional[np.ndarray]):
    """KD-tree over the query's descriptors.

    Uncapping the query means thousands of descriptors, which brute force cannot afford
    once every reference and scale searches them. The tree is built once per query and
    reused, so more keypoints cost detection time but not matching time.
    """
    if descriptors is None or len(descriptors) < 2:
        return None
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=4), dict(checks=48)
    )
    flann.add([np.asarray(descriptors, dtype=np.float32)])
    flann.train()
    return flann


def _rootsift(descriptors: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """L1-normalize then square-root, which makes L2 distance a Hellinger distance.

    Same descriptors, same cost, consistently better matching under the illumination
    changes this badge suffers from.
    """
    if descriptors is None or len(descriptors) == 0:
        return descriptors
    d = descriptors.astype(np.float32)
    d /= np.linalg.norm(d, ord=1, axis=1, keepdims=True) + 1e-7
    return np.sqrt(d)


def _prepare(img: np.ndarray, variant: str) -> np.ndarray:
    if variant == "plain":
        return _as_gray(img)
    gray = _as_gray(_despecular(img))
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)


def _quad_is_sane(
    quad: np.ndarray,
    crop_shape: Tuple[int, ...],
    cfg: Settings,
    expected_aspect: Optional[float] = None,
) -> Tuple[bool, str]:
    if not np.all(np.isfinite(quad)):
        return False, "non-finite homography"

    edges = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(edges, axis=1)
    if float(lengths.min()) < 1e-3:
        return False, "collapsed edge"
    if float(lengths.max() / lengths.min()) > cfg.badge_max_edge_ratio:
        return False, "extreme edge ratio"

    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False, "non-convex quad"

    # A badge keeps its proportions. A fit that stretched the reference into a sliver
    # matched something that is not this badge, however many inliers agreed on it.
    width = float((lengths[0] + lengths[2]) / 2.0)
    height = float((lengths[1] + lengths[3]) / 2.0)
    if expected_aspect and height > 1e-6:
        skew = (width / height) / expected_aspect
        if skew < 1.0 / cfg.badge_max_aspect_skew or skew > cfg.badge_max_aspect_skew:
            return False, "badge shape distorted"

    area = float(abs(cv2.contourArea(quad.astype(np.float32))))
    crop_area = float(crop_shape[0] * crop_shape[1])
    if crop_area <= 0:
        return False, "empty crop"
    # Absolute, not a fraction of the search region. A badge worn on a shirt covers well
    # under 1% of a 1280x720 frame, so a fractional floor threw away correct matches for
    # no reason other than the frame being wide. What actually matters is whether enough
    # pixels landed on the badge to trust the fit, and that is a pixel count.
    if area < cfg.badge_min_area_px:
        return False, "projection too small"
    if area / crop_area > cfg.badge_max_area_ratio:
        return False, "projection too large"
    return True, "ok"


def _chroma_signature(bgr: np.ndarray, size: int = 24) -> Optional[np.ndarray]:
    """A coarse map of *colour* across the badge, ignoring brightness.

    Below roughly 70px of badge width there is no longer enough print detail for SIFT
    to reach a confident inlier count, but the badge's colour layout -- navy wordmark,
    red arrow, cyan arrow, gold field -- survives far smaller. Lab's a/b channels carry
    that layout while dropping L, which is exactly the channel a mirror finish ruins.
    """
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return None
    lab = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2LAB)
    small = cv2.resize(lab, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    ab = small[:, :, 1:3].reshape(-1)
    ab -= ab.mean()
    norm = float(np.linalg.norm(ab))
    return ab / norm if norm > 1e-6 else None


def _fit(
    src: np.ndarray,
    dst: np.ndarray,
    corners: np.ndarray,
    crop_shape: Tuple[int, ...],
    cfg: Settings,
    expected_aspect: Optional[float] = None,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], str]:
    """Best sane projection of the reference corners, trying two motion models.

    A full homography has 8 degrees of freedom, which a domed badge photographed at an
    angle happily overfits into a skewed quad that the sanity check then throws away --
    losing a detection that was really there. A 4-DoF similarity cannot bend that way,
    so it is tried as well and whichever survives with more inliers wins.
    """
    attempts = (
        ("homography", lambda: cv2.findHomography(src, dst, cv2.USAC_MAGSAC, cfg.badge_ransac_reproj)),
        ("similarity", lambda: cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.badge_ransac_reproj)),
    )

    best_inliers, best_quad, best_matrix, best_reason = 0, None, None, "no homography"
    for name, run in attempts:
        matrix, mask = run()
        if matrix is None or mask is None:
            continue
        if matrix.shape == (2, 3):
            matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
        inliers = int(mask.sum())
        quad = cv2.perspectiveTransform(corners, matrix).reshape(4, 2)
        sane, reason = _quad_is_sane(quad, crop_shape, cfg, expected_aspect)
        if not sane:
            if best_quad is None and inliers >= best_inliers:
                best_inliers, best_reason = inliers, reason
            continue
        if best_quad is None or inliers > best_inliers:
            best_inliers, best_quad, best_matrix, best_reason = inliers, quad, matrix, name
    return best_inliers, best_quad, best_matrix, best_reason


class BadgeMatcher:
    """Reference SIFT features are computed once at construction."""

    def __init__(
        self, reference: Optional[np.ndarray] = None, settings: Optional[Settings] = None
    ):
        cfg = settings or get_settings()
        if reference is None:
            if not cfg.badge_ref_path.exists():
                raise FileNotFoundError(f"badge reference not found: {cfg.badge_ref_path}")
            reference = cv2.imread(str(cfg.badge_ref_path), cv2.IMREAD_COLOR)
            if reference is None:
                raise BadgeError(f"could not read badge reference: {cfg.badge_ref_path}")

        self.settings = cfg
        self._sift = _make_sift(cfg)
        self._query_sift = _make_sift(cfg, cfg.badge_query_nfeatures)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._lock = threading.Lock()

        # One fixed reference size only matches badges that happen to appear at roughly
        # that scale; anything much smaller or larger degenerates. Descriptors are built
        # at several scales instead, so the badge can be any size on screen.
        self._levels: List[_Level] = []
        for variant in VARIANTS:
            gray = _prepare(reference, variant)
            longest = max(gray.shape[:2])
            for side in sorted({min(s, longest) for s in cfg.badge_ref_scales}, reverse=True):
                factor = side / float(longest)
                scaled = gray if factor >= 1.0 else cv2.resize(
                    gray,
                    (max(1, round(gray.shape[1] * factor)), max(1, round(gray.shape[0] * factor))),
                    interpolation=cv2.INTER_AREA,
                )
                keypoints, descriptors = self._sift.detectAndCompute(scaled, None)
                if descriptors is None or len(keypoints) < 4:
                    continue
                if cfg.badge_rootsift:
                    descriptors = _rootsift(descriptors)
                h, w = scaled.shape[:2]
                self._levels.append(
                    _Level(
                        shape=(h, w),
                        points=np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2),
                        descriptors=descriptors,
                        corners=np.float32(
                            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
                        ).reshape(-1, 1, 2),
                        variant=variant,
                    )
                )
        if not self._levels:
            raise BadgeError("badge reference has too little texture for SIFT")
        self._ref_shape = self._levels[0].shape
        self._ref_sig = _chroma_signature(reference)
        self._preferred = 0

    @property
    def reference_keypoints(self) -> int:
        return len(self._levels[0].descriptors)

    @property
    def scales(self) -> List[int]:
        return sorted({max(level.shape) for level in self._levels}, reverse=True)

    @property
    def variants(self) -> List[str]:
        return sorted({level.variant for level in self._levels})

    def _check_level(self, level: "_Level", query: "Query") -> BadgeResult:
        cfg = self.settings
        features = query.features(level.variant)
        keypoints, descriptors = features.keypoints, features.descriptors
        if descriptors is None or len(keypoints) < 2:
            return BadgeResult(False, 0, 0, "no features in crop")
        gray = _Shaped(features.shape)
        with self._lock:
            if features.index is not None:
                knn = features.index.knnMatch(level.descriptors, k=2)
            else:
                knn = self._matcher.knnMatch(level.descriptors, descriptors, k=2)

        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            nearest, second = pair
            if nearest.distance < cfg.badge_ratio * second.distance:
                good.append(nearest)
        if len(good) < 4:
            return BadgeResult(False, 0, len(good), "too few ratio-test matches")

        src = np.float32([level.points[m.queryIdx][0] for m in good]).reshape(-1, 1, 2)
        dst = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        aspect = level.shape[1] / float(level.shape[0])
        inliers, quad, matrix, reason = _fit(
            src, dst, level.corners, gray.shape, cfg, aspect
        )
        if quad is None:
            return BadgeResult(False, inliers, len(good), reason)
        if inliers >= cfg.badge_min_inliers:
            return BadgeResult(True, inliers, len(good), "ok", quad)

        # Not enough inliers on their own. The geometry may still be right -- a badge too
        # far away to yield inliers is usually still located correctly -- so ask colour,
        # which is an independent signal that survives at sizes SIFT cannot carry.
        if cfg.badge_colour_rescue and inliers >= cfg.badge_colour_min_inliers:
            agreement = self._colour_agreement(query.image, matrix, level)
            if agreement is not None and agreement >= cfg.badge_colour_agreement:
                return BadgeResult(True, inliers, len(good), "ok (colour)", quad)
            if agreement is not None:
                return BadgeResult(
                    False, inliers, len(good),
                    f"too few inliers, colour {agreement:.2f}", quad,
                )
        return BadgeResult(False, inliers, len(good), "too few inliers", quad)

    def _colour_agreement(
        self, colour: np.ndarray, matrix: Optional[np.ndarray], level: "_Level"
    ) -> Optional[float]:
        """Rectify the matched region back onto the reference and compare colour layout.

        The homography is expressed in this level's coordinates, not the full-resolution
        reference's, so the warp target is the level. Both signatures are resampled to a
        fixed grid afterwards, which makes the comparison scale-free either way.
        """
        if matrix is None or self._ref_sig is None or colour is None or colour.ndim != 3:
            return None
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None
        height, width = level.shape
        warped = cv2.warpPerspective(colour, inverse, (width, height))
        signature = _chroma_signature(warped)
        if signature is None:
            return None
        return float(np.dot(self._ref_sig, signature))

    def describe(self, crop: np.ndarray) -> "Query":
        """Wrap the query so its SIFT passes are computed once, and only if needed.

        Detection dominates the cost (~86ms on a 1280x720 frame versus ~5ms to match a
        reference), so it is shared across references and campaigns, and the second
        variant is never computed when the first already matched.
        """
        return Query(crop, self._query_sift, self._lock, self.settings.badge_rootsift)

    def check(self, crop: np.ndarray) -> BadgeResult:
        return self.check_prepared(self.describe(crop))

    def check_prepared(self, query: "Query") -> BadgeResult:
        # The level that worked last time is tried first, so the steady state costs one
        # match and one SIFT pass rather than a sweep of scales and variants.
        order = sorted(range(len(self._levels)), key=lambda i: i != self._preferred)
        best = BadgeResult(False, 0, 0, "no features in crop")
        for index in order:
            result = self._check_level(self._levels[index], query)
            if result.ok:
                self._preferred = index
                return result
            if result.inliers > best.inliers or (best.quad is None and result.quad is not None):
                best = result
        return best
