from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.config import Settings, get_settings


class ShirtError(ValueError):
    pass


@dataclass(frozen=True)
class ShirtResult:
    ok: bool
    delta_e: float
    rgb: Tuple[int, int, int]
    expected_rgb: Tuple[int, int, int]
    coverage: float


def parse_hex(value: str) -> Tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        raise ShirtError(f"invalid hex colour: {value!r}")
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        raise ShirtError(f"invalid hex colour: {value!r}") from None


def _to_lab(bgr: np.ndarray) -> np.ndarray:
    patch = np.asarray(bgr, dtype=np.float32).reshape(1, 1, 3) / 255.0
    return cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float64)


def chest_patch(crop: np.ndarray, settings: Optional[Settings] = None) -> np.ndarray:
    """The part of the torso box that is reliably shirt.

    The box is sized from the face, so it holds its proportions at any distance -- but
    it is cut to shoulder width, and the further away someone stands the more of its
    edges fall on the room behind them rather than on them. Past roughly half coverage
    the dominant cluster starts describing the wall. Insetting horizontally drops the
    shoulder edges, and trimming top and bottom drops the neck and the belt line.
    """
    cfg = settings or get_settings()
    if crop is None or crop.size == 0:
        raise ShirtError("empty crop")
    height, width = crop.shape[:2]
    x0 = int(round(width * cfg.shirt_sample_inset_x))
    x1 = int(round(width * (1.0 - cfg.shirt_sample_inset_x)))
    y0 = int(round(height * cfg.shirt_sample_top))
    y1 = int(round(height * cfg.shirt_sample_bottom))
    # A box too small to inset is used whole; a tiny sample is worse than a wide one.
    if x1 - x0 < 8 or y1 - y0 < 8:
        return crop
    return crop[y0:y1, x0:x1]


def color_clusters(
    crop: np.ndarray, settings: Optional[Settings] = None
) -> List[Tuple[np.ndarray, float]]:
    """Every colour present on the chest, with the share of the patch it covers.

    A garment is not one colour. A printed logo, a plaid, or a bold stripe can easily
    own more of the chest than the fabric does, so reducing the patch to its single
    biggest cluster answers the wrong question. Returning all of them lets the caller
    ask what actually matters: is an allowed colour substantially present here?
    """
    cfg = settings or get_settings()
    if crop is None or crop.size == 0:
        raise ShirtError("empty crop")
    if crop.ndim != 3 or crop.shape[2] != 3:
        raise ShirtError(f"expected a 3-channel BGR crop, got shape {crop.shape}")

    size = cfg.shirt_size
    small = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3).astype(np.float32)

    k = min(cfg.shirt_k, len(np.unique(data, axis=0)))
    if k < 1:
        raise ShirtError("crop has no usable pixels")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    counts = np.bincount(labels.reshape(-1), minlength=k)
    total = float(len(data))
    clusters = [(centers[i], float(counts[i]) / total) for i in range(k)]
    clusters.sort(key=lambda c: c[1], reverse=True)
    return _merge_similar(clusters, cfg)


def _merge_similar(
    clusters: List[Tuple[np.ndarray, float]], cfg: Settings
) -> List[Tuple[np.ndarray, float]]:
    """Fold clusters that are the same colour into one, summing their shares.

    k-means splits a single expanse of fabric into several near-identical clusters --
    shading across a sleeve is enough to do it. Their shares are then each a fraction of
    what that colour really covers, which makes any judgement about area meaningless.
    """
    merged: List[Tuple[np.ndarray, float]] = []
    for center, share in clusters:
        for index, (other, weight) in enumerate(merged):
            if _distance(center, tuple(int(v) for v in np.clip(other, 0, 255)[::-1]), cfg) <= cfg.shirt_merge_delta:
                total = weight + share
                merged[index] = ((other * weight + center * share) / total, total)
                break
        else:
            merged.append((center, share))
    merged.sort(key=lambda c: c[1], reverse=True)
    return merged


def dominant_color(crop: np.ndarray, settings: Optional[Settings] = None) -> Tuple[np.ndarray, float]:
    return color_clusters(crop, settings)[0]


def _distance(center_bgr: np.ndarray, expected_rgb: Tuple[int, int, int], cfg: Settings) -> float:
    diff = _to_lab(center_bgr) - _to_lab(np.array(expected_rgb[::-1], dtype=np.float32))
    # Exposure moves L* far more than it moves hue, so a correctly-coloured shirt in a
    # dim or warm room fails on brightness alone unless lightness is down-weighted.
    diff[0] *= cfg.shirt_lightness_weight
    return float(np.linalg.norm(diff))


def check_any(
    crop: np.ndarray,
    expected_hexes: Sequence[str],
    settings: Optional[Settings] = None,
) -> Tuple[ShirtResult, str]:
    """Nearest of several allowed colours. The dominant cluster is computed once."""
    cfg = settings or get_settings()
    if not expected_hexes:
        raise ShirtError("no expected colours configured")

    clusters = color_clusters(chest_patch(crop, cfg), cfg)

    # A colour counts as the garment only if it covers a fair share of the chest AND is
    # comparable in area to the most common colour there. Area alone is not enough: a
    # white logo on a navy shirt can cover a third of the chest, and without the second
    # condition it would pass as a white shirt. Requiring it to rival the dominant
    # colour keeps stripes and plaids -- where both colours are fabric -- while
    # rejecting a print that sits on top of a garment of another colour.
    floor = max(cfg.shirt_min_cluster, clusters[0][1] * cfg.shirt_cluster_ratio)
    eligible = [c for c in clusters if c[1] >= floor] or clusters[:1]

    best_hex, best_rgb, best_delta = "", (0, 0, 0), float("inf")
    center_bgr, coverage = eligible[0]
    for candidate in expected_hexes:
        rgb = parse_hex(candidate)
        for cluster_bgr, share in eligible:
            delta = _distance(cluster_bgr, rgb, cfg)
            if delta < best_delta:
                best_hex, best_rgb, best_delta = candidate, rgb, delta
                center_bgr, coverage = cluster_bgr, share
    measured = tuple(int(round(float(v))) for v in np.clip(center_bgr, 0, 255)[::-1])

    return (
        ShirtResult(
            ok=best_delta <= cfg.shirt_delta_e_max,
            delta_e=best_delta,
            rgb=measured,  # type: ignore[arg-type]
            expected_rgb=best_rgb,
            coverage=coverage,
        ),
        best_hex,
    )


def check(
    crop: np.ndarray, expected_hex: Optional[str] = None, settings: Optional[Settings] = None
) -> ShirtResult:
    cfg = settings or get_settings()
    result, _ = check_any(crop, [expected_hex or cfg.shirt_expected_hex], cfg)
    return result
