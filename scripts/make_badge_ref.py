"""Prepare assets/badge_ref.png, the SIFT reference for the badge check.

    python scripts/make_badge_ref.py path/to/real_madrid_logo.png   # use a real logo
    python scripts/make_badge_ref.py                                # crest-shaped placeholder

Drop the real crest in and re-run; nothing else in the project needs to change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR  # noqa: E402

TARGET_SIDE = 600


def make_badge(width: int = 480, height: int = 600, seed: int = 7) -> np.ndarray:
    """A circular crest placeholder, standing in for the real club logo."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 250, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    radius = min(cx, cy) - 30

    navy, gold = (110, 40, 20), (40, 170, 220)
    cv2.circle(img, (cx, cy), radius, navy, 6)
    cv2.circle(img, (cx, cy), radius - 22, gold, 3)
    cv2.circle(img, (cx, cy), radius - 70, navy, 2)

    # Crown-ish arcs above centre.
    for i, dx in enumerate((-58, 0, 58)):
        cv2.circle(img, (cx + dx, cy - radius + 74), 20 - i % 2 * 4, gold, 3)
    cv2.rectangle(img, (cx - 74, cy - radius + 92), (cx + 74, cy - radius + 106), navy, -1)

    cv2.putText(img, "RM", (cx - 74, cy + 34), cv2.FONT_HERSHEY_TRIPLEX, 2.6, navy, 7)
    cv2.line(img, (cx + 66, cy - 46), (cx + 66, cy + 60), (60, 60, 200), 8)

    # Deliberately asymmetric: evenly spaced radial marks make the crest look the
    # same under rotation, and RANSAC then collapses the homography to a point.
    cv2.ellipse(img, (cx - 30, cy + 40), (96, 54), 28, 0, 235, (70, 130, 60), 5)
    cv2.line(img, (cx - radius + 40, cy - 60), (cx + 40, cy + radius - 50), (30, 30, 160), 4)

    for _ in range(150):
        px, py = int(rng.integers(30, width - 30)), int(rng.integers(30, height - 30))
        if (px - cx) ** 2 + (py - cy) ** 2 < (radius - 20) ** 2:
            color = tuple(int(c) for c in rng.integers(0, 190, 3))
            if rng.random() < 0.6:
                cv2.circle(img, (px, py), int(rng.integers(2, 8)), color, -1)
            else:
                cv2.rectangle(img, (px, py), (px + int(rng.integers(4, 18)), py + int(rng.integers(4, 18))), color, -1)

    cv2.putText(img, "STAFF 4417", (cx - 118, height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, navy, 2)
    return img


def autocrop_to_crest(img: np.ndarray, margin: float = 0.08) -> np.ndarray:
    """Trim to the coloured crest so plain shirt fabric contributes no features.

    Fabric weave generates plenty of SIFT keypoints that describe the shirt, not the
    badge, which both dilutes the match and invites false positives on bare fabric.
    """
    saturation = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = (saturation > 60).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w < 40 or h < 40:
        return img
    pad_x, pad_y = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1 = min(img.shape[1], x + w + pad_x)
    y1 = min(img.shape[0], y + h + pad_y)
    return img[y0:y1, x0:x1]


def prepare(img: np.ndarray, autocrop: bool = True) -> np.ndarray:
    """Flatten transparency onto white, trim to the crest, normalise the size."""
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if autocrop:
        img = autocrop_to_crest(img)
    h, w = img.shape[:2]
    scale = TARGET_SIDE / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", help="real logo image; omitted = placeholder")
    parser.add_argument("--no-autocrop", action="store_true", help="keep the full frame")
    parser.add_argument(
        "--test", metavar="FRAME", action="append",
        help="a photo of the badge being worn; scores the new reference against it "
             "(repeatable). Use this to check a reference before trusting it.",
    )
    args = parser.parse_args()

    if args.source:
        loaded = cv2.imread(args.source, cv2.IMREAD_UNCHANGED)
        if loaded is None:
            print(f"could not read {args.source}", file=sys.stderr)
            return 2
        badge, origin = prepare(loaded, autocrop=not args.no_autocrop), args.source
    else:
        badge, origin = make_badge(), "generated placeholder"

    keypoints, _ = cv2.SIFT.create(nfeatures=1500).detectAndCompute(
        cv2.cvtColor(badge, cv2.COLOR_BGR2GRAY), None
    )
    out = BASE_DIR / "assets" / "badge_ref.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), badge)
    print(f"wrote {out} ({badge.shape[1]}x{badge.shape[0]}) from {origin}")
    print(f"SIFT keypoints: {len(keypoints)}")
    if len(keypoints) < 150:
        print("WARNING: sparse features - badge matching will be unreliable", file=sys.stderr)

    if args.test:
        print()
        return score_against(args.test)
    return 0


def score_against(frames: list) -> int:
    """Run the real matcher over worn-badge photos so a reference can be judged."""
    from app.badge import BadgeMatcher
    from app.config import Settings

    cfg = Settings.from_env()
    matcher = BadgeMatcher(settings=cfg)
    print(f"scoring against {len(frames)} frame(s), need >= {cfg.badge_min_inliers} inliers")
    worst = None
    for path in frames:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  {path}: could not read", file=sys.stderr)
            continue
        result = matcher.check(frame)
        verdict = "OK" if result.ok else "FAIL"
        print(f"  {Path(path).name:<28} {verdict:<5} {result.inliers:>4} inliers  ({result.reason})")
        worst = result.inliers if worst is None else min(worst, result.inliers)
    if worst is None:
        return 2
    if worst < cfg.badge_min_inliers:
        print("\nThis reference is too weak. Reshoot it closer, flatter and better lit.")
        return 1
    print("\nReference looks usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
