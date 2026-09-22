"""Debug harness: run the full pipeline over one image and dump what each stage saw.

usage: python scripts/try_image.py <path> [--shirt-hex 1E3A8A] [--out out/]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import badge as badge_mod  # noqa: E402
from app import shirt as shirt_mod  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.face import FaceEngine, FaceError, downscale, pose_label, torso_roi  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--shirt-hex", default=None, help="defaults to SHIRT_EXPECTED_HEX")
    parser.add_argument("--out", default="out")
    args = parser.parse_args()

    cfg = get_settings()
    img = cv2.imread(args.path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"could not read image: {args.path}", file=sys.stderr)
        return 2

    img, scale = downscale(img, cfg.max_side)
    print(f"image      : {img.shape[1]}x{img.shape[0]} (downscale factor {scale:.3f})")

    engine = FaceEngine(cfg)
    try:
        face = engine.detect_single(img)
    except FaceError as exc:
        print(f"face       : {exc}", file=sys.stderr)
        return 1

    pose = pose_label(face, cfg)
    print(f"face box   : x={face.x} y={face.y} w={face.w} h={face.h} score={face.score:.3f}")
    print(f"pose       : {pose.label} (yaw={pose.yaw:+.3f} pitch={pose.pitch:+.3f})")

    embedding = engine.embed(img, face)
    print(f"embedding  : dim={embedding.shape[0]} norm={float((embedding ** 2).sum()) ** 0.5:.6f}")

    tx, ty, tw, th = torso_roi(face, img.shape, cfg)
    torso = img[ty : ty + th, tx : tx + tw].copy()
    print(f"torso box  : x={tx} y={ty} w={tw} h={th}")

    try:
        result = badge_mod.BadgeMatcher(settings=cfg).check(torso)
        print(
            f"badge      : ok={result.ok} inliers={result.inliers} "
            f"matches={result.matches} ({result.reason})"
        )
    except (badge_mod.BadgeError, FileNotFoundError) as exc:
        print(f"badge      : unavailable - {exc}")

    shirt = shirt_mod.check(torso, args.shirt_hex, cfg)
    print(
        f"shirt      : ok={shirt.ok} delta_e={shirt.delta_e:.2f} "
        f"rgb={shirt.rgb} expected={shirt.expected_rgb} coverage={shirt.coverage:.2f}"
    )

    annotated = img.copy()
    cv2.rectangle(annotated, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 220, 0), 2)
    for px, py in face.landmarks.astype(int):
        cv2.circle(annotated, (px, py), 2, (0, 0, 255), -1)
    cv2.rectangle(annotated, (tx, ty), (tx + tw, ty + th), (255, 140, 0), 2)
    cv2.putText(
        annotated, pose.label, (face.x, max(18, face.y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "annotated.png"), annotated)
    cv2.imwrite(str(out_dir / "torso.png"), torso)
    print(f"wrote      : {out_dir / 'annotated.png'}, {out_dir / 'torso.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
