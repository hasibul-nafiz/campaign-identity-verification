"""Score a labelled image set and suggest thresholds from the separation it finds.

Expected layout (missing folders are skipped):

    dataset/
      genuine/       you, wearing the badge and a white shirt
      no_badge/      you, white shirt, no badge
      wrong_shirt/   you, badge, wrong shirt colour
      other_person/  somebody else

    python scripts/evaluate.py dataset/ [--enroll dataset/enroll] [--expected-color FFFFFF]

Unlike /detect, every stage runs on every image even when an earlier one fails --
a threshold cannot be calibrated from stages that were skipped.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import shirt as shirt_mod  # noqa: E402
from app.badge import BadgeMatcher  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.db import Database  # noqa: E402
from app.face import FaceEngine, FaceError, downscale, pose_label, torso_roi  # noqa: E402
from app.store import TemplateStore  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# is_self: the face should match the enrolled identity
# has_badge / shirt_ok: what the badge and shirt stages should conclude
GROUPS: Dict[str, Dict[str, Optional[bool]]] = {
    "genuine": {"is_self": True, "has_badge": True, "shirt_ok": True},
    "no_badge": {"is_self": True, "has_badge": False, "shirt_ok": True},
    "wrong_shirt": {"is_self": True, "has_badge": True, "shirt_ok": False},
    "other_person": {"is_self": False, "has_badge": None, "shirt_ok": None},
}


@dataclass
class Case:
    path: Path
    group: str
    error: Optional[str] = None
    pose: Optional[str] = None
    face_score: Optional[float] = None
    badge_inliers: Optional[int] = None
    badge_reason: Optional[str] = None
    shirt_delta_e: Optional[float] = None

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class Suggestion:
    threshold: float
    separable: bool
    margin: float
    pos: Tuple[float, float]
    neg: Tuple[float, float]


def find_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)


def _suggest(pos: Sequence[float], neg: Sequence[float]) -> Optional[Suggestion]:
    """Higher scores pass. Feed negated values for 'lower is better' metrics."""
    if not pos or not neg:
        return None
    pos, neg = list(pos), list(neg)
    span = (min(pos), max(pos)), (min(neg), max(neg))
    if min(pos) > max(neg):
        return Suggestion((min(pos) + max(neg)) / 2.0, True, min(pos) - max(neg), *span)

    candidates = sorted(set(pos + neg))
    midpoints = [(a + b) / 2.0 for a, b in zip(candidates, candidates[1:])]
    best, best_key = candidates[0], (-1.0, -1.0)
    for t in candidates + midpoints:
        tpr = sum(v >= t for v in pos) / len(pos)
        tnr = sum(v < t for v in neg) / len(neg)
        key = ((tpr + tnr) / 2.0, min(abs(v - t) for v in pos + neg))
        if key > best_key:
            best, best_key = t, key
    return Suggestion(best, False, min(pos) - max(neg), *span)


def run_case(path: Path, group: str, engine, store, matcher, cfg, expected_hex) -> Case:
    case = Case(path=path, group=group)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        case.error = "unreadable"
        return case
    image, _ = downscale(image, cfg.max_side)

    try:
        face = engine.detect_single(image)
    except FaceError as exc:
        case.error = str(exc)
        return case

    case.pose = pose_label(face, cfg).label
    match = store.identify(engine.embed(image, face))
    case.face_score = match.score if match else 0.0

    try:
        x, y, w, h = torso_roi(face, image.shape, cfg)
    except FaceError as exc:
        case.error = f"torso: {exc}"
        return case
    torso = image[y : y + h, x : x + w]

    badge = matcher.check(torso)
    case.badge_inliers = badge.inliers
    case.badge_reason = badge.reason
    case.shirt_delta_e = shirt_mod.check(torso, expected_hex, cfg).delta_e
    return case


def collect(values: Sequence[Optional[float]]) -> List[float]:
    return [v for v in values if v is not None]


def describe(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    return f"min={min(values):.3f} med={statistics.median(values):.3f} max={max(values):.3f}"


def verdict(case: Case, face_t: float, badge_t: float, shirt_t: float) -> Tuple[bool, str]:
    if case.error:
        return False, case.error
    if case.face_score < face_t:
        return False, "face"
    if case.badge_inliers < badge_t:
        return False, "badge"
    if case.shirt_delta_e > shirt_t:
        return False, "shirt"
    return True, "pass"


def report_errors(cases: List[Case], face_t: float, badge_t: float, shirt_t: float, label: str) -> None:
    false_accept, false_reject = [], []
    for case in cases:
        passed, why = verdict(case, face_t, badge_t, shirt_t)
        should_pass = case.group == "genuine"
        if passed and not should_pass:
            false_accept.append(case)
        elif not passed and should_pass:
            false_reject.append(f"{case.name} (failed on {why})")

    print(f"\n  {label}: face>={face_t:.3f}  badge>={badge_t:.0f}  shirt<={shirt_t:.2f}")
    if false_accept:
        print(f"    FALSE ACCEPTS ({len(false_accept)}): " + ", ".join(
            f"{c.name} [{c.group}]" for c in false_accept))
    else:
        print("    false accepts: none")
    if false_reject:
        print(f"    FALSE REJECTS ({len(false_reject)}): " + ", ".join(false_reject))
    else:
        print("    false rejects: none")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--enroll", help="folder of enrollment images (default: first genuine image)")
    parser.add_argument("--expected-color", default=None)
    args = parser.parse_args()

    cfg: Settings = get_settings()
    expected_hex = args.expected_color or cfg.shirt_expected_hex
    root = Path(args.dataset)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    engine = FaceEngine(cfg)
    matcher = BadgeMatcher(settings=cfg)
    store = TemplateStore(Database(Path(":memory:")), cfg)

    # Enrollment must be held out: scoring an image against its own template
    # returns 1.0 and tells you nothing about the threshold.
    if args.enroll:
        enroll_paths = find_images(Path(args.enroll))
        held_out = set()
    else:
        genuine = find_images(root / "genuine")
        if not genuine:
            print("no genuine/ images to enroll from; pass --enroll", file=sys.stderr)
            return 2
        enroll_paths = genuine[:1]
        held_out = {genuine[0]}
        print(f"enrolling from {genuine[0].name} (held out of scoring)")

    enrolled = 0
    for path in enroll_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image, _ = downscale(image, cfg.max_side)
        try:
            face = engine.detect_single(image)
        except FaceError as exc:
            print(f"  skipped {path.name}: {exc}", file=sys.stderr)
            continue
        store.register("reference", f"{pose_label(face, cfg).label}_{enrolled}", engine.embed(image, face))
        enrolled += 1
    if enrolled == 0:
        print("no usable enrollment images", file=sys.stderr)
        return 2
    print(f"enrolled {enrolled} template(s); expected shirt colour {expected_hex}\n")

    cases: List[Case] = []
    for group in GROUPS:
        paths = [p for p in find_images(root / group) if p not in held_out]
        if not paths:
            continue
        print(f"{group}/  ({len(paths)} images)")
        print(f"  {'file':<28} {'pose':<8} {'face':>7} {'badge':>7} {'deltaE':>8}  note")
        for path in paths:
            case = run_case(path, group, engine, store, matcher, cfg, expected_hex)
            cases.append(case)
            if case.error:
                print(f"  {case.name:<28} {'-':<8} {'-':>7} {'-':>7} {'-':>8}  ERROR: {case.error}")
            else:
                print(
                    f"  {case.name:<28} {case.pose:<8} {case.face_score:>7.3f} "
                    f"{case.badge_inliers:>7d} {case.shirt_delta_e:>8.2f}  {case.badge_reason}"
                )
        print()

    scored = [c for c in cases if not c.error]
    if not scored:
        print("nothing scored successfully", file=sys.stderr)
        return 1
    errored = [c for c in cases if c.error]
    if errored:
        print(f"{len(errored)} image(s) could not be scored:")
        for case in errored:
            print(f"  {case.group}/{case.name}: {case.error}")
        print()

    by_group = {g: [c for c in scored if c.group == g] for g in GROUPS}

    # Each stage is calibrated from the folders that actually exercise it:
    # wrong_shirt still carries a badge, and no_badge still has a white shirt.
    face_pos = collect([c.face_score for g in ("genuine", "no_badge", "wrong_shirt") for c in by_group[g]])
    face_neg = collect([c.face_score for c in by_group["other_person"]])
    badge_pos = collect([c.badge_inliers for g in ("genuine", "wrong_shirt") for c in by_group[g]])
    badge_neg = collect([c.badge_inliers for c in by_group["no_badge"]])
    shirt_pos = collect([c.shirt_delta_e for g in ("genuine", "no_badge") for c in by_group[g]])
    shirt_neg = collect([c.shirt_delta_e for c in by_group["wrong_shirt"]])

    print("score distributions")
    print(f"  face  self    : {describe(face_pos)}")
    print(f"  face  others  : {describe(face_neg)}")
    print(f"  badge present : {describe(badge_pos)}")
    print(f"  badge absent  : {describe(badge_neg)}")
    print(f"  shirt correct : {describe(shirt_pos)}")
    print(f"  shirt wrong   : {describe(shirt_neg)}")

    face_s = _suggest(face_pos, face_neg)
    badge_s = _suggest(badge_pos, badge_neg)
    shirt_raw = _suggest([-v for v in shirt_pos], [-v for v in shirt_neg])

    print("\nsuggested thresholds")
    rows = [
        ("FACE_MATCH_THRESH", face_s, cfg.face_match_thresh, "{:.3f}"),
        ("BADGE_MIN_INLIERS", badge_s, float(cfg.badge_min_inliers), "{:.0f}"),
    ]
    for env, sug, current, fmt in rows:
        if sug is None:
            print(f"  {env:<20} insufficient data (need both positive and negative examples)")
            continue
        state = f"separable, margin {sug.margin:.3f}" if sug.separable else "OVERLAP - not separable"
        print(f"  {env:<20} {fmt.format(sug.threshold):>8}  (current {fmt.format(current)})  {state}")

    if shirt_raw is None:
        print(f"  {'SHIRT_DELTA_E_MAX':<20} insufficient data (need both positive and negative examples)")
        shirt_t = cfg.shirt_delta_e_max
    else:
        shirt_t = -shirt_raw.threshold
        state = f"separable, margin {shirt_raw.margin:.3f}" if shirt_raw.separable else "OVERLAP - not separable"
        print(f"  {'SHIRT_DELTA_E_MAX':<20} {shirt_t:>8.2f}  (current {cfg.shirt_delta_e_max:.2f})  {state}")

    face_t = face_s.threshold if face_s else cfg.face_match_thresh
    badge_t = badge_s.threshold if badge_s else float(cfg.badge_min_inliers)

    print("\nerrors at each threshold set")
    report_errors(scored, cfg.face_match_thresh, float(cfg.badge_min_inliers), cfg.shirt_delta_e_max, "current")
    report_errors(scored, face_t, badge_t, shirt_t, "suggested")

    print("\nexport the suggested set with:")
    print(f"  export FACE_MATCH_THRESH={face_t:.3f} BADGE_MIN_INLIERS={badge_t:.0f} SHIRT_DELTA_E_MAX={shirt_t:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
