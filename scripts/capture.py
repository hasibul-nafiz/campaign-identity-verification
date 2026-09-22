"""Grab stills from the webcam, for building a badge reference and test frames.

    python scripts/capture.py crest.jpg          # SPACE to shoot, ESC to quit
    python scripts/capture.py worn.jpg --delay 5 # countdown instead, no window

Numbered files (crest_1.jpg, crest_2.jpg, ...) are written when you shoot more than once.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {index}.\n"
            "On macOS, grant camera access to your terminal in\n"
            "System Settings > Privacy & Security > Camera, then run this again."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def save(frame, out: Path, shot: int) -> Path:
    path = out if shot == 0 else out.with_name(f"{out.stem}_{shot + 1}{out.suffix}")
    cv2.imwrite(str(path), frame)
    print(f"  saved {path}  ({frame.shape[1]}x{frame.shape[0]})")
    return path


def countdown(cap: cv2.VideoCapture, seconds: int):
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining}...", flush=True)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            cap.read()  # keep the sensor warm so exposure settles
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read a frame from the camera.")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="where to save, e.g. crest.jpg")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--delay", type=int, help="countdown seconds instead of a preview window")
    args = parser.parse_args()

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.camera)

    try:
        if args.delay:
            print(f"Capturing in {args.delay}s — get into position.")
            save(countdown(cap, args.delay), out, 0)
            return 0

        print("SPACE = capture   ESC/q = quit")
        shot = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.", file=sys.stderr)
                return 1
            preview = frame.copy()
            cv2.putText(
                preview, "SPACE = capture    ESC = quit", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32:
                save(frame, out, shot)
                shot += 1
        if shot == 0:
            print("Nothing captured.")
            return 1
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
