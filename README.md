# Campaign Identity Verification

A verification service for field/promo campaigns: confirm that the person in front of
the camera is (a) a registered person, (b) wearing the correct campaign badge, and
(c) wearing a shirt in one of the campaign's approved colours — all from a single
webcam/phone capture, in real time, with free and permissive computer-vision models.

No cloud AI, no paid models, no training. Everything runs locally on OpenCV.

## What it's for

Field campaigns (promoters, brand reps, event staff) need a quick way to confirm someone
is who they say they are and is properly kitted out — the right badge, the right coloured
shirt — before letting them log activity. This service takes one photo and answers three
questions:

1. **Face match** — does this face match a person already registered in the system?
2. **Badge match** — is the campaign's badge/logo visible somewhere on their body, in
   any position, lighting, or angle?
3. **Shirt colour match** — is the shirt one of the campaign's approved colours (solid,
   striped, plaid, or printed)?

Campaigns are configured independently: each has its own badge reference images and its
own list of approved shirt colours, so the same service can run several campaigns (each
with a different badge and uniform) at once.

## How it works

### Face verification
- [`YuNet`](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  detects faces; [`SFace`](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
  produces a 128-d embedding per face.
- Embeddings are L2-normalized `float32`, stored as BLOBs in SQLite (WAL mode), and also
  kept in one in-memory matrix so matching a new face against every registered person is a
  single matrix multiply — no per-person loop, no external vector DB.
- Both models are loaded once at process startup (FastAPI `lifespan`), never per request.

### Badge detection
Badges can be printed, embossed, or a mirror-finish pin, in any position on the shirt,
under low light or glare, at any distance the camera can resolve. That rules out colour-
or template-matching, so badge detection uses **SIFT keypoint matching**:

- Reference badge images are described at multiple scales (`BADGE_REF_SCALES`) so both
  close-up and small/distant badges have a matching reference size.
- The live query frame is described with an **uncapped** keypoint budget
  (`BADGE_QUERY_NFEATURES=0`) and matched via a FLANN KD-tree index, so a cluttered scene
  (face, background, badge) doesn't starve the actual badge of keypoints — a capped budget
  was found to put most keypoints on the wearer's face instead of a small badge.
- Matches are RootSIFT-normalized (L1 + sqrt) for better robustness to lighting.
- A geometric fit is attempted both as a **homography** (`USAC_MAGSAC`) and as an
  **affine/similarity transform** (`RANSAC`), and whichever survives sanity checks (area,
  aspect skew, edge ratio) with more inliers wins — the affine fallback matters for
  domed/curved badges that don't fit a flat homography.
- If a match has too few geometric inliers to pass outright but still clears a lower bar,
  a **colour rescue** step checks whether the matched region's Lab a/b "chroma signature"
  (lightness-independent, since mirror-finish badges have unstable brightness) agrees with
  the reference — this recovers otherwise-borderline low-light/glare matches without
  weakening the geometric threshold for badges that don't have any inlier support at all.
- Search order is **torso first, full frame second** — a badge is almost always on the
  torso, so trying that crop first is both faster (~6x) and more accurate (less background
  clutter to compete with).
- Badge references are uploaded pre-cropped (the frontend requires a tight crop before
  upload; the server rejects anything that doesn't fill enough of the frame) — an
  uncropped reference was found to match background texture instead of the badge itself.

### Shirt colour matching
- The torso region (sized off the detected face, so it holds its proportions at any
  distance) is inset to a **chest patch** — trimmed on the sides and top/bottom — to
  exclude shoulders, neck, and background that creep into the box's edges the further away
  someone stands.
- The chest patch is clustered with k-means into several dominant colours (not just one),
  because a shirt is often not a single colour — stripes, plaid, and printed logos all
  contribute their own clusters. Near-identical clusters (k-means splitting one fabric
  colour into several due to shading) are merged before comparison.
- A colour counts as "the garment colour" if it covers a fair share of the chest **and**
  is comparable in area to the most common colour there — this keeps stripes/plaid
  (both colours are fabric) while rejecting a small printed logo from being read as the
  whole shirt's colour.
- Colour distance is measured in **CIE Lab space** (ΔE) with lightness down-weighted, since
  exposure/room lighting moves L* far more than it moves hue — a correctly-coloured shirt
  in a dim or warm room shouldn't fail on brightness alone.

### Campaigns
- Each campaign has its own badge reference image(s) and its own list of approved shirt
  colours (hex), and can optionally override badge/shirt thresholds.
- Badge references and per-campaign settings are pre-loaded and kept in memory, rebuilt
  whenever a campaign or its references change.

## Tech stack

**Backend:** Python, FastAPI, OpenCV, NumPy, SQLite (WAL mode), pytest.
**Frontend:** Vite, React, TypeScript, Tailwind CSS.

## Project layout

```
app/                  FastAPI application
  main.py             Routes (thin — request/response only, no CV logic)
  face.py             Face detection + embedding (pure functions, no FastAPI)
  badge.py            SIFT badge matcher (pure functions, no FastAPI)
  shirt.py            Shirt colour matcher (pure functions, no FastAPI)
  campaigns.py        Campaign registry: loads refs/colours, runs badge/shirt checks
  db.py               SQLite schema + queries
  config.py            All thresholds, env-overridable
models/                YuNet + SFace ONNX weights (see Setup)
assets/                Badge reference images (per-campaign, git-ignored)
data/                  SQLite DB + debug frame dumps (git-ignored)
scripts/               Offline helpers: capture, evaluate, make_badge_ref, try_image
tests/                 pytest suite — one file per module
frontend/              Vite + React + TS client
```

## API

All endpoints return JSON; errors are `{"detail": "..."}` with a 4xx status.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/people` | List registered people |
| POST | `/register` | Register a person (name + face photo(s)) |
| DELETE | `/people/{id}` | Remove a person |
| POST | `/detect` | Main verification call: face match + badge + shirt colour |
| GET | `/campaigns` | List campaigns |
| POST | `/campaigns` | Create a campaign (name, approved colours, thresholds) |
| GET | `/campaigns/{id}` | Campaign detail |
| DELETE | `/campaigns/{id}` | Remove a campaign |
| POST | `/campaigns/{id}/badges` | Upload a badge reference image (cropped) |
| DELETE | `/campaigns/{id}/badges/{badge_id}` | Remove a badge reference |
| GET | `/campaigns/{id}/badges/{badge_id}/image` | Fetch a stored badge reference image |
| GET | `/badge-ref` | Legacy single-badge reference (pre-campaigns) |

## Setup

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the model weights (not committed to this repo) into `models/`:

- `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` from the
  [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models) (`face_detection_yunet`,
  `face_recognition_sface`) — both released under an Apache-2.0-style permissive licence.

```bash
uvicorn app.main:app --port 8001 --reload
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env   # adjust VITE_API_URL if needed
npm run dev
```

Runs on `http://localhost:5173` and talks to the backend on `http://localhost:8001` by
default. `getUserMedia` requires a secure context to use a phone's camera over the LAN —
the dev server is configured with `@vitejs/plugin-basic-ssl` and proxies `/api` to the
backend so a phone on the same network can open `https://<your-lan-ip>:5173` directly.

### Tests

```bash
pytest -q
```

## Configuration

Every threshold lives in `app/config.py` and can be overridden with an environment
variable of the same name in upper snake case (e.g. `BADGE_MIN_INLIERS=10`,
`SHIRT_DELTA_E_MAX=12`). See `config.py` for the full list and defaults. Per-campaign
overrides for badge inlier count and shirt ΔE are also available via the campaign API.

Set `DEBUG=1` to have every `/detect` call dump the raw frame and torso crop to
`data/debug/last_detect.png` / `last_torso.png`, so a failed detection can be reproduced
offline instead of only being visible as a screenshot.
