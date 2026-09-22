# Face verify test service
FastAPI + OpenCV. Free/permissive models only (YuNet, SFace, SIFT). No training, no paid or non-commercial weights.

## Rules
- Load models once at startup (lifespan), never per request.
- CV code is blocking: use plain `def` routes (threadpool), not `async def`.
- Downscale input so the longest side is <= 960px before any processing.
- Embeddings: float32, L2-normalized, stored as BLOB in SQLite (WAL mode).
- Keep all templates in an in-memory numpy matrix; match with one matrix multiply. Rebuild on register/delete.
- Thresholds live in config.py (env-overridable), never hardcoded.
- Pure functions in face/badge/shirt.py (take numpy image, return dataclass). No FastAPI imports there.
- Every module gets pytest tests. Run tests before saying a phase is done.
- Errors: JSON {detail}, 4xx for user errors (no face, multiple faces, bad image).
- Show only changed code in explanations; keep replies short.

## Commands
- run: uvicorn app.main:app --port 8001 --reload
- test: pytest -q

## Frontend (frontend/)
Vite + React + TypeScript + Tailwind. API base URL from VITE_API_URL (default http://localhost:8001).
- One api.ts client with typed responses; no fetch calls inside components.
- Webcam via getUserMedia in a reusable <Camera> component (ref exposes capture() -> Blob).
- Preview is mirrored with CSS only; the captured frame sent to the API is NOT mirrored.
- Capture at 1280x720, JPEG quality 0.9.
- Show loading and error states for every request. Keep components small.
- Run: cd frontend && npm run dev (port 5173). Check with `npm run build` and `npx tsc --noEmit`.