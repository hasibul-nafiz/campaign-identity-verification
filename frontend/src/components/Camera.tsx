import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

// CLAUDE.md fixes capture at 1280x720; these allow raising it where the camera
// supports it, because a badge only ~90px wide is near the limit of what matching
// can resolve and resolution is the one lever that needs nothing from the wearer.
const CAPTURE_WIDTH = Number(import.meta.env.VITE_CAPTURE_WIDTH ?? 1280);
const CAPTURE_HEIGHT = Number(import.meta.env.VITE_CAPTURE_HEIGHT ?? 720);

export interface CameraHandle {
  capture: () => Promise<Blob>;
  ready: boolean;
}

type State =
  | { kind: "starting" }
  | { kind: "live" }
  | { kind: "error"; title: string; detail: string; retryable: boolean };

function describe(err: unknown): { title: string; detail: string; retryable: boolean } {
  const name = err instanceof DOMException ? err.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return {
        title: "Camera permission denied",
        detail:
          "Allow camera access for this site in your browser's address bar, then try again.",
        retryable: true,
      };
    case "NotFoundError":
    case "OverconstrainedError":
      return {
        title: "No camera found",
        detail: "Connect a webcam and try again.",
        retryable: true,
      };
    case "NotReadableError":
      return {
        title: "Camera is busy",
        detail: "Another application is using the camera. Close it and try again.",
        retryable: true,
      };
    default:
      return {
        title: "Could not start the camera",
        detail: err instanceof Error ? err.message : String(err),
        retryable: true,
      };
  }
}

export const Camera = forwardRef<CameraHandle, { className?: string }>(
  function Camera({ className }, ref) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const [state, setState] = useState<State>({ kind: "starting" });
    const [attempt, setAttempt] = useState(0);
    // The preview box takes the stream's own shape. A phone hands back a portrait
    // stream, and forcing that into a fixed 16:9 box with object-cover crops most of
    // the height away -- which on this screen means the wearer cannot see whether the
    // badge is even in frame, while the captured frame is untouched and looks fine.
    const [aspect, setAspect] = useState<number | null>(null);

    useEffect(() => {
      let cancelled = false;

      async function start() {
        setState({ kind: "starting" });
        setAspect(null);
        if (!navigator.mediaDevices?.getUserMedia) {
          setState({
            kind: "error",
            title: "Camera unavailable",
            detail:
              "getUserMedia needs a secure context. Use http://localhost or serve over HTTPS.",
            retryable: false,
          });
          return;
        }
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: "user",
              width: { ideal: CAPTURE_WIDTH },
              height: { ideal: CAPTURE_HEIGHT },
            },
            audio: false,
          });
          if (cancelled) {
            stream.getTracks().forEach((t) => t.stop());
            return;
          }
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play().catch(() => undefined);
          }
          setState({ kind: "live" });
        } catch (err) {
          if (!cancelled) setState({ kind: "error", ...describe(err) });
        }
      }

      void start();
      return () => {
        cancelled = true;
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      };
    }, [attempt]);

    const capture = useCallback(async (): Promise<Blob> => {
      const video = videoRef.current;
      if (!video || !video.videoWidth || !video.videoHeight) {
        throw new Error("Camera is not ready yet.");
      }
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Could not get a 2D canvas context.");
      // The preview is mirrored in CSS only; the frame sent to the API must not be.
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
          (blob) => (blob ? resolve(blob) : reject(new Error("Could not encode the frame."))),
          "image/jpeg",
          0.9,
        );
      });
    }, []);

    useImperativeHandle(ref, () => ({ capture, ready: state.kind === "live" }), [
      capture,
      state.kind,
    ]);

    return (
      <div
        className={`relative overflow-hidden rounded-xl bg-slate-900 ${className ?? ""}`}
        style={{ aspectRatio: aspect ?? 16 / 9 }}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          onLoadedMetadata={(e) => {
            const video = e.currentTarget;
            if (video.videoWidth && video.videoHeight) {
              setAspect(video.videoWidth / video.videoHeight);
            }
          }}
          className="h-full w-full -scale-x-100 object-cover"
        />
        {state.kind !== "live" && (
          <div className="absolute inset-0 grid place-items-center bg-slate-900/95 p-6 text-center">
            {state.kind === "starting" ? (
              <p className="text-sm text-slate-300">Starting camera…</p>
            ) : (
              <div className="max-w-sm">
                <p className="font-medium text-rose-300">{state.title}</p>
                <p className="mt-1 text-sm text-slate-400">{state.detail}</p>
                {state.retryable && (
                  <button
                    type="button"
                    onClick={() => setAttempt((a) => a + 1)}
                    className="mt-4 rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium text-white hover:bg-slate-600"
                  >
                    Try again
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    );
  },
);
