import { useRef, useState } from "react";
import { Camera } from "./Camera";
import type { CameraHandle } from "./Camera";
import { CropPicker } from "./CropPicker";

/**
 * Builds a badge reference from the webcam rather than an uploaded photo.
 *
 * A glossy badge reflects its surroundings, so a studio or desk photo of it shares
 * very little pixel detail with what this camera sees on a chest. Capturing through
 * the same camera in the same room is what makes those two views comparable.
 */
export function CaptureRef({
  onCancel,
  onCaptured,
}: {
  onCancel: () => void;
  onCaptured: (blob: Blob) => void;
}) {
  const cameraRef = useRef<CameraHandle>(null);
  const [shot, setShot] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function grab() {
    setError(null);
    try {
      const blob = await cameraRef.current!.capture();
      setShot(new File([blob], "capture.jpg", { type: "image/jpeg" }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  if (shot) {
    return (
      <CropPicker
        file={shot}
        onCancel={() => setShot(null)}
        onCropped={(blob) => {
          setShot(null);
          onCaptured(blob);
        }}
      />
    );
  }

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/70 p-4">
      <div className="w-full max-w-xl rounded-xl bg-white p-4 shadow-xl">
        <h3 className="text-base font-semibold text-slate-900">Capture the badge</h3>
        <p className="mt-1 text-sm text-slate-600">
          Wear or hold the badge as it will be worn, then capture. You will crop to the badge
          next. This works far better than a photo taken elsewhere, because the reference and
          the live view then share the same camera and lighting.
        </p>
        <div className="mt-3">
          <Camera ref={cameraRef} />
        </div>
        {error && (
          <p className="mt-2 rounded-lg border border-rose-200 bg-rose-50 p-2 text-sm text-rose-800">
            {error}
          </p>
        )}
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-lg px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={grab}
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-700"
          >
            Capture
          </button>
        </div>
      </div>
    </div>
  );
}
