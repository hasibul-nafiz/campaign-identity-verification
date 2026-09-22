import { useEffect, useRef, useState } from "react";

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Drag a box over a photo and get just that region back as a PNG Blob. */
export function CropPicker({
  file,
  onCancel,
  onCropped,
}: {
  file: File;
  onCancel: () => void;
  onCropped: (blob: Blob) => void;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const [dragging, setDragging] = useState(false);
  const startRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  function relative(e: React.MouseEvent): { x: number; y: number } {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: Math.min(Math.max(e.clientX - rect.left, 0), rect.width),
      y: Math.min(Math.max(e.clientY - rect.top, 0), rect.height),
    };
  }

  function onDown(e: React.MouseEvent) {
    const point = relative(e);
    startRef.current = point;
    setBox({ x: point.x, y: point.y, w: 0, h: 0 });
    setDragging(true);
  }

  function onMove(e: React.MouseEvent) {
    if (!dragging || !startRef.current) return;
    const point = relative(e);
    const start = startRef.current;
    setBox({
      x: Math.min(start.x, point.x),
      y: Math.min(start.y, point.y),
      w: Math.abs(point.x - start.x),
      h: Math.abs(point.y - start.y),
    });
  }

  function crop() {
    const img = imgRef.current;
    if (!img || !box || box.w < 8 || box.h < 8) return;
    // The box is in displayed pixels; scale it up to the image's natural size so the
    // crop keeps full resolution.
    const scaleX = img.naturalWidth / img.clientWidth;
    const scaleY = img.naturalHeight / img.clientHeight;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(box.w * scaleX);
    canvas.height = Math.round(box.h * scaleY);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(
      img,
      box.x * scaleX,
      box.y * scaleY,
      canvas.width,
      canvas.height,
      0,
      0,
      canvas.width,
      canvas.height,
    );
    canvas.toBlob((blob) => blob && onCropped(blob), "image/png");
  }

  const size =
    box && imgRef.current
      ? `${Math.round((box.w * imgRef.current.naturalWidth) / imgRef.current.clientWidth)}×${Math.round(
          (box.h * imgRef.current.naturalHeight) / imgRef.current.clientHeight,
        )}`
      : null;

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/70 p-4">
      <div className="w-full max-w-2xl rounded-xl bg-white p-4 shadow-xl">
        <h3 className="text-base font-semibold text-slate-900">Crop to the badge</h3>
        <p className="mt-1 text-sm text-slate-600">
          Drag a box around the badge only. A whole photo makes a useless reference — the badge
          would end up a few pixels wide.
        </p>

        <div
          className="relative mt-3 select-none overflow-hidden rounded-lg bg-slate-100"
          onMouseDown={onDown}
          onMouseMove={onMove}
          onMouseUp={() => setDragging(false)}
          onMouseLeave={() => setDragging(false)}
        >
          {url && (
            <img
              ref={imgRef}
              src={url}
              alt="Crop source"
              draggable={false}
              className="block max-h-[55vh] w-full object-contain"
            />
          )}
          {box && box.w > 2 && (
            <div
              className="pointer-events-none absolute border-2 border-amber-400 bg-amber-400/20"
              style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
            />
          )}
        </div>

        <div className="mt-4 flex items-center justify-end gap-2">
          {size && <span className="mr-auto text-xs text-slate-500">selection {size}px</span>}
          <button
            type="button"
            onClick={onCancel}
            className="rounded-lg px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={crop}
            disabled={!box || box.w < 8 || box.h < 8}
            className="rounded-lg bg-slate-900 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-700 disabled:opacity-40"
          >
            Use this crop
          </button>
        </div>
      </div>
    </div>
  );
}
