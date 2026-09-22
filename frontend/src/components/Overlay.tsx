import type { DetectResponse } from "../api";

/**
 * The SVG viewBox is the ORIGINAL image size the API reported, so every box and
 * polygon is drawn in API pixel coordinates with no manual scaling.
 */
export function Overlay({ frameUrl, result }: { frameUrl: string; result: DetectResponse }) {
  const [width, height] = result.image_size;
  const [fx, fy, fw, fh] = result.face.box;
  const torso = result.torso_box;
  const polygon = result.badge?.box ?? null;
  const stroke = Math.max(2, Math.round(width / 300));

  return (
    <div className="relative overflow-hidden rounded-xl bg-slate-900">
      <img src={frameUrl} alt="Captured frame" className="block w-full" />
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="pointer-events-none absolute inset-0 h-full w-full"
      >
        {torso && (
          <rect
            x={torso[0]}
            y={torso[1]}
            width={torso[2] - torso[0]}
            height={torso[3] - torso[1]}
            fill="none"
            stroke="#38bdf8"
            strokeWidth={stroke}
            strokeDasharray={`${stroke * 4} ${stroke * 3}`}
          />
        )}
        <rect
          x={fx}
          y={fy}
          width={fw}
          height={fh}
          fill="none"
          stroke="#4ade80"
          strokeWidth={stroke}
        />
        {polygon && (
          <polygon
            points={polygon.map(([x, y]) => `${x},${y}`).join(" ")}
            fill="rgba(251,191,36,0.15)"
            stroke="#fbbf24"
            strokeWidth={stroke}
          />
        )}
      </svg>
      <div className="absolute bottom-2 left-2 flex flex-wrap gap-2 text-[11px] font-medium">
        <Chip color="bg-green-400" label="Face" />
        <Chip color="bg-sky-400" label="Torso" />
        <Chip color="bg-amber-400" label="Badge" />
      </div>
    </div>
  );
}

function Chip({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 rounded-full bg-slate-900/80 px-2 py-1 text-slate-100">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {label}
    </span>
  );
}
