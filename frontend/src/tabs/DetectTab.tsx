import { useEffect, useRef, useState } from "react";
import { assetUrl, detect } from "../api";
import type { Campaign, DetectResponse } from "../api";
import { Camera } from "../components/Camera";
import type { CameraHandle } from "../components/Camera";
import { Overlay } from "../components/Overlay";

export function DetectTab({ campaigns }: { campaigns: Campaign[] }) {
  const cameraRef = useRef<CameraHandle>(null);
  const [campaignId, setCampaignId] = useState<number | "">("");
  const [frame, setFrame] = useState<string | null>(null);
  const [result, setResult] = useState<DetectResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => () => { if (frame) URL.revokeObjectURL(frame); }, [frame]);

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const blob = await cameraRef.current!.capture();
      setFrame((old) => {
        if (old) URL.revokeObjectURL(old);
        return URL.createObjectURL(blob);
      });
      setResult(await detect(blob, campaignId === "" ? undefined : campaignId));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        {frame && result ? (
          <Overlay frameUrl={frame} result={result} />
        ) : (
          <Camera ref={cameraRef} />
        )}
        <label className="block">
          <span className="text-xs font-medium text-slate-600">Campaign</span>
          <select
            value={campaignId}
            onChange={(e) => setCampaignId(e.target.value === "" ? "" : Number(e.target.value))}
            className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-900"
          >
            <option value="">Auto — identify from the badge</option>
            {campaigns.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
                {c.active ? "" : " (inactive)"} — {c.badge_ref_count} ref
                {c.badge_ref_count === 1 ? "" : "s"}
              </option>
            ))}
          </select>
        </label>

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={run}
            disabled={busy}
            className="rounded-lg bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white hover:bg-slate-700 disabled:opacity-40"
          >
            {busy ? "Detecting…" : "Detect"}
          </button>
          {frame && (
            <button
              type="button"
              onClick={() => {
                setFrame((old) => { if (old) URL.revokeObjectURL(old); return null; });
                setResult(null);
                setError(null);
              }}
              className="rounded-lg px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-200"
            >
              Back to camera
            </button>
          )}
        </div>
        {error && (
          <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
            {error}
          </div>
        )}
      </div>

      <div className="space-y-4">
        {result ? <Results result={result} /> : (
          <p className="text-sm text-slate-500">
            Press Detect to capture a frame and run the pipeline.
          </p>
        )}
      </div>
    </div>
  );
}

function Results({ result }: { result: DetectResponse }) {
  const { badge, shirt, person, face } = result;
  const referenceUrl =
    badge?.matched_ref_id != null && result.campaign
      ? assetUrl(`/campaigns/${result.campaign.id}/badges/${badge.matched_ref_id}/image`)
      : assetUrl("/badge-ref");
  return (
    <>
      <div
        className={`rounded-xl p-4 text-center text-lg font-bold tracking-wide ${
          result.passed ? "bg-emerald-600 text-white" : "bg-rose-600 text-white"
        }`}
      >
        {result.passed ? "PASSED" : "FAILED"}
        {result.campaign && (
          <span className="mt-1 block text-xs font-medium opacity-90">
            {result.campaign.name}
            {result.campaign.auto ? " (identified from the badge)" : " (selected)"}
          </span>
        )}
        {result.reason && (
          <span className="mt-1 block text-xs font-medium opacity-90">{result.reason}</span>
        )}
      </div>

      <div className="divide-y divide-slate-100 overflow-hidden rounded-xl border border-slate-200">
        <StatusRow
          label="Face"
          ok={person !== null}
          value={
            person
              ? `${person.name} · ${person.score.toFixed(3)}`
              : `no match${result.best_score !== null ? ` · best ${result.best_score.toFixed(3)}` : ""}`
          }
          note={`pose ${face.pose}`}
        />
        <StatusRow
          label="Badge"
          ok={badge?.ok ?? false}
          value={badge ? `${badge.inliers} inliers` : "not checked"}
          note={badge?.reason ?? "skipped — face did not match"}
        />
        <StatusRow
          label="Shirt"
          ok={shirt?.ok ?? false}
          value={shirt ? `ΔE ${shirt.delta_e.toFixed(2)}` : "not checked"}
          note={shirt ? `nearest ${shirt.matched_color}` : "skipped — face did not match"}
        />
      </div>

      {shirt && (
        <div className="rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Shirt colour
          </p>
          <div className="mt-3 flex items-center gap-4">
            <Swatch label="Measured" rgb={shirt.rgb} />
            <span className="text-sm font-medium text-slate-400">vs</span>
            <Swatch label="Expected" rgb={shirt.expected_rgb} />
            <div className="ml-auto text-right text-xs text-slate-500">
              <p>ΔE {shirt.delta_e.toFixed(2)}</p>
              <p>{Math.round(shirt.coverage * 100)}% of chest patch</p>
            </div>
          </div>
          {shirt.coverage < 0.5 && (
            <p className="mt-2 text-xs text-amber-700">
              The dominant colour covers under half the chest patch — it may be
              background rather than your shirt. Step closer or centre yourself.
            </p>
          )}
        </div>
      )}

      {result.campaign_candidates && result.campaign_candidates.length > 1 && (
        <div className="rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Campaign ranking
          </p>
          <ul className="mt-2 space-y-1">
            {result.campaign_candidates.map((c, i) => (
              <li key={c.id} className="flex items-center gap-2 text-sm">
                <span
                  className={`h-2 w-2 rounded-full ${c.ok ? "bg-emerald-500" : "bg-slate-300"}`}
                />
                <span className={i === 0 ? "font-semibold text-slate-800" : "text-slate-600"}>
                  {c.name}
                </span>
                <span className="ml-auto tabular-nums text-xs text-slate-500">
                  {c.inliers} inliers{c.ok ? "" : " · rejected"}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[11px] text-slate-500">
            Inliers alone do not decide it — a runner-up can score well and still be rejected
            by the homography check.
          </p>
        </div>
      )}

      <div className="rounded-xl border border-slate-200 p-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Badge match
        </p>
        <div className="mt-3 grid grid-cols-2 gap-3">
          <figure>
            <img
              src={referenceUrl}
              alt="Reference badge"
              className="h-36 w-full rounded-lg border border-slate-200 bg-slate-50 object-contain"
            />
            <figcaption className="mt-1 text-center text-xs text-slate-500">
              Reference{badge?.matched_ref_id != null ? ` #${badge.matched_ref_id}` : ""}
            </figcaption>
          </figure>
          <figure>
            {result.badge_crop_b64 ? (
              <img
                src={`data:image/jpeg;base64,${result.badge_crop_b64}`}
                alt="Matched region"
                className="h-36 w-full rounded-lg border border-slate-200 bg-slate-50 object-contain"
              />
            ) : (
              <div className="grid h-36 place-items-center rounded-lg border border-dashed border-slate-300 text-xs text-slate-400">
                no match
              </div>
            )}
            <figcaption className="mt-1 text-center text-xs text-slate-500">
              Matched {badge ? `· ${badge.inliers} inliers` : ""}
            </figcaption>
          </figure>
        </div>
      </div>

      <p className="text-[11px] leading-relaxed text-slate-400">
        {Object.entries(result.timings_ms)
          .map(([k, v]) => `${k} ${v}ms`)
          .join(" · ")}
      </p>
    </>
  );
}

function StatusRow({
  label,
  ok,
  value,
  note,
}: {
  label: string;
  ok: boolean;
  value: string;
  note?: string;
}) {
  return (
    <div className="flex items-center gap-3 p-3">
      <span
        className={`grid h-7 w-7 shrink-0 place-items-center rounded-full text-sm font-bold text-white ${
          ok ? "bg-emerald-600" : "bg-rose-600"
        }`}
        aria-hidden
      >
        {ok ? "✓" : "✕"}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold text-slate-800">{label}</p>
        {note && <p className="truncate text-xs text-slate-500">{note}</p>}
      </div>
      <p className="shrink-0 text-sm font-medium tabular-nums text-slate-700">{value}</p>
    </div>
  );
}

function Swatch({ label, rgb }: { label: string; rgb: [number, number, number] }) {
  const [r, g, b] = rgb;
  return (
    <div className="text-center">
      <div
        className="h-12 w-12 rounded-lg border border-slate-300"
        style={{ backgroundColor: `rgb(${r},${g},${b})` }}
      />
      <p className="mt-1 text-[11px] font-medium text-slate-600">{label}</p>
      <p className="text-[10px] tabular-nums text-slate-400">
        {r},{g},{b}
      </p>
    </div>
  );
}
