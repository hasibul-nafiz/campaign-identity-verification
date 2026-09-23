import { useEffect, useState } from "react";
import {
  addBadgeRefs,
  createCampaign,
  deleteBadgeRef,
  deleteCampaign,
  getCampaign,
  updateCampaign,
  useAssetImage,
} from "../api";
import type { BadgeRefSummary, Campaign, UploadRow } from "../api";
import { CaptureRef } from "../components/CaptureRef";
import { CropPicker } from "../components/CropPicker";
import { Modal } from "../components/Modal";

export function CampaignsTab({
  campaigns,
  loading,
  error,
  onChanged,
}: {
  campaigns: Campaign[];
  loading: boolean;
  error: string | null;
  onChanged: () => void;
}) {
  const [selected, setSelected] = useState<number | null>(null);
  const [detail, setDetail] = useState<Campaign | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Campaign | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const [uploads, setUploads] = useState<UploadRow[] | null>(null);

  useEffect(() => {
    if (selected === null) {
      setDetail(null);
      return;
    }
    let live = true;
    getCampaign(selected)
      .then((c) => live && setDetail(c))
      .catch((e) => live && setFailure(String(e.message ?? e)));
    return () => {
      live = false;
    };
  }, [selected, campaigns]);

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setFailure(null);
    try {
      await action();
      onChanged();
    } catch (err) {
      setFailure(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
      <div className="space-y-4">
        <CreateForm
          busy={busy}
          onCreate={(name, colors, files) =>
            run(async () => {
              const created = await createCampaign(name, colors, files);
              setUploads(created.uploads);
              setSelected(created.id);
            })
          }
        />

        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-slate-700">
            Campaigns {campaigns.length > 0 && `(${campaigns.length})`}
          </h2>
          {error && <Banner tone="error">{error}</Banner>}
          {loading ? (
            <p className="text-sm text-slate-500">Loading…</p>
          ) : campaigns.length === 0 ? (
            <p className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">
              No campaigns yet.
            </p>
          ) : (
            <ul className="divide-y divide-slate-100 overflow-hidden rounded-xl border border-slate-200">
              {campaigns.map((c) => (
                <li key={c.id}>
                  <button
                    type="button"
                    onClick={() => setSelected(c.id)}
                    className={`flex w-full items-center gap-3 p-3 text-left hover:bg-slate-50 ${
                      selected === c.id ? "bg-slate-50" : ""
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold text-slate-800">
                        {c.name}
                        {!c.active && (
                          <span className="ml-2 rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-medium text-slate-600">
                            inactive
                          </span>
                        )}
                      </p>
                      <p className="text-xs text-slate-500">
                        {c.badge_ref_count} reference{c.badge_ref_count === 1 ? "" : "s"} ·{" "}
                        {c.colors.length} colour{c.colors.length === 1 ? "" : "s"}
                      </p>
                    </div>
                    <div className="flex gap-1">
                      {c.colors.map((col) => (
                        <span
                          key={col.id}
                          title={`${col.hex}${col.label ? ` (${col.label})` : ""}`}
                          className="h-5 w-5 rounded border border-slate-300"
                          style={{ backgroundColor: col.hex }}
                        />
                      ))}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="space-y-4">
        {failure && <Banner tone="error">{failure}</Banner>}
        {uploads && <UploadReport rows={uploads} onDismiss={() => setUploads(null)} />}
        {detail ? (
          <Detail
            campaign={detail}
            busy={busy}
            onPatch={(fields) => run(() => updateCampaign(detail.id, fields))}
            onAddRefs={(files) =>
              run(async () => setUploads((await addBadgeRefs(detail.id, files)).uploads))
            }
            onDeleteRef={(badgeId) => run(() => deleteBadgeRef(detail.id, badgeId))}
            onDelete={() => setPendingDelete(detail)}
          />
        ) : (
          <p className="text-sm text-slate-500">Select a campaign to edit it.</p>
        )}
      </div>

      {pendingDelete && (
        <Modal
          title={`Delete ${pendingDelete.name}?`}
          body={`This removes the campaign, its ${pendingDelete.badge_ref_count} badge reference image(s) and its colours. Enrolled faces are not affected.`}
          confirmLabel="Delete"
          danger
          busy={busy}
          onCancel={() => setPendingDelete(null)}
          onConfirm={() =>
            run(async () => {
              await deleteCampaign(pendingDelete.id);
              setPendingDelete(null);
              setSelected(null);
            })
          }
        />
      )}
    </div>
  );
}

function Banner({ tone, children }: { tone: "error" | "info"; children: React.ReactNode }) {
  const styles =
    tone === "error"
      ? "border-rose-200 bg-rose-50 text-rose-800"
      : "border-slate-200 bg-slate-50 text-slate-700";
  return <div className={`rounded-lg border p-3 text-sm ${styles}`}>{children}</div>;
}

function UploadReport({ rows, onDismiss }: { rows: UploadRow[]; onDismiss: () => void }) {
  return (
    <div className="rounded-xl border border-slate-200 p-3">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Reference uploads
        </p>
        <button type="button" onClick={onDismiss} className="text-xs text-slate-500 hover:underline">
          dismiss
        </button>
      </div>
      <ul className="space-y-1 text-sm">
        {rows.map((row, i) => (
          <li key={i} className="flex gap-2">
            <span className={row.ok ? "text-emerald-700" : "text-rose-700"}>
              {row.ok ? "✓" : "✕"}
            </span>
            <span className="min-w-0 flex-1">
              <span className="font-medium text-slate-700">{row.filename ?? `#${i}`}</span>{" "}
              <span
                className="text-xs text-slate-500"
                title="Distinctive detail points found in the image. More means more to match on; under 60 is rejected."
              >
                {row.keypoints} detail points
                {row.cropped ? " after cropping to the badge" : ""}
              </span>
              {row.error && <span className="block text-xs text-rose-700">{row.error}</span>}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function CreateForm({
  busy,
  onCreate,
}: {
  busy: boolean;
  onCreate: (name: string, colors: string[], files: Blob[]) => void;
}) {
  const [name, setName] = useState("");
  const [colors, setColors] = useState<{ hex: string; label: string }[]>([
    { hex: "#FFFFFF", label: "home" },
  ]);
  const [files, setFiles] = useState<File[]>([]);
  const [toCrop, setToCrop] = useState<File | null>(null);
  const [capturing, setCapturing] = useState(false);

  const specs = colors.filter((c) => c.hex).map((c) => (c.label ? `${c.hex}:${c.label}` : c.hex));

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onCreate(name.trim(), specs, files);
        setName("");
        setFiles([]);
      }}
      className="space-y-3 rounded-xl border border-slate-200 p-4"
    >
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">New campaign</p>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Campaign name"
        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-900"
      />

      <div className="space-y-2">
        {colors.map((color, i) => (
          <div key={i} className="flex items-center gap-2">
            <input
              type="color"
              value={color.hex}
              onChange={(e) =>
                setColors((cs) => cs.map((c, j) => (j === i ? { ...c, hex: e.target.value.toUpperCase() } : c)))
              }
              className="h-9 w-12 rounded border border-slate-300"
            />
            <input
              value={color.label}
              onChange={(e) =>
                setColors((cs) => cs.map((c, j) => (j === i ? { ...c, label: e.target.value } : c)))
              }
              placeholder="label"
              className="flex-1 rounded-lg border border-slate-300 px-2 py-1.5 text-sm outline-none focus:border-slate-900"
            />
            <button
              type="button"
              onClick={() => setColors((cs) => cs.filter((_, j) => j !== i))}
              disabled={colors.length === 1}
              className="rounded px-2 py-1 text-xs text-slate-500 hover:bg-slate-100 disabled:opacity-30"
            >
              remove
            </button>
          </div>
        ))}
        <button
          type="button"
          onClick={() => setColors((cs) => [...cs, { hex: "#1E3A8A", label: "" }])}
          className="text-xs font-medium text-slate-600 hover:underline"
        >
          + add colour
        </button>
      </div>

      <label className="block">
        <span className="text-xs text-slate-500">
          First badge reference — crop to the badge
        </span>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            const picked = e.target.files?.[0];
            if (picked) setToCrop(picked);
            e.target.value = "";
          }}
          className="mt-1 block w-full text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-slate-900 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-white"
        />
        {files.length > 0 && (
          <span className="mt-1 block text-[11px] text-emerald-700">
            {files.length} crop{files.length === 1 ? "" : "s"} ready
          </span>
        )}
        <span className="mt-1 block text-[11px] text-slate-500">
          Once a campaign has one reference you can add more from any full photo — the badge is
          then located automatically.
        </span>
      </label>
      <button
        type="button"
        onClick={() => setCapturing(true)}
        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-100"
      >
        Capture from camera instead
      </button>
      {capturing && (
        <CaptureRef
          onCancel={() => setCapturing(false)}
          onCaptured={(blob) => {
            setCapturing(false);
            setFiles((f) => [...f, new File([blob], `ref_${f.length}.png`, { type: "image/png" })]);
          }}
        />
      )}
      {toCrop && (
        <CropPicker
          file={toCrop}
          onCancel={() => setToCrop(null)}
          onCropped={(blob) => {
            setToCrop(null);
            setFiles((f) => [...f, new File([blob], `ref_${f.length}.png`, { type: "image/png" })]);
          }}
        />
      )}

      <button
        type="submit"
        disabled={busy || !name.trim() || specs.length === 0}
        className="w-full rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-700 disabled:opacity-40"
      >
        {busy ? "Working…" : "Create campaign"}
      </button>
    </form>
  );
}

function Detail({
  campaign,
  busy,
  onPatch,
  onAddRefs,
  onDeleteRef,
  onDelete,
}: {
  campaign: Campaign;
  busy: boolean;
  onPatch: (fields: Parameters<typeof updateCampaign>[1]) => void;
  onAddRefs: (files: Blob[]) => void;
  onDeleteRef: (badgeId: number) => void;
  onDelete: () => void;
}) {
  const [toCrop, setToCrop] = useState<File | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [minInliers, setMinInliers] = useState<string>(
    campaign.overrides.badge_min_inliers?.toString() ?? "",
  );
  const [deltaE, setDeltaE] = useState<string>(
    campaign.overrides.shirt_delta_e_max?.toString() ?? "",
  );

  useEffect(() => {
    setMinInliers(campaign.overrides.badge_min_inliers?.toString() ?? "");
    setDeltaE(campaign.overrides.shirt_delta_e_max?.toString() ?? "");
  }, [campaign.id, campaign.overrides.badge_min_inliers, campaign.overrides.shirt_delta_e_max]);

  return (
    <div className="space-y-4 rounded-xl border border-slate-200 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-base font-semibold text-slate-900">{campaign.name}</h3>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => onPatch({ active: !campaign.active })}
            disabled={busy}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100"
          >
            {campaign.active ? "Deactivate" : "Activate"}
          </button>
          <button
            type="button"
            onClick={onDelete}
            className="rounded-lg border border-rose-200 px-3 py-1.5 text-xs font-semibold text-rose-700 hover:bg-rose-50"
          >
            Delete
          </button>
        </div>
      </div>

      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Badge references ({campaign.badge_refs?.length ?? 0})
        </p>
        <div className="mt-2 grid grid-cols-3 gap-2 sm:grid-cols-4">
          {(campaign.badge_refs ?? []).map((ref) => (
            <BadgeRefThumb key={ref.id} badge={ref} onDelete={() => onDeleteRef(ref.id)} />
          ))}
          {(campaign.badge_refs?.length ?? 0) === 0 && (
            <p className="col-span-full rounded-lg border border-dashed border-slate-300 p-4 text-center text-xs text-slate-500">
              No references — detection will fail for this campaign.
            </p>
          )}
        </div>
        <input
          type="file"
          accept="image/*"
          disabled={busy}
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            // Every reference is cropped by hand. An uncropped photo spends its whole
            // feature budget on the table or the wall behind the badge -- measured at
            // 4% of keypoints landing on a badge that filled 11% of the frame -- and
            // matching then compares backgrounds rather than badges.
            if (files.length) setToCrop(files[0]);
            e.target.value = "";
          }}
          className="mt-2 block w-full text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-slate-100 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-slate-700"
        />
        <p className="mt-1 text-[11px] text-slate-500">
          Pick a photo and drag a box tightly around the badge — edge to edge, with as
          little background as you can manage. Crop one at a time.
        </p>
        <button
          type="button"
          onClick={() => setCapturing(true)}
          disabled={busy}
          className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-100 disabled:opacity-40"
        >
          Capture from camera
        </button>
        <p className="mt-1 text-[11px] text-slate-500">
          Best for glossy badges: a photo taken elsewhere shares little pixel detail with what
          this camera sees, because the shiny surface reflects different surroundings.
        </p>
        {capturing && (
          <CaptureRef onCancel={() => setCapturing(false)} onCaptured={(blob) => {
            setCapturing(false);
            setToCrop(new File([blob], "capture.png", { type: blob.type }));
          }} />
        )}
        {toCrop && (
          <CropPicker
            file={toCrop}
            onCancel={() => setToCrop(null)}
            onCropped={(blob) => {
              setToCrop(null);
              onAddRefs([blob]);
            }}
          />
        )}
      </div>

      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Threshold overrides
        </p>
        <div className="mt-2 flex flex-wrap items-end gap-3">
          <label className="text-xs text-slate-600" title="How many matched detail points must agree on one consistent position and angle before the badge counts as found. Higher is stricter.">
            Min matching points
            <input
              value={minInliers}
              onChange={(e) => setMinInliers(e.target.value)}
              placeholder={String(campaign.effective.badge_min_inliers)}
              className="mt-1 block w-28 rounded-lg border border-slate-300 px-2 py-1.5 text-sm outline-none focus:border-slate-900"
            />
          </label>
          <label className="text-xs text-slate-600" title="How far the measured shirt colour may sit from the expected colour. 0 is identical; about 14 is a noticeable but acceptable difference. Higher is more forgiving.">
            Max colour difference
            <input
              value={deltaE}
              onChange={(e) => setDeltaE(e.target.value)}
              placeholder={String(campaign.effective.shirt_delta_e_max)}
              className="mt-1 block w-28 rounded-lg border border-slate-300 px-2 py-1.5 text-sm outline-none focus:border-slate-900"
            />
          </label>
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              onPatch({
                badge_min_inliers: minInliers === "" ? "" : Number(minInliers),
                shirt_delta_e_max: deltaE === "" ? "" : Number(deltaE),
              })
            }
            className="rounded-lg bg-slate-900 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-700 disabled:opacity-40"
          >
            Save
          </button>
        </div>
        <p className="mt-1 text-[11px] text-slate-500">
          Blank uses the global default ({campaign.effective.badge_min_inliers} matching points,
          colour difference {campaign.effective.shirt_delta_e_max}). Raise the first to demand a
          clearer badge; raise the second to tolerate more colour variation.
        </p>
      </div>
    </div>
  );
}

function BadgeRefThumb({ badge, onDelete }: { badge: BadgeRefSummary; onDelete: () => void }) {
  const src = useAssetImage(badge.url);
  return (
    <figure className="group relative">
      <img
        src={src ?? undefined}
        alt={`reference ${badge.id}`}
        className="h-24 w-full rounded-lg border border-slate-200 bg-slate-50 object-contain"
      />
      <figcaption className="mt-1 text-center text-[10px] text-slate-500">
        {badge.keypoints} kp
      </figcaption>
      <button
        type="button"
        onClick={onDelete}
        className="absolute right-1 top-1 rounded bg-white/90 px-1.5 py-0.5 text-[10px] font-semibold text-rose-700 opacity-0 transition group-hover:opacity-100"
      >
        remove
      </button>
    </figure>
  );
}
