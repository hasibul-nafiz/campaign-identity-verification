import { useEffect, useRef, useState } from "react";
import { ApiError, registerPerson } from "../api";
import type { ImageStatus, RegisterError, RegisterResponse } from "../api";
import { Camera } from "../components/Camera";
import type { CameraHandle } from "../components/Camera";

const STEPS = [
  { pose: "front", instruction: "Look straight at the camera." },
  { pose: "left", instruction: "Turn your head to your left." },
  { pose: "right", instruction: "Turn your head to your right." },
  { pose: "up", instruction: "Tilt your chin up." },
  { pose: "down", instruction: "Tilt your chin down." },
] as const;

interface Shot {
  blob: Blob;
  url: string;
}

export function RegisterTab({ onRegistered }: { onRegistered: () => void }) {
  const cameraRef = useRef<CameraHandle>(null);
  const [name, setName] = useState("");
  const [shots, setShots] = useState<(Shot | null)[]>(() => STEPS.map(() => null));
  const [step, setStep] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RegisterResponse | null>(null);
  const [rows, setRows] = useState<ImageStatus[] | null>(null);

  useEffect(
    () => () => shots.forEach((s) => s && URL.revokeObjectURL(s.url)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const captured = shots.filter((s): s is Shot => s !== null).length;

  async function capture() {
    setError(null);
    try {
      const blob = await cameraRef.current!.capture();
      setShots((prev) => {
        const next = [...prev];
        if (next[step]) URL.revokeObjectURL(next[step]!.url);
        next[step] = { blob, url: URL.createObjectURL(blob) };
        return next;
      });
      setStep((s) => Math.min(s + 1, STEPS.length - 1));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function retake(index: number) {
    setShots((prev) => {
      const next = [...prev];
      if (next[index]) URL.revokeObjectURL(next[index]!.url);
      next[index] = null;
      return next;
    });
    setStep(index);
    setResult(null);
    setRows(null);
  }

  async function submit() {
    setBusy(true);
    setError(null);
    setResult(null);
    setRows(null);
    try {
      // One request at the end: the API only commits if a 'front' pose is present.
      const blobs = shots.filter((s): s is Shot => s !== null).map((s) => s.blob);
      const res = await registerPerson(name.trim(), blobs);
      setResult(res);
      setRows(res.images);
      onRegistered();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
        const body = err.body as RegisterError | undefined;
        if (body?.images) setRows(body.images);
      } else {
        setError(String(err));
      }
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    shots.forEach((s) => s && URL.revokeObjectURL(s.url));
    setShots(STEPS.map(() => null));
    setStep(0);
    setName("");
    setResult(null);
    setRows(null);
    setError(null);
  }

  // Server rows arrive in upload order, which is the order of the captured steps.
  const capturedStepIndexes = shots
    .map((shot, i) => (shot ? i : -1))
    .filter((i) => i >= 0);

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        <label className="block">
          <span className="text-sm font-medium text-slate-700">Name</span>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Ana Ruiz"
            className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-900"
          />
        </label>

        <Camera ref={cameraRef} />

        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Step {step + 1} of {STEPS.length} — {STEPS[step].pose}
          </p>
          <p className="mt-1 text-sm text-slate-700">{STEPS[step].instruction}</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={capture}
              className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-700"
            >
              Capture {STEPS[step].pose}
            </button>
            <button
              type="button"
              onClick={submit}
              disabled={busy || captured === 0 || !name.trim()}
              className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-40"
            >
              {busy ? "Registering…" : `Register (${captured})`}
            </button>
            <button
              type="button"
              onClick={reset}
              className="rounded-lg px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-200"
            >
              Reset
            </button>
          </div>
          {!name.trim() && captured > 0 && (
            <p className="mt-2 text-xs text-amber-700">Enter a name to enable registering.</p>
          )}
        </div>
      </div>

      <div className="space-y-4">
        <div className="grid grid-cols-5 gap-2">
          {STEPS.map((s, i) => (
            <button
              key={s.pose}
              type="button"
              onClick={() => setStep(i)}
              className={`rounded-lg border p-1 text-center ${
                i === step ? "border-slate-900 ring-2 ring-slate-900/20" : "border-slate-200"
              }`}
            >
              <div className="aspect-square overflow-hidden rounded bg-slate-100">
                {shots[i] ? (
                  <img src={shots[i]!.url} alt={s.pose} className="h-full w-full object-cover" />
                ) : (
                  <div className="grid h-full place-items-center text-[10px] text-slate-400">
                    empty
                  </div>
                )}
              </div>
              <span className="mt-1 block text-[11px] font-medium text-slate-600">{s.pose}</span>
            </button>
          ))}
        </div>

        {error && (
          <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
            {error}
          </div>
        )}

        {result && (
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900">
            Registered <strong>{result.person.name}</strong> with{" "}
            <strong>{result.templates_written}</strong> template
            {result.templates_written === 1 ? "" : "s"} ({result.poses.join(", ")}).
          </div>
        )}

        {rows && (
          <div className="overflow-hidden rounded-lg border border-slate-200">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-3 py-2">Asked for</th>
                  <th className="px-3 py-2">Detected</th>
                  <th className="px-3 py-2">Result</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {rows.map((row) => {
                  const stepIndex = capturedStepIndexes[row.index] ?? row.index;
                  const asked = STEPS[stepIndex]?.pose ?? "?";
                  const mismatch = row.ok && row.pose !== asked;
                  return (
                    <tr key={row.index}>
                      <td className="px-3 py-2 font-medium text-slate-700">{asked}</td>
                      <td className="px-3 py-2 text-slate-600">{row.pose ?? "—"}</td>
                      <td className="px-3 py-2">
                        {!row.ok ? (
                          <span className="text-rose-700">{row.error}</span>
                        ) : mismatch ? (
                          <span className="text-amber-700">
                            stored as “{row.pose}”
                            {row.replaced_earlier_image ? " (overwrote an earlier pose)" : ""}
                          </span>
                        ) : (
                          <span className="text-emerald-700">ok</span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {(!row.ok || mismatch) && (
                          <button
                            type="button"
                            onClick={() => retake(stepIndex)}
                            className="rounded border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100"
                          >
                            Retake
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <p className="text-xs text-slate-500">
          The server decides each pose from facial landmarks, so a capture can be stored under a
          different pose than the one requested. Poses are unique per person — a duplicate replaces
          the earlier one.
        </p>
      </div>
    </div>
  );
}
