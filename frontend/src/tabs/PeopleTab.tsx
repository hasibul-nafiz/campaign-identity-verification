import { useState } from "react";
import { deletePerson } from "../api";
import type { Person } from "../api";
import { Modal } from "../components/Modal";

export function PeopleTab({
  people,
  loading,
  error,
  onChanged,
}: {
  people: Person[];
  loading: boolean;
  error: string | null;
  onChanged: () => void;
}) {
  const [pending, setPending] = useState<Person | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  async function confirmDelete() {
    if (!pending) return;
    setBusy(true);
    setFailure(null);
    try {
      await deletePerson(pending.id);
      setPending(null);
      onChanged();
    } catch (err) {
      setFailure(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700">
          Registered people {people.length > 0 && `(${people.length})`}
        </h2>
        <button
          type="button"
          onClick={onChanged}
          className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
        >
          Refresh
        </button>
      </div>

      {(error || failure) && (
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
          {error ?? failure}
        </div>
      )}

      {loading ? (
        <p className="text-sm text-slate-500">Loading…</p>
      ) : people.length === 0 ? (
        <p className="rounded-lg border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">
          Nobody registered yet. Use the Register tab to enroll someone.
        </p>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-200">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-3">Name</th>
                <th className="px-4 py-3">Poses captured</th>
                <th className="px-4 py-3">Templates</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {people.map((person) => (
                <tr key={person.id}>
                  <td className="px-4 py-3 font-medium text-slate-800">{person.name}</td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-1">
                      {person.poses.length === 0 ? (
                        <span className="text-xs text-slate-400">none</span>
                      ) : (
                        person.poses.map((pose) => (
                          <span
                            key={pose}
                            className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700"
                          >
                            {pose}
                          </span>
                        ))
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3 tabular-nums text-slate-600">
                    {person.template_count}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <button
                      type="button"
                      onClick={() => setPending(person)}
                      className="rounded-lg border border-rose-200 px-3 py-1.5 text-xs font-semibold text-rose-700 hover:bg-rose-50"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {pending && (
        <Modal
          title={`Delete ${pending.name}?`}
          body={`This removes the person and all ${pending.template_count} face template${
            pending.template_count === 1 ? "" : "s"
          }. This cannot be undone.`}
          confirmLabel="Delete"
          danger
          busy={busy}
          onConfirm={confirmDelete}
          onCancel={() => setPending(null)}
        />
      )}
    </div>
  );
}
