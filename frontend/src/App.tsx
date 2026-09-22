import { useCallback, useEffect, useState } from "react";
import { listCampaigns, listPeople } from "./api";
import type { Campaign, Person } from "./api";
import { CampaignsTab } from "./tabs/CampaignsTab";
import { DetectTab } from "./tabs/DetectTab";
import { PeopleTab } from "./tabs/PeopleTab";
import { RegisterTab } from "./tabs/RegisterTab";

const TABS = ["Register", "Detect", "Campaigns", "People"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Register");
  const [people, setPeople] = useState<Person[]>([]);
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextPeople, nextCampaigns] = await Promise.all([listPeople(), listCampaigns()]);
      setPeople(nextPeople);
      setCampaigns(nextCampaigns);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-4 py-4">
          <div>
            <h1 className="text-lg font-bold tracking-tight">Face Verify</h1>
            <p className="text-xs text-slate-500">Face, badge and shirt verification</p>
          </div>
          <nav className="flex rounded-lg bg-slate-100 p-1" role="tablist">
            {TABS.map((name) => (
              <button
                key={name}
                type="button"
                role="tab"
                aria-selected={tab === name}
                onClick={() => setTab(name)}
                className={`rounded-md px-4 py-1.5 text-sm font-medium transition ${
                  tab === name
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {name}
                {name === "People" && people.length > 0 && (
                  <span className="ml-1.5 text-xs text-slate-400">{people.length}</span>
                )}
                {name === "Campaigns" && campaigns.length > 0 && (
                  <span className="ml-1.5 text-xs text-slate-400">{campaigns.length}</span>
                )}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-6">
        {tab === "Register" && <RegisterTab onRegistered={refresh} />}
        {tab === "Detect" && <DetectTab campaigns={campaigns} />}
        {tab === "Campaigns" && (
          <CampaignsTab
            campaigns={campaigns}
            loading={loading}
            error={error}
            onChanged={refresh}
          />
        )}
        {tab === "People" && (
          <PeopleTab people={people} loading={loading} error={error} onChanged={refresh} />
        )}
      </main>
    </div>
  );
}
