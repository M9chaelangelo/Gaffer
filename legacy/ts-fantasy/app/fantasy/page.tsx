"use client";

/**
 * Fantasy-dashboardet. Ét skærmbillede der svarer på "hvad gør jeg inden
 * deadline" for både FPL og UCL.
 *
 * Siden henter rådet fra /api/fantasy/advice, som kører hele modellen.
 * Den gemmer ikke noget selv — cron-jobbet skriver til Firestore, så to
 * kilder aldrig kan være uenige om hvad der er gemt.
 */

import { useCallback, useEffect, useState } from "react";

import { Pitch } from "@/components/fantasy/Pitch";
import {
  CaptainAdvice,
  ChipAdvice,
  Projection,
  TransferAdvice,
} from "@/components/fantasy/Recommendations";
import type { DeadlineAdvice } from "@/lib/fantasy/advice";

type Game = "FPL" | "UCL";

interface ApiResponse {
  ok: boolean;
  advice?: DeadlineAdvice;
  warnings?: string[];
  error?: string;
}

export default function FantasyPage() {
  const [game, setGame] = useState<Game>("FPL");
  const [aggression, setAggression] = useState(0.5);
  const [freeTransfers, setFreeTransfers] = useState(1);
  const [bank, setBank] = useState(0);

  const [advice, setAdvice] = useState<DeadlineAdvice | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        game,
        aggression: String(aggression),
        ft: String(freeTransfers),
        bank: String(bank),
      });
      const res = await fetch(`/api/fantasy/advice?${params}`);
      const data = (await res.json()) as ApiResponse;
      if (!data.ok || !data.advice) {
        setError(data.error ?? "Ukendt fejl");
        setAdvice(null);
      } else {
        setAdvice(data.advice);
        setWarnings(data.warnings ?? []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Kunne ikke nå API'et");
    } finally {
      setLoading(false);
    }
  }, [game, aggression, freeTransfers, bank]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <main className="shell">
      <header className="spread">
        <div>
          <h1>Fantasy</h1>
          <p className="muted">
            {advice
              ? `${advice.game} · runde ${advice.round}${advice.deadline ? ` · deadline ${formatDeadline(advice.deadline)}` : ""}`
              : "Henter…"}
          </p>
        </div>
        <button className="secondary" onClick={() => void load()} disabled={loading}>
          {loading ? "Regner…" : "Opdatér"}
        </button>
      </header>

      <div className="controls">
        <fieldset className="control">
          <legend className="label">Spil</legend>
          <div className="segmented">
            {(["FPL", "UCL"] as Game[]).map((g) => (
              <button
                key={g}
                className={game === g ? "seg seg-on" : "seg"}
                onClick={() => setGame(g)}
                aria-pressed={game === g}
              >
                {g}
              </button>
            ))}
          </div>
        </fieldset>

        <fieldset className="control">
          <legend className="label">Strategi</legend>
          <input
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={aggression}
            onChange={(e) => setAggression(Number(e.target.value))}
            aria-label="Aggression: 0 er sikre point, 1 er jagt på rang"
          />
          <span className="mono">
            {aggression < 0.34 ? "Sikre point" : aggression < 0.67 ? "Balanceret" : "Jagt rang"}
          </span>
        </fieldset>

        <fieldset className="control">
          <legend className="label">Gratis skift</legend>
          <input
            type="number"
            min={0}
            max={5}
            value={freeTransfers}
            onChange={(e) => setFreeTransfers(Number(e.target.value))}
          />
        </fieldset>

        <fieldset className="control">
          <legend className="label">Bank</legend>
          <input
            type="number"
            min={0}
            step={0.1}
            value={bank}
            onChange={(e) => setBank(Number(e.target.value))}
          />
        </fieldset>
      </div>

      {error && (
        <p className="notice status-critical">
          <strong>Kunne ikke hente rådet.</strong> {error}
        </p>
      )}

      {warnings.length > 0 && (
        <section className="block">
          <div className="block-head">
            <h2>Forbehold</h2>
            <span className="label">{warnings.length}</span>
          </div>
          <ul className="warning-list">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </section>
      )}

      {advice && (
        <>
          <Projection advice={advice} />

          <section className="block block-wide">
            <div className="block-head">
              <h2>Truppen</h2>
              <span className="label">Formation {advice.lineup.formation}</span>
            </div>
            <Pitch
              squad={advice.squad}
              starters={advice.lineup.starters}
              bench={advice.lineup.bench}
              captainId={advice.captain.pick.playerId}
              viceCaptainId={advice.captain.vice?.playerId ?? null}
              formation={advice.lineup.formation}
            />
          </section>

          <TransferAdvice advice={advice} />
          <CaptainAdvice advice={advice} />
          <ChipAdvice advice={advice} />

          <section className="block">
            <div className="block-head">
              <h2>Modellen</h2>
              <span className="label">{advice.modelSummary.matchesFitted} kampe fittet</span>
            </div>
            <p className="muted">
              Hjemmebanefordel {advice.modelSummary.homeAdvantage.toFixed(2)} ·
              lavscore-korrektion {advice.modelSummary.rho.toFixed(3)} ·
              basisrate {advice.modelSummary.baseRate.toFixed(2)} mål pr. hold
            </p>
            <div className="table-scroll">
              <table className="captain-table">
                <caption className="sr-only">Holdstyrker efter modellen</caption>
                <thead>
                  <tr>
                    <th scope="col">Hold</th>
                    <th scope="col">Angreb</th>
                    <th scope="col">Forsvar</th>
                  </tr>
                </thead>
                <tbody>
                  {advice.modelSummary.attackRanking.slice(0, 10).map((t) => (
                    <tr key={t.team}>
                      <th scope="row">{t.team}</th>
                      <td className="mono">{t.attack >= 0 ? "+" : ""}{t.attack.toFixed(2)}</td>
                      <td className="mono">{t.defence >= 0 ? "+" : ""}{t.defence.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {advice.caveats.length > 0 && (
              <ul className="warning-list">
                {advice.caveats.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </main>
  );
}

function formatDeadline(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("da-DK", {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}
