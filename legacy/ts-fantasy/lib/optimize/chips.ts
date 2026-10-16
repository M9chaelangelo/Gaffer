/**
 * Chip-rådgiveren.
 *
 * Chips er den største enkeltkilde til point man kan planlægge sig til, og
 * den mest spildte. De fleste spiller Bench Boost i en tilfældig uge og får
 * 12 point ud af noget der kunne have givet 30.
 *
 * Reglen der styrer alt: en chip er kun værd at spille, når MERVÆRDIEN over
 * en normal runde er stor nok til at slå den bedste fremtidige mulighed.
 * Derfor beregner vi merværdien eksplicit for hver chip og sammenligner den
 * med en tærskel, der stiger jo flere runder der er tilbage at bruge den i.
 */

import type { PointsDistribution } from "@/lib/fantasy/model/simulate";
import type { GameRules } from "@/lib/fantasy/rules/types";

export interface ChipEvaluation {
  chipId: string;
  name: string;
  /** Point vundet ved at spille chippen NU frem for slet ikke. */
  gain: number;
  /** Tærsklen den skal slå for at være værd at bruge i denne runde. */
  threshold: number;
  recommend: boolean;
  reason: string;
}

export interface ChipContext {
  rules: GameRules;
  /** Runder tilbage hvor chippen stadig kan bruges. */
  roundsRemaining: number;
  /** Fordelinger for hele truppen. */
  dists: Map<number, PointsDistribution>;
  starters: number[];
  bench: number[];
  captainId: number | null;
  /** Chips der stadig er i behold. */
  available: string[];
  /** Antal spillere i truppen uden kamp i runden — afgør Free Hit. */
  blanks?: number;
  /** Antal spillere med to kampe — afgør Bench Boost og Triple Captain. */
  doubles?: number;
}

/**
 * Tærsklen for at bruge en chip nu. Med mange runder tilbage skal gevinsten
 * være stor (der kommer en bedre uge); med få runder tilbage skal den bare
 * være positiv, for ellers går chippen til spilde.
 *
 * Formen er en simpel option-værdi: jo flere fremtidige forsøg, jo højere er
 * værdien af at vente.
 */
export function waitThreshold(base: number, roundsRemaining: number): number {
  if (roundsRemaining <= 1) return 0;
  return base * (1 - Math.exp(-0.18 * (roundsRemaining - 1)));
}

export function evaluateChips(ctx: ChipContext): ChipEvaluation[] {
  const out: ChipEvaluation[] = [];
  const { rules, dists, starters, bench, captainId, roundsRemaining } = ctx;

  const sum = (ids: number[]): number =>
    ids.reduce((s, id) => s + (dists.get(id)?.mean ?? 0), 0);

  for (const chip of rules.chips) {
    if (!ctx.available.includes(chip.id)) continue;

    if (chip.id === "bboost") {
      // Bench Boost giver præcis bænkens point — intet andet.
      const gain = sum(bench);
      const threshold = waitThreshold(18, roundsRemaining);
      out.push({
        chipId: chip.id,
        name: chip.name,
        gain,
        threshold,
        recommend: gain > threshold,
        reason:
          gain > threshold
            ? `Bænken er sat til ${gain.toFixed(1)} point. Over tærsklen på ${threshold.toFixed(1)} — spil den.`
            : `Bænken giver kun ${gain.toFixed(1)} point. Gem den til en dobbeltrunde, hvor fire spillere har to kampe (tærskel ${threshold.toFixed(1)}).`,
      });
      continue;
    }

    if (chip.id === "3xc") {
      // Triple Captain giver ét ekstra sæt af anførerens point.
      const d = captainId !== null ? dists.get(captainId) : undefined;
      const gain = d?.mean ?? 0;
      const threshold = waitThreshold(9, roundsRemaining);
      out.push({
        chipId: chip.id,
        name: chip.name,
        gain,
        threshold,
        recommend: gain > threshold,
        reason: d
          ? gain > threshold
            ? `Anføreren står til ${gain.toFixed(1)} point ekstra, og rammer 10+ i ${Math.round(d.pHaul * 100)} % af trækkene.`
            : `Anføreren står kun til ${gain.toFixed(1)} ekstra. Vent på en dobbeltrunde eller en hjemmekamp mod bunden (tærskel ${threshold.toFixed(1)}).`
          : "Ingen anfører valgt endnu.",
      });
      continue;
    }

    if (chip.id === "freehit") {
      // Free Hit er værd mest når truppen er tom for kampe.
      const blanks = ctx.blanks ?? 0;
      const gain = blanks * 4.2;
      const threshold = waitThreshold(20, roundsRemaining);
      out.push({
        chipId: chip.id,
        name: chip.name,
        gain,
        threshold,
        recommend: blanks >= 4 && gain > threshold,
        reason:
          blanks >= 4
            ? `${blanks} spillere uden kamp i runden. Free Hit henter cirka ${gain.toFixed(0)} point tilbage.`
            : `Kun ${blanks} spillere uden kamp. Free Hit er spildt her — den hører til en blank runde med 5+ uden kamp.`,
      });
      continue;
    }

    if (chip.id === "wildcard" || chip.id === "limitless") {
      // Wildcard vurderes ikke på én runde. Den er værd, når truppen er
      // strukturelt forkert — mange skift ville koste for meget i fradrag.
      const threshold = waitThreshold(12, roundsRemaining);
      out.push({
        chipId: chip.id,
        name: chip.name,
        gain: Number.NaN,
        threshold,
        recommend: false,
        reason:
          chip.id === "limitless"
            ? "Limitless er størst i en runde hvor de dyreste spillere har de bedste kampe — typisk sidste spilledag i ligafasen, hvor alt er afgjort og de store hold stadig spiller. Vurder den mod kampprogrammet, ikke mod én runde."
            : "Wildcard vurderes mod hele truppen, ikke mod én runde. Brug den når 4+ skift trænger sig på samtidig — kør trupbyggeren for at se hvad den ville give.",
      });
    }
  }

  return out.sort((a, b) => (b.gain || 0) - (a.gain || 0));
}
