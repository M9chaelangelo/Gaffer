/**
 * Startopstilling og bænkorden.
 *
 * Opgaven ligner et simpelt "vælg de 11 højeste", men formationsreglerne gør
 * den ikke-triviel: mindst 1 målmand, 3 forsvarere, 2 midtbaner og 1 angriber.
 * Grådigt valg kan male sig op i et hjørne, hvor den sidste plads skal fyldes
 * af en forsvarer, og der kun er dyre alternativer tilbage.
 *
 * Feltet er lille nok til at gøre det eksakt: der findes kun en håndfuld
 * lovlige formationer, og inden for hver er valget grådigt og optimalt.
 * Vi opregner alle formationer og tager den bedste. Det er både hurtigere og
 * mere sikkert end en heuristik.
 */

import type { Position } from "@/lib/fantasy/rules/types";
import type { SquadRules } from "@/lib/fantasy/rules/types";

export interface LineupCandidate {
  playerId: number;
  position: Position;
  /** Forventede point. Bænkorden bruger samme tal. */
  mean: number;
  /** Sandsynlighed for overhovedet at spille — afgør bænkorden. */
  pAppear: number;
}

export interface Lineup {
  starters: number[];
  bench: number[];
  formation: string;
  /** Summen af starternes forventede point, uden anfører. */
  expected: number;
}

/** Alle lovlige formationer under reglerne. */
export function legalFormations(squad: SquadRules): Record<Position, number>[] {
  const out: Record<Position, number>[] = [];
  const { minStarting: lo, maxStarting: hi, startingSize } = squad;

  for (let def = lo.DEF; def <= hi.DEF; def++) {
    for (let mid = lo.MID; mid <= hi.MID; mid++) {
      for (let fwd = lo.FWD; fwd <= hi.FWD; fwd++) {
        const gk = startingSize - def - mid - fwd;
        if (gk < lo.GK || gk > hi.GK) continue;
        out.push({ GK: gk, DEF: def, MID: mid, FWD: fwd });
      }
    }
  }
  return out;
}

/**
 * Bedste startopstilling. `valueOf` gør funktionen brugbar både til ren
 * forventning og til den rang-bevidste målfunktion — kaldere bestemmer selv
 * hvad "bedst" betyder.
 */
export function bestLineup(
  candidates: LineupCandidate[],
  squad: SquadRules,
  valueOf: (c: LineupCandidate) => number = (c) => c.mean,
): Lineup {
  const byPosition = new Map<Position, LineupCandidate[]>();
  for (const c of candidates) {
    const list = byPosition.get(c.position) ?? [];
    list.push(c);
    byPosition.set(c.position, list);
  }
  // Sortér én gang; inden for en formation er grådigt valg optimalt.
  for (const list of byPosition.values()) {
    list.sort((a, b) => valueOf(b) - valueOf(a));
  }

  let best: Lineup | null = null;

  for (const formation of legalFormations(squad)) {
    const starters: LineupCandidate[] = [];
    let feasible = true;
    let total = 0;

    for (const position of ["GK", "DEF", "MID", "FWD"] as Position[]) {
      const need = formation[position];
      const available = byPosition.get(position) ?? [];
      if (available.length < need) {
        feasible = false;
        break;
      }
      for (let i = 0; i < need; i++) {
        starters.push(available[i]);
        total += valueOf(available[i]);
      }
    }
    if (!feasible) continue;

    if (!best || total > best.expected) {
      const startingIds = new Set(starters.map((s) => s.playerId));
      best = {
        starters: starters.map((s) => s.playerId),
        bench: orderBench(candidates.filter((c) => !startingIds.has(c.playerId)), valueOf),
        formation: `${formation.DEF}-${formation.MID}-${formation.FWD}`,
        expected: total,
      };
    }
  }

  if (!best) {
    throw new Error(
      "Ingen lovlig startopstilling kunne dannes — truppen mangler spillere i mindst én position.",
    );
  }
  return best;
}

/**
 * Bænkorden. Den er ikke bare "næstbedste først": en bænkspiller er kun
 * værd noget, hvis han FAKTISK spiller, og kun hvis en starter udebliver.
 * Derfor vægtes han med sin egen spillesandsynlighed. En stjerne med 30 %
 * chance for at spille er en dårligere første bænkmand end en sikker starter
 * i en mindre klub.
 *
 * Målmanden kan kun erstatte målmanden, så han holdes altid sidst blandt
 * markspillerne — spillet bytter ham kun ind for den anden keeper.
 */
function orderBench(
  bench: LineupCandidate[],
  valueOf: (c: LineupCandidate) => number,
): number[] {
  const outfield = bench.filter((b) => b.position !== "GK");
  const keepers = bench.filter((b) => b.position === "GK");
  outfield.sort((a, b) => valueOf(b) * b.pAppear - valueOf(a) * a.pAppear);
  return [...outfield.map((b) => b.playerId), ...keepers.map((b) => b.playerId)];
}
