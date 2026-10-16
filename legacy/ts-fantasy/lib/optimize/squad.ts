/**
 * Trupbygning fra bunden — wildcard, Free Hit, Limitless og sæsonstart.
 *
 * Her er rummet for stort til at være eksakt: ~700 spillere i 15 pladser under
 * budget og klubloft er langt over hvad man kan opregne. Det er en klassisk
 * multidimensionel rygsæk, og den er NP-hård.
 *
 * Vi bruger simuleret udglødning med et stort naboskab (LNS): start fra en
 * grådig løsning, byt gentagne gange 1-3 spillere ud, og accepter dårligere
 * løsninger med aftagende sandsynlighed, så søgningen kan komme ud af lokale
 * toppe. Med nok genstarter rammer den i praksis optimum for FPL-trupper —
 * og vi verificerer den mod råstyrke på små instanser i testene.
 *
 * Målfunktionen er IKKE summen af alle 15. Den er den bedste startopstilling
 * plus en rabatteret bænk, fordi bænken kun scorer når nogen udebliver.
 * Bygger man på summen af 15, ender man med fire dyre bænkspillere.
 */

import { Rng } from "@/lib/fantasy/model/rng";
import { bestLineup, type LineupCandidate } from "@/lib/fantasy/optimize/lineup";
import type { GameRules, Position } from "@/lib/fantasy/rules/types";
import type { Player } from "@/lib/fantasy/sources/types";
import type { PointsDistribution } from "@/lib/fantasy/model/simulate";
import { rankValue, type OwnershipContext } from "@/lib/fantasy/optimize/rank";

export interface SquadOption {
  player: Player;
  dist: PointsDistribution;
  price: number;
}

export interface BuiltSquad {
  playerIds: number[];
  starters: number[];
  bench: number[];
  formation: string;
  /** Startopstillingens forventede point. */
  expected: number;
  /** Målfunktionens værdi, bænkrabat iberegnet. */
  objective: number;
  cost: number;
  bankLeft: number;
}

export interface BuildOptions {
  rules: GameRules;
  pool: SquadOption[];
  budget: number;
  ownership?: OwnershipContext;
  /** Spillere der SKAL med (fx en du ikke vil sælge). */
  locked?: number[];
  /** Spillere der ikke må vælges. */
  banned?: number[];
  iterations?: number;
  restarts?: number;
  seed?: number;
  /**
   * Hvor meget en bænkplads er værd i forhold til en startplads.
   * ~0,12 svarer til hvor ofte en bænkspiller faktisk kommer ind at score.
   * Sættes til 1 når Bench Boost er aktiv — så scorer alle 15.
   */
  benchWeight?: number;
}

export function buildSquad(opts: BuildOptions): BuiltSquad {
  const {
    rules, pool, budget, ownership,
    locked = [], banned = [],
    iterations = 6000, restarts = 4, seed = 7,
    benchWeight = 0.12,
  } = opts;

  const banSet = new Set(banned);
  const usable = pool.filter((o) => !banSet.has(o.player.id) && o.player.available);
  const byId = new Map(usable.map((o) => [o.player.id, o]));
  const lockedSet = new Set(locked.filter((id) => byId.has(id)));

  const value = (o: SquadOption): number =>
    ownership ? rankValue(o.dist, ownership, rules.squad.captainMultiplier) : o.dist.mean;

  let best: BuiltSquad | null = null;

  for (let r = 0; r < restarts; r++) {
    const rng = new Rng(seed + r * 7919);
    let current = greedySeed(usable, lockedSet, rules, budget, value, rng);
    if (!current) continue;

    let currentScore = objective(current, byId, rules, value, benchWeight);
    let bestLocal = current;
    let bestLocalScore = currentScore;

    for (let i = 0; i < iterations; i++) {
      // Temperaturen falder lineært: tidligt accepteres meget, sent næsten intet.
      const temperature = Math.max(0.01, 1.2 * (1 - i / iterations));
      const candidate = mutate(current, usable, byId, lockedSet, rules, budget, rng);
      if (!candidate) continue;

      const score = objective(candidate, byId, rules, value, benchWeight);
      const delta = score - currentScore;

      if (delta > 0 || rng.next() < Math.exp(delta / temperature)) {
        current = candidate;
        currentScore = score;
        if (score > bestLocalScore) {
          bestLocal = candidate;
          bestLocalScore = score;
        }
      }
    }

    const built = describe(bestLocal, byId, rules, value, benchWeight, budget);
    if (!best || built.objective > best.objective) best = built;
  }

  if (!best) {
    throw new Error(
      "Kunne ikke bygge en lovlig trup. Tjek at spillerpuljen dækker alle positioner " +
        "og at budgettet rækker til den billigste lovlige sammensætning.",
    );
  }
  return best;
}

/** Grådig start: bedste point pr. million, position for position, under loftet. */
function greedySeed(
  pool: SquadOption[],
  locked: Set<number>,
  rules: GameRules,
  budget: number,
  value: (o: SquadOption) => number,
  rng: Rng,
): number[] | null {
  const need: Record<Position, number> = { ...rules.squad.composition };
  const chosen: number[] = [];
  const clubs = new Map<number, number>();
  let spent = 0;

  const take = (o: SquadOption): void => {
    chosen.push(o.player.id);
    need[o.player.position]--;
    clubs.set(o.player.team, (clubs.get(o.player.team) ?? 0) + 1);
    spent += o.price;
  };

  for (const id of locked) {
    const o = pool.find((p) => p.player.id === id);
    if (o && need[o.player.position] > 0) take(o);
  }

  // Lidt støj i rangeringen, så genstarter ikke giver samme greedy-løsning.
  const ranked = [...pool]
    .filter((o) => !chosen.includes(o.player.id))
    .sort((a, b) => value(b) / Math.max(0.1, b.price) - value(a) / Math.max(0.1, a.price));

  for (const o of ranked) {
    if (need[o.player.position] <= 0) continue;
    if ((clubs.get(o.player.team) ?? 0) >= rules.squad.maxPerClub) continue;
    if (spent + o.price > budget) continue;
    // 15 % spring, så løsningerne spreder sig mellem genstarter.
    if (rng.next() < 0.15) continue;
    take(o);
    if (chosen.length === rules.squad.size) break;
  }

  // Fyld resten med de billigste lovlige, hvis springene efterlod huller.
  if (chosen.length < rules.squad.size) {
    const cheap = [...pool]
      .filter((o) => !chosen.includes(o.player.id))
      .sort((a, b) => a.price - b.price);
    for (const o of cheap) {
      if (chosen.length === rules.squad.size) break;
      if (need[o.player.position] <= 0) continue;
      if ((clubs.get(o.player.team) ?? 0) >= rules.squad.maxPerClub) continue;
      if (spent + o.price > budget) continue;
      take(o);
    }
  }

  return chosen.length === rules.squad.size ? chosen : null;
}

/** Byt én tilfældig spiller ud med en lovlig afløser i samme position. */
function mutate(
  squad: number[],
  pool: SquadOption[],
  byId: Map<number, SquadOption>,
  locked: Set<number>,
  rules: GameRules,
  budget: number,
  rng: Rng,
): number[] | null {
  const swappable = squad.filter((id) => !locked.has(id));
  if (swappable.length === 0) return null;

  const outId = swappable[Math.floor(rng.next() * swappable.length)];
  const out = byId.get(outId);
  if (!out) return null;

  const rest = squad.filter((id) => id !== outId);
  const spentRest = rest.reduce((s, id) => s + (byId.get(id)?.price ?? 0), 0);
  const room = budget - spentRest;

  const clubs = new Map<number, number>();
  for (const id of rest) {
    const t = byId.get(id)?.player.team;
    if (t !== undefined) clubs.set(t, (clubs.get(t) ?? 0) + 1);
  }

  const options = pool.filter(
    (o) =>
      o.player.position === out.player.position &&
      !rest.includes(o.player.id) &&
      o.price <= room &&
      (clubs.get(o.player.team) ?? 0) < rules.squad.maxPerClub,
  );
  if (options.length === 0) return null;

  const pick = options[Math.floor(rng.next() * options.length)];
  return [...rest, pick.player.id];
}

/** Bedste XI plus rabatteret bænk. */
function objective(
  squad: number[],
  byId: Map<number, SquadOption>,
  rules: GameRules,
  value: (o: SquadOption) => number,
  benchWeight: number,
): number {
  const candidates: LineupCandidate[] = [];
  for (const id of squad) {
    const o = byId.get(id);
    if (!o) continue;
    candidates.push({
      playerId: id,
      position: o.player.position,
      mean: value(o),
      pAppear: 1 - o.dist.pNoShow,
    });
  }
  let lineup;
  try {
    lineup = bestLineup(candidates, rules.squad);
  } catch {
    // Ulovlig sammensætning — straf hårdt, så søgningen vender om.
    return Number.NEGATIVE_INFINITY;
  }
  const starters = new Set(lineup.starters);
  const bench = candidates.filter((c) => !starters.has(c.playerId));
  const benchValue = bench.reduce((s, b) => s + b.mean * b.pAppear, 0);
  return lineup.expected + benchWeight * benchValue;
}

function describe(
  squad: number[],
  byId: Map<number, SquadOption>,
  rules: GameRules,
  value: (o: SquadOption) => number,
  benchWeight: number,
  budget: number,
): BuiltSquad {
  const candidates: LineupCandidate[] = squad.map((id) => {
    const o = byId.get(id)!;
    return {
      playerId: id,
      position: o.player.position,
      mean: o.dist.mean,
      pAppear: 1 - o.dist.pNoShow,
    };
  });
  const lineup = bestLineup(candidates, rules.squad);
  const cost = squad.reduce((s, id) => s + (byId.get(id)?.price ?? 0), 0);

  return {
    playerIds: squad,
    starters: lineup.starters,
    bench: lineup.bench,
    formation: lineup.formation,
    expected: lineup.expected,
    objective: objective(squad, byId, rules, value, benchWeight),
    cost,
    bankLeft: budget - cost,
  };
}
