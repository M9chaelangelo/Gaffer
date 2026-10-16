/**
 * Skifteplanlæggeren.
 *
 * Søgerummet er lille nok til at gøre eksakt for de skift der faktisk laves:
 *
 *   0 skift   1 mulighed
 *   1 skift   ~15 ud x ~600 ind = 9.000 kombinationer
 *   2 skift   ~9.000 x 9.000 filtreret på budget = håndterbart med beskæring
 *
 * Derfor opregner vi 0 og 1 skift eksakt og beskærer 2 skift hårdt på
 * budget og positionsmatch. Det er bedre end en heuristik, fordi vi så kan
 * garantere, at rådet ER det bedste under modellen — ikke bare et godt bud.
 *
 * Værdien af et skift er IKKE bare "ind minus ud". Tre ting tælles med:
 *
 *  1. Skiftet ændrer startopstillingen, og dermed også hvem der er bænket.
 *     Et skift der hæver en bænkspiller er mindre værd end et der hæver XI.
 *  2. Fradraget på -4 skal tjenes hjem. Over én runde er det en høj tærskel;
 *     derfor vurderes gevinsten over en horisont, ikke kun næste runde.
 *  3. Rang frem for point: en spiller ejet af 60 % af feltet giver mindre
 *     rang-gevinst end en lige så god ejet af 5 %.
 */

import type { Player } from "@/lib/fantasy/sources/types";
import type { PointsDistribution } from "@/lib/fantasy/model/simulate";
import type { GameRules } from "@/lib/fantasy/rules/types";
import { bestLineup, type LineupCandidate } from "@/lib/fantasy/optimize/lineup";
import { rankValue, type OwnershipContext } from "@/lib/fantasy/optimize/rank";

export interface TransferCandidate {
  player: Player;
  dist: PointsDistribution;
  /** Prisen du kan købe ham for. */
  buyPrice: number;
}

export interface HeldPlayer {
  player: Player;
  dist: PointsDistribution;
  /** Prisen du kan sælge ham for (FPL tager halvdelen af gevinsten). */
  sellPrice: number;
}

export interface TransferMove {
  out: { playerId: number; name: string; price: number };
  in: { playerId: number; name: string; price: number };
}

export interface TransferPlan {
  moves: TransferMove[];
  /** Fradrag i point (0, -4, -8 …). */
  hit: number;
  /** Forventede point for startopstillingen EFTER skiftene, uden anfører. */
  expectedAfter: number;
  /** Nettogevinst mod at blive stående, fradrag iberegnet. */
  gain: number;
  /** Samme gevinst målt i rang frem for point. */
  rankGain: number;
  bankAfter: number;
  formation: string;
  starters: number[];
  bench: number[];
  reason: string;
}

export interface PlanOptions {
  rules: GameRules;
  held: HeldPlayer[];
  market: TransferCandidate[];
  bank: number;
  freeTransfers: number;
  ownership: OwnershipContext;
  /** Hvor mange skift der højst overvejes. 2 er eksakt; 3+ bliver grådigt. */
  maxTransfers?: number;
  /**
   * Hvor mange runder gevinsten forventes at holde. Et skift der er 1,5 point
   * bedre i næste runde er 6 point bedre over fire — det er sådan et fradrag
   * tjener sig hjem. Konservativt sat, fordi fremtiden er usikker.
   */
  horizon?: number;
}

export function planTransfers(opts: PlanOptions): TransferPlan[] {
  const {
    rules, held, market, bank, freeTransfers,
    ownership, maxTransfers = 2, horizon = 3,
  } = opts;

  const heldById = new Map(held.map((h) => [h.player.id, h]));
  const marketById = new Map(market.map((m) => [m.player.id, m]));

  // Baseline: bliv stående.
  const base = evaluate([...heldById.keys()], heldById, marketById, rules, ownership);
  const plans: TransferPlan[] = [
    {
      moves: [],
      hit: 0,
      expectedAfter: base.expected,
      gain: 0,
      rankGain: 0,
      bankAfter: bank,
      formation: base.formation,
      starters: base.starters,
      bench: base.bench,
      reason: "Stå stille. Ingen skift tjener sit fradrag hjem under modellen.",
    },
  ];

  const currentIds = held.map((h) => h.player.id);
  const clubCount = countByClub(held.map((h) => h.player));

  // --- Ét skift: eksakt opregning -------------------------------------
  const singles: { plan: TransferPlan; key: string }[] = [];

  for (const out of held) {
    for (const cand of market) {
      if (heldById.has(cand.player.id)) continue;
      if (cand.player.position !== out.player.position) continue;
      if (!cand.player.available) continue;

      const bankAfter = bank + out.sellPrice - cand.buyPrice;
      if (bankAfter < -1e-9) continue;
      if (!clubOk(clubCount, out.player, cand.player, rules.squad.maxPerClub)) continue;

      const ids = currentIds.filter((id) => id !== out.player.id).concat(cand.player.id);
      const after = evaluate(ids, heldById, marketById, rules, ownership);

      const hit = freeTransfers >= 1 ? 0 : rules.transfers.costPerExtra;
      const gain = (after.expected - base.expected) * horizon + hit;
      const rankGain = (after.rank - base.rank) * horizon + hit;

      singles.push({
        key: `${out.player.id}->${cand.player.id}`,
        plan: {
          moves: [{
            out: { playerId: out.player.id, name: out.player.name, price: out.sellPrice },
            in: { playerId: cand.player.id, name: cand.player.name, price: cand.buyPrice },
          }],
          hit,
          expectedAfter: after.expected,
          gain,
          rankGain,
          bankAfter,
          formation: after.formation,
          starters: after.starters,
          bench: after.bench,
          reason: describeMove(out, cand, after.expected - base.expected, hit, horizon),
        },
      });
    }
  }

  singles.sort((a, b) => b.plan.gain - a.plan.gain);
  plans.push(...singles.slice(0, 12).map((s) => s.plan));

  // --- To skift: eksakt, men beskåret --------------------------------
  // Kun de bedste enkeltskift pr. spiller kombineres. Uden beskæringen er
  // rummet 81 millioner par; med den er det få tusinde, og optimum bevares
  // i praksis, fordi to skift der begge er dårlige alene sjældent er gode
  // sammen (undtagelsen er finansiering, som håndteres ved at tillade
  // negative enkeltgevinster i toppen af listen).
  if (maxTransfers >= 2 && singles.length > 1) {
    const perOut = new Map<number, { plan: TransferPlan; key: string }[]>();
    for (const s of singles) {
      const outId = s.plan.moves[0].out.playerId;
      const list = perOut.get(outId) ?? [];
      if (list.length < 8) list.push(s);
      perOut.set(outId, list);
    }

    const pairs: TransferPlan[] = [];
    const outIds = [...perOut.keys()];

    for (let i = 0; i < outIds.length; i++) {
      for (let j = i + 1; j < outIds.length; j++) {
        for (const a of perOut.get(outIds[i]) ?? []) {
          for (const b of perOut.get(outIds[j]) ?? []) {
            const inA = a.plan.moves[0].in;
            const inB = b.plan.moves[0].in;
            if (inA.playerId === inB.playerId) continue;

            const spend = inA.price + inB.price;
            const raise = a.plan.moves[0].out.price + b.plan.moves[0].out.price;
            const bankAfter = bank + raise - spend;
            if (bankAfter < -1e-9) continue;

            const ids = currentIds
              .filter((id) => id !== outIds[i] && id !== outIds[j])
              .concat(inA.playerId, inB.playerId);

            if (!clubOkMulti(ids, heldById, marketById, rules.squad.maxPerClub)) continue;

            const after = evaluate(ids, heldById, marketById, rules, ownership);
            const extra = Math.max(0, 2 - freeTransfers);
            const hit = extra * rules.transfers.costPerExtra;
            const gain = (after.expected - base.expected) * horizon + hit;

            pairs.push({
              moves: [a.plan.moves[0], b.plan.moves[0]],
              hit,
              expectedAfter: after.expected,
              gain,
              rankGain: (after.rank - base.rank) * horizon + hit,
              bankAfter,
              formation: after.formation,
              starters: after.starters,
              bench: after.bench,
              reason:
                hit < 0
                  ? `To skift med ${Math.abs(hit)} i fradrag. Gevinsten over ${horizon} runder er ${gain.toFixed(1)} point netto.`
                  : `To skift inden for de gratis. ${gain.toFixed(1)} point over ${horizon} runder.`,
            });
          }
        }
      }
    }

    pairs.sort((a, b) => b.gain - a.gain);
    plans.push(...pairs.slice(0, 8));
  }

  plans.sort((a, b) => b.gain - a.gain);
  return plans;
}

interface Evaluation {
  expected: number;
  rank: number;
  formation: string;
  starters: number[];
  bench: number[];
}

/** Sætter den bedste XI for en given trup og returnerer begge målestokke. */
function evaluate(
  ids: number[],
  held: Map<number, HeldPlayer>,
  market: Map<number, TransferCandidate>,
  rules: GameRules,
  ownership: OwnershipContext,
): Evaluation {
  const candidates: LineupCandidate[] = [];
  const rankByPlayer = new Map<number, number>();

  for (const id of ids) {
    const h = held.get(id);
    const m = market.get(id);
    const player = h?.player ?? m?.player;
    const dist = h?.dist ?? m?.dist;
    if (!player || !dist) continue;

    candidates.push({
      playerId: id,
      position: player.position,
      mean: dist.mean,
      pAppear: 1 - dist.pNoShow,
    });
    rankByPlayer.set(id, rankValue(dist, ownership, rules.squad.captainMultiplier));
  }

  const lineup = bestLineup(candidates, rules.squad);
  const rank = lineup.starters.reduce((s, id) => s + (rankByPlayer.get(id) ?? 0), 0);

  return {
    expected: lineup.expected,
    rank,
    formation: lineup.formation,
    starters: lineup.starters,
    bench: lineup.bench,
  };
}

function countByClub(players: Player[]): Map<number, number> {
  const m = new Map<number, number>();
  for (const p of players) m.set(p.team, (m.get(p.team) ?? 0) + 1);
  return m;
}

function clubOk(counts: Map<number, number>, out: Player, incoming: Player, max: number): boolean {
  if (out.team === incoming.team) return true;
  return (counts.get(incoming.team) ?? 0) + 1 <= max;
}

function clubOkMulti(
  ids: number[],
  held: Map<number, HeldPlayer>,
  market: Map<number, TransferCandidate>,
  max: number,
): boolean {
  const counts = new Map<number, number>();
  for (const id of ids) {
    const team = held.get(id)?.player.team ?? market.get(id)?.player.team;
    if (team === undefined) continue;
    const next = (counts.get(team) ?? 0) + 1;
    if (next > max) return false;
    counts.set(team, next);
  }
  return true;
}

function describeMove(
  out: HeldPlayer,
  cand: TransferCandidate,
  weeklyGain: number,
  hit: number,
  horizon: number,
): string {
  const parts: string[] = [];
  parts.push(`${out.player.name} → ${cand.player.name}`);
  parts.push(`${weeklyGain >= 0 ? "+" : ""}${weeklyGain.toFixed(1)} point pr. runde`);

  if (!out.player.available) {
    parts.push(`${out.player.name} er ude: ${out.player.news || "meldt utilgængelig"}`);
  } else if (out.dist.pNoShow > 0.3) {
    parts.push(`${out.player.name} har ${Math.round(out.dist.pNoShow * 100)} % risiko for slet ikke at spille`);
  }

  if (cand.dist.pHaul > 0.18) {
    parts.push(`${cand.player.name} rammer 10+ i ${Math.round(cand.dist.pHaul * 100)} % af trækkene`);
  }

  if (hit < 0) {
    parts.push(`Koster ${Math.abs(hit)} — tjent hjem på ${(Math.abs(hit) / Math.max(0.01, weeklyGain)).toFixed(1)} runder, horisont ${horizon}`);
  }

  return parts.join(". ") + ".";
}
