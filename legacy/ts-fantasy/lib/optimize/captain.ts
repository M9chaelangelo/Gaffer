/**
 * Anførervalget.
 *
 * Det enkeltvalg der flytter flest point over en sæson — og det, folk
 * oftest tager på gennemsnit. Gennemsnittet er det forkerte tal:
 *
 *  - Med dobbelte point vægtes halen dobbelt. En spiller med 6,0 i snit og
 *    20 % chance for et to-cifret træk er et bedre bind end en med 6,4 i snit
 *    og 6 % chance, hvis man jagter.
 *  - Ejerskab betyder alt. Er du bagud i en minirække, giver det ingen mening
 *    at anføre samme spiller som de andre — du kan ikke vinde på det.
 *    Er du foran, er det præcis det du skal.
 *
 * Funktionen giver derfor tre svar: det sikre, det aggressive, og det som
 * den valgte aggression peger på.
 */

import type { PointsDistribution } from "@/lib/fantasy/model/simulate";
import { effectiveOwnership, probabilityBeats, type OwnershipContext } from "@/lib/fantasy/optimize/rank";

export interface CaptainOption {
  playerId: number;
  name: string;
  mean: number;
  /** Forventede point med bindet på. */
  captainedMean: number;
  pHaul: number;
  ceiling: number;
  /** Andel af feltet der også anfører ham. */
  fieldCaptaincy: number;
  /** Point vundet relativt til feltet ved at anføre ham. */
  rankEdge: number;
  /** P(han slår det mest anførte alternativ). */
  pBeatsTemplate: number;
  reason: string;
}

export interface CaptainAdvice {
  /** Valget under den aktuelle aggression. */
  pick: CaptainOption;
  /** Højeste forventede point, uanset ejerskab. */
  safest: CaptainOption;
  /** Bedste differential — højest rang-gevinst hvis det rammer. */
  aggressive: CaptainOption;
  /** FPL: viceanfører. Den med højeste forventning blandt resten. */
  vice: CaptainOption | null;
  all: CaptainOption[];
}

export function chooseCaptain(
  starters: { playerId: number; name: string }[],
  dists: Map<number, PointsDistribution>,
  ctx: OwnershipContext,
  captainMultiplier: number,
): CaptainAdvice {
  const options: CaptainOption[] = [];

  // Feltets valg: den spiller flest anfører. Alt måles op mod ham, fordi det
  // er ham du reelt konkurrerer imod.
  let templateId: number | null = null;
  let templateShare = -1;
  for (const s of starters) {
    const share = ctx.captaincy.get(s.playerId) ?? 0;
    if (share > templateShare) {
      templateShare = share;
      templateId = s.playerId;
    }
  }
  const templateDist = templateId !== null ? dists.get(templateId) : undefined;

  for (const s of starters) {
    const d = dists.get(s.playerId);
    if (!d) continue;

    const fieldCaptaincy = ctx.captaincy.get(s.playerId) ?? 0;
    const captainedMean = d.mean * captainMultiplier;
    const eo = effectiveOwnership(s.playerId, ctx, captainMultiplier);
    // Rang-kanten: dine dobbelte point minus det feltet i gennemsnit får ud
    // af samme spiller.
    const rankEdge = captainedMean - d.mean * eo;

    const pBeatsTemplate = templateDist && templateId !== s.playerId
      ? probabilityBeats(d, templateDist)
      : 0.5;

    options.push({
      playerId: s.playerId,
      name: s.name,
      mean: d.mean,
      captainedMean,
      pHaul: d.pHaul,
      ceiling: d.ceiling,
      fieldCaptaincy,
      rankEdge,
      pBeatsTemplate,
      reason: explain(d, fieldCaptaincy, pBeatsTemplate),
    });
  }

  if (options.length === 0) {
    throw new Error("Ingen anførerkandidater — startopstillingen er tom.");
  }

  const safest = [...options].sort((a, b) => b.captainedMean - a.captainedMean)[0];
  const aggressive = [...options].sort((a, b) => b.rankEdge - a.rankEdge)[0];

  // Det blandede valg: vægt forventning mod rang-kant efter aggression.
  const pick = [...options].sort((a, b) => score(b, ctx) - score(a, ctx))[0];

  const vice = [...options]
    .filter((o) => o.playerId !== pick.playerId)
    .sort((a, b) => b.captainedMean - a.captainedMean)[0] ?? null;

  return { pick, safest, aggressive, vice, all: options.sort((a, b) => score(b, ctx) - score(a, ctx)) };
}

function score(o: CaptainOption, ctx: OwnershipContext): number {
  return ctx.aggression * o.rankEdge + (1 - ctx.aggression) * o.captainedMean;
}

function explain(d: PointsDistribution, fieldCaptaincy: number, pBeats: number): string {
  const haul = Math.round(d.pHaul * 100);
  const owned = Math.round(fieldCaptaincy * 100);
  const beats = Math.round(pBeats * 100);
  if (fieldCaptaincy > 0.4) {
    return `${haul} % chance for 10+. ${owned} % af feltet har samme bind — du følger med, du vinder ikke rang på ham.`;
  }
  if (fieldCaptaincy < 0.08 && d.pHaul > 0.15) {
    return `${haul} % chance for 10+ og kun ${owned} % af feltet har ham. ${beats} % chance for at slå feltets bind.`;
  }
  return `${haul} % chance for 10+, ${owned} % af feltet anfører ham.`;
}
