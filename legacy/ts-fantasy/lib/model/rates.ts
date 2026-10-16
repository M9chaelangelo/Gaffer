/**
 * Spillerrater pr. 90 minutter, krympet mod en positions-prior.
 *
 * Problemet: en angriber der har scoret 2 mål på 90 minutter har en rate på
 * 2,0 mål/90. Det er ikke hans evne, det er støj. Med fire kampe mere ligger
 * han på 0,6. Naive rater gør præcis den fejl, og den er dyr: den køber
 * spilleren på toppen af hans varians.
 *
 * Løsningen er empirisk Bayes. Vi lægger en prior på positionen (hvad en
 * gennemsnitlig midtbane i denne liga scorer pr. 90) og krymper spillerens
 * observerede rate mod den, vægtet af hvor mange minutter han har spillet:
 *
 *   rate = (observeret * minutter + prior * pseudo-minutter) / (minutter + pseudo)
 *
 * Med 90 minutter spillet vejer prioren tungest; med 1500 minutter gør den
 * næsten ingenting. Det er den rigtige kurve, fordi det er sådan tilliden
 * faktisk vokser.
 *
 * Hvor xG og xA findes (FPL leverer dem), bruges de frem for faktiske mål:
 * xG konvergerer hurtigere end mål, fordi det ikke straffer en spiller for
 * ikke at have scoret på fire store chancer.
 */

import type { Player, PlayerMatch } from "@/lib/fantasy/sources/types";
import type { Position } from "@/lib/fantasy/rules/types";

export interface PlayerRates {
  playerId: number;
  /** Mål pr. 90, krympet. */
  goals90: number;
  /** Assists pr. 90, krympet. */
  assists90: number;
  /** Redninger pr. 90 (kun målmænd). */
  saves90: number;
  /** Defensive handlinger pr. 90 — CBIT/CBIRT alt efter position. */
  defensive90: number;
  /** Boldgenerobringer pr. 90 (UCL scorer dem direkte). */
  recoveries90: number;
  /** Gule kort pr. 90. */
  yellows90: number;
  /** Andel af holdets straffespark, hvis han er fast skytte. */
  penaltyShare: number;
  /** Bonuspoint pr. 90 (FPL). Ren observation — BPS er for kringlet at modellere. */
  bonus90: number;
  /** Vægtede minutter bag estimaterne. */
  sampleMinutes: number;
  /** 0-1: hvor meget der er måling frem for prior. */
  confidence: number;
}

/**
 * Priors pr. 90 minutter pr. position. Sat fra typiske PL-gennemsnit.
 * De behøver ikke være præcise — de skal blot være rimelige, fordi de kun
 * dominerer når datagrundlaget er tyndt.
 */
const PRIORS: Record<Position, Omit<PlayerRates, "playerId" | "sampleMinutes" | "confidence">> = {
  GK: {
    goals90: 0.001, assists90: 0.01, saves90: 3.0, defensive90: 1.2,
    recoveries90: 2.5, yellows90: 0.05, penaltyShare: 0, bonus90: 0.35,
  },
  DEF: {
    goals90: 0.05, assists90: 0.06, saves90: 0, defensive90: 6.5,
    recoveries90: 4.5, yellows90: 0.18, penaltyShare: 0, bonus90: 0.30,
  },
  MID: {
    goals90: 0.14, assists90: 0.13, saves90: 0, defensive90: 5.0,
    recoveries90: 4.0, yellows90: 0.16, penaltyShare: 0, bonus90: 0.32,
  },
  FWD: {
    goals90: 0.33, assists90: 0.13, saves90: 0, defensive90: 2.2,
    recoveries90: 2.0, yellows90: 0.12, penaltyShare: 0, bonus90: 0.40,
  },
};

/**
 * Pseudo-minutter: hvor mange minutters data prioren svarer til.
 * 450 = fem hele kampe. Under det stoler vi mest på positionen; over det
 * mest på spilleren. Tallet er valgt, så en spiller med ti kampe har cirka
 * to tredjedele egen signal — som er dér, mål/90 holder op med at svinge
 * vildt fra uge til uge.
 */
const PSEUDO_MINUTES = 450;

/** Halveringstid i runder. Form er reel, men svagere end folk tror. */
const ROUND_HALF_LIFE = 8;

export function computeRates(
  player: Player,
  opts: { asOfRound?: number } = {},
): PlayerRates {
  const prior = PRIORS[player.position];
  const rows = opts.asOfRound === undefined
    ? player.history
    : player.history.filter((m) => m.round < opts.asOfRound!);

  const played = rows.filter((m) => m.minutes > 0);

  if (played.length === 0) {
    return {
      playerId: player.id,
      ...prior,
      penaltyShare: penaltyShare(player),
      sampleMinutes: 0,
      confidence: 0,
    };
  }

  const latestRound = Math.max(...played.map((m) => m.round));
  const decay = Math.log(2) / ROUND_HALF_LIFE;

  let minutes = 0;
  let goals = 0;
  let assists = 0;
  let saves = 0;
  let defensive = 0;
  let recoveries = 0;
  let yellows = 0;
  let bonus = 0;
  // Separat tæller: xG findes ikke for alle kampe, så den må ikke dele
  // nævner med mål, ellers fortyndes raten af kampe uden xG-data.
  let xgMinutes = 0;
  let xg = 0;
  let xaMinutes = 0;
  let xa = 0;
  let defMinutes = 0;
  let recMinutes = 0;

  for (const m of played) {
    const w = Math.exp(-decay * Math.max(0, latestRound - m.round));
    const wm = w * m.minutes;
    minutes += wm;
    goals += w * m.goals;
    assists += w * m.assists;
    saves += w * m.saves;
    yellows += w * m.yellowCards;
    bonus += w * m.bonus;

    if (m.xG !== null) { xg += w * m.xG; xgMinutes += wm; }
    if (m.xA !== null) { xa += w * m.xA; xaMinutes += wm; }
    if (m.defensiveActions !== null) { defensive += w * m.defensiveActions; defMinutes += wm; }
    if (m.ballRecoveries !== null) { recoveries += w * m.ballRecoveries; recMinutes += wm; }
  }

  // xG/xA foretrækkes over mål/assists — de er mindre støjende pr. minut.
  const goalSignal = xgMinutes >= 270 ? per90(xg, xgMinutes) : per90(goals, minutes);
  const goalMinutes = xgMinutes >= 270 ? xgMinutes : minutes;
  const assistSignal = xaMinutes >= 270 ? per90(xa, xaMinutes) : per90(assists, minutes);
  const assistMinutes = xaMinutes >= 270 ? xaMinutes : minutes;

  return {
    playerId: player.id,
    goals90: shrink(goalSignal, goalMinutes, prior.goals90),
    assists90: shrink(assistSignal, assistMinutes, prior.assists90),
    saves90: shrink(per90(saves, minutes), minutes, prior.saves90),
    defensive90: defMinutes > 0
      ? shrink(per90(defensive, defMinutes), defMinutes, prior.defensive90)
      : prior.defensive90,
    recoveries90: recMinutes > 0
      ? shrink(per90(recoveries, recMinutes), recMinutes, prior.recoveries90)
      : prior.recoveries90,
    yellows90: shrink(per90(yellows, minutes), minutes, prior.yellows90),
    penaltyShare: penaltyShare(player),
    bonus90: shrink(per90(bonus, minutes), minutes, prior.bonus90),
    sampleMinutes: minutes,
    confidence: minutes / (minutes + PSEUDO_MINUTES),
  };
}

function per90(events: number, minutes: number): number {
  return minutes > 0 ? (events * 90) / minutes : 0;
}

/** Krympning mod prioren, vægtet af hvor mange minutter der ligger bag. */
export function shrink(observed: number, minutes: number, prior: number): number {
  const w = minutes / (minutes + PSEUDO_MINUTES);
  return w * observed + (1 - w) * prior;
}

/**
 * Straffesparksandel. En fast skytte på et hold der får mange straffe er
 * blandt de mest undervurderede kilder til point, fordi straffe ikke ligner
 * evne i xG-tallene — de ligner held, indtil man ser, hvem der tager dem.
 */
function penaltyShare(player: Player): number {
  if (player.penaltyOrder === null) return 0;
  if (player.penaltyOrder === 1) return 0.85;
  if (player.penaltyOrder === 2) return 0.12;
  return 0.03;
}

/** Eksponeret til test og til UI'et, der viser hvilken prior der blev brugt. */
export function priorFor(position: Position) {
  return PRIORS[position];
}
