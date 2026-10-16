/**
 * Minutmodellen. Den vigtigste model i hele systemet.
 *
 * Alle andre modeller regner point pr. 90 minutter. En spiller der ikke
 * spiller, scorer nul — uanset hvor god han er. Størstedelen af de point
 * almindelige managere taber, tabes på spillere der sad på bænken, ikke på
 * spillere der spillede dårligt. Derfor er rotationsrisiko modelleret
 * eksplicit i stedet for at blive gemt i et gennemsnit.
 *
 * Modellen giver tre tal:
 *   pStart   sandsynlighed for at starte
 *   pBench   sandsynlighed for at komme ind fra bænken
 *   minutter forventede minutter betinget af hver af de to
 *
 * Kilderne er spillerens seneste kampe (vægtet, nyeste tungest) korrigeret med
 * spillets egen skadesindikator, som er den eneste friske information vi har
 * om holdnyt.
 */

import type { Player, PlayerMatch } from "@/lib/fantasy/sources/types";

export interface MinutesForecast {
  pStart: number;
  pBench: number;
  /** P(spiller overhovedet) = pStart + pBench. */
  pAppear: number;
  /** Forventede minutter, ubetinget. */
  expected: number;
  /** P(mindst 60 minutter) — grænsen for clean sheet og det andet mødepoint. */
  p60: number;
  /** Hvor meget af estimatet der hviler på data frem for prior. */
  confidence: number;
  /** Menneskelig forklaring, vises i UI. */
  reason: string;
}

/** Hvor mange af de seneste kampe der tæller, og hvor hurtigt de falder af. */
const WINDOW = 6;
const DECAY = 0.82;

/**
 * En startende spiller spiller sjældent alle 90. Tallene er robuste
 * gennemsnit over PL-data og bruges kun som betinget forventning; den
 * observerede historik vinder når den findes.
 */
const DEFAULT_MINUTES_IF_START = 78;
const DEFAULT_MINUTES_IF_BENCH = 18;

export function forecastMinutes(
  player: Player,
  opts: { asOfRound?: number } = {},
): MinutesForecast {
  const history = relevantHistory(player, opts.asOfRound);

  // Spillets egen melding er den friskeste information der findes. 0 betyder
  // udelukket; 25/50/75 er tvivl. null betyder "ingen nyhed", ikke "rask".
  const flag = player.chanceOfPlaying;
  if (flag === 0 || !player.available) {
    return {
      pStart: 0,
      pBench: 0,
      pAppear: 0,
      expected: 0,
      p60: 0,
      confidence: 1,
      reason: player.news?.trim() || "Meldt ude af spillet.",
    };
  }

  if (history.length === 0) {
    // Ingen historik: en nysigneret eller en der ikke har spillet endnu.
    // Vi gætter ikke på en stamplads — vi siger, at vi ikke ved det.
    return {
      pStart: 0.25,
      pBench: 0.3,
      pAppear: 0.55,
      expected: 0.25 * DEFAULT_MINUTES_IF_START + 0.3 * DEFAULT_MINUTES_IF_BENCH,
      p60: 0.25,
      confidence: 0,
      reason: "Ingen kampe i denne sæson — estimatet er en prior, ikke en måling.",
    };
  }

  let wStart = 0;
  let wBench = 0;
  let wNone = 0;
  let minutesIfStart = 0;
  let minutesIfStartW = 0;
  let minutesIfBench = 0;
  let minutesIfBenchW = 0;
  let totalWeight = 0;

  history.forEach((m, i) => {
    const w = Math.pow(DECAY, i);
    totalWeight += w;
    if (m.starts > 0) {
      wStart += w;
      minutesIfStart += w * m.minutes;
      minutesIfStartW += w;
    } else if (m.minutes > 0) {
      wBench += w;
      minutesIfBench += w * m.minutes;
      minutesIfBenchW += w;
    } else {
      wNone += w;
    }
  });

  // Laplace-udjævning mod prioren. Uden den giver tre kampe i træk som starter
  // pStart = 1,0, hvilket aldrig er sandt — der findes altid rotationsrisiko.
  const prior = 1.5;
  const priorStart = 0.45;
  const denom = totalWeight + prior;
  let pStart = (wStart + prior * priorStart) / denom;
  let pBench = (wBench + prior * 0.3) / denom;

  const mStart = minutesIfStartW > 0 ? minutesIfStart / minutesIfStartW : DEFAULT_MINUTES_IF_START;
  const mBench = minutesIfBenchW > 0 ? minutesIfBench / minutesIfBenchW : DEFAULT_MINUTES_IF_BENCH;

  // Tvivl fra spillets side skalerer sandsynligheden for at spille. 75 %
  // betyder ikke "75 % af minutterne", men "75 % chance for at være med".
  let reason = describe(wStart, wBench, wNone, totalWeight);
  if (flag !== null && flag < 100) {
    const scale = flag / 100;
    pStart *= scale;
    pBench *= scale;
    reason = `${player.news?.trim() || `Tvivlsom (${flag} %)`} — ${reason}`;
  }

  const pAppear = Math.min(1, pStart + pBench);
  const expected = pStart * mStart + pBench * mBench;

  // P(60+) er næsten kun startende spillere. En indskiftet spiller når 60
  // minutter kun hvis han kommer ind før det 30. minut, hvilket er sjældent.
  const p60 = pStart * fractionAtLeast60(history) + pBench * 0.04;

  return {
    pStart: clamp01(pStart),
    pBench: clamp01(pBench),
    pAppear: clamp01(pAppear),
    expected,
    p60: clamp01(p60),
    confidence: Math.min(1, totalWeight / 4),
    reason,
  };
}

/** Kampe før den runde vi forudsiger, nyeste først. */
function relevantHistory(player: Player, asOfRound?: number): PlayerMatch[] {
  const rows = asOfRound === undefined
    ? player.history
    : player.history.filter((m) => m.round < asOfRound);
  return [...rows].sort((a, b) => b.round - a.round).slice(0, WINDOW);
}

/** Andel af STARTS der nåede 60 minutter — ikke af alle kampe. */
function fractionAtLeast60(history: PlayerMatch[]): number {
  const starts = history.filter((m) => m.starts > 0);
  if (starts.length === 0) return 0.85;
  const reached = starts.filter((m) => m.minutes >= 60).length;
  // Udjævnet, så to tidlige udskiftninger ikke giver 0.
  return (reached + 1.5) / (starts.length + 2);
}

function describe(wStart: number, wBench: number, wNone: number, total: number): string {
  const pct = (v: number) => Math.round((v / Math.max(1e-9, total)) * 100);
  if (wStart / total > 0.85) return `startet i ${pct(wStart)} % af de seneste kampe`;
  if (wNone / total > 0.5) return `uden for truppen i ${pct(wNone)} % af de seneste kampe`;
  if (wBench > wStart) return `mest indskiftning — startet ${pct(wStart)} %`;
  return `roterer: startet ${pct(wStart)} %, bænket ${pct(wBench + wNone)} %`;
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}
