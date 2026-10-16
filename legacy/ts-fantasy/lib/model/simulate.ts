/**
 * Monte Carlo over pointreglerne.
 *
 * Her ligger forskellen på dette system og alle de projekterede pointtabeller,
 * der findes i forvejen. De giver ét tal: forventede point. Det tal er
 * ubrugeligt til to af de vigtigste beslutninger i spillet.
 *
 *   ANFØRER. Man vælger ikke anfører efter gennemsnit, men efter loft.
 *   To spillere med 6,0 i forventning er ikke ens, hvis den ene har 18 %
 *   chance for et to-cifret træk og den anden har 4 %. Med dobbelte point
 *   er det halen, der afgør runden.
 *
 *   RANGERING. En minirække vindes ikke på flest point i gennemsnit, men på
 *   at slå feltet. Feltet ejer de samme fem spillere som dig. Det du skal
 *   maksimere er P(du slår dem), og den kan kun beregnes af fordelingen.
 *
 * Derfor simulerer vi hver kamp mål for mål, fordeler dem på spillere efter
 * deres rater, kører hændelserne gennem den rigtige pointmotor og gemmer
 * hele fordelingen. Det er dyrere end en formel. Det er også det eneste, der
 * ser outliers.
 */

import { Rng } from "@/lib/fantasy/model/rng";
import { scoreStatLine, type StatLine } from "@/lib/fantasy/rules/score";
import type { ScoringRules, Position } from "@/lib/fantasy/rules/types";
import type { PoissonModel } from "@/lib/fantasy/model/poisson";
import { expectedGoals } from "@/lib/fantasy/model/poisson";
import type { MinutesForecast } from "@/lib/fantasy/model/minutes";
import type { PlayerRates } from "@/lib/fantasy/model/rates";
import type { Fixture, Player } from "@/lib/fantasy/sources/types";

export interface PointsDistribution {
  playerId: number;
  mean: number;
  sd: number;
  /** Hele den empiriske fordeling, indeks = point (negative klippes til 0). */
  histogram: number[];
  p10: number;
  median: number;
  p90: number;
  /** P(≥ 10 point) — "haul". Det der vinder runder. */
  pHaul: number;
  /** P(≤ 2 point) — "blank". Det der taber dem. */
  pBlank: number;
  /** P(spiller ikke) — adskilt fra blank, fordi årsagen er en anden. */
  pNoShow: number;
  /** Bedste observerede udfald i simuleringen. */
  ceiling: number;
}

export interface SimulationInput {
  player: Player;
  rates: PlayerRates;
  minutes: MinutesForecast;
  /** Kampene spilleren har i runden. Dobbeltrunder giver to eller flere. */
  fixtures: { fixture: Fixture; isHome: boolean }[];
}

export interface SimulationContext {
  rules: ScoringRules;
  model: PoissonModel;
  iterations?: number;
  seed?: number;
  /**
   * Holdkammerater pr. hold, så mål kan fordeles konkurrerende: scorer
   * Haaland, scorer Foden ikke det samme mål. Uden det overtæller modellen
   * angrebspoint systematisk for hold med mange gode spillere.
   */
  squadByTeam?: Map<number, SimulationInput[]>;
}

const DEFAULT_ITERATIONS = 4000;

/**
 * Simulerer én spiller. Hvis `squadByTeam` er sat, normaliseres hans andel af
 * holdets mål mod holdkammeraternes — ellers bruges hans rate direkte, hvilket
 * er rigtigt nok for enkeltopslag, men ikke for en hel trup.
 */
export function simulatePlayer(
  input: SimulationInput,
  ctx: SimulationContext,
): PointsDistribution {
  const iterations = ctx.iterations ?? DEFAULT_ITERATIONS;
  const rng = new Rng((ctx.seed ?? 1) ^ (input.player.id * 2654435761));
  const position = input.player.position;

  const samples = new Int32Array(iterations);
  let noShows = 0;

  // Andele beregnes én gang uden for løkken — de afhænger ikke af trækket.
  const shares = input.fixtures.map((f) =>
    involvementShares(input, ctx, f.fixture, f.isHome),
  );

  for (let it = 0; it < iterations; it++) {
    let total = 0;
    let appearedAnywhere = false;

    for (let fi = 0; fi < input.fixtures.length; fi++) {
      const { fixture, isHome } = input.fixtures[fi];
      const share = shares[fi];

      // 1. Spiller han? Og hvor længe?
      const roll = rng.next();
      let minutes: number;
      if (roll < input.minutes.pStart) {
        // Startende: minutter omkring det observerede snit, aldrig over 90.
        minutes = Math.min(90, Math.max(1, Math.round(share.minutesIfStart + rng.normal() * 12)));
      } else if (roll < input.minutes.pStart + input.minutes.pBench) {
        minutes = Math.min(45, Math.max(1, Math.round(share.minutesIfBench + rng.normal() * 8)));
      } else {
        continue; // Ude af truppen i denne kamp.
      }
      appearedAnywhere = true;

      // 2. Hvor mange mål falder der i kampen?
      const teamGoals = rng.poisson(share.teamLambda);
      const oppGoals = rng.poisson(share.oppLambda);

      // 3. Hvor mange af holdets mål er hans? Binomial over hans andel,
      //    skaleret med den andel af kampen han faktisk var på banen.
      const onPitch = minutes / 90;
      let goals = 0;
      let assists = 0;
      for (let g = 0; g < teamGoals; g++) {
        if (!rng.chance(onPitch)) continue;
        if (rng.chance(share.goalShare)) goals++;
        // Et mål kan ikke både scores og assisteres af samme spiller.
        else if (rng.chance(share.assistShare)) assists++;
      }

      // 4. Straffespark oven i det åbne spil. Faste skytter får en separat
      //    chance, fordi straffe ikke fordeles efter angrebsandel.
      if (share.penaltyShare > 0 && rng.chance(share.penaltyRate * onPitch)) {
        if (rng.chance(0.79)) goals++;
        else {
          // Brændt straffe: minus point, og i UCL taber man også for at
          // have vundet den forgæves — nej, den tæller stadig som vundet.
          const line = emptyLine(minutes);
          line.penaltiesMissed = 1;
          total += scoreStatLine(ctx.rules, position, line) - baseAppearance(ctx.rules, minutes);
        }
      }

      const stat: StatLine = {
        minutes,
        goals,
        goalsOutsideBox: ctx.rules.goalOutsideBox > 0 ? countOutsideBox(goals, position, rng) : 0,
        assists,
        cleanSheet: oppGoals === 0,
        goalsConceded: oppGoals,
        saves: position === "GK" ? sampleSaves(share, oppGoals, onPitch, rng) : 0,
        penaltiesSaved: position === "GK" && rng.chance(0.012 * onPitch) ? 1 : 0,
        penaltiesMissed: 0,
        penaltiesWon: ctx.rules.penaltyWon > 0 && rng.chance(0.03 * onPitch) ? 1 : 0,
        penaltiesConceded: ctx.rules.penaltyConceded < 0 && rng.chance(0.015 * onPitch) ? 1 : 0,
        yellowCards: rng.chance((share.yellows90 * minutes) / 90) ? 1 : 0,
        redCards: rng.chance(0.004 * onPitch) ? 1 : 0,
        ownGoals: rng.chance(0.006 * onPitch) ? 1 : 0,
        ballRecoveries: samplePer90(share.recoveries90, minutes, rng),
        defensiveActions: samplePer90(share.defensive90, minutes, rng),
        bonus: ctx.rules.bonus ? sampleBonus(share, goals, assists, oppGoals === 0, rng) : 0,
        playerOfTheMatch:
          ctx.rules.playerOfTheMatch > 0 ? sampleMotm(goals, assists, minutes, rng) : false,
      };

      total += scoreStatLine(ctx.rules, position, stat);
    }

    if (!appearedAnywhere) noShows++;
    samples[it] = total;
  }

  return summarize(input.player.id, samples, noShows / iterations);
}

interface Shares {
  teamLambda: number;
  oppLambda: number;
  goalShare: number;
  assistShare: number;
  penaltyShare: number;
  penaltyRate: number;
  minutesIfStart: number;
  minutesIfBench: number;
  yellows90: number;
  recoveries90: number;
  defensive90: number;
  saves90: number;
  bonus90: number;
}

/**
 * Oversætter rater til andele af HOLDETS mål. En spiller med 0,5 mål/90 på et
 * hold der scorer 2,0 pr. kamp tager 25 % af målene — ikke 0,5 mål uanset
 * hvor godt holdet spiller. Det er koblingen der gør, at en god kamp mod et
 * dårligt forsvar hæver alle på holdet samtidig, som det også sker i
 * virkeligheden.
 */
function involvementShares(
  input: SimulationInput,
  ctx: SimulationContext,
  fixture: Fixture,
  isHome: boolean,
): Shares {
  const teamId = isHome ? fixture.homeTeam : fixture.awayTeam;
  const oppId = isHome ? fixture.awayTeam : fixture.homeTeam;
  const xg = expectedGoals(ctx.model, fixture.homeTeam, fixture.awayTeam);
  const teamLambda = isHome ? xg.home : xg.away;
  const oppLambda = isHome ? xg.away : xg.home;

  const r = input.rates;

  // Holdets samlede rate, hvis vi kender holdkammeraterne. Andelen
  // normaliseres, så summen af holdets spillere ikke overstiger 1.
  let goalShare = safeShare(r.goals90, teamLambda);
  let assistShare = safeShare(r.assists90, teamLambda);

  const teammates = ctx.squadByTeam?.get(teamId);
  if (teammates && teammates.length > 1) {
    const totalGoals = teammates.reduce((s, t) => s + t.rates.goals90 * t.minutes.pStart, 0);
    const totalAssists = teammates.reduce((s, t) => s + t.rates.assists90 * t.minutes.pStart, 0);
    if (totalGoals > 0) {
      goalShare = (r.goals90 * input.minutes.pStart) / totalGoals;
    }
    if (totalAssists > 0) {
      // Kun cirka tre fjerdedele af mål har en registreret assist.
      assistShare = ((r.assists90 * input.minutes.pStart) / totalAssists) * 0.75;
    }
  }

  return {
    teamLambda,
    oppLambda,
    goalShare: clamp01(goalShare),
    assistShare: clamp01(assistShare),
    penaltyShare: r.penaltyShare,
    // Cirka 0,23 straffespark pr. hold pr. kamp i topfodbold.
    penaltyRate: r.penaltyShare * 0.23,
    minutesIfStart: input.minutes.expected > 0 ? Math.max(45, Math.min(90, input.minutes.expected / Math.max(0.2, input.minutes.pStart))) : 78,
    minutesIfBench: 18,
    yellows90: r.yellows90,
    recoveries90: r.recoveries90,
    defensive90: r.defensive90,
    saves90: r.saves90,
    bonus90: r.bonus90,
  };
}

function safeShare(rate: number, lambda: number): number {
  if (lambda <= 0) return 0;
  return Math.min(0.9, rate / lambda);
}

/** Redninger hænger sammen med hvor meget modstanderen skyder, ikke kun mål. */
function sampleSaves(share: Shares, oppGoals: number, onPitch: number, rng: Rng): number {
  const base = share.saves90 * onPitch;
  // Flere mål imod betyder som regel også flere skud — og flere redninger.
  const pressure = 1 + 0.25 * (oppGoals - share.oppLambda);
  return Math.max(0, rng.poisson(Math.max(0, base * pressure)));
}

function samplePer90(rate: number, minutes: number, rng: Rng): number {
  return rng.poisson(Math.max(0, (rate * minutes) / 90));
}

/**
 * Bonuspoint. BPS er en lang vægtet sum vi ikke kan genskabe uden hele
 * kampens data, så vi modellerer resultatet i stedet for mekanikken:
 * sandsynligheden for at ende i top 3 stiger kraftigt med mål og assists.
 * Den observerede bonus90 bærer alt det, mål og assists ikke forklarer
 * (redninger, tacklinger, afleveringer).
 */
function sampleBonus(share: Shares, goals: number, assists: number, cleanSheet: boolean, rng: Rng): number {
  let score = share.bonus90 * 0.8;
  score += goals * 1.5;
  score += assists * 0.8;
  if (cleanSheet) score += 0.35;
  const p3 = clamp01(score * 0.22);
  const p2 = clamp01(score * 0.18);
  const p1 = clamp01(score * 0.16);
  const roll = rng.next();
  if (roll < p3) return 3;
  if (roll < p3 + p2) return 2;
  if (roll < p3 + p2 + p1) return 1;
  return 0;
}

/** Player of the Match i UCL: næsten altid en målscorer. */
function sampleMotm(goals: number, assists: number, minutes: number, rng: Rng): boolean {
  if (minutes < 45) return false;
  const p = 0.01 + goals * 0.16 + assists * 0.07;
  return rng.chance(Math.min(0.6, p));
}

/** Cirka en femtedel af mål falder uden for feltet. */
function countOutsideBox(goals: number, position: Position, rng: Rng): number {
  if (goals === 0) return 0;
  const p = position === "MID" || position === "DEF" ? 0.28 : 0.13;
  let n = 0;
  for (let i = 0; i < goals; i++) if (rng.chance(p)) n++;
  return n;
}

function emptyLine(minutes: number): StatLine {
  return {
    minutes, goals: 0, assists: 0, cleanSheet: false, goalsConceded: 0, saves: 0,
    penaltiesSaved: 0, penaltiesMissed: 0, yellowCards: 0, redCards: 0, ownGoals: 0,
  };
}

function baseAppearance(rules: ScoringRules, minutes: number): number {
  return rules.appearance + (minutes >= 60 ? rules.appearance60 : 0);
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

/** Empiriske kvantiler og halesandsynligheder fra trækkene. */
export function summarize(
  playerId: number,
  samples: Int32Array,
  pNoShow: number,
): PointsDistribution {
  const n = samples.length;
  const sorted = Int32Array.from(samples).sort();

  let sum = 0;
  let hauls = 0;
  let blanks = 0;
  let max = Number.NEGATIVE_INFINITY;
  const histogram: number[] = [];

  for (let i = 0; i < n; i++) {
    const v = samples[i];
    sum += v;
    if (v >= 10) hauls++;
    if (v <= 2) blanks++;
    if (v > max) max = v;
    const bucket = Math.max(0, v);
    histogram[bucket] = (histogram[bucket] ?? 0) + 1;
  }
  for (let i = 0; i < histogram.length; i++) histogram[i] = (histogram[i] ?? 0) / n;

  const mean = sum / n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (samples[i] - mean) ** 2;
  variance /= Math.max(1, n - 1);

  const q = (p: number) => sorted[Math.min(n - 1, Math.max(0, Math.floor(p * n)))];

  return {
    playerId,
    mean,
    sd: Math.sqrt(variance),
    histogram,
    p10: q(0.1),
    median: q(0.5),
    p90: q(0.9),
    pHaul: hauls / n,
    pBlank: blanks / n,
    pNoShow,
    ceiling: Number.isFinite(max) ? max : 0,
  };
}

/** Kører hele truppen og returnerer fordelinger pr. spiller. */
export function simulateSquad(
  inputs: SimulationInput[],
  ctx: SimulationContext,
): Map<number, PointsDistribution> {
  // Hold-gruppering laves én gang, så andele normaliseres mod holdkammerater.
  const byTeam = new Map<number, SimulationInput[]>();
  for (const i of inputs) {
    const list = byTeam.get(i.player.team) ?? [];
    list.push(i);
    byTeam.set(i.player.team, list);
  }
  const withTeams: SimulationContext = { ...ctx, squadByTeam: ctx.squadByTeam ?? byTeam };

  const out = new Map<number, PointsDistribution>();
  for (const input of inputs) {
    out.set(input.player.id, simulatePlayer(input, withTeams));
  }
  return out;
}
