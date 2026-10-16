/**
 * Holdstyrke-modellen: Dixon-Coles.
 *
 * Hvorfor ikke bare bruge FPL's egen sværhedsgrad (FDR 1-5): den er sat i
 * hånden før sæsonen og opdateres ikke efter resultater. Et oprykkerhold der
 * viser sig at være godt, bliver ved med at hedde "2" hele sæsonen. Det er
 * præcis der, markedet tager fejl, og præcis der, en model tjener point.
 *
 * Modellen estimerer for hvert hold et angreb (alpha) og et forsvar (beta),
 * plus en fælles hjemmebanefordel. Forventede mål i en kamp:
 *
 *   lambda_hjemme = exp(alpha_h - beta_a + hjemmefordel)
 *   lambda_ude    = exp(alpha_a - beta_h)
 *
 * To ting adskiller Dixon-Coles fra naiv Poisson, og begge betyder noget:
 *
 *  1. TIDSVÆGTNING. En kamp fra sidste sæson siger mindre end en fra i går.
 *     Vægten falder eksponentielt med alderen (xi). Uden den reagerer
 *     modellen alt for langsomt på en ny træner eller en skadet nøglespiller.
 *
 *  2. LAVSCORE-KORREKTION (tau). Uafhængige Poisson-fordelinger undervurderer
 *     systematisk 0-0 og 1-1 og overvurderer 1-0 og 0-1. rho retter de fire
 *     felter. Det lyder som en detalje, men clean sheets er 4 point i begge
 *     spil, så hele forsvarsværdien hænger i netop de felter.
 */

import type { Fixture, Team } from "@/lib/fantasy/sources/types";

export interface TeamStrength {
  teamId: number;
  attack: number;
  defence: number;
  /** Hvor mange vægtede kampe estimatet hviler på. Lav værdi = lav tillid. */
  weight: number;
}

export interface PoissonModel {
  strengths: Map<number, TeamStrength>;
  homeAdvantage: number;
  /** Lavscore-korrektionen. 0 = ren uafhængig Poisson. */
  rho: number;
  /** Gennemsnitligt antal mål pr. hold pr. kamp i datasættet. */
  baseRate: number;
  /** Antal kampe modellen er fittet på. */
  matches: number;
  logLikelihood: number;
}

export interface FitOptions {
  /** Halveringstid i dage for tidsvægtningen. */
  halfLifeDays?: number;
  /** Referencedato — kampe efter den ignoreres (bruges til backtest). */
  asOf?: Date;
  iterations?: number;
  learningRate?: number;
  /** L2 mod 0. Holder hold med få kampe tæt på middelvejen. */
  regularization?: number;
}

/**
 * Dixon-Coles' tau. Retter kun de fire felter hvor begge hold har 0 eller 1.
 * rho > 0 flytter masse mod uafgjort; rho < 0 væk fra det.
 */
export function tau(homeGoals: number, awayGoals: number, lh: number, la: number, rho: number): number {
  if (homeGoals === 0 && awayGoals === 0) return 1 - lh * la * rho;
  if (homeGoals === 0 && awayGoals === 1) return 1 + lh * rho;
  if (homeGoals === 1 && awayGoals === 0) return 1 + la * rho;
  if (homeGoals === 1 && awayGoals === 1) return 1 - rho;
  return 1;
}

function poissonPmf(k: number, lambda: number): number {
  if (lambda <= 0) return k === 0 ? 1 : 0;
  let logP = -lambda + k * Math.log(lambda);
  for (let i = 2; i <= k; i++) logP -= Math.log(i);
  return Math.exp(logP);
}

/**
 * Fitter modellen med gradient descent på den tidsvægtede log-likelihood.
 *
 * Gradienterne for alpha og beta er analytiske (Poisson-delen), mens rho
 * og hjemmefordelen opdateres numerisk — de er to skalarer, så en central
 * differens er billigere end at udlede dem i hånden og lettere at holde rigtig.
 */
export function fitPoissonModel(
  teams: Team[],
  fixtures: Fixture[],
  opts: FitOptions = {},
): PoissonModel {
  const halfLife = opts.halfLifeDays ?? 120;
  const asOf = opts.asOf ?? new Date();
  const iterations = opts.iterations ?? 400;
  const lr = opts.learningRate ?? 0.05;
  const lambda2 = opts.regularization ?? 0.02;

  const played = fixtures.filter(
    (f) =>
      f.finished &&
      f.homeGoals !== null &&
      f.awayGoals !== null &&
      (!f.kickoff || new Date(f.kickoff) <= asOf),
  );

  const attack = new Map<number, number>();
  const defence = new Map<number, number>();
  const weightSum = new Map<number, number>();
  for (const t of teams) {
    attack.set(t.id, 0);
    defence.set(t.id, 0);
    weightSum.set(t.id, 0);
  }

  // Tidsvægt: halveres for hver `halfLife` dage tilbage i tiden.
  const decay = Math.log(2) / halfLife;
  const weights = played.map((f) => {
    if (!f.kickoff) return 1;
    const ageDays = (asOf.getTime() - new Date(f.kickoff).getTime()) / 86_400_000;
    return Math.exp(-decay * Math.max(0, ageDays));
  });

  let totalGoals = 0;
  let totalWeight = 0;
  played.forEach((f, i) => {
    const w = weights[i];
    totalGoals += w * ((f.homeGoals ?? 0) + (f.awayGoals ?? 0));
    totalWeight += 2 * w;
    weightSum.set(f.homeTeam, (weightSum.get(f.homeTeam) ?? 0) + w);
    weightSum.set(f.awayTeam, (weightSum.get(f.awayTeam) ?? 0) + w);
  });

  // Uden kampe kan intet estimeres. Vi gætter ikke — vi returnerer en flad
  // model med baseRate fra ligagennemsnittet og weight 0, så alt der bruger
  // den kan se, at den ikke ved noget.
  const baseRate = totalWeight > 0 ? totalGoals / totalWeight : 1.35;
  const logBase = Math.log(Math.max(0.05, baseRate));

  let homeAdv = 0.25;
  let rho = -0.05;

  if (played.length === 0) {
    return {
      strengths: new Map(
        teams.map((t) => [t.id, { teamId: t.id, attack: 0, defence: 0, weight: 0 }]),
      ),
      homeAdvantage: homeAdv,
      rho: 0,
      baseRate,
      matches: 0,
      logLikelihood: 0,
    };
  }

  const lambdasFor = (f: Fixture, ha: number): [number, number] => {
    const ah = attack.get(f.homeTeam) ?? 0;
    const aa = attack.get(f.awayTeam) ?? 0;
    const dh = defence.get(f.homeTeam) ?? 0;
    const da = defence.get(f.awayTeam) ?? 0;
    return [
      Math.exp(logBase + ah - da + ha),
      Math.exp(logBase + aa - dh),
    ];
  };

  const logLik = (ha: number, r: number): number => {
    let ll = 0;
    played.forEach((f, i) => {
      const [lh, la] = lambdasFor(f, ha);
      const hg = f.homeGoals ?? 0;
      const ag = f.awayGoals ?? 0;
      const t = Math.max(1e-9, tau(hg, ag, lh, la, r));
      const p = Math.max(1e-12, poissonPmf(hg, lh) * poissonPmf(ag, la) * t);
      ll += weights[i] * Math.log(p);
    });
    return ll;
  };

  for (let iter = 0; iter < iterations; iter++) {
    const gAttack = new Map<number, number>();
    const gDefence = new Map<number, number>();

    played.forEach((f, i) => {
      const w = weights[i];
      const [lh, la] = lambdasFor(f, homeAdv);
      const hg = f.homeGoals ?? 0;
      const ag = f.awayGoals ?? 0;

      // d/dalpha_h log P = w * (mål_h - lambda_h); symmetrisk for de øvrige.
      // tau afhænger også af lambda, men kun i fire felter og svagt — den del
      // overlades til de numeriske opdateringer af rho/hjemmefordel nedenfor.
      gAttack.set(f.homeTeam, (gAttack.get(f.homeTeam) ?? 0) + w * (hg - lh));
      gDefence.set(f.awayTeam, (gDefence.get(f.awayTeam) ?? 0) - w * (hg - lh));
      gAttack.set(f.awayTeam, (gAttack.get(f.awayTeam) ?? 0) + w * (ag - la));
      gDefence.set(f.homeTeam, (gDefence.get(f.homeTeam) ?? 0) - w * (ag - la));
    });

    for (const t of teams) {
      const w = Math.max(1, weightSum.get(t.id) ?? 0);
      const a = attack.get(t.id) ?? 0;
      const d = defence.get(t.id) ?? 0;
      // Gradienten normaliseres med holdets egen vægt, så et hold med få
      // kampe ikke tager lige så store skridt som et med mange.
      attack.set(t.id, a + (lr * ((gAttack.get(t.id) ?? 0) / w - lambda2 * a)));
      defence.set(t.id, d + (lr * ((gDefence.get(t.id) ?? 0) / w - lambda2 * d)));
    }

    // Identifikation: alpha og beta er kun bestemt op til en konstant, fordi
    // (alpha+c, beta+c) giver samme lambdaer. Uden centrering driver de mod
    // uendelig sammen, og tallene bliver umulige at sammenligne mellem hold.
    center(attack, teams);
    center(defence, teams);

    // Hjemmefordel og rho hver 10. iteration — de er billige at flytte, men
    // dyre at evaluere, fordi hver evaluering koster en fuld likelihood.
    if (iter % 10 === 9) {
      homeAdv = lineSearch(homeAdv, 0.02, (v) => logLik(v, rho), 0, 0.8);
      rho = lineSearch(rho, 0.02, (v) => logLik(homeAdv, v), -0.2, 0.2);
    }
  }

  const strengths = new Map<number, TeamStrength>(
    teams.map((t) => [
      t.id,
      {
        teamId: t.id,
        attack: attack.get(t.id) ?? 0,
        defence: defence.get(t.id) ?? 0,
        weight: weightSum.get(t.id) ?? 0,
      },
    ]),
  );

  return {
    strengths,
    homeAdvantage: homeAdv,
    rho,
    baseRate,
    matches: played.length,
    logLikelihood: logLik(homeAdv, rho),
  };
}

function center(m: Map<number, number>, teams: Team[]): void {
  let sum = 0;
  for (const t of teams) sum += m.get(t.id) ?? 0;
  const mean = sum / Math.max(1, teams.length);
  for (const t of teams) m.set(t.id, (m.get(t.id) ?? 0) - mean);
}

/** Ét skridt op eller ned, hvis det hæver likelihooden. Klippet til [lo,hi]. */
function lineSearch(
  current: number,
  step: number,
  f: (v: number) => number,
  lo: number,
  hi: number,
): number {
  const base = f(current);
  const up = Math.min(hi, current + step);
  const down = Math.max(lo, current - step);
  const fUp = f(up);
  const fDown = f(down);
  if (fUp > base && fUp >= fDown) return up;
  if (fDown > base) return down;
  return current;
}

/** Forventede mål for begge hold i én kamp. */
export function expectedGoals(
  model: PoissonModel,
  homeTeam: number,
  awayTeam: number,
): { home: number; away: number } {
  const logBase = Math.log(Math.max(0.05, model.baseRate));
  const ah = model.strengths.get(homeTeam)?.attack ?? 0;
  const dh = model.strengths.get(homeTeam)?.defence ?? 0;
  const aa = model.strengths.get(awayTeam)?.attack ?? 0;
  const da = model.strengths.get(awayTeam)?.defence ?? 0;
  return {
    home: Math.exp(logBase + ah - da + model.homeAdvantage),
    away: Math.exp(logBase + aa - dh),
  };
}

/**
 * Hele resultatfordelingen som en matrix P[i][j] = P(hjemme i, ude j).
 * Herfra kommer clean sheet-sandsynligheder, som er hele forsvarsværdien.
 */
export function scoreMatrix(
  model: PoissonModel,
  homeTeam: number,
  awayTeam: number,
  maxGoals = 8,
): number[][] {
  const { home: lh, away: la } = expectedGoals(model, homeTeam, awayTeam);
  const matrix: number[][] = [];
  let total = 0;

  for (let i = 0; i <= maxGoals; i++) {
    matrix[i] = [];
    for (let j = 0; j <= maxGoals; j++) {
      const p = poissonPmf(i, lh) * poissonPmf(j, la) * tau(i, j, lh, la, model.rho);
      matrix[i][j] = Math.max(0, p);
      total += matrix[i][j];
    }
  }

  // Normalisér: tau og afkortningen ved maxGoals gør, at summen ikke er 1.
  if (total > 0) {
    for (let i = 0; i <= maxGoals; i++) {
      for (let j = 0; j <= maxGoals; j++) matrix[i][j] /= total;
    }
  }
  return matrix;
}

/** P(clean sheet) for hvert hold, læst af resultatmatricen. */
export function cleanSheetProbabilities(
  model: PoissonModel,
  homeTeam: number,
  awayTeam: number,
): { home: number; away: number } {
  const m = scoreMatrix(model, homeTeam, awayTeam);
  let home = 0;
  let away = 0;
  for (let i = 0; i < m.length; i++) {
    // Hjemmeholdet holder nullet, når udeholdet scorer 0.
    home += m[i][0];
    away += m[0][i];
  }
  return { home, away };
}
