/**
 * Orkestratoren: fra rå data til ét svar på "hvad gør jeg inden deadline".
 *
 * Rækkefølgen er bevidst. Hvert trin bruger det forrige, og intet trin må
 * gætte, når det forrige ikke kunne levere. Hvis holdmodellen ikke har kampe
 * nok, siger rådet det — det finder ikke på en sværhedsgrad.
 */

import { fitPoissonModel, type PoissonModel } from "@/lib/fantasy/model/poisson";
import { forecastMinutes } from "@/lib/fantasy/model/minutes";
import { computeRates } from "@/lib/fantasy/model/rates";
import { simulateSquad, type PointsDistribution, type SimulationInput } from "@/lib/fantasy/model/simulate";
import { bestLineup, type LineupCandidate } from "@/lib/fantasy/optimize/lineup";
import { chooseCaptain, type CaptainAdvice } from "@/lib/fantasy/optimize/captain";
import { planTransfers, type TransferPlan, type HeldPlayer, type TransferCandidate } from "@/lib/fantasy/optimize/transfers";
import { evaluateChips, type ChipEvaluation } from "@/lib/fantasy/optimize/chips";
import { teamDistribution, probabilityAtLeast, type OwnershipContext } from "@/lib/fantasy/optimize/rank";
import type { GameRules } from "@/lib/fantasy/rules/types";
import type { GameSnapshot, ManagerSquad, Player } from "@/lib/fantasy/sources/types";

export interface AdviceOptions {
  snapshot: GameSnapshot;
  squad: ManagerSquad;
  rules: GameRules;
  /** 0 = spil sikkert, 1 = jagt rang. Default 0,5. */
  aggression?: number;
  /** Runder gevinsten regnes over. */
  horizon?: number;
  iterations?: number;
  seed?: number;
  /** Chips der stadig er i behold. */
  chipsAvailable?: string[];
  roundsRemaining?: number;
}

export interface PlayerAdviceRow {
  playerId: number;
  name: string;
  position: string;
  team: string;
  price: number;
  ownership: number;
  dist: PointsDistribution;
  minutesReason: string;
  /** Advarsel der skal vises med ord, ikke kun farve. */
  flag: string | null;
}

export interface DeadlineAdvice {
  game: string;
  round: number;
  deadline: string | null;
  generatedAt: string;
  /** Advarsler om datagrundlaget. Tom liste = modellen stod på rigtige tal. */
  caveats: string[];
  modelSummary: {
    matchesFitted: number;
    homeAdvantage: number;
    rho: number;
    baseRate: number;
    /** Hold sorteret efter angrebsstyrke — vises som modellens egen FDR. */
    attackRanking: { team: string; attack: number; defence: number }[];
  };
  squad: PlayerAdviceRow[];
  lineup: { starters: number[]; bench: number[]; formation: string; expected: number };
  captain: CaptainAdvice;
  transfers: TransferPlan[];
  chips: ChipEvaluation[];
  /** Hele holdets fordeling, så man kan se loft og gulv for runden. */
  projection: {
    expected: number;
    p10: number;
    p90: number;
    pAbove60: number;
    pAbove80: number;
  };
}

export function buildAdvice(opts: AdviceOptions): DeadlineAdvice {
  const {
    snapshot, squad, rules,
    aggression = 0.5, horizon = 3,
    iterations = 4000, seed = 20262027,
    chipsAvailable = squad.chipsAvailable,
    roundsRemaining = 30,
  } = opts;

  const caveats: string[] = [];
  const playersById = new Map(snapshot.players.map((p) => [p.id, p]));
  const teamsById = new Map(snapshot.teams.map((t) => [t.id, t]));
  const round = snapshot.nextRound;

  // --- 1. Holdmodellen ------------------------------------------------
  const model = fitPoissonModel(snapshot.teams, snapshot.fixtures, {
    asOf: new Date(snapshot.fetchedAt),
  });

  if (model.matches === 0) {
    caveats.push(
      "Holdmodellen har ingen spillede kampe at fitte på. Alle hold behandles som lige stærke, " +
        "så modstanderjusteringen er slået fra — rådene hviler kun på spillernes egne rater.",
    );
  } else if (model.matches < snapshot.teams.length * 2) {
    caveats.push(
      `Holdmodellen er fittet på kun ${model.matches} kampe. Angrebs- og forsvarsstyrker ` +
        "er stadig tæt på gennemsnittet — stol mere på dem om et par runder.",
    );
  }

  // --- 2. Kampe i runden ----------------------------------------------
  const roundFixtures = snapshot.fixtures.filter((f) => f.round === round);
  if (roundFixtures.length === 0) {
    caveats.push(
      `Der findes ingen kampe for runde ${round} i data. Uden kampprogram kan hverken ` +
        "clean sheets eller modstanderstyrke beregnes — rådene er ikke pålidelige.",
    );
  }

  const fixturesByTeam = new Map<number, { fixture: typeof roundFixtures[number]; isHome: boolean }[]>();
  for (const f of roundFixtures) {
    push(fixturesByTeam, f.homeTeam, { fixture: f, isHome: true });
    push(fixturesByTeam, f.awayTeam, { fixture: f, isHome: false });
  }

  // --- 3. Ejerskab -----------------------------------------------------
  const ownership: OwnershipContext = {
    ownership: new Map(snapshot.players.map((p) => [p.id, p.selectedByPercent / 100])),
    // Anførerandele udstilles ikke af spillene. Vi tilnærmer dem: bindet
    // følger i høj grad ejerskab blandt de dyreste angribere. Det er en
    // tilnærmelse, og den er markeret som sådan.
    captaincy: approximateCaptaincy(snapshot.players),
    aggression,
  };
  caveats.push(
    "Anførerandele er estimeret ud fra ejerskab og pris — spillene udstiller dem ikke. " +
      "Differential-rådene er derfor retningsgivende, ikke målte.",
  );

  // --- 4. Simulér truppen + markedet ----------------------------------
  const squadIds = squad.picks.map((p) => p.playerId);
  const missing = squadIds.filter((id) => !playersById.has(id));
  if (missing.length > 0) {
    caveats.push(
      `${missing.length} spiller(e) i truppen findes ikke i spillets data (id ${missing.join(", ")}). ` +
        "De er udeladt af beregningen.",
    );
  }

  const toInput = (p: Player): SimulationInput => ({
    player: p,
    rates: computeRates(p, { asOfRound: round }),
    minutes: forecastMinutes(p, { asOfRound: round }),
    fixtures: fixturesByTeam.get(p.team) ?? [],
  });

  // Markedet beskæres: kun spillere der realistisk kan købes. Uden det
  // simulerer vi 700 spillere for at finde 20 der betyder noget.
  const market = snapshot.players.filter(
    (p) => p.available && (p.selectedByPercent > 0.4 || p.totalPoints > 8 || squadIds.includes(p.id)),
  );

  const inputs = [...new Set([...squadIds.map((id) => playersById.get(id)), ...market])]
    .filter((p): p is Player => Boolean(p))
    .map(toInput);

  const dists = simulateSquad(inputs, { rules: rules.scoring, model, iterations, seed });

  // --- 5. Startopstilling ---------------------------------------------
  const squadPlayers = squadIds
    .map((id) => playersById.get(id))
    .filter((p): p is Player => Boolean(p));

  const candidates: LineupCandidate[] = squadPlayers.map((p) => {
    const d = dists.get(p.id)!;
    return { playerId: p.id, position: p.position, mean: d.mean, pAppear: 1 - d.pNoShow };
  });
  const lineup = bestLineup(candidates, rules.squad);

  // --- 6. Anfører ------------------------------------------------------
  const starterMeta = lineup.starters.map((id) => ({
    playerId: id,
    name: playersById.get(id)?.name ?? String(id),
  }));
  const captain = chooseCaptain(starterMeta, dists, ownership, rules.squad.captainMultiplier);

  // --- 7. Skift --------------------------------------------------------
  const held: HeldPlayer[] = squadPlayers.map((p) => {
    const pick = squad.picks.find((x) => x.playerId === p.id);
    return {
      player: p,
      dist: dists.get(p.id)!,
      // Salgsprisen er den spillet oplyser; findes den ikke, bruges dagsprisen.
      sellPrice: pick?.sellingPrice ?? p.price,
    };
  });

  const marketCandidates: TransferCandidate[] = market
    .filter((p) => dists.has(p.id))
    .map((p) => ({ player: p, dist: dists.get(p.id)!, buyPrice: p.price }));

  const transfers = planTransfers({
    rules,
    held,
    market: marketCandidates,
    bank: squad.bank,
    freeTransfers: squad.freeTransfers,
    ownership,
    horizon,
  });

  // --- 8. Chips --------------------------------------------------------
  const blanks = squadPlayers.filter((p) => (fixturesByTeam.get(p.team) ?? []).length === 0).length;
  const doubles = squadPlayers.filter((p) => (fixturesByTeam.get(p.team) ?? []).length > 1).length;

  const chips = evaluateChips({
    rules,
    roundsRemaining,
    dists,
    starters: lineup.starters,
    bench: lineup.bench,
    captainId: captain.pick.playerId,
    available: chipsAvailable,
    blanks,
    doubles,
  });

  // --- 9. Holdets samlede fordeling ------------------------------------
  const starterDists = lineup.starters
    .map((id) => dists.get(id))
    .filter((d): d is PointsDistribution => Boolean(d));
  const teamHist = teamDistribution(starterDists, captain.pick.playerId, rules.squad.captainMultiplier);

  const expected = starterDists.reduce((s, d) => s + d.mean, 0)
    + (dists.get(captain.pick.playerId)?.mean ?? 0) * (rules.squad.captainMultiplier - 1);

  return {
    game: rules.scoring.game,
    round,
    deadline: snapshot.deadline,
    generatedAt: new Date().toISOString(),
    caveats,
    modelSummary: {
      matchesFitted: model.matches,
      homeAdvantage: model.homeAdvantage,
      rho: model.rho,
      baseRate: model.baseRate,
      attackRanking: rankTeams(model, teamsById),
    },
    squad: squadPlayers.map((p) => {
      const d = dists.get(p.id)!;
      const m = forecastMinutes(p, { asOfRound: round });
      return {
        playerId: p.id,
        name: p.name,
        position: p.position,
        team: teamsById.get(p.team)?.shortName ?? String(p.team),
        price: p.price,
        ownership: p.selectedByPercent,
        dist: d,
        minutesReason: m.reason,
        flag: flagFor(p, d),
      };
    }),
    lineup: {
      starters: lineup.starters,
      bench: lineup.bench,
      formation: lineup.formation,
      expected: lineup.expected,
    },
    captain,
    transfers,
    chips,
    projection: {
      expected,
      p10: quantile(teamHist, 0.1),
      p90: quantile(teamHist, 0.9),
      pAbove60: probabilityAtLeast(teamHist, 60),
      pAbove80: probabilityAtLeast(teamHist, 80),
    },
  };
}

function push<K, V>(map: Map<K, V[]>, key: K, value: V): void {
  const list = map.get(key) ?? [];
  list.push(value);
  map.set(key, list);
}

/**
 * Tilnærmet anførerfordeling. Bindet koncentrerer sig om få spillere:
 * i praksis tager de to-tre dyreste angribere med højest ejerskab omkring
 * 70 % af alle bind. Vi fordeler efter ejerskab gange pris i anden, hvilket
 * rammer den koncentration rimeligt.
 */
function approximateCaptaincy(players: Player[]): Map<number, number> {
  const weights = players.map((p) => ({
    id: p.id,
    w: p.available ? (p.selectedByPercent / 100) * Math.pow(p.price, 2) : 0,
  }));
  const total = weights.reduce((s, x) => s + x.w, 0);
  const out = new Map<number, number>();
  if (total <= 0) return out;
  for (const { id, w } of weights) out.set(id, w / total);
  return out;
}

function rankTeams(
  model: PoissonModel,
  teams: Map<number, { shortName: string }>,
): { team: string; attack: number; defence: number }[] {
  return [...model.strengths.values()]
    .map((s) => ({
      team: teams.get(s.teamId)?.shortName ?? String(s.teamId),
      attack: s.attack,
      defence: s.defence,
    }))
    .sort((a, b) => b.attack - a.attack);
}

/** Advarsler i ord. Farve alene er ikke tilgængeligt og ikke nok. */
function flagFor(p: Player, d: PointsDistribution): string | null {
  if (!p.available) return p.news?.trim() || "Meldt ude";
  if (p.chanceOfPlaying !== null && p.chanceOfPlaying < 100) {
    return `${p.chanceOfPlaying} % spilchance${p.news ? ` — ${p.news}` : ""}`;
  }
  if (d.pNoShow > 0.35) {
    return `${Math.round(d.pNoShow * 100)} % risiko for slet ikke at spille`;
  }
  return null;
}

function quantile(histogram: number[], q: number): number {
  let cumulative = 0;
  for (let i = 0; i < histogram.length; i++) {
    cumulative += histogram[i] ?? 0;
    if (cumulative >= q) return i;
  }
  return histogram.length - 1;
}
