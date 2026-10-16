/**
 * FPL's officielle API. Ingen nøgle, ingen dokumentation, ingen garantier —
 * derfor validerer vi formen ved indgangen i stedet for at stole på den.
 *
 * Endepunkter vi bruger:
 *   /api/bootstrap-static/            spillere, hold, runder, regler
 *   /api/fixtures/                    alle kampe med sværhedsgrad
 *   /api/element-summary/{id}/        én spillers kamphistorik
 *   /api/entry/{id}/                  managerens profil
 *   /api/entry/{id}/event/{gw}/picks/ managerens trup i en runde
 *
 * Bemærk: FPL angiver priser i tiendedele (125 = 12,5 mio.). Vi normaliserer
 * til millioner ét sted — her — så resten af koden regner i spillets egne tal.
 */

import type {
  Fixture,
  GameSnapshot,
  ManagerSquad,
  Player,
  PlayerMatch,
  Team,
} from "@/lib/fantasy/sources/types";
import type { Position } from "@/lib/fantasy/rules/types";

export const FPL_BASE = "https://fantasy.premierleague.com/api";

/** FPL's element_type → vores position. */
const POSITION_BY_ELEMENT_TYPE: Record<number, Position> = {
  1: "GK",
  2: "DEF",
  3: "MID",
  4: "FWD",
};

export interface FetchOptions {
  /** Injiceres i test og lader et cron-job genbruge en cachet fetch. */
  fetchImpl?: typeof fetch;
  /** Hvor længe svar må caches af Next's data-cache. 0 = altid frisk. */
  revalidateSeconds?: number;
}

async function getJson<T>(url: string, opts: FetchOptions = {}): Promise<T> {
  const impl = opts.fetchImpl ?? fetch;
  const res = await impl(url, {
    headers: {
      // FPL afviser klienter uden User-Agent med 403.
      "User-Agent": "jgk-fantasy-assistant/1.0 (+https://github.com/m9chaelangelo)",
      Accept: "application/json",
    },
    ...(opts.revalidateSeconds !== undefined
      ? { next: { revalidate: opts.revalidateSeconds } }
      : {}),
  } as RequestInit);

  if (!res.ok) {
    throw new Error(
      `FPL API svarede ${res.status} på ${url}. ` +
        (res.status === 403
          ? "403 betyder som regel at kaldet kom uden User-Agent, eller at udgående trafik er blokeret."
          : "Prøv igen — FPL lukker af omkring deadline."),
    );
  }
  return (await res.json()) as T;
}

/* --- Rå former, kun det vi faktisk læser ------------------------------- */

interface RawBootstrap {
  events: {
    id: number;
    deadline_time: string;
    is_current: boolean;
    is_next: boolean;
    finished: boolean;
  }[];
  teams: {
    id: number;
    name: string;
    short_name: string;
    strength_attack_home: number;
    strength_attack_away: number;
    strength_defence_home: number;
    strength_defence_away: number;
  }[];
  elements: {
    id: number;
    web_name: string;
    first_name: string;
    second_name: string;
    team: number;
    element_type: number;
    now_cost: number;
    selected_by_percent: string;
    total_points: number;
    minutes: number;
    chance_of_playing_next_round: number | null;
    news: string;
    status: string;
    penalties_order: number | null;
    corners_and_indirect_freekicks_order: number | null;
    direct_freekicks_order: number | null;
  }[];
}

interface RawFixture {
  id: number;
  event: number | null;
  team_h: number;
  team_a: number;
  kickoff_time: string | null;
  finished: boolean;
  team_h_score: number | null;
  team_a_score: number | null;
  team_h_difficulty: number;
  team_a_difficulty: number;
}

interface RawHistoryRow {
  round: number;
  fixture: number;
  opponent_team: number;
  was_home: boolean;
  minutes: number;
  starts: number;
  goals_scored: number;
  assists: number;
  clean_sheets: number;
  goals_conceded: number;
  saves: number;
  yellow_cards: number;
  red_cards: number;
  own_goals: number;
  penalties_missed: number;
  penalties_saved: number;
  bonus: number;
  bps: number;
  total_points: number;
  expected_goals?: string;
  expected_assists?: string;
  clearances_blocks_interceptions?: number;
  tackles?: number;
  recoveries?: number;
  defensive_contribution?: number;
}

/* --- Normalisering ------------------------------------------------------ */

function toTeam(t: RawBootstrap["teams"][number]): Team {
  return {
    id: t.id,
    name: t.name,
    shortName: t.short_name,
    strengthAttackHome: t.strength_attack_home,
    strengthAttackAway: t.strength_attack_away,
    strengthDefenceHome: t.strength_defence_home,
    strengthDefenceAway: t.strength_defence_away,
  };
}

function toFixture(f: RawFixture): Fixture {
  return {
    id: f.id,
    // Kampe uden runde er udsatte og hører ingen steder hjemme endnu.
    round: f.event ?? 0,
    homeTeam: f.team_h,
    awayTeam: f.team_a,
    kickoff: f.kickoff_time,
    finished: f.finished,
    homeGoals: f.team_h_score,
    awayGoals: f.team_a_score,
    homeDifficulty: f.team_h_difficulty,
    awayDifficulty: f.team_a_difficulty,
  };
}

function num(v: string | number | null | undefined): number | null {
  if (v === null || v === undefined) return null;
  const n = typeof v === "number" ? v : Number.parseFloat(v);
  return Number.isFinite(n) ? n : null;
}

export function toPlayerMatch(row: RawHistoryRow, teamId: number): PlayerMatch {
  // FPL udstiller defensive_contribution som et færdigt tal i nyere sæsoner.
  // Findes det ikke, bygger vi det af delene — CBI + tacklinger, plus
  // generobringer for midt/angreb (som `defensiveActions` selv rummer, fordi
  // pointmotoren kender tærsklen pr. position).
  const cbi = row.clearances_blocks_interceptions ?? 0;
  const tackles = row.tackles ?? 0;
  const recoveries = row.recoveries ?? 0;
  const derived = cbi + tackles;

  return {
    round: row.round,
    fixture: row.fixture,
    team: teamId,
    opponent: row.opponent_team,
    wasHome: row.was_home,
    minutes: row.minutes,
    starts: row.starts,
    goals: row.goals_scored,
    assists: row.assists,
    cleanSheet: row.clean_sheets > 0,
    goalsConceded: row.goals_conceded,
    saves: row.saves,
    yellowCards: row.yellow_cards,
    redCards: row.red_cards,
    ownGoals: row.own_goals,
    penaltiesMissed: row.penalties_missed,
    penaltiesSaved: row.penalties_saved,
    bonus: row.bonus,
    bps: row.bps,
    xG: num(row.expected_goals),
    xA: num(row.expected_assists),
    defensiveActions: row.defensive_contribution ?? (derived > 0 ? derived : null),
    ballRecoveries: recoveries > 0 ? recoveries : null,
    totalPoints: row.total_points,
  };
}

function toPlayer(e: RawBootstrap["elements"][number]): Player {
  return {
    id: e.id,
    name: e.web_name,
    fullName: `${e.first_name} ${e.second_name}`.trim(),
    team: e.team,
    position: POSITION_BY_ELEMENT_TYPE[e.element_type] ?? "MID",
    // Tiendedele → millioner. Gøres her og kun her.
    price: e.now_cost / 10,
    selectedByPercent: num(e.selected_by_percent) ?? 0,
    totalPoints: e.total_points,
    minutes: e.minutes,
    chanceOfPlaying: e.chance_of_playing_next_round,
    news: e.news ?? "",
    // status: a=available, d=doubtful, i=injured, s=suspended, u=unavailable
    available: e.status === "a" || e.status === "d",
    penaltyOrder: e.penalties_order,
    cornersOrder: e.corners_and_indirect_freekicks_order,
    freekicksOrder: e.direct_freekicks_order,
    history: [],
  };
}

/* --- Offentligt API ----------------------------------------------------- */

/**
 * Henter hele spillets tilstand. Kamphistorik pr. spiller ligger bag et kald
 * pr. spiller (~700 kald), så den hentes ikke her — `attachHistories` gør det
 * for den delmængde modellen faktisk skal bruge.
 */
export async function fetchFplSnapshot(opts: FetchOptions = {}): Promise<GameSnapshot> {
  const [boot, fixtures] = await Promise.all([
    getJson<RawBootstrap>(`${FPL_BASE}/bootstrap-static/`, opts),
    getJson<RawFixture[]>(`${FPL_BASE}/fixtures/`, opts),
  ]);

  const current = boot.events.find((e) => e.is_current);
  const next = boot.events.find((e) => e.is_next);

  return {
    game: "FPL",
    season: seasonLabel(boot.events[0]?.deadline_time),
    currentRound: current?.id ?? 0,
    nextRound: next?.id ?? (current?.id ?? 0) + 1,
    deadline: next?.deadline_time ?? null,
    teams: boot.teams.map(toTeam),
    players: boot.elements.map(toPlayer),
    fixtures: fixtures.map(toFixture),
    fetchedAt: new Date().toISOString(),
  };
}

/** "2026/27" udledt af sæsonens første deadline. */
export function seasonLabel(firstDeadline: string | undefined): string {
  if (!firstDeadline) return "ukendt";
  const d = new Date(firstDeadline);
  if (Number.isNaN(d.getTime())) return "ukendt";
  // Sæsonen starter i august, så kalenderåret er første halvdel.
  const start = d.getUTCMonth() >= 6 ? d.getUTCFullYear() : d.getUTCFullYear() - 1;
  return `${start}/${String((start + 1) % 100).padStart(2, "0")}`;
}

/**
 * Henter kamphistorik for udvalgte spillere. Kaldene serialiseres i små
 * portioner: FPL rate-limiter aggressivt, og 700 samtidige kald får os
 * blokeret midt i en deadline.
 */
export async function attachHistories(
  players: Player[],
  ids: number[],
  opts: FetchOptions & { concurrency?: number } = {},
): Promise<void> {
  const byId = new Map(players.map((p) => [p.id, p]));
  const concurrency = opts.concurrency ?? 6;
  const queue = [...new Set(ids)];

  async function worker(): Promise<void> {
    for (;;) {
      const id = queue.shift();
      if (id === undefined) return;
      const player = byId.get(id);
      if (!player) continue;
      try {
        const summary = await getJson<{ history: RawHistoryRow[] }>(
          `${FPL_BASE}/element-summary/${id}/`,
          opts,
        );
        player.history = summary.history.map((row) => toPlayerMatch(row, player.team));
      } catch {
        // En enkelt spiller uden historik må ikke vælte hele opdateringen —
        // modellen falder tilbage på positions-prioren for netop ham.
        player.history = [];
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, queue.length) }, worker));
}

/** Managerens trup i en given runde. */
export async function fetchFplSquad(
  entryId: string,
  round: number,
  opts: FetchOptions = {},
): Promise<ManagerSquad> {
  interface RawPicks {
    picks: {
      element: number;
      position: number;
      is_captain: boolean;
      is_vice_captain: boolean;
      selling_price?: number;
      purchase_price?: number;
    }[];
    entry_history: { bank: number; value: number; event_transfers: number };
    active_chip: string | null;
  }

  const data = await getJson<RawPicks>(
    `${FPL_BASE}/entry/${entryId}/event/${round}/picks/`,
    opts,
  );

  return {
    game: "FPL",
    entryId,
    round,
    picks: data.picks.map((p) => ({
      playerId: p.element,
      slot: p.position,
      isCaptain: p.is_captain,
      isViceCaptain: p.is_vice_captain,
      purchasePrice: p.purchase_price !== undefined ? p.purchase_price / 10 : undefined,
      sellingPrice: p.selling_price !== undefined ? p.selling_price / 10 : undefined,
    })),
    bank: data.entry_history.bank / 10,
    squadValue: data.entry_history.value / 10,
    // FPL udstiller ikke antallet af gratis skift i dette endepunkt; det
    // udledes af transferhistorikken eller sættes manuelt i UI'et.
    freeTransfers: 1,
    chipsAvailable: [],
    chipActive: data.active_chip,
  };
}
