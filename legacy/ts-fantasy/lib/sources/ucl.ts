/**
 * UEFA Champions League Fantasy. Feedet ligger under gaming.uefa.com og er
 * ikke dokumenteret; formen herunder er udledt af de kald spillets egen
 * frontend laver.
 *
 *   GET /services/api/Feed/players?gamedayId={n}&language=en
 *   GET /services/api/Gameplay/user/{guid}/team?matchdayId={n}   (kræver session)
 *
 * UEFA bruger "skill" 1-4 til position og "value" til pris i millioner euro.
 * Feltnavnene er korte og uregelmæssige, så al oversættelse sker her.
 *
 * Vigtigt: truppen bag `Gameplay/user` kræver login. Vi henter den ikke —
 * appen tager truppen fra manuel indtastning eller fra Firestore i stedet,
 * så der aldrig ligger UEFA-credentials i miljøet.
 */

import type { GameSnapshot, Player, Team } from "@/lib/fantasy/sources/types";
import type { Position } from "@/lib/fantasy/rules/types";
import type { FetchOptions } from "@/lib/fantasy/sources/fpl";

export const UCL_BASE = "https://gaming.uefa.com/en/uclfantasy";

const POSITION_BY_SKILL: Record<number, Position> = {
  1: "GK",
  2: "DEF",
  3: "MID",
  4: "FWD",
};

/** Ét element i UEFA's spillerfeed. Kun de felter vi faktisk læser. */
interface RawUclPlayer {
  id: string | number;
  /** Visningsnavn, fx "H. Kane". */
  pDName: string;
  /** Fulde navn, når feedet har det. */
  pFName?: string;
  pLName?: string;
  /** Klubkode, fx "BAY". */
  cCode: string;
  /** Klubbens id i feedet. */
  teamId?: string | number;
  /** 1=GK, 2=DEF, 3=MID, 4=FWD. */
  skill: number;
  /** Pris i millioner. */
  value: number | string;
  /** 1 = tilgængelig. */
  isActive: number;
  totPts: number | string;
  avgPlayerPts?: number | string;
  lastGdPoints?: number | string;
  /** Ejerskabsprocent, hvis feedet leverer den. */
  selPer?: number | string;
  /** Minutter spillet i turneringen. */
  minsPlyd?: number | string;
  goals?: number | string;
  assists?: number | string;
  cleanSheet?: number | string;
  balRec?: number | string;
  /** Skadesstatus, hvis feedet har den. */
  injStatus?: string;
  trained?: number;
}

interface RawUclFeed {
  data: { value: { playerList: RawUclPlayer[] } };
}

function n(v: number | string | null | undefined, fallback = 0): number {
  if (v === null || v === undefined) return fallback;
  const parsed = typeof v === "number" ? v : Number.parseFloat(v);
  return Number.isFinite(parsed) ? parsed : fallback;
}

async function getJson<T>(url: string, opts: FetchOptions = {}): Promise<T> {
  const impl = opts.fetchImpl ?? fetch;
  const res = await impl(url, {
    headers: {
      Accept: "application/json",
      // UEFA afviser kald uden Referer fra deres eget domæne.
      Referer: `${UCL_BASE}/services/index.html`,
      "User-Agent": "jgk-fantasy-assistant/1.0",
    },
    ...(opts.revalidateSeconds !== undefined
      ? { next: { revalidate: opts.revalidateSeconds } }
      : {}),
  } as RequestInit);

  if (!res.ok) {
    throw new Error(
      `UEFA-feedet svarede ${res.status} på ${url}. ` +
        "Feedet er udokumenteret og ændrer form mellem sæsoner — tjek at " +
        "gamedayId er rigtigt, og at udgående trafik til gaming.uefa.com er tilladt.",
    );
  }
  return (await res.json()) as T;
}

/**
 * Klubberne findes ikke som eget feed; de udledes af spillerlisten. Det er
 * nok til modellen, som kun skal bruge en stabil identitet pr. hold.
 */
function teamsFromPlayers(raw: RawUclPlayer[]): { teams: Team[]; idByCode: Map<string, number> } {
  const codes = [...new Set(raw.map((p) => p.cCode))].sort();
  const idByCode = new Map<string, number>();
  const teams = codes.map((code, i) => {
    idByCode.set(code, i + 1);
    return { id: i + 1, name: code, shortName: code };
  });
  return { teams, idByCode };
}

export function normalizeUclPlayers(raw: RawUclPlayer[]): {
  players: Player[];
  teams: Team[];
} {
  const { teams, idByCode } = teamsFromPlayers(raw);

  const players = raw.map((p): Player => {
    const full = [p.pFName, p.pLName].filter(Boolean).join(" ").trim();
    return {
      id: n(p.id),
      name: p.pDName,
      fullName: full || p.pDName,
      team: idByCode.get(p.cCode) ?? 0,
      position: POSITION_BY_SKILL[p.skill] ?? "MID",
      price: n(p.value),
      selectedByPercent: n(p.selPer),
      totalPoints: n(p.totPts),
      minutes: n(p.minsPlyd),
      // UEFA har ingen procentvis skadesindikator — kun aktiv/inaktiv.
      chanceOfPlaying: p.isActive === 1 ? null : 0,
      news: p.injStatus ?? "",
      available: p.isActive === 1,
      penaltyOrder: null,
      cornersOrder: null,
      freekicksOrder: null,
      // Feedet giver sæsonsummer, ikke kamp-for-kamp. Historikken fyldes
      // derfor fra vores egne snapshots over tid — se store.ts.
      history: [],
    };
  });

  return { players, teams };
}

export async function fetchUclSnapshot(
  matchday: number,
  opts: FetchOptions = {},
): Promise<GameSnapshot> {
  const feed = await getJson<RawUclFeed>(
    `${UCL_BASE}/services/api/Feed/players?gamedayId=${matchday}&language=en`,
    opts,
  );

  const list = feed?.data?.value?.playerList;
  if (!Array.isArray(list)) {
    throw new Error(
      "UEFA-feedet havde ikke formen data.value.playerList — feedet er ændret. " +
        "Ret normalizeUclPlayers i lib/fantasy/sources/ucl.ts.",
    );
  }

  const { players, teams } = normalizeUclPlayers(list);

  return {
    game: "UCL",
    season: "2026/27",
    currentRound: matchday,
    nextRound: matchday,
    // Feedet oplyser ikke deadline; den sættes af kalenderen i store.ts.
    deadline: null,
    teams,
    players,
    // Kampprogrammet kommer ikke med spillerfeedet. Uden det falder
    // holdmodellen tilbage på sæsonsummer i stedet for modstanderjustering.
    fixtures: [],
    fetchedAt: new Date().toISOString(),
  };
}
