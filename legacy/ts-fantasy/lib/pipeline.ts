/**
 * Hele kæden i én funktion: hent, modellér, rådgiv, gem.
 *
 * Bruges af både cron-jobbet og den manuelle opdatering i UI'et, så de to
 * aldrig kan komme til at gøre noget forskelligt.
 */

import { fetchFplSnapshot, attachHistories, fetchFplSquad } from "@/lib/fantasy/sources/fpl";
import { fetchUclSnapshot } from "@/lib/fantasy/sources/ucl";
import { FPL_RULES } from "@/lib/fantasy/rules/fpl";
import { UCL_RULES } from "@/lib/fantasy/rules/ucl";
import { buildAdvice, type DeadlineAdvice } from "@/lib/fantasy/advice";
import {
  FPL_SQUAD_GW5, FPL_CHIPS_AVAILABLE, FPL_FLAGS_GW5,
  UCL_SQUAD_MD2, UCL_CHIPS_AVAILABLE,
  matchSeedToPlayers, type SeedPick,
} from "@/lib/fantasy/data/squads";
import type { GameSnapshot, ManagerSquad } from "@/lib/fantasy/sources/types";

export interface RunOptions {
  game: "FPL" | "UCL";
  /** FPL manager-id. Er det sat, hentes truppen fra spillet i stedet for seed. */
  entryId?: string;
  aggression?: number;
  horizon?: number;
  iterations?: number;
  /** UCL: hvilken spilledag. */
  matchday?: number;
  freeTransfers?: number;
  bank?: number;
}

export interface RunResult {
  advice: DeadlineAdvice;
  snapshot: GameSnapshot;
  squad: ManagerSquad;
  warnings: string[];
}

export async function runPipeline(opts: RunOptions): Promise<RunResult> {
  const warnings: string[] = [];

  if (opts.game === "FPL") {
    const snapshot = await fetchFplSnapshot({ revalidateSeconds: 300 });

    const teamCodeById = new Map(snapshot.teams.map((t) => [t.id, t.shortName]));
    const { squad, warnings: squadWarnings } = await resolveFplSquad(
      snapshot, teamCodeById, opts,
    );
    warnings.push(...squadWarnings);

    // Historik hentes kun for dem der betyder noget: truppen plus det
    // marked vi faktisk overvejer. Ellers er det 700 kald mod et API der
    // rate-limiter omkring deadline.
    const squadIds = squad.picks.map((p) => p.playerId);
    const marketIds = snapshot.players
      .filter((p) => p.available && (p.selectedByPercent > 1.5 || p.totalPoints > 15))
      .map((p) => p.id);
    await attachHistories(snapshot.players, [...new Set([...squadIds, ...marketIds])], {
      revalidateSeconds: 900,
    });

    applySeedFlags(snapshot, FPL_FLAGS_GW5, warnings);

    const advice = buildAdvice({
      snapshot,
      squad,
      rules: FPL_RULES,
      aggression: opts.aggression,
      horizon: opts.horizon,
      iterations: opts.iterations,
      chipsAvailable: squad.chipsAvailable,
      roundsRemaining: Math.max(1, 38 - snapshot.nextRound),
    });

    return { advice, snapshot, squad, warnings };
  }

  // --- UCL ------------------------------------------------------------
  const matchday = opts.matchday ?? 2;
  const snapshot = await fetchUclSnapshot(matchday, { revalidateSeconds: 900 });
  const teamCodeById = new Map(snapshot.teams.map((t) => [t.id, t.shortName]));

  const { matched, unmatched } = matchSeedToPlayers(UCL_SQUAD_MD2, snapshot.players, teamCodeById);
  if (unmatched.length > 0) {
    warnings.push(
      `Kunne ikke matche ${unmatched.length} spiller(e) i UCL-truppen mod UEFA's feed: ` +
        `${unmatched.map((u) => u.name).join(", ")}. Rådene dækker kun de ${matched.length} der blev fundet.`,
    );
  }
  if (snapshot.fixtures.length === 0) {
    warnings.push(
      "UEFA's spillerfeed indeholder ikke kampprogrammet, så clean sheet- og " +
        "modstanderberegningen er slået fra for UCL. Rådene hviler på spillernes " +
        "egne rater alene og er svagere end FPL-siden.",
    );
  }

  const squad: ManagerSquad = {
    game: "UCL",
    entryId: null,
    round: matchday,
    picks: matched.map(({ pick, player }) => ({
      playerId: player.id,
      slot: pick.slot,
      isCaptain: pick.isCaptain ?? false,
      isViceCaptain: false,
    })),
    bank: opts.bank ?? 0,
    squadValue: 100,
    freeTransfers: opts.freeTransfers ?? UCL_RULES.transfers.freePerRound,
    chipsAvailable: UCL_CHIPS_AVAILABLE,
    chipActive: null,
  };

  const advice = buildAdvice({
    snapshot,
    squad,
    rules: UCL_RULES,
    aggression: opts.aggression,
    horizon: opts.horizon,
    iterations: opts.iterations,
    chipsAvailable: UCL_CHIPS_AVAILABLE,
    // 8 spilledage i ligafasen plus knockout.
    roundsRemaining: Math.max(1, 8 - matchday),
  });

  return { advice, snapshot, squad, warnings };
}

async function resolveFplSquad(
  snapshot: GameSnapshot,
  teamCodeById: Map<number, string>,
  opts: RunOptions,
): Promise<{ squad: ManagerSquad; warnings: string[] }> {
  const warnings: string[] = [];

  // Har vi et manager-id, er spillets egen trup altid sandheden.
  if (opts.entryId) {
    try {
      const squad = await fetchFplSquad(opts.entryId, snapshot.currentRound, {
        revalidateSeconds: 60,
      });
      return {
        squad: {
          ...squad,
          round: snapshot.nextRound,
          freeTransfers: opts.freeTransfers ?? squad.freeTransfers,
          chipsAvailable: FPL_CHIPS_AVAILABLE,
        },
        warnings,
      };
    } catch (err) {
      warnings.push(
        `Kunne ikke hente truppen fra FPL for manager ${opts.entryId} ` +
          `(${err instanceof Error ? err.message : "ukendt fejl"}). Bruger den indtastede trup i stedet.`,
      );
    }
  }

  const { matched, unmatched } = matchSeedToPlayers(FPL_SQUAD_GW5, snapshot.players, teamCodeById);
  if (unmatched.length > 0) {
    warnings.push(
      `Kunne ikke matche ${unmatched.length} spiller(e) i FPL-truppen: ` +
        `${unmatched.map((u) => `${u.name} (${u.club})`).join(", ")}. ` +
        "Ret navnet i lib/fantasy/data/squads.ts, eller sæt FPL_ENTRY_ID så truppen hentes fra spillet.",
    );
  }

  return {
    squad: {
      game: "FPL",
      entryId: null,
      round: snapshot.nextRound,
      picks: matched.map(({ pick, player }) => ({
        playerId: player.id,
        slot: pick.slot,
        isCaptain: pick.isCaptain ?? false,
        isViceCaptain: pick.isViceCaptain ?? false,
      })),
      bank: opts.bank ?? 0,
      squadValue: 100,
      freeTransfers: opts.freeTransfers ?? 1,
      chipsAvailable: FPL_CHIPS_AVAILABLE,
      chipActive: null,
    },
    warnings,
  };
}

/**
 * Lægger de manuelt aflæste flag oven på spillets data.
 *
 * Grunden: FPL's `chance_of_playing_next_round` opdateres i ryk, og et flag
 * kan stå i appen før det står i API'et. Ser vi et flag i appen som API'et
 * ikke har, vinder det strengeste af de to — og vi siger det højt.
 */
function applySeedFlags(
  snapshot: GameSnapshot,
  flags: Record<string, { level: "out" | "doubt"; note: string }>,
  warnings: string[],
): void {
  for (const [name, flag] of Object.entries(flags)) {
    const player = snapshot.players.find((p) => p.name === name);
    if (!player) continue;

    const apiChance = player.chanceOfPlaying;
    const seedChance = flag.level === "out" ? 0 : 50;

    if (apiChance === null || seedChance < apiChance) {
      player.chanceOfPlaying = seedChance;
      if (flag.level === "out") player.available = false;
      player.news = player.news || flag.note;
      warnings.push(
        `${name}: manuelt aflæst flag er strengere end FPL's API (${apiChance ?? "intet flag"} %) — ` +
          `bruger ${seedChance} %. ${flag.note}`,
      );
    }
  }
}
