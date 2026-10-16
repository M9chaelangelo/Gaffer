/**
 * Den normaliserede datamodel. FPL og UEFA leverer vidt forskellige felter;
 * alt herunder er oversat, så modellen og optimeringen kun kender ét skema og
 * ikke skal vide hvilket spil den regner på.
 */

import type { Position } from "@/lib/fantasy/rules/types";

export type GameId = "FPL" | "UCL";

export interface Team {
  id: number;
  name: string;
  shortName: string;
  /**
   * Holdstyrke som spillet selv angiver den, hvis den findes (FPL har
   * strength_attack_home osv.). Modellen bruger den kun som prior — de
   * faktiske styrker estimeres fra resultater.
   */
  strengthAttackHome?: number;
  strengthAttackAway?: number;
  strengthDefenceHome?: number;
  strengthDefenceAway?: number;
}

export interface Fixture {
  id: number;
  /** Runde: FPL-gameweek eller UCL-matchday. */
  round: number;
  homeTeam: number;
  awayTeam: number;
  kickoff: string | null;
  finished: boolean;
  homeGoals: number | null;
  awayGoals: number | null;
  /** Spillets egen sværhedsgrad 1-5, hvis den findes. */
  homeDifficulty?: number;
  awayDifficulty?: number;
}

/** Hvad en spiller faktisk gjorde i én kamp. Grundlaget for alle rater. */
export interface PlayerMatch {
  round: number;
  fixture: number;
  team: number;
  opponent: number;
  wasHome: boolean;
  minutes: number;
  starts: number;
  goals: number;
  assists: number;
  cleanSheet: boolean;
  goalsConceded: number;
  saves: number;
  yellowCards: number;
  redCards: number;
  ownGoals: number;
  penaltiesMissed: number;
  penaltiesSaved: number;
  bonus: number;
  bps: number;
  /** xG/xA hvis kilden leverer dem — FPL gør siden 2022/23. */
  xG: number | null;
  xA: number | null;
  /** Defensive handlinger; FPL leverer dem som clearances_blocks_interceptions m.fl. */
  defensiveActions: number | null;
  ballRecoveries: number | null;
  totalPoints: number;
}

export interface Player {
  id: number;
  name: string;
  /** Fulde navn når kilden har det — bruges til at matche på tværs af spil. */
  fullName: string;
  team: number;
  position: Position;
  /** Pris i spillets egen enhed (FPL: millioner, altså 12.5 ikke 125). */
  price: number;
  /** Ejerskabsprocent. Kernen i den rang-bevidste optimering. */
  selectedByPercent: number;
  totalPoints: number;
  minutes: number;
  /**
   * Spillets egen skadesindikator: 0-100, eller null når der ikke er nyt.
   * FPL sætter chance_of_playing_next_round; UEFA har kun isActive.
   */
  chanceOfPlaying: number | null;
  /** Nyhedstekst om skade/karantæne, hvis kilden har den. */
  news: string;
  available: boolean;
  /** Fast dødboldstager? FPL leverer rangeringer 1-3; null hvis ukendt. */
  penaltyOrder: number | null;
  cornersOrder: number | null;
  freekicksOrder: number | null;
  history: PlayerMatch[];
}

/** Ét snapshot af hele spillets tilstand på ét tidspunkt. */
export interface GameSnapshot {
  game: GameId;
  season: string;
  /** Runden der er åben for skift lige nu. */
  currentRound: number;
  nextRound: number;
  /** ISO-tidspunkt for næste deadline. */
  deadline: string | null;
  teams: Team[];
  players: Player[];
  fixtures: Fixture[];
  /** Hvornår data blev hentet — vises i UI så gamle tal ikke ligner friske. */
  fetchedAt: string;
}

/** Managerens egen trup, som spillet ser den. */
export interface ManagerSquad {
  game: GameId;
  entryId: string | null;
  round: number;
  picks: SquadPick[];
  bank: number;
  squadValue: number;
  freeTransfers: number;
  chipsAvailable: string[];
  chipActive: string | null;
}

export interface SquadPick {
  playerId: number;
  /** 1-11 starter, 12-15 bænk. Rækkefølgen på bænken er substitutionsordenen. */
  slot: number;
  isCaptain: boolean;
  isViceCaptain: boolean;
  /** Prisen spilleren blev købt til — afgør fortjeneste ved salg. */
  purchasePrice?: number;
  sellingPrice?: number;
}
