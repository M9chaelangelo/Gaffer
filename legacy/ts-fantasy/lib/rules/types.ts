/**
 * Regelmotoren for begge spil. Pointtabellen er data, ikke kode, fordi FPL og
 * UEFA justerer den mellem sæsoner — 2026/27 fjernede fx straffen for at blive
 * tacklet af med bolden, og DefCon-tærsklerne blev kalibreret om.
 *
 * Hver værdi har en kilde i docs/fantasy-rules.md. Værdier vi ikke har kunnet
 * bekræfte fra en autoritativ kilde er markeret `unverified` i konfigurationen,
 * så de kan rettes ét sted i stedet for at være spredt ud i modellen.
 */

/** Positionerne deles af begge spil. UEFA kalder dem "skill" 1-4. */
export const POSITIONS = ["GK", "DEF", "MID", "FWD"] as const;
export type Position = (typeof POSITIONS)[number];

/** Point pr. position — én tabel pr. hændelse der afhænger af positionen. */
export type ByPosition<T> = Record<Position, T>;

export interface ScoringRules {
  /** Spillets navn, til visning og til at vælge regelsæt. */
  game: "FPL" | "UCL";
  season: string;

  /** Point for at komme på banen overhovedet. */
  appearance: number;
  /** Ekstra point ved 60+ minutter (lægges oven i `appearance`). */
  appearance60: number;

  goal: ByPosition<number>;
  /** UCL giver +1 for mål uden for feltet. FPL har ingen sådan regel (0). */
  goalOutsideBox: number;
  assist: number;
  cleanSheet: ByPosition<number>;

  /** Antal redninger pr. point (3 = 1 point pr. 3 redninger). */
  savesPerPoint: number;
  penaltySave: number;
  penaltyMiss: number;
  /** UCL belønner at vinde et straffespark; FPL gør ikke (0). */
  penaltyWon: number;
  /** UCL straffer at begå et; FPL gør ikke (0). */
  penaltyConceded: number;

  /** Antal mål lukket ind pr. minuspoint, kun GK/DEF. */
  goalsConcededPerPoint: number;
  yellowCard: number;
  redCard: number;
  ownGoal: number;

  /** UCL: 1 point pr. N boldgenerobringer. FPL tæller dem i DefCon i stedet. */
  ballRecoveriesPerPoint: number | null;
  /** UCL: Player of the Match. FPL har bonuspoint i stedet. */
  playerOfTheMatch: number;

  /**
   * FPL's defensive bidrag. Forsvarere tæller CBIT (clearances, blocks,
   * interceptions, tackles); midt/angreb tæller CBIRT — samme plus
   * boldgenerobringer. Præmien er fast og loftet er den samme værdi, så
   * 20 handlinger giver ikke det dobbelte.
   */
  defensiveContribution: {
    threshold: ByPosition<number | null>;
    points: number;
  } | null;

  /** FPL's bonuspoint fra BPS-rangeringen: 3/2/1 til de tre bedste. */
  bonus: readonly number[] | null;
}

export interface SquadRules {
  size: number;
  /** Præcist antal pr. position i truppen (ikke i startopstillingen). */
  composition: ByPosition<number>;
  /** Startopstillingens størrelse. */
  startingSize: number;
  /** Mindste antal pr. position i startopstillingen. */
  minStarting: ByPosition<number>;
  maxStarting: ByPosition<number>;
  budget: number;
  maxPerClub: number;
  /** Multiplikator på anførerens point. */
  captainMultiplier: number;
  /** FPL har viceanfører; UCL lader dig flytte bindet undervejs i stedet. */
  hasViceCaptain: boolean;
}

export interface TransferRules {
  /** Gratis skift pr. runde efter den første. */
  freePerRound: number;
  /** Hvor mange gratis skift der kan spares op. */
  maxBanked: number;
  /** Fradrag pr. skift ud over de gratis. */
  costPerExtra: number;
}

export interface ChipRule {
  id: string;
  name: string;
  /** Hvad chippen gør, i én sætning — vises i rådgivningen. */
  effect: string;
  /** Hvor mange gange chippen kan spilles i sæsonen. */
  uses: number;
}

export interface GameRules {
  scoring: ScoringRules;
  squad: SquadRules;
  transfers: TransferRules;
  chips: readonly ChipRule[];
  /** Værdier vi ikke har kunnet bekræfte fra spillets egen dokumentation. */
  unverified: readonly string[];
}
