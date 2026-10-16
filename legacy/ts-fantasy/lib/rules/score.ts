/**
 * Pointmotoren: én statlinje ind, point ud. Den er ren og deterministisk, så
 * Monte Carlo-simuleringen kan kalde den millioner af gange, og så en test kan
 * fodre den med en kendt kamp og kræve det rigtige facit.
 *
 * Hvorfor ikke bare bruge spillets egne `total_points`: de findes først EFTER
 * kampen. Simuleringen skal score hændelser der endnu ikke er sket.
 */

import type { Position } from "@/lib/fantasy/rules/types";
import type { ScoringRules } from "@/lib/fantasy/rules/types";

/** En spillers hændelser i én kamp — alt pointmotoren har brug for at vide. */
export interface StatLine {
  minutes: number;
  goals: number;
  /** Delmængde af `goals` scoret uden for feltet. Kun UCL giver ekstra for dem. */
  goalsOutsideBox?: number;
  assists: number;
  cleanSheet: boolean;
  goalsConceded: number;
  saves: number;
  penaltiesSaved: number;
  penaltiesMissed: number;
  penaltiesWon?: number;
  penaltiesConceded?: number;
  yellowCards: number;
  redCards: number;
  ownGoals: number;
  ballRecoveries?: number;
  /** FPL: CBIT for forsvarere, CBIRT for midt/angreb. */
  defensiveActions?: number;
  /** FPL-bonus 0-3. Simuleringen trækker den fra en BPS-model. */
  bonus?: number;
  playerOfTheMatch?: boolean;
}

/**
 * Rent clean sheet-kriterium: FPL og UEFA kræver begge, at spilleren har
 * været på banen i mindst 60 minutter. En indskiftning i det 70. minut på et
 * 0-0 giver altså ikke clean sheet-point.
 */
const CLEAN_SHEET_MIN_MINUTES = 60;

export function scoreStatLine(rules: ScoringRules, position: Position, s: StatLine): number {
  if (s.minutes <= 0) return 0;

  let points = rules.appearance;
  if (s.minutes >= 60) points += rules.appearance60;

  points += s.goals * rules.goal[position];
  if (rules.goalOutsideBox > 0 && s.goalsOutsideBox) {
    points += s.goalsOutsideBox * rules.goalOutsideBox;
  }
  points += s.assists * rules.assist;

  if (s.cleanSheet && s.minutes >= CLEAN_SHEET_MIN_MINUTES) {
    points += rules.cleanSheet[position];
  }

  // Mål lukket ind rammer kun målmænd og forsvarere, og tælles i hele par:
  // tre indkasserede mål koster ét point, ikke halvandet.
  if (position === "GK" || position === "DEF") {
    points -= Math.floor(s.goalsConceded / rules.goalsConcededPerPoint);
  }

  if (position === "GK") {
    points += Math.floor(s.saves / rules.savesPerPoint);
    points += s.penaltiesSaved * rules.penaltySave;
  }

  points += s.penaltiesMissed * rules.penaltyMiss;
  if (rules.penaltyWon && s.penaltiesWon) points += s.penaltiesWon * rules.penaltyWon;
  if (rules.penaltyConceded && s.penaltiesConceded) {
    points += s.penaltiesConceded * rules.penaltyConceded;
  }

  points += s.yellowCards * rules.yellowCard;
  points += s.redCards * rules.redCard;
  points += s.ownGoals * rules.ownGoal;

  if (rules.ballRecoveriesPerPoint && s.ballRecoveries) {
    points += Math.floor(s.ballRecoveries / rules.ballRecoveriesPerPoint);
  }

  if (rules.defensiveContribution) {
    const threshold = rules.defensiveContribution.threshold[position];
    if (threshold !== null && (s.defensiveActions ?? 0) >= threshold) {
      // Loftet er selve præmien: 20 handlinger giver 2 point, ikke 4.
      points += rules.defensiveContribution.points;
    }
  }

  if (rules.playerOfTheMatch && s.playerOfTheMatch) points += rules.playerOfTheMatch;
  if (rules.bonus && s.bonus) points += s.bonus;

  return points;
}

/**
 * Hvor mange defensive handlinger en spiller mangler for at ramme tærsklen.
 * Bruges i UI'et: "Gravenberch er 1,4 handlinger fra DefCon i snit" siger mere
 * end et forventet pointtal.
 */
export function defconGap(
  rules: ScoringRules,
  position: Position,
  actionsPer90: number,
): { threshold: number; gap: number } | null {
  const threshold = rules.defensiveContribution?.threshold[position];
  if (threshold === null || threshold === undefined) return null;
  return { threshold, gap: threshold - actionsPer90 };
}
