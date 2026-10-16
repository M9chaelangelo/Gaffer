/**
 * Firestore-laget for fantasy-delen.
 *
 * Tre ting gemmes, og de har hver sin grund:
 *
 *  SNAPSHOTS  hele spillets tilstand pr. runde. Uden dem kan vi aldrig
 *             backteste modellen, fordi FPL overskriver priser og ejerskab
 *             løbende — historikken findes ikke andre steder end hos os.
 *  ADVICE     det færdige råd pr. runde. Så dashboardet er øjeblikkeligt,
 *             og så vi bagefter kan se, hvad modellen faktisk anbefalede.
 *  RESULTS    hvad der så skete. Grundlaget for at måle om modellen virker.
 */

import { adminDb, athleteUid } from "@/lib/firebase/admin";
import type { DeadlineAdvice } from "@/lib/fantasy/advice";
import type { GameSnapshot, ManagerSquad } from "@/lib/fantasy/sources/types";

export const FANTASY_COLLECTIONS = {
  snapshots: "fantasy_snapshots",
  advice: "fantasy_advice",
  squads: "fantasy_squads",
  results: "fantasy_results",
} as const;

function docId(game: string, round: number): string {
  // Nul-padding gør leksikalsk sortering enig med numerisk.
  return `${game}-${String(round).padStart(2, "0")}`;
}

function base(uid: string, key: keyof typeof FANTASY_COLLECTIONS): string {
  return `users/${uid}/${FANTASY_COLLECTIONS[key]}`;
}

export async function saveSnapshot(snapshot: GameSnapshot, uid = athleteUid()): Promise<void> {
  // Spillerhistorik fylder megabytes og kan hentes igen; den skrives ikke med.
  // Firestore har en hård grænse på 1 MB pr. dokument.
  const slim: GameSnapshot = {
    ...snapshot,
    players: snapshot.players.map((p) => ({ ...p, history: [] })),
  };
  await adminDb()
    .doc(`${base(uid, "snapshots")}/${docId(snapshot.game, snapshot.nextRound)}`)
    .set(slim, { merge: false });
}

export async function loadSnapshot(
  game: string,
  round: number,
  uid = athleteUid(),
): Promise<GameSnapshot | null> {
  const snap = await adminDb().doc(`${base(uid, "snapshots")}/${docId(game, round)}`).get();
  return snap.exists ? (snap.data() as GameSnapshot) : null;
}

export async function saveAdvice(advice: DeadlineAdvice, uid = athleteUid()): Promise<void> {
  // Histogrammerne er lange arrays pr. spiller. De er interessante i UI'et,
  // men ikke værd at gemme for alle 15 — vi beholder kvantilerne.
  const slim = {
    ...advice,
    squad: advice.squad.map((row) => ({
      ...row,
      dist: { ...row.dist, histogram: [] },
    })),
  };
  await adminDb()
    .doc(`${base(uid, "advice")}/${docId(advice.game, advice.round)}`)
    .set(slim, { merge: false });
}

export async function loadAdvice(
  game: string,
  round: number,
  uid = athleteUid(),
): Promise<DeadlineAdvice | null> {
  const snap = await adminDb().doc(`${base(uid, "advice")}/${docId(game, round)}`).get();
  return snap.exists ? (snap.data() as DeadlineAdvice) : null;
}

/** Seneste råd for et spil, uanset runde. */
export async function latestAdvice(game: string, uid = athleteUid()): Promise<DeadlineAdvice | null> {
  const query = await adminDb()
    .collection(base(uid, "advice"))
    .where("game", "==", game)
    .orderBy("round", "desc")
    .limit(1)
    .get();
  return query.empty ? null : (query.docs[0].data() as DeadlineAdvice);
}

export async function saveSquad(squad: ManagerSquad, uid = athleteUid()): Promise<void> {
  await adminDb()
    .doc(`${base(uid, "squads")}/${docId(squad.game, squad.round)}`)
    .set(squad, { merge: false });
}

export async function loadSquad(
  game: string,
  round: number,
  uid = athleteUid(),
): Promise<ManagerSquad | null> {
  const snap = await adminDb().doc(`${base(uid, "squads")}/${docId(game, round)}`).get();
  return snap.exists ? (snap.data() as ManagerSquad) : null;
}

/**
 * Registrerer hvad der faktisk skete, så modellen kan måles.
 * Uden det er "outperform alle andre" en påstand, ikke en måling.
 */
export interface RoundResult {
  game: string;
  round: number;
  /** Hvad modellen forudsagde for startopstillingen. */
  predicted: number;
  /** Hvad der faktisk blev scoret. */
  actual: number | null;
  captainId: number | null;
  captainPoints: number | null;
  transfersMade: number;
  hitTaken: number;
  recordedAt: string;
}

export async function saveResult(result: RoundResult, uid = athleteUid()): Promise<void> {
  await adminDb()
    .doc(`${base(uid, "results")}/${docId(result.game, result.round)}`)
    .set(result, { merge: true });
}

export async function loadResults(game: string, uid = athleteUid()): Promise<RoundResult[]> {
  const query = await adminDb()
    .collection(base(uid, "results"))
    .where("game", "==", game)
    .orderBy("round", "asc")
    .get();
  return query.docs.map((d) => d.data() as RoundResult);
}

/**
 * Modellens kalibrering: hvor tæt forudsigelserne har ligget på virkeligheden.
 * Vises i UI'et, fordi et råd uden et sporet track record er en mening.
 */
export function calibration(results: RoundResult[]): {
  rounds: number;
  meanError: number;
  meanAbsoluteError: number;
  bias: string;
} {
  const scored = results.filter((r) => r.actual !== null);
  if (scored.length === 0) {
    return { rounds: 0, meanError: 0, meanAbsoluteError: 0, bias: "Ingen afsluttede runder endnu." };
  }
  let sum = 0;
  let absSum = 0;
  for (const r of scored) {
    const error = (r.actual ?? 0) - r.predicted;
    sum += error;
    absSum += Math.abs(error);
  }
  const meanError = sum / scored.length;
  return {
    rounds: scored.length,
    meanError,
    meanAbsoluteError: absSum / scored.length,
    bias:
      Math.abs(meanError) < 3
        ? "Modellen rammer uden systematisk skævhed."
        : meanError > 0
          ? `Modellen undervurderer med ${meanError.toFixed(1)} point pr. runde.`
          : `Modellen overvurderer med ${Math.abs(meanError).toFixed(1)} point pr. runde.`,
  };
}
