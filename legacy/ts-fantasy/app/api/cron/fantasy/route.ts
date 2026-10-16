/**
 * Opdaterer begge spil og gemmer rådet i Firestore.
 *
 * Ruten er IKKE koblet på Vercels cron-plan, og det er med vilje: Hobby giver
 * to cron-jobs pr. projekt, og Garmin-synken og ugerapporten har dem. I stedet
 * skriver /api/fantasy/advice hver gang dashboardet åbnes.
 *
 * Ruten bliver her alligevel, fordi den gør to ting dashboardet ikke gør: den
 * opdaterer BEGGE spil i ét kald, og den kan køres uden en browser. Sæt den på
 * en schedule hvis projektet flytter til Pro:
 *
 *   { "path": "/api/cron/fantasy", "schedule": "0 7 * * *" }
 *
 * eller kald den fra en ekstern scheduler med CRON_SECRET som Bearer-token.
 */

import { NextResponse } from "next/server";
import { runPipeline } from "@/lib/fantasy/pipeline";
import { saveAdvice, saveSnapshot, saveSquad } from "@/lib/fantasy/store";

export const runtime = "nodejs";
export const maxDuration = 300;
export const dynamic = "force-dynamic";

export async function GET(request: Request): Promise<NextResponse> {
  // Vercel signerer cron-kald med CRON_SECRET. Uden tjekket kan hvem som
  // helst trigge en fuld opdatering mod FPL's API på vores vegne.
  const secret = process.env.CRON_SECRET;
  if (secret) {
    const auth = request.headers.get("authorization");
    if (auth !== `Bearer ${secret}`) {
      return NextResponse.json({ ok: false, error: "Uautoriseret" }, { status: 401 });
    }
  }

  const results: Record<string, unknown> = {};

  for (const game of ["FPL", "UCL"] as const) {
    try {
      const run = await runPipeline({
        game,
        entryId: game === "FPL" ? process.env.FPL_ENTRY_ID ?? undefined : undefined,
      });
      await Promise.all([
        saveSnapshot(run.snapshot),
        saveAdvice(run.advice),
        saveSquad(run.squad),
      ]);
      results[game] = {
        ok: true,
        round: run.advice.round,
        expected: run.advice.projection.expected,
        warnings: run.warnings,
      };
    } catch (err) {
      // Ét spil der fejler må ikke stoppe det andet.
      results[game] = { ok: false, error: err instanceof Error ? err.message : "Ukendt fejl" };
    }
  }

  return NextResponse.json({ ok: true, results, ranAt: new Date().toISOString() });
}
