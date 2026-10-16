/**
 * Rådgivningen som API. GET kører hele kæden og svarer med det færdige råd.
 *
 * Kører i Node-runtime, ikke Edge: simuleringen er CPU-tung, og Edge har et
 * hårdere loft. Ruten er dynamisk, fordi et råd der caches henover et
 * holdnyt er værre end intet råd.
 *
 * Ruten SKRIVER også, ikke kun læser. Grunden er Vercels Hobby-plan: den
 * tillader to cron-jobs pr. projekt, og de to er taget af Garmin-synken og
 * ugerapporten. I stedet for at ofre et af dem gemmes snapshot og råd her,
 * hver gang dashboardet åbnes. Det giver den historik, backtesten skal bruge,
 * uden at koste en cron — og i praksis kigger man alligevel på dashboardet
 * oftere end tre gange om dagen op mod en deadline.
 */

import { NextResponse } from "next/server";
import { runPipeline } from "@/lib/fantasy/pipeline";
import { saveAdvice, saveSnapshot, saveSquad } from "@/lib/fantasy/store";
import type { RunResult } from "@/lib/fantasy/pipeline";

export const runtime = "nodejs";
export const maxDuration = 60;
export const dynamic = "force-dynamic";

export async function GET(request: Request): Promise<NextResponse> {
  const url = new URL(request.url);
  const game = url.searchParams.get("game") === "UCL" ? "UCL" : "FPL";

  const numberParam = (key: string): number | undefined => {
    const raw = url.searchParams.get(key);
    if (raw === null) return undefined;
    const parsed = Number.parseFloat(raw);
    return Number.isFinite(parsed) ? parsed : undefined;
  };

  try {
    const result = await runPipeline({
      game,
      entryId: url.searchParams.get("entry") ?? process.env.FPL_ENTRY_ID ?? undefined,
      aggression: numberParam("aggression"),
      horizon: numberParam("horizon"),
      iterations: numberParam("iterations"),
      matchday: numberParam("matchday"),
      freeTransfers: numberParam("ft"),
      bank: numberParam("bank"),
    });

    const persistError = await persist(result);

    return NextResponse.json({
      ok: true,
      advice: result.advice,
      warnings: persistError ? [...result.warnings, persistError] : result.warnings,
    });
  } catch (err) {
    // Fejlen vises i UI'et, ikke kun i loggen: et tomt dashboard på
    // deadline-dagen er værre end en fejlbesked der siger hvad der mangler.
    const message = err instanceof Error ? err.message : "Ukendt fejl";
    return NextResponse.json({ ok: false, error: message }, { status: 502 });
  }
}

/**
 * Gemmer bedst muligt. Rådet er allerede regnet færdigt på dette tidspunkt,
 * og det er hele pointen med kaldet — en manglende Firestore-nøgle må derfor
 * ikke koste brugeren svaret. Men den må heller ikke være tavs, for så tror
 * man at der bygges historik, mens der ikke gør.
 */
async function persist(result: RunResult): Promise<string | null> {
  try {
    await Promise.all([
      saveSnapshot(result.snapshot),
      saveAdvice(result.advice),
      saveSquad(result.squad),
    ]);
    return null;
  } catch (err) {
    const message = err instanceof Error ? err.message : "ukendt fejl";
    return `Rådet blev regnet, men kunne ikke gemmes i Firestore: ${message} — ` +
      "der bygges ingen historik til backtest før det er løst.";
  }
}
