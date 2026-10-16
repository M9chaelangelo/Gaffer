/**
 * Den rang-bevidste målfunktion.
 *
 * Det her er kernen i at VINDE frem for at score pænt. De to ting er ikke ens,
 * og forskellen er effektivt ejerskab.
 *
 * Hvis 78 % af feltet ejer Haaland, og du også gør, tjener du ingen rang på
 * hans 15 point — du følger bare med. Ejer du ham ikke, taber du 15 point til
 * 78 % af feltet i én runde. Den asymmetri er hele spillet i en minirække.
 *
 * Den rigtige størrelse at maksimere er ikke E[point], men
 *
 *     E[point] - ejerandel * E[point]      (bidrag til rang)
 *
 * altså de point du vinder RELATIVT til feltet. For en anfører tælles
 * ejerskabet med anførermultiplikatoren, fordi feltet også fordobler ham.
 *
 * Og fordi vi har hele fordelingen og ikke kun et gennemsnit, kan vi gøre det
 * bedre endnu: vi kan regne P(du slår feltet) direkte.
 */

import type { PointsDistribution } from "@/lib/fantasy/model/simulate";

export interface OwnershipContext {
  /** Ejerandel 0-1 pr. spiller i den population du konkurrerer mod. */
  ownership: Map<number, number>;
  /** Andel af feltet der har spilleren som anfører, 0-1. */
  captaincy: Map<number, number>;
  /**
   * Hvor aggressiv strategien skal være. 0 = maksimér forventede point
   * (rigtigt hvis du er langt foran). 1 = maksimér rang-bidrag (rigtigt hvis
   * du jagter). Mellem 0 og 1 vægtes de to.
   */
  aggression: number;
}

/** Effektivt ejerskab: ejerandel plus den ekstra vægt fra anførerbind. */
export function effectiveOwnership(
  playerId: number,
  ctx: OwnershipContext,
  captainMultiplier: number,
): number {
  const owned = ctx.ownership.get(playerId) ?? 0;
  const captained = ctx.captaincy.get(playerId) ?? 0;
  // En spiller ejet af 50 % og anført af 30 % tæller som 50 % + 30 % ekstra,
  // fordi de 30 % får ham dobbelt.
  return owned + captained * (captainMultiplier - 1);
}

/**
 * Værdien af at eje en spiller, målt i rang frem for i point.
 * Positiv = han flytter dig op i forhold til feltet.
 */
export function rankValue(
  dist: PointsDistribution,
  ctx: OwnershipContext,
  captainMultiplier = 1,
  isCaptain = false,
): number {
  const multiplier = isCaptain ? captainMultiplier : 1;
  const mine = dist.mean * multiplier;
  const eo = effectiveOwnership(dist.playerId, ctx, captainMultiplier);
  // Feltets forventede bidrag fra samme spiller.
  const field = dist.mean * eo;
  const differential = mine - field;
  return ctx.aggression * differential + (1 - ctx.aggression) * mine;
}

/**
 * P(A slår B) ud fra to fordelinger, under antagelse af uafhængighed.
 * Bruges til at sammenligne to anførervalg eller to skiftemuligheder direkte
 * på det der betyder noget: hvem der ender øverst, ikke hvem der har det
 * højeste gennemsnit.
 */
export function probabilityBeats(a: PointsDistribution, b: PointsDistribution): number {
  let p = 0;
  for (let i = 0; i < a.histogram.length; i++) {
    const pa = a.histogram[i] ?? 0;
    if (pa === 0) continue;
    let cumulativeB = 0;
    for (let j = 0; j < i && j < b.histogram.length; j++) {
      cumulativeB += b.histogram[j] ?? 0;
    }
    // Uafgjort deles: halvdelen af masse ved samme point tæller som gevinst.
    const tie = (b.histogram[i] ?? 0) / 2;
    p += pa * (cumulativeB + tie);
  }
  return p;
}

/**
 * Fold to uafhængige fordelinger sammen. Gør det muligt at bygge hele
 * holdets fordeling af de enkelte spilleres — og dermed at svare på
 * "hvad er sandsynligheden for at jeg slår 60 point i denne runde".
 */
export function convolve(a: number[], b: number[]): number[] {
  const out = new Array<number>(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) {
    const pa = a[i];
    if (!pa) continue;
    for (let j = 0; j < b.length; j++) {
      const pb = b[j];
      if (!pb) continue;
      out[i + j] += pa * pb;
    }
  }
  return out;
}

/**
 * Hele holdets pointfordeling. Spillerne er ikke uafhængige i virkeligheden
 * (to Liverpool-spillere hænger sammen), så dette er en tilnærmelse — den
 * undervurderer halerne en smule. Det er stadig langt bedre end at lægge
 * gennemsnit sammen, som slet ingen hale har.
 */
export function teamDistribution(
  dists: PointsDistribution[],
  captainId: number | null,
  captainMultiplier: number,
): number[] {
  let acc = [1];
  for (const d of dists) {
    let hist = d.histogram;
    if (d.playerId === captainId && captainMultiplier > 1) {
      // Anføreren: skalér pointaksen, ikke sandsynlighederne.
      const scaled: number[] = [];
      for (let i = 0; i < hist.length; i++) {
        const p = hist[i];
        if (!p) continue;
        const idx = i * captainMultiplier;
        scaled[idx] = (scaled[idx] ?? 0) + p;
      }
      for (let i = 0; i < scaled.length; i++) scaled[i] = scaled[i] ?? 0;
      hist = scaled;
    }
    acc = convolve(acc, hist);
  }
  return acc;
}

/** P(score ≥ tærskel) læst af en foldet fordeling. */
export function probabilityAtLeast(histogram: number[], threshold: number): number {
  let p = 0;
  for (let i = Math.max(0, Math.ceil(threshold)); i < histogram.length; i++) {
    p += histogram[i] ?? 0;
  }
  return p;
}
