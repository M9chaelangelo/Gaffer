/**
 * Michaels to trupper, indtastet fra appens egne skærmbilleder.
 *
 * Hvorfor de ligger i koden: FPL's entry-endepunkt kræver et manager-id vi
 * ikke har, og UEFA's trup-endepunkt kræver login. Indtil begge er koblet på,
 * er det her sandheden om truppen — og den er nok, fordi modellen kun skal
 * bruge HVEM der er i truppen, ikke hvordan den kom derind.
 *
 * Spiller-id'erne matches ved navn mod spillets egne data ved indlæsning, så
 * listen overlever at FPL skifter interne id'er mellem sæsoner.
 */

import type { Position } from "@/lib/fantasy/rules/types";

export interface SeedPick {
  /** Navnet som spillet viser det. Matches mod `web_name`/`pDName`. */
  name: string;
  position: Position;
  /** Klubkode, til at skelne navnesammenfald. */
  club: string;
  slot: number;
  isCaptain?: boolean;
  isViceCaptain?: boolean;
}

/**
 * FPL — Gameweek 5, deadline fredag 18. september 19:30.
 * Aflæst 18. september kl. 11:05.
 */
export const FPL_SQUAD_GW5: SeedPick[] = [
  { name: "Kinsky", position: "GK", club: "TOT", slot: 1 },
  { name: "N.Williams", position: "DEF", club: "NFO", slot: 2 },
  { name: "Egan", position: "DEF", club: "HUL", slot: 3 },
  { name: "Thomas", position: "DEF", club: "COV", slot: 4 },
  { name: "Elanga", position: "MID", club: "NEW", slot: 5 },
  { name: "Ndiaye", position: "MID", club: "MCI", slot: 6 },
  { name: "Rogers", position: "MID", club: "CHE", slot: 7 },
  { name: "Szoboszlai", position: "MID", club: "LIV", slot: 8, isCaptain: true },
  { name: "Haaland", position: "FWD", club: "MCI", slot: 9, isViceCaptain: true },
  { name: "João Pedro", position: "FWD", club: "CHE", slot: 10 },
  { name: "Isak", position: "FWD", club: "LIV", slot: 11 },
  // Bænken i den rækkefølge spillet bytter dem ind.
  { name: "Lammens", position: "GK", club: "MUN", slot: 12 },
  { name: "Wirtz", position: "MID", club: "LIV", slot: 13 },
  { name: "Maguire", position: "DEF", club: "MUN", slot: 14 },
  { name: "Ajer", position: "DEF", club: "BRE", slot: 15 },
];

/**
 * Advarsler aflæst direkte på skærmbilledet 18. september kl. 11:05.
 * De er friskere end noget vi kan hente, indtil API'et er koblet på — og
 * João Pedros flag kom til mellem de to skærmbilleder samme morgen.
 */
export const FPL_FLAGS_GW5: Record<string, { level: "out" | "doubt"; note: string }> = {
  Elanga: {
    level: "out",
    // Rødt flag i appen = 0 % spilchance.
    note: "Rødt flag i FPL 18/9 kl. 11:05 — 0 % spilchance. Skal ud af startopstillingen.",
  },
  "João Pedro": {
    level: "doubt",
    // Gult flag kom til mellem kl. 09:38 og 11:05 samme dag.
    note: "Gult flag opstod i FPL mellem 09:38 og 11:05 den 18/9 — 75 % eller lavere spilchance. Tjek holdnyt igen inden 19:30.",
  },
};

/** FPL-chips: Triple Captain er brugt i GW3, resten er i behold. */
export const FPL_CHIPS_AVAILABLE = ["wildcard", "freehit", "bboost"];

/**
 * UCL Fantasy — "BarCola tak", Matchday 2, deadline 13. oktober 18:45.
 * 105 point efter MD1.
 */
export const UCL_SQUAD_MD2: SeedPick[] = [
  { name: "Z. Suzuki", position: "GK", club: "PAR", slot: 1 },
  { name: "H. Maguire", position: "DEF", club: "MUN", slot: 2 },
  { name: "Sergi Altimira", position: "DEF", club: "SPO", slot: 3 },
  { name: "N. Brown", position: "DEF", club: "BAY", slot: 4 },
  { name: "Bruno Fernandes", position: "MID", club: "MUN", slot: 5 },
  { name: "Raphinha", position: "MID", club: "BAR", slot: 6 },
  { name: "M. Olise", position: "MID", club: "BAY", slot: 7 },
  { name: "R. Zalazar", position: "MID", club: "SPO", slot: 8 },
  { name: "H. Kane", position: "FWD", club: "BAY", slot: 9, isCaptain: true },
  { name: "S. Guirassy", position: "FWD", club: "BVB", slot: 10 },
  { name: "D. Undav", position: "FWD", club: "VFB", slot: 11 },
  { name: "F. Bredlow", position: "GK", club: "VFB", slot: 12 },
  { name: "A. Hakimi", position: "DEF", club: "PSG", slot: 13 },
  { name: "K. Kvaratskhelia", position: "MID", club: "PSG", slot: 14 },
  { name: "M. Mittelstädt", position: "DEF", club: "VFB", slot: 15 },
];

export const UCL_CHIPS_AVAILABLE = ["wildcard", "limitless"];

/**
 * Matcher en indtastet trup mod spillets egne spillere.
 *
 * Navnematchning er skrøbelig — derfor er den eksplicit om, hvad den ikke
 * kunne finde, i stedet for stiltiende at droppe en spiller. En trup på 14
 * spillere ville give råd der ser rigtige ud og er forkerte.
 */
export function matchSeedToPlayers<T extends { id: number; name: string; fullName: string; position: Position; team: number }>(
  seed: SeedPick[],
  players: T[],
  teamCodeById: Map<number, string>,
): { matched: { pick: SeedPick; player: T }[]; unmatched: SeedPick[] } {
  const matched: { pick: SeedPick; player: T }[] = [];
  const unmatched: SeedPick[] = [];
  const used = new Set<number>();

  const normalize = (s: string): string =>
    s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z ]/g, "").trim();

  for (const pick of seed) {
    const target = normalize(pick.name);
    const candidates = players.filter(
      (p) =>
        !used.has(p.id) &&
        p.position === pick.position &&
        (normalize(p.name) === target ||
          normalize(p.fullName).endsWith(target) ||
          normalize(p.fullName).split(" ").includes(target.split(" ").pop() ?? "")),
    );

    // Klubkoden bryder navnesammenfald (to spillere kan hedde "Thomas").
    const byClub = candidates.filter((p) => teamCodeById.get(p.team) === pick.club);
    const chosen = byClub[0] ?? candidates[0];

    if (chosen) {
      used.add(chosen.id);
      matched.push({ pick, player: chosen });
    } else {
      unmatched.push(pick);
    }
  }

  return { matched, unmatched };
}
