/**
 * Fantasy Premier League 2026/27.
 *
 * Kilder: premierleague.com's egne regelsider om defensive bidrag samt
 * spillets pointtabel. To ændringer fra 2025/26 er indarbejdet:
 *  - DefCon-formlen er kalibreret om (holdende midtbaner op, backs ned).
 *  - Straffen for at blive tacklet af med bolden er fjernet — den ramte
 *    driblende kantspillere for hårdt. Den indgik kun i BPS, så den optræder
 *    ikke i pointtabellen her, men i `bps.ts`.
 */

import type { GameRules } from "@/lib/fantasy/rules/types";

export const FPL_RULES: GameRules = {
  scoring: {
    game: "FPL",
    season: "2026/27",

    appearance: 1,
    appearance60: 1,

    goal: { GK: 6, DEF: 6, MID: 5, FWD: 4 },
    goalOutsideBox: 0,
    assist: 3,
    cleanSheet: { GK: 4, DEF: 4, MID: 1, FWD: 0 },

    savesPerPoint: 3,
    penaltySave: 5,
    penaltyMiss: -2,
    penaltyWon: 0,
    penaltyConceded: 0,

    goalsConcededPerPoint: 2,
    yellowCard: -1,
    redCard: -3,
    ownGoal: -2,

    // FPL tæller boldgenerobringer inde i DefCon for MID/FWD, ikke separat.
    ballRecoveriesPerPoint: null,
    playerOfTheMatch: 0,

    defensiveContribution: {
      // Forsvarere: 10 CBIT. Midt/angreb: 12 CBIRT (samme plus generobringer).
      // Målmænd har ingen tærskel — de får point for redninger i stedet.
      threshold: { GK: null, DEF: 10, MID: 12, FWD: 12 },
      points: 2,
    },

    bonus: [3, 2, 1],
  },

  squad: {
    size: 15,
    composition: { GK: 2, DEF: 5, MID: 5, FWD: 3 },
    startingSize: 11,
    minStarting: { GK: 1, DEF: 3, MID: 2, FWD: 1 },
    maxStarting: { GK: 1, DEF: 5, MID: 5, FWD: 3 },
    budget: 100,
    maxPerClub: 3,
    captainMultiplier: 2,
    hasViceCaptain: true,
  },

  transfers: {
    freePerRound: 1,
    // FPL lod opsparingen stige til fem gratis skift fra 2024/25.
    maxBanked: 5,
    costPerExtra: -4,
  },

  chips: [
    {
      id: "wildcard",
      name: "Wildcard",
      effect: "Ubegrænsede skift i én runde uden fradrag; truppen beholdes bagefter.",
      uses: 2,
    },
    {
      id: "freehit",
      name: "Free Hit",
      effect: "Ubegrænsede skift for én runde; truppen ruller tilbage bagefter.",
      uses: 1,
    },
    {
      id: "bboost",
      name: "Bench Boost",
      effect: "Alle 15 spillere scorer i stedet for de 11 i startopstillingen.",
      uses: 1,
    },
    {
      id: "3xc",
      name: "Triple Captain",
      effect: "Anføreren scorer tredobbelt i stedet for dobbelt.",
      uses: 1,
    },
  ],

  unverified: [
    // Disse tal er uændrede gennem mange sæsoner og er de facto standard, men
    // vi har ikke kunnet hente FPL's egen regelside i dette miljø.
    "yellowCard/redCard/ownGoal: standardværdier (-1/-3/-2), ikke verificeret mod FPL's regelside for 2026/27",
    "transfers.maxBanked: 5 gratis skift, indført 2024/25 — bekræft at loftet er uændret",
  ],
};
