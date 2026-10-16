/**
 * UEFA Champions League Fantasy 2026/27.
 *
 * Adskiller sig fra FPL på fire måder der ændrer strategien:
 *  - Målmænd får 6 point for mål, ikke 6 til forsvarere alene.
 *  - Boldgenerobringer giver point direkte (1 pr. 3), ikke via en tærskel.
 *    Det gør defensive midtbaner til et gulv, ikke et lotteri.
 *  - Player of the Match er 3 point, vinderen tager det hele.
 *  - Anførerbindet kan flyttes undervejs i en spilledag, så længe den nye
 *    anfører ikke har spillet endnu. Det er en option, ikke et valg — og den
 *    har værdi, som modellen prissætter.
 */

import type { GameRules } from "@/lib/fantasy/rules/types";

export const UCL_RULES: GameRules = {
  scoring: {
    game: "UCL",
    season: "2026/27",

    appearance: 1,
    appearance60: 1,

    goal: { GK: 6, DEF: 6, MID: 5, FWD: 4 },
    goalOutsideBox: 1,
    assist: 3,
    cleanSheet: { GK: 4, DEF: 4, MID: 1, FWD: 0 },

    savesPerPoint: 3,
    penaltySave: 5,
    penaltyMiss: -2,
    penaltyWon: 2,
    penaltyConceded: -1,

    goalsConcededPerPoint: 2,
    yellowCard: -1,
    redCard: -3,
    ownGoal: -2,

    ballRecoveriesPerPoint: 3,
    playerOfTheMatch: 3,

    // UEFA har ingen DefCon-tærskel; generobringer tæller direkte ovenfor.
    defensiveContribution: null,
    // Ingen bonuspoint — Player of the Match udfylder den rolle.
    bonus: null,
  },

  squad: {
    size: 15,
    composition: { GK: 2, DEF: 5, MID: 5, FWD: 3 },
    startingSize: 11,
    minStarting: { GK: 1, DEF: 3, MID: 2, FWD: 1 },
    maxStarting: { GK: 1, DEF: 5, MID: 5, FWD: 3 },
    budget: 100,
    // Ligafasen. Grænsen løftes i knockout-fasen — se KNOCKOUT_CLUB_LIMITS.
    maxPerClub: 3,
    captainMultiplier: 2,
    hasViceCaptain: false,
  },

  transfers: {
    freePerRound: 2,
    // Kun ét ubrugt skift kan tages med videre i ligafasen.
    maxBanked: 1,
    costPerExtra: -4,
  },

  chips: [
    {
      id: "wildcard",
      name: "Wildcard",
      effect: "Ubegrænsede skift i én spilledag inden for budgettet; truppen beholdes.",
      uses: 1,
    },
    {
      id: "limitless",
      name: "Limitless",
      effect: "Ubegrænsede skift uden budgetloft for én spilledag; truppen ruller tilbage bagefter.",
      uses: 1,
    },
  ],

  unverified: [
    "yellowCard/redCard/ownGoal: standardværdier (-1/-3/-2), ikke verificeret mod UEFA's regelside for 2026/27",
    "squad.maxPerClub: 3 i ligafasen — bekræft mod UEFA's regler for det nye 36-holds format",
    "chips.uses: antaget 1 af hver; bekræft om Wildcard kan spilles igen i knockout-fasen",
  ],
};

/**
 * Klub-loftet løftes gennem knockout-fasen, fordi feltet skrumper: til sidst
 * er der kun to hold tilbage at vælge fra. Tallene er fra det tidligere
 * format og skal bekræftes for 2026/27 — derfor står de også i `unverified`.
 */
export const KNOCKOUT_CLUB_LIMITS: Record<string, number> = {
  "league-phase": 3,
  "round-of-16": 4,
  "quarter-finals": 5,
  "semi-finals": 6,
  final: 8,
};
