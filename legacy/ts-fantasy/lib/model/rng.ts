/**
 * Deterministisk tilfældighed. Monte Carlo skal kunne gentages: to kørsler på
 * samme data skal give samme råd, ellers kan man hverken teste modellen eller
 * forklare hvorfor den skiftede mening.
 *
 * mulberry32 er hurtig, har fuld 32-bit periode og består de statistiske
 * test der betyder noget her. Math.random() kan ikke seedes og er derfor
 * ubrugelig til formålet.
 */
export class Rng {
  private state: number;

  constructor(seed = 0x9e3779b9) {
    this.state = seed >>> 0;
  }

  /** Uniform i [0,1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Standardnormal via Box-Muller. */
  normal(): number {
    // u må ikke være 0, ellers giver log() minus uendelig.
    const u = 1 - this.next();
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /**
   * Poisson. Knuths multiplikative metode under lambda 30, hvor den er
   * hurtigst; derover ville e^-lambda underflow'e, så vi bruger en
   * normaltilnærmelse — fodbold når aldrig derop, men modellen bruges også
   * til at summere hold over en hel sæson.
   */
  poisson(lambda: number): number {
    if (lambda <= 0) return 0;
    if (lambda < 30) {
      const limit = Math.exp(-lambda);
      let k = 0;
      let p = 1;
      do {
        k++;
        p *= this.next();
      } while (p > limit);
      return k - 1;
    }
    return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * this.normal()));
  }

  /** Bernoulli. */
  chance(p: number): boolean {
    return this.next() < p;
  }

  /** Vælg et indeks ud fra vægte der ikke behøver summere til 1. */
  weightedIndex(weights: number[]): number {
    let total = 0;
    for (const w of weights) total += Math.max(0, w);
    if (total <= 0) return -1;
    let r = this.next() * total;
    for (let i = 0; i < weights.length; i++) {
      r -= Math.max(0, weights[i]);
      if (r <= 0) return i;
    }
    return weights.length - 1;
  }
}
