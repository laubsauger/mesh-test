import { describe, it, expect } from 'vitest';
import { CHOREOGRAPHIES, computeWalkerPhase, hashRandom } from '../src/choreography.js';

const bounds = { centerX: 0, centerZ: 0, halfX: 10, halfZ: 10, radius: 14.142135 };
const mk = (index, seed, x = 0, z = 0) => ({ index, seed, basePosition: { x, z } });
const opts = (choreography, choreoDelay = 0.6, count = 7) => ({ choreography, choreoDelay, count, bounds });
const DUR = 2;

describe('computeWalkerPhase — V7: pure, reproducible', () => {
  it('same inputs → same output (deterministic, no Math.random)', () => {
    const w = mk(3, 0.42, 4, -2);
    const a = computeWalkerPhase(w, DUR, opts('ripple'));
    const b = computeWalkerPhase(w, DUR, opts('ripple'));
    expect(a).toBe(b);
  });

  it('does not mutate walker or opts', () => {
    const w = mk(3, 0.42, 4, -2);
    const o = opts('wave');
    const wSnap = JSON.stringify(w);
    const oSnap = JSON.stringify(o);
    computeWalkerPhase(w, DUR, o);
    expect(JSON.stringify(w)).toBe(wSnap);
    expect(JSON.stringify(o)).toBe(oSnap);
  });

  it('every CHOREOGRAPHIES entry returns finite offset', () => {
    for (const c of CHOREOGRAPHIES) {
      const v = computeWalkerPhase(mk(2, 0.7, 3, 5), DUR, opts(c));
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

describe('computeWalkerPhase — pattern semantics', () => {
  it('synced: 0 for all walkers', () => {
    expect(computeWalkerPhase(mk(0, 0.1), DUR, opts('synced'))).toBe(0);
    expect(computeWalkerPhase(mk(5, 0.9, 8, 8), DUR, opts('synced'))).toBe(0);
  });

  it('choreoDelay <= 0 disables all offsets', () => {
    for (const c of CHOREOGRAPHIES) {
      expect(computeWalkerPhase(mk(4, 0.8, 6, 6), DUR, opts(c, 0))).toBe(0);
    }
  });

  it('desync: legacy parity = seed * duration * delay', () => {
    const w = mk(3, 0.42);
    expect(computeWalkerPhase(w, DUR, opts('desync'))).toBeCloseTo(0.42 * DUR * 0.6, 10);
  });

  it('ripple: 0 at center, grows toward rim, capped at span', () => {
    const center = computeWalkerPhase(mk(0, 0.5, 0, 0), DUR, opts('ripple'));
    const mid = computeWalkerPhase(mk(1, 0.5, 5, 5), DUR, opts('ripple'));
    const rim = computeWalkerPhase(mk(2, 0.5, 100, 100), DUR, opts('ripple'));
    expect(center).toBe(0);
    expect(mid).toBeGreaterThan(center);
    expect(rim).toBeGreaterThan(mid);
    expect(rim).toBeLessThanOrEqual(DUR * 0.6 + 1e-9);
  });

  it('wave: increases along +x', () => {
    const left = computeWalkerPhase(mk(0, 0.5, -10, 0), DUR, opts('wave'));
    const center = computeWalkerPhase(mk(1, 0.5, 0, 0), DUR, opts('wave'));
    const right = computeWalkerPhase(mk(2, 0.5, 10, 0), DUR, opts('wave'));
    expect(left).toBeLessThan(center);
    expect(center).toBeLessThan(right);
    expect(left).toBeGreaterThanOrEqual(0);
  });

  it('index: monotonic by spawn order, 0..span', () => {
    const first = computeWalkerPhase(mk(0, 0.5), DUR, opts('index', 0.6, 7));
    const last = computeWalkerPhase(mk(6, 0.5), DUR, opts('index', 0.6, 7));
    expect(first).toBe(0);
    expect(last).toBeCloseTo(DUR * 0.6, 10);
  });

  it('random: deterministic per index, distinct stream from seed', () => {
    const w = mk(3, 0.42);
    const v = computeWalkerPhase(w, DUR, opts('random'));
    expect(v).toBeCloseTo(hashRandom(3 * 31 + 7) * DUR * 0.6, 10);
    // distinct from desync (seed-based) for same walker
    expect(v).not.toBeCloseTo(computeWalkerPhase(w, DUR, opts('desync')), 6);
  });
});
