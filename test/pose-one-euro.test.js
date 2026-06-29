import { describe, it, expect } from 'vitest';
import { OneEuro, CanonicalSmoother } from '../src/pose/one-euro.js';

describe('OneEuro — T10 smoothing', () => {
  it('first sample passes through unchanged', () => {
    const f = new OneEuro({ minCutoff: 1, beta: 0 });
    expect(f.filter(5, 1 / 30)).toBe(5);
  });

  it('converges to a constant signal', () => {
    const f = new OneEuro({ minCutoff: 1, beta: 0 });
    f.filter(0, 1 / 30);
    let v = 0;
    for (let i = 0; i < 200; i += 1) v = f.filter(10, 1 / 30);
    expect(v).toBeGreaterThan(9.9);
    expect(v).toBeLessThanOrEqual(10);
  });

  it('attenuates jitter (output variance < input variance)', () => {
    const f = new OneEuro({ minCutoff: 1, beta: 0 });
    const noisy = [0, 1, -1, 0.8, -0.9, 1.1, -1, 0.5, -0.6, 1, -1];
    const out = noisy.map((x) => f.filter(x, 1 / 30));
    const variance = (a) => { const m = a.reduce((s, v) => s + v, 0) / a.length; return a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length; };
    expect(variance(out.slice(2))).toBeLessThan(variance(noisy.slice(2)));
  });

  it('reset clears state', () => {
    const f = new OneEuro({ minCutoff: 1, beta: 0 });
    f.filter(100, 1 / 30);
    f.reset();
    expect(f.filter(3, 1 / 30)).toBe(3);
  });
});

describe('CanonicalSmoother', () => {
  const mkCanon = (t, v) => ({
    timestampMs: t,
    joints: [{ x: v, y: v, z: v, confidence: 0.9 }]
  });

  it('smooths joint positions toward the signal, preserves confidence', () => {
    const s = new CanonicalSmoother(1, { minCutoff: 1, beta: 0 });
    s.smooth(mkCanon(0, 0));
    let out;
    for (let i = 1; i <= 50; i += 1) out = s.smooth(mkCanon(i * 33, 10));
    expect(out.joints[0].x).toBeGreaterThan(9);
    expect(out.joints[0].confidence).toBe(0.9);
  });
});
