import { describe, it, expect } from 'vitest';
import { toCanonical } from '../src/pose/observation-adapter.js';
import { KPT, NUM_KPTS } from '../src/pose/rtmw-constants.js';

// Minimal frame: full 133 kpts, only hips + shoulders meaningful.
function makeFrame() {
  const keypoints3D = Array.from({ length: NUM_KPTS }, () => ({ x: 0, y: 0, z: 0, confidence: 0.9 }));
  // Image-space y is DOWN: shoulders (higher up) have SMALLER y than hips.
  keypoints3D[KPT.leftHip] = { x: -1, y: 4, z: 0.2, confidence: 0.9 };
  keypoints3D[KPT.rightHip] = { x: 1, y: 4, z: -0.2, confidence: 0.8 };
  keypoints3D[KPT.leftShoulder] = { x: -1, y: 2, z: 0, confidence: 0.7 };
  keypoints3D[KPT.rightShoulder] = { x: 1, y: 2, z: 0, confidence: 0.6 };
  return { timestampMs: 123, keypoints2D: [], keypoints3D, boundingBox: null };
}

describe('toCanonical — V21/V22: pelvis-center + canonical basis', () => {
  const canon = toCanonical(makeFrame());

  it('pelvis origin = (0,0,0) by construction', () => {
    expect(canon.pelvisCenter).toEqual({ x: 0, y: 0, z: 0 });
    // hip midpoint was (0,4,0) → recentered hips average to 0
    const lh = canon.joints[KPT.leftHip];
    const rh = canon.joints[KPT.rightHip];
    expect((lh.x + rh.x) / 2).toBeCloseTo(0, 6);
    expect((lh.y + rh.y) / 2).toBeCloseTo(0, 6);
  });

  it('y flips image-down → up (shoulders above pelvis)', () => {
    // raw shoulder y=2 < hip mid y=4 (image-down); after -(y-pelvis) shoulders +up
    expect(canon.shoulderCenter.y).toBeGreaterThan(0);
  });

  it('x = image-right, pelvis-centered (no negate — matches rig)', () => {
    // raw leftHip x=-1, pelvis x=0 → canonical -1 (preserved, just centered)
    expect(canon.joints[KPT.leftHip].x).toBeCloseTo(-1, 6);
    expect(canon.joints[KPT.rightHip].x).toBeCloseTo(1, 6);
  });

  it('carries confidence + timestamp', () => {
    expect(canon.joints[KPT.leftHip].confidence).toBe(0.9);
    expect(canon.timestampMs).toBe(123);
  });

  it('throws if hips missing (no pelvis origin)', () => {
    const bad = { timestampMs: 0, keypoints3D: [{ x: 0, y: 0, z: 0, confidence: 0 }] };
    expect(() => toCanonical(bad)).toThrow(/hip/);
  });
});
