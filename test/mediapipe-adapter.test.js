import { describe, it, expect } from 'vitest';
import { toCanonicalFromMediaPipe, blendshapesToExpr, COCO_FROM_BLAZE } from '../src/pose/mediapipe-adapter.js';
import { KPT } from '../src/pose/rtmw-constants.js';

// Minimal 33-landmark BlazePose world set: hips at ±0.1 x, shoulders up at y=-0.4 (MP y is
// DOWN), nose higher. Everything else 0. visibility 0.9.
function blazePose() {
  const w = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0.9 }));
  w[0] = { x: 0, y: -0.6, z: 0, visibility: 0.95 }; // nose (up in image = -y)
  w[11] = { x: 0.2, y: -0.4, z: 0, visibility: 0.9 }; // left shoulder
  w[12] = { x: -0.2, y: -0.4, z: 0, visibility: 0.9 }; // right shoulder
  w[23] = { x: 0.1, y: 0, z: 0, visibility: 0.9 }; // left hip
  w[24] = { x: -0.1, y: 0, z: 0, visibility: 0.9 }; // right hip
  w[27] = { x: 0.1, y: 0.9, z: 0, visibility: 0.9 }; // left ankle (down = +y)
  return w;
}

describe('mediapipe-adapter', () => {
  it('maps BlazePose → COCO indices, pelvis-centered, +y up', () => {
    const canon = toCanonicalFromMediaPipe({ timestampMs: 5, poseWorld: blazePose() });
    // pelvis (hip midpoint) is the origin
    expect(canon.pelvisCenter).toEqual({ x: 0, y: 0, z: 0 });
    // nose maps to COCO 0 and is ABOVE the pelvis (+y) after the y-flip
    expect(canon.joints[KPT.nose].y).toBeGreaterThan(0.3);
    // ankle is BELOW the pelvis (−y)
    expect(canon.joints[KPT.leftAnkle].y).toBeLessThan(-0.5);
    // shoulders landed at the right COCO indices with confidence carried from visibility
    expect(canon.joints[KPT.leftShoulder].confidence).toBeCloseTo(0.9);
    expect(canon.joints[KPT.rightShoulder].x).toBeCloseTo(-0.2); // right shoulder on −x
  });

  it('shoulderCenter is the shoulder midpoint', () => {
    const canon = toCanonicalFromMediaPipe({ timestampMs: 0, poseWorld: blazePose() });
    expect(canon.shoulderCenter.x).toBeCloseTo(0);
    expect(canon.shoulderCenter.y).toBeCloseTo(0.4); // +y up (MP had −0.4)
  });

  it('throws on missing pose landmarks (fail loud, no silent empty pose)', () => {
    expect(() => toCanonicalFromMediaPipe({ poseWorld: [] })).toThrow();
  });

  it('every rig-read COCO joint has a BlazePose source', () => {
    for (const k of [KPT.nose, KPT.leftEar, KPT.rightEar, KPT.leftShoulder, KPT.rightHip, KPT.leftKnee, KPT.rightAnkle]) {
      expect(COCO_FROM_BLAZE[k]).toBeTypeOf('number');
    }
  });

  it('blendshapes → the 7 face scalars', () => {
    const e = blendshapesToExpr({ jawOpen: 0.7, mouthSmileLeft: 0.4, mouthSmileRight: 0.6, mouthPucker: 0.2, eyeBlinkLeft: 1, eyeBlinkRight: 0 });
    expect(e.jawOpen).toBeCloseTo(0.7);
    expect(e.smile).toBeCloseTo(0.5);
    expect(e.blinkL).toBe(1);
    expect(e.blinkR).toBe(0);
    expect(e.jawOpen).toBeLessThanOrEqual(1); // clamped
  });
});
