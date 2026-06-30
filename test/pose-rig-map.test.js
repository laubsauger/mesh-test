import { describe, it, expect } from 'vitest';
import { MESHY_BONES, SEGMENTS, resolveRigBones } from '../src/pose/rig-map.js';

// Real 24-bone Meshy skeleton (extracted from every shipped GLB).
const MESHY_SKELETON_BONES = [
  'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
  'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
  'Spine02', 'Spine01', 'Spine',
  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
  'neck', 'Head', 'head_end', 'headfront'
];

const fakeSkeleton = (names) => ({ bones: names.map((name) => ({ name })) });

describe('rig-map — T11: canonical → Meshy bone', () => {
  it('every mapped bone exists in the real skeleton', () => {
    const set = new Set(MESHY_SKELETON_BONES);
    for (const name of Object.values(MESHY_BONES)) expect(set.has(name)).toBe(true);
  });

  it('resolveRigBones resolves all 22 logical bones', () => {
    const rig = resolveRigBones(fakeSkeleton(MESHY_SKELETON_BONES));
    expect(Object.keys(rig).length).toBe(Object.keys(MESHY_BONES).length);
    expect(rig.hips.name).toBe('Hips');
    expect(rig.leftForeArm.name).toBe('LeftForeArm');
  });

  it('throws on a skeleton missing a mapped bone (no fallback)', () => {
    const missing = MESHY_SKELETON_BONES.filter((n) => n !== 'LeftArm');
    expect(() => resolveRigBones(fakeSkeleton(missing))).toThrow(/LeftArm/);
  });

  it('SEGMENTS reference declared rig keys with resolver endpoints', () => {
    for (const seg of SEGMENTS) {
      expect(MESHY_BONES[seg.bone]).toBeDefined();
      expect(typeof seg.from).toBe('function');
      expect(typeof seg.to).toBe('function');
    }
  });

  it('SEGMENTS exclude spine (180° flip) + neck/head (driven by full-basis _aimHead/_aimHips)', () => {
    const bones = SEGMENTS.map((s) => s.bone);
    expect(bones).not.toContain('spine'); // §B: Spine child is below it → 180° flip
    expect(bones).not.toContain('neck'); // head yaw/roll via _aimHead basis
    expect(bones).toContain('leftFoot'); // feet still direction-aimed
  });
});
