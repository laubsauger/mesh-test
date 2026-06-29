import { describe, it, expect } from 'vitest';
import { POSE_EPS, assertPoseEP } from '../src/pose/ep.js';
import { KPT, NUM_KPTS, RTMW3D_MODEL } from '../src/pose/rtmw-constants.js';

describe('assertPoseEP — V23: explicit EP, no silent fallback', () => {
  it('accepts known EPs', () => {
    expect(assertPoseEP('webgpu')).toBe('webgpu');
    expect(assertPoseEP('wasm')).toBe('wasm');
  });

  it('throws on unknown EP (loud, never auto-fallback)', () => {
    expect(() => assertPoseEP('cuda')).toThrow(/unknown pose EP/);
    expect(() => assertPoseEP(undefined)).toThrow(/unknown pose EP/);
  });

  it('declares exactly the two web EPs', () => {
    expect(POSE_EPS).toEqual(['webgpu', 'wasm']);
  });
});

describe('rtmw-constants — COCO-WholeBody contract', () => {
  it('133 keypoints, hips at 11/12 (V21 index map)', () => {
    expect(NUM_KPTS).toBe(133);
    expect(KPT.leftHip).toBe(11);
    expect(KPT.rightHip).toBe(12);
  });

  it('rtmw3d-x input is CHW [1,3,resH,resW] = 384x288', () => {
    expect(RTMW3D_MODEL.resH).toBe(384);
    expect(RTMW3D_MODEL.resW).toBe(288);
    expect(RTMW3D_MODEL.inputName).toBe('input');
  });
});
