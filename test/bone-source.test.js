import { describe, it, expect } from 'vitest';
import { BONE_SOURCES, boneSourceForIndex } from '../src/bone-source.js';

describe('boneSourceForIndex — V20: per-walker source select', () => {
  it('poseCount 0 → all walkers clip', () => {
    for (let i = 0; i < 10; i++) expect(boneSourceForIndex(i, 0)).toBe('clip');
  });

  it('first poseCount walkers are pose, rest clip', () => {
    expect(boneSourceForIndex(0, 2)).toBe('pose');
    expect(boneSourceForIndex(1, 2)).toBe('pose');
    expect(boneSourceForIndex(2, 2)).toBe('clip');
    expect(boneSourceForIndex(7, 2)).toBe('clip');
  });

  it('clip + pose coexist in one batch (mixed roster)', () => {
    const roster = Array.from({ length: 5 }, (_, i) => boneSourceForIndex(i, 1));
    expect(roster).toContain('pose');
    expect(roster).toContain('clip');
  });

  it('only declares known sources', () => {
    for (let i = 0; i < 5; i++) expect(BONE_SOURCES).toContain(boneSourceForIndex(i, 2));
  });
});
