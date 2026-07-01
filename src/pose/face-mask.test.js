// T27 (V27, V28): face-mask generation is mesh-derived + region-correct, the .bin
// round-trips, and the vertexCount guard fails loud on a stale sidecar.
import { describe, it, expect } from 'vitest';
import { generateFaceMask, encodeFaceMask, decodeFaceMask, assertMaskFits, FACE_REGIONS } from './face-mask.js';

const HEAD = 5; // head bone index for the synthetic rig

// Build a synthetic head: 8 AABB corners (pin the box to x,z∈[-0.5,0.5], y∈[0,1]) plus
// named feature verts. All bound to the Head bone except one non-head vert.
function synthetic() {
  const verts = [];
  const idx = {};
  const add = (name, x, y, z, bone = HEAD) => { idx[name] = verts.length; verts.push({ x, y, z, bone }); };
  for (const x of [-0.5, 0.5]) for (const y of [0, 1]) for (const z of [-0.5, 0.5]) add(`corner${verts.length}`, x, y, z);
  add('chin', 0, 0.05, 0.45);
  add('mouth', 0, 0.29, 0.45);
  add('corner_mouthL', 0.42, 0.30, 0.42);
  add('eyeL', 0.25, 0.56, 0.45);
  add('eyeR', -0.25, 0.56, 0.45);
  add('browL', 0.25, 0.68, 0.45);
  add('crown', 0, 1.0, 0.0);
  add('backHead', 0, 0.30, -0.48);
  add('notHead', 0, 0.30, 0.45, 2); // bound to a different bone → excluded

  const count = verts.length;
  const positions = new Float32Array(count * 3);
  const skinIndices = new Uint16Array(count * 4);
  const skinWeights = new Float32Array(count * 4);
  verts.forEach((v, i) => {
    positions[i * 3] = v.x; positions[i * 3 + 1] = v.y; positions[i * 3 + 2] = v.z;
    skinIndices[i * 4] = v.bone; skinWeights[i * 4] = 1;
  });
  return { positions, skinIndices, skinWeights, count, idx };
}

const w = (mask, region, i) => mask.data[FACE_REGIONS.indexOf(region) * mask.vertexCount + i];

describe('generateFaceMask', () => {
  const s = synthetic();
  const mask = generateFaceMask({ ...s, headBoneIndex: HEAD });

  it('tags the chin into the jaw region, not the crown', () => {
    expect(w(mask, 'jaw', s.idx.chin)).toBeGreaterThan(120);
    expect(w(mask, 'jaw', s.idx.crown)).toBe(0);
  });

  it('splits eyelids by side (eyeL→upperLidL, not upperLidR)', () => {
    expect(w(mask, 'upperLidL', s.idx.eyeL)).toBeGreaterThan(120);
    expect(w(mask, 'upperLidR', s.idx.eyeL)).toBe(0);
    expect(w(mask, 'upperLidR', s.idx.eyeR)).toBeGreaterThan(120);
  });

  it('puts the mouth corner in mouthCorner, the center in lowerLip', () => {
    expect(w(mask, 'mouthCorner', s.idx.corner_mouthL)).toBeGreaterThan(w(mask, 'lowerLip', s.idx.corner_mouthL));
    expect(w(mask, 'lowerLip', s.idx.mouth)).toBeGreaterThan(w(mask, 'mouthCorner', s.idx.mouth));
  });

  it('never deforms the back of the head or non-head verts', () => {
    for (let r = 0; r < FACE_REGIONS.length; r += 1) {
      expect(w(mask, FACE_REGIONS[r], s.idx.backHead)).toBe(0);
      expect(w(mask, FACE_REGIONS[r], s.idx.notHead)).toBe(0);
    }
  });

  it('throws on a wrong head bone index', () => {
    expect(() => generateFaceMask({ ...s, headBoneIndex: 99 })).toThrow(/no head-weighted/);
  });
});

describe('encode/decode + guard', () => {
  const s = synthetic();
  const mask = generateFaceMask({ ...s, headBoneIndex: HEAD });

  it('round-trips through the .bin format', () => {
    const back = decodeFaceMask(encodeFaceMask(mask));
    expect(back.vertexCount).toBe(mask.vertexCount);
    expect(back.headHeight).toBeCloseTo(mask.headHeight, 5);
    expect(Array.from(back.data)).toEqual(Array.from(mask.data));
    // per-region config (v2) round-trips
    expect(back.config.length).toBe(mask.config.length);
    expect(back.config[0].type).toBe(mask.config[0].type); // jaw = hinge
    expect(back.config[0].driver).toBe(mask.config[0].driver);
    expect(back.config[2].mirrorX).toBe(mask.config[2].mirrorX); // mouthCorner mirrored
    expect(back.config[1].amount).toBeCloseTo(mask.config[1].amount, 5);
    back.hinge.origin.forEach((v, k) => expect(v).toBeCloseTo(mask.hinge.origin[k], 5)); // f32 round-trip
    back.hinge.axis.forEach((v, k) => expect(v).toBeCloseTo(mask.hinge.axis[k], 5));
  });

  it('assertMaskFits throws on a vertexCount mismatch (stale sidecar)', () => {
    expect(() => assertMaskFits(mask, mask.vertexCount + 1)).toThrow(/vertexCount/);
    expect(assertMaskFits(mask, mask.vertexCount)).toBe(mask);
  });

  it('decode throws on bad magic', () => {
    const bad = new ArrayBuffer(64);
    expect(() => decodeFaceMask(bad)).toThrow(/bad magic/);
  });
});
