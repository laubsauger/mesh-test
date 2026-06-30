import { describe, it, expect } from 'vitest';
import { bboxToRect, decode3d, letterboxRect } from '../src/pose/decode.js';
import { Z_RANGE, RTMW3D_MODEL } from '../src/pose/rtmw-constants.js';

const { resW, resH } = RTMW3D_MODEL; // 288 x 384

// Build a 1-keypoint SimCC tensor set with argmax at given bins + peak values.
function makeOut(bx, by, bz, vx, vy, vz) {
  const Wx = resW * 2; // 576 (split_ratio 2)
  const Hy = resH * 2; // 768
  const Wz = resW * 2; // 576
  const ax = new Float32Array(Wx); ax[bx] = vx;
  const ay = new Float32Array(Hy); ay[by] = vy;
  const az = new Float32Array(Wz); az[bz] = vz;
  return {
    [RTMW3D_MODEL.outX]: { data: ax, dims: [1, 1, Wx] },
    [RTMW3D_MODEL.outY]: { data: ay, dims: [1, 1, Hy] },
    [RTMW3D_MODEL.outZ]: { data: az, dims: [1, 1, Wz] }
  };
}

describe('bboxToRect — aspect pad + padding (V_pre)', () => {
  it('widens to model aspect then applies padding 1.25', () => {
    const rect = bboxToRect({ x: 0, y: 0, w: 100, h: 100 }, resW, resH, 1.25);
    // aspect 0.75: 100 > 100*0.75 → h = 100/0.75 = 133.33; ×1.25 → w 125, h 166.67
    expect(rect.sw).toBeCloseTo(125, 4);
    expect(rect.sh).toBeCloseTo(166.6667, 3);
    // centered on box center (50,50)
    expect(rect.sx).toBeCloseTo(50 - 62.5, 4);
    expect(rect.sy).toBeCloseTo(50 - 83.3333, 3);
  });
});

describe('letterboxRect — aspect-preserved fit + inverse map', () => {
  it('16:9 video into 384 square: fits width, centers vertically', () => {
    const lb = letterboxRect(1280, 720, 384);
    expect(lb.scale).toBeCloseTo(384 / 1280, 6); // longest side (width) fills
    expect(lb.drawW).toBeCloseTo(384, 4);
    expect(lb.offsetX).toBeCloseTo(0, 4);
    expect(lb.drawH).toBeCloseTo(216, 4);
    expect(lb.offsetY).toBeCloseTo(84, 4); // (384-216)/2
  });

  it('inverse maps a detection-space point back to source px', () => {
    const lb = letterboxRect(1280, 720, 384);
    // a point at the letterbox center maps to video center
    const xs = (192 - lb.offsetX) / lb.scale;
    const ys = (192 - lb.offsetY) / lb.scale;
    expect(xs).toBeCloseTo(640, 3);
    expect(ys).toBeCloseTo(360, 3);
  });
});

describe('decode3d — SimCC argmax → k2d/k3d (V22, faithful port)', () => {
  const rect = { sx: 100, sy: 50, sw: 288, sh: 384 };
  const out = makeOut(288, 384, 144, 2, 3, 9);
  const { k2d, k3d } = decode3d(out, rect, resW, resH);

  it('k2d maps model px back to source frame px', () => {
    expect(k2d[0].x).toBeCloseTo(244, 4); // 100 + (144/288)*288
    expect(k2d[0].y).toBeCloseTo(242, 4); // 50 + (192/384)*384
  });

  it('k3d is root-relative normalized, z scaled by Z_RANGE', () => {
    expect(k3d[0].x).toBeCloseTo(144 / 192, 6); // mx/(resH/2)
    expect(k3d[0].y).toBeCloseTo(192 / 192, 6);
    expect(k3d[0].z).toBeCloseTo((72 / 144 - 1) * Z_RANGE, 6);
  });

  it('score = mean of x,y maxima (z ignored)', () => {
    expect(k2d[0].score).toBeCloseTo(2.5, 6);
    expect(k3d[0].score).toBeCloseTo(2.5, 6);
  });
});
