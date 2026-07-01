// FaceExpressionExtractor (T26 / V26): derive a small set of expression scalars
// from the RTMW3D face landmarks (COCO-WholeBody idx 23-90 = iBUG-68, +23 offset).
//
// DETERMINISTIC 2D GEOMETRY, no model (V26 / global "code answers"). Every metric
// is a RATIO (mouth/eye aspect, corner width, brow gap) normalized by the
// inter-ocular span → scale-invariant and mostly rotation-invariant, so it reads
// EXPRESSION not head size/distance. Monocular z is unreliable (whole codebase
// fights it) → we use x,y only.
//
// Output `FaceExpression {jawOpen, smile, pucker, blinkL, blinkR, browL, browR}`,
// each 0..1 = deviation from a NEUTRAL baseline (captured once via
// recalibrateNeutral(), like the retargeter's recalibrateFacing). Per-scalar One
// Euro smoothing (V25 pattern). Pure — no three/DOM, unit-testable.
import { OneEuro } from './one-euro.js';

// iBUG-68 face landmarks are RTMW indices 23..90. Constants below are the iBUG
// (0-based) ids; FACE_OFFSET maps them into the 133-keypoint canonical array.
export const FACE_OFFSET = 23;
const F = (i) => i + FACE_OFFSET;

// Semantic landmark groups (iBUG-68 layout).
const FACE = {
  // outer eye corners → scale reference (inter-ocular span, stable).
  eyeOuterR: F(36), eyeOuterL: F(45),
  // Right eye ring p1..p6 (EAR): outer, top×2, inner, bottom×2.
  eyeR: [F(36), F(37), F(38), F(39), F(40), F(41)],
  // Left eye ring p1..p6: inner, top×2, outer, bottom×2.
  eyeL: [F(42), F(43), F(44), F(45), F(46), F(47)],
  // Mouth outer corners (width) + inner lip ring (aperture).
  mouthCornerL: F(48), mouthCornerR: F(54),
  innerTop: [F(61), F(62), F(63)], innerBot: [F(67), F(66), F(65)],
  innerCornerL: F(60), innerCornerR: F(64),
  // Brow centers + eye centers (brow-raise gap).
  browR: F(19), browL: F(24)
};

const dist2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mean = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const median = (arr) => {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

// Eye-aspect-ratio from a 6-point ring: (|p2-p6|+|p3-p5|) / (2|p1-p4|). Open ≈ 0.3,
// closed ≈ 0.1 → a blink is a DROP below the neutral open value.
function ear(j, ring) {
  const p = ring.map((i) => j[i]);
  const w = dist2(p[0], p[3]);
  if (w < 1e-6) return 0;
  return (dist2(p[1], p[5]) + dist2(p[2], p[4])) / (2 * w);
}

function eyeCenter(j, ring) {
  return { x: mean(ring.map((i) => j[i].x)), y: mean(ring.map((i) => j[i].y)) };
}

// Raw (pre-neutral) geometric metrics for one frame. All ratios (iod-normalized
// where a length, not already a ratio). Returns null if the face points aren't
// confident enough (→ caller holds last).
function rawMetrics(joints, thresh) {
  const need = [FACE.eyeOuterR, FACE.eyeOuterL, FACE.mouthCornerL, FACE.mouthCornerR,
    ...FACE.eyeR, ...FACE.eyeL, ...FACE.innerTop, ...FACE.innerBot,
    FACE.innerCornerL, FACE.innerCornerR, FACE.browR, FACE.browL];
  for (const i of need) {
    const p = joints[i];
    if (!p || p.confidence < thresh) return null;
  }
  const iod = dist2(joints[FACE.eyeOuterR], joints[FACE.eyeOuterL]);
  if (iod < 1e-6) return null;

  // Mouth aspect: mean inner-lip vertical gap / inner-lip width (a ratio already).
  const mw = dist2(joints[FACE.innerCornerL], joints[FACE.innerCornerR]);
  const gap = mean(FACE.innerTop.map((t, k) => dist2(joints[t], joints[FACE.innerBot[k]])));
  const mar = mw > 1e-6 ? gap / mw : 0;

  // Mouth width (smile/pucker axis) normalized by iod.
  const mouthWidth = dist2(joints[FACE.mouthCornerL], joints[FACE.mouthCornerR]) / iod;

  const earR = ear(joints, FACE.eyeR);
  const earL = ear(joints, FACE.eyeL);

  // Brow-raise gap: brow center → eye center vertical distance / iod.
  const browGapR = Math.abs(joints[FACE.browR].y - eyeCenter(joints, FACE.eyeR).y) / iod;
  const browGapL = Math.abs(joints[FACE.browL].y - eyeCenter(joints, FACE.eyeL).y) / iod;

  return { mar, mouthWidth, earR, earL, browGapR, browGapL };
}

const CALIB_FRAMES = 15; // ~0.5s of neutral samples → median baseline (noise-robust)

export class FaceExpressionExtractor {
  // Gains map a raw ratio DEVIATION from neutral into 0..1. Tunable; defaults are
  // sane starting points (refine against live webcam in the Inspector).
  constructor({ jawGain = 3, smileGain = 6, puckerGain = 6, browGain = 10,
    oneEuro = { minCutoff: 1.5, beta: 0.05 } } = {}) {
    this.gains = { jawGain, smileGain, puckerGain, browGain };
    this.neutral = null;
    this._calib = [];      // rolling neutral samples while (re)calibrating
    this._calibN = CALIB_FRAMES; // >0 → still capturing baseline
    this.value = { jawOpen: 0, smile: 0, pucker: 0, blinkL: 0, blinkR: 0, browL: 0, browR: 0 };
    this.filters = {};
    for (const k of Object.keys(this.value)) this.filters[k] = new OneEuro(oneEuro);
  }

  // Start (re)capturing the neutral baseline from the next CALIB_FRAMES confident
  // frames. Call on a "rest face" calibration hold (beside recalibrateFacing).
  recalibrateNeutral() {
    this.neutral = null;
    this._calib = [];
    this._calibN = CALIB_FRAMES;
  }

  // canon = CanonicalPoseObservation (joints indexed by COCO-WholeBody id). dt in
  // seconds. Returns the smoothed FaceExpression (also stored on .value). Low face
  // confidence or missing neutral → holds the last value.
  update(canon, dt, { thresh = 0.3 } = {}) {
    const m = rawMetrics(canon.joints, thresh);
    if (!m) return this.value; // face not confident → hold last

    // Auto/explicit neutral capture: median over a short window so a single noisy
    // frame can't skew the rest baseline (jawOpen etc. must read ~0 at rest).
    if (this._calibN > 0) {
      this._calib.push(m);
      this._calibN -= 1;
      if (this._calibN === 0) {
        this.neutral = {};
        for (const key of Object.keys(m)) this.neutral[key] = median(this._calib.map((s) => s[key]));
        this._calib = [];
      }
      return this.value; // output stays 0 until neutral is set
    }
    if (!this.neutral) return this.value;

    const n = this.neutral;
    const g = this.gains;
    // Signed deviations from neutral → 0..1 expression scalars. Blink is a DROP in
    // EAR (eye closes) relative to neutral-open; jaw/brow/smile are increases.
    const raw = {
      jawOpen: clamp01((m.mar - n.mar) * g.jawGain),
      smile: clamp01((m.mouthWidth - n.mouthWidth) * g.smileGain),
      pucker: clamp01((n.mouthWidth - m.mouthWidth) * g.puckerGain),
      blinkR: clamp01(n.earR > 1e-6 ? (n.earR - m.earR) / n.earR : 0),
      blinkL: clamp01(n.earL > 1e-6 ? (n.earL - m.earL) / n.earL : 0),
      browR: clamp01((m.browGapR - n.browGapR) * g.browGain),
      browL: clamp01((m.browGapL - n.browGapL) * g.browGain)
    };
    for (const k of Object.keys(raw)) this.value[k] = this.filters[k].filter(raw[k], dt > 0 ? dt : 1 / 30);
    return this.value;
  }

  reset() {
    for (const k of Object.keys(this.filters)) this.filters[k].reset();
    this.value = { jawOpen: 0, smile: 0, pucker: 0, blinkL: 0, blinkR: 0, browL: 0, browR: 0 };
  }
}
