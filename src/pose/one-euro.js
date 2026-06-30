// One Euro filter (Casiez et al.) — low-latency adaptive smoothing. Smooths
// slow motion hard (kills jitter) but lets fast motion through (low lag). Pure,
// per-scalar; dt in seconds. T10.
function alpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

export class OneEuro {
  constructor({ minCutoff = 1.0, beta = 0.0, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0;
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = 0;
  }

  filter(x, dt) {
    if (this.xPrev === null || dt <= 0) {
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }
    const dx = (x - this.xPrev) / dt;
    const edx = this.dxPrev + alpha(this.dCutoff, dt) * (dx - this.dxPrev);
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    const a = alpha(cutoff, dt);
    const xHat = this.xPrev + a * (x - this.xPrev);
    this.xPrev = xHat;
    this.dxPrev = edx;
    return xHat;
  }
}

// Smooths a canonical observation's joints (x,y,z) over time with per-axis One
// Euro filters. Confidence passes through. Recomputes nothing else — the
// retargeter derives centers from joints.
export class CanonicalSmoother {
  constructor(numJoints, params = {}) {
    this.params = params;
    this.fx = [];
    this.fy = [];
    this.fz = [];
    for (let i = 0; i < numJoints; i += 1) {
      this.fx.push(new OneEuro(params));
      this.fy.push(new OneEuro(params));
      this.fz.push(new OneEuro(params));
    }
    this.prevT = null;
  }

  setParams({ minCutoff, beta }) {
    for (let i = 0; i < this.fx.length; i += 1) {
      this.fx[i].minCutoff = this.fy[i].minCutoff = this.fz[i].minCutoff = minCutoff;
      this.fx[i].beta = this.fy[i].beta = this.fz[i].beta = beta;
    }
  }

  reset() {
    for (let i = 0; i < this.fx.length; i += 1) {
      this.fx[i].reset();
      this.fy[i].reset();
      this.fz[i].reset();
    }
    this.prevT = null;
  }

  smooth(canon) {
    let dt = this.prevT === null ? 0 : (canon.timestampMs - this.prevT) / 1000;
    this.prevT = canon.timestampMs;
    // `!(dt > 0)` (not `dt <= 0`) so a NaN/undefined timestamp also falls back — NaN
    // comparisons are false, so the old guard let NaN dt through → NaN-poisoned output.
    if (!(dt > 0) || dt > 1) dt = 1 / 30; // first frame / stale / missing ts → ~30fps

    const joints = canon.joints.map((j, i) => ({
      x: this.fx[i].filter(j.x, dt),
      y: this.fy[i].filter(j.y, dt),
      z: this.fz[i].filter(j.z, dt),
      confidence: j.confidence
    }));
    return { ...canon, joints };
  }
}
