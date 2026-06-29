// yolo26n person detector (top-down stage). Runs on the same EP as the pose
// model. Returns person boxes in un-mirrored source px, score-sorted desc.
// Faithful port of I.reference detPersonBoxes.
import { ort, createPoseSession } from './ort-session.js';
import { YOLO_DET_MODEL } from './rtmw-constants.js';
import { assetUrl } from '../asset-url.js';

export class PersonDetector {
  constructor(ep, threshold = 0.3) {
    this.ep = ep;
    this.threshold = threshold;
    this.res = YOLO_DET_MODEL.res;
    this.session = null;
    this.buf = null;
    this.canvas = null;
    this.ctx = null;
  }

  async load() {
    const res = await fetch(assetUrl(YOLO_DET_MODEL.url));
    if (!res.ok) throw new Error(`detector fetch ${res.status}: ${YOLO_DET_MODEL.url}`);
    this.session = await createPoseSession(await res.arrayBuffer(), this.ep);
    this.buf = new Float32Array(3 * this.res * this.res);
    this.canvas = new OffscreenCanvas(this.res, this.res);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
  }

  async detect(video, vidW, vidH) {
    const R = this.res;
    this.ctx.drawImage(video, 0, 0, R, R);
    const d = this.ctx.getImageData(0, 0, R, R).data;
    const hw = R * R;
    for (let i = 0; i < hw; i += 1) {
      this.buf[i] = d[i * 4] / 255;
      this.buf[hw + i] = d[i * 4 + 1] / 255;
      this.buf[2 * hw + i] = d[i * 4 + 2] / 255;
    }
    const t = new ort.Tensor('float32', this.buf, [1, 3, R, R]);
    const out = await this.session.run({ [YOLO_DET_MODEL.inputName]: t });
    t.dispose();

    const o0 = out.output0;
    const data = o0.data;
    const stride = o0.dims[2];
    const boxes = [];
    for (let q = 0; q < data.length / stride; q += 1) {
      const o = q * stride;
      const score = data[o + 4];
      if (score < this.threshold) break; // rows score-sorted desc → done
      if (Math.round(data[o + 5]) !== YOLO_DET_MODEL.personClassId) continue;
      const x1 = (data[o] / R) * vidW;
      const y1 = (data[o + 1] / R) * vidH;
      const x2 = (data[o + 2] / R) * vidW;
      const y2 = (data[o + 3] / R) * vidH;
      boxes.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1, score });
    }
    for (const k in out) out[k].dispose();
    boxes.sort((a, b) => b.score - a.score);
    return boxes;
  }

  dispose() {
    this.session?.release?.();
    this.session = null;
  }
}
