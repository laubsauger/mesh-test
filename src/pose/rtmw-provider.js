// RTMWPoseProvider (T5): webcam → yolo26n detect → affine crop → rtmw3d →
// decode3d → RTMWPoseFrame. Single performer (v1): top-scoring person box.
// Faithful port of I.reference; the EP is explicit (state.poseEP), no fallback.
import { ort, createPoseSession } from './ort-session.js';
import { RTMW3D_MODEL, POSE_MEAN, POSE_STD } from './rtmw-constants.js';
import { assetUrl } from '../asset-url.js';
import { bboxToRect, decode3d } from './decode.js';
import { PersonDetector } from './detector.js';
import { startWebcam, stopWebcam } from './webcam.js';

export class RTMWPoseProvider {
  constructor({ ep, kptThresh = 0.3 } = {}) {
    if (!ep) throw new Error('RTMWPoseProvider needs an explicit EP (state.poseEP)');
    this.ep = ep;
    this.kptThresh = kptThresh;
    this.resW = RTMW3D_MODEL.resW;
    this.resH = RTMW3D_MODEL.resH;
    this.session = null;
    this.detector = null;
    this.video = null;
    this.stream = null;
    this.inputBuf = new Float32Array(3 * this.resW * this.resH);
    this.canvas = null;
    this.ctx = null;
    this.latestFrame = null;
    this.running = false;
    this.timings = { detect: 0, preprocess: 0, inference: 0, decode: 0, total: 0 };
    this.detectEveryN = 1; // run yolo every N frames; reuse last box between (person barely moves)
    this._frameCount = 0;
    this._lastBoxes = null;
  }

  async start() {
    const cam = await startWebcam();
    this.video = cam.video;
    this.stream = cam.stream;

    const res = await fetch(assetUrl(RTMW3D_MODEL.url));
    if (!res.ok) throw new Error(`rtmw3d fetch ${res.status}: ${RTMW3D_MODEL.url}`);
    this.session = await createPoseSession(await res.arrayBuffer(), this.ep);

    this.detector = new PersonDetector(this.ep, this.kptThresh);
    await this.detector.load();

    this.canvas = new OffscreenCanvas(this.resW, this.resH);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.running = true;
  }

  // Run one inference on the current video frame → RTMWPoseFrame | null.
  async infer() {
    if (!this.running || !this.video) return null;
    const vidW = this.video.videoWidth;
    const vidH = this.video.videoHeight;
    if (!vidW || !vidH) return null;

    const t0 = performance.now();
    this._frameCount += 1;
    let boxes;
    if (this.detectEveryN <= 1 || !this._lastBoxes || this._frameCount % this.detectEveryN === 0) {
      boxes = await this.detector.detect(this.video, vidW, vidH);
      this._lastBoxes = boxes;
    } else {
      boxes = this._lastBoxes; // reuse last box (padding 1.25 absorbs small motion)
    }
    const t1 = performance.now();
    if (boxes.length === 0) {
      this.timings = { detect: t1 - t0, preprocess: 0, inference: 0, decode: 0, total: t1 - t0 };
      this.latestFrame = null;
      return null;
    }

    const box = boxes[0];
    const rect = bboxToRect(box, this.resW, this.resH);
    this.ctx.clearRect(0, 0, this.resW, this.resH);
    this.ctx.drawImage(this.video, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, this.resW, this.resH);
    const d = this.ctx.getImageData(0, 0, this.resW, this.resH).data;
    const hw = this.resW * this.resH;
    for (let i = 0; i < hw; i += 1) {
      this.inputBuf[i] = (d[i * 4] - POSE_MEAN[0]) / POSE_STD[0];
      this.inputBuf[hw + i] = (d[i * 4 + 1] - POSE_MEAN[1]) / POSE_STD[1];
      this.inputBuf[2 * hw + i] = (d[i * 4 + 2] - POSE_MEAN[2]) / POSE_STD[2];
    }
    const t2 = performance.now();

    const tensor = new ort.Tensor('float32', this.inputBuf, [1, 3, this.resH, this.resW]);
    const out = await this.session.run({ [RTMW3D_MODEL.inputName]: tensor });
    tensor.dispose();
    const t3 = performance.now();
    const { k2d, k3d } = decode3d(out, rect, this.resW, this.resH);
    for (const key in out) out[key].dispose();
    const t4 = performance.now();
    this.timings = { detect: t1 - t0, preprocess: t2 - t1, inference: t3 - t2, decode: t4 - t3, total: t4 - t0 };

    const frame = {
      timestampMs: performance.now(),
      keypoints2D: k2d.map((p) => ({ x: p.x, y: p.y, confidence: p.score })),
      keypoints3D: k3d.map((p) => ({ x: p.x, y: p.y, z: p.z, confidence: p.score })),
      boundingBox: { x: box.x, y: box.y, width: box.w, height: box.h, confidence: box.score }
    };
    this.latestFrame = frame;
    return frame;
  }

  stop() {
    this.running = false;
    stopWebcam(this.stream);
    this.detector?.dispose();
    this.session?.release?.();
    this.session = null;
    this.detector = null;
    this.video = null;
    this.stream = null;
  }
}
