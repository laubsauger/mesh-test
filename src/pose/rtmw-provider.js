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

    const boxes = await this.detector.detect(this.video, vidW, vidH);
    if (boxes.length === 0) {
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

    const tensor = new ort.Tensor('float32', this.inputBuf, [1, 3, this.resH, this.resW]);
    const out = await this.session.run({ [RTMW3D_MODEL.inputName]: tensor });
    tensor.dispose();
    const { k2d, k3d } = decode3d(out, rect, this.resW, this.resH);
    for (const key in out) out[key].dispose();

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
