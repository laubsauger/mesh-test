// Worker-backed pose provider (T20). Same API as RTMWPoseProvider (start / infer /
// stop / timings) so the pose loop is unchanged — but inference runs in
// pose-worker.js (own GPU device, off the main thread). Webcam stays on the main
// thread (workers can't getUserMedia); frames cross as transferred ImageBitmaps.
import { startWebcam, stopWebcam } from './webcam.js';

export class WorkerPoseProvider {
  constructor({ ep, kptThresh = 0.3, yoloRes, rtmwVariant } = {}) {
    if (!ep) throw new Error('WorkerPoseProvider needs an explicit EP');
    this.ep = ep;
    this.kptThresh = kptThresh;
    this.yoloRes = yoloRes;
    this.rtmwVariant = rtmwVariant;
    this.detectEveryN = 1;
    this.video = null;
    this.stream = null;
    this.worker = null;
    this.running = false;
    this.latestFrame = null;
    this.timings = { detect: 0, preprocess: 0, inference: 0, decode: 0, total: 0 };
    this._pending = null;
  }

  async start() {
    const cam = await startWebcam();
    this.video = cam.video;
    this.stream = cam.stream;

    this.worker = new Worker(new URL('./pose-worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => this._onMessage(e.data);

    await new Promise((resolve, reject) => {
      this._readyResolve = resolve;
      this.worker.onerror = (err) => reject(new Error(`pose worker: ${err.message}`));
      this.worker.postMessage({ type: 'init', ep: this.ep, kptThresh: this.kptThresh, yoloRes: this.yoloRes, rtmwVariant: this.rtmwVariant });
    });
    this.running = true;
  }

  _onMessage(msg) {
    if (msg.type === 'ready') {
      this._readyResolve?.();
      this._readyResolve = null;
    } else if (msg.type === 'pose') {
      this.timings = msg.timings;
      this.latestFrame = msg.frame;
      const resolve = this._pending;
      this._pending = null;
      resolve?.(msg.frame);
    } else if (msg.type === 'error') {
      console.error('[pose-worker]', msg.message);
      const resolve = this._pending;
      this._pending = null;
      resolve?.(null);
    }
  }

  // Grab the current video frame, hand it to the worker, resolve when it returns.
  async infer() {
    if (!this.running || !this.video) return null;
    const vidW = this.video.videoWidth;
    const vidH = this.video.videoHeight;
    if (!vidW || !vidH) return null;
    if (this._pending) return null; // one in flight (newest-only)

    const bitmap = await createImageBitmap(this.video);
    return new Promise((resolve) => {
      this._pending = resolve;
      this.worker.postMessage(
        { type: 'frame', bitmap, detectEveryN: this.detectEveryN, timestampMs: performance.now() },
        [bitmap]
      );
    });
  }

  stop() {
    this.running = false;
    this._pending = null;
    if (this.worker) { this.worker.terminate(); this.worker = null; }
    stopWebcam(this.stream);
    this.video = null;
    this.stream = null;
  }
}
