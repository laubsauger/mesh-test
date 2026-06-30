// Sidecar pose provider — same API as WorkerPoseProvider (start/infer/stop/
// timings/latestFrame), but inference runs in the NATIVE Python sidecar
// (sidecar/pose_sidecar.py) over a localhost WebSocket, so the 4090 box can use
// the TensorRT/CUDA EPs that onnxruntime-web can't reach.
//
// Transport: the browser owns the webcam, downscales each frame, reads back RAW
// RGBA (no codec, no pre-shaped tensor — the pose crop depends on the detector
// result computed IN the sidecar), and ships header+pixels as one binary message.
// Sidecar returns keypoints (tiny JSON). On loopback the ~0.4MB frame is
// readback-dominated, sub-ms wire (see the design note from the user).
import { startWebcam, stopWebcam } from './webcam.js';

const HEADER_BYTES = 20; // ts:f64(8) | w:u32(4) | h:u32(4) | detectEveryN:u32(4)

export class SidecarPoseProvider {
  constructor({ kptThresh = 0.3, url = 'ws://127.0.0.1:8787', sendMaxSide = 512 } = {}) {
    this.kptThresh = kptThresh;
    this.url = url;
    this.sendMaxSide = sendMaxSide; // longest side of the downscaled frame we ship
    this.detectEveryN = 1;
    this.video = null;
    this.stream = null;
    this.ws = null;
    this.running = false;
    this.latestFrame = null;
    this.ep = 'sidecar';
    // detect/preprocess/inference/decode = NATIVE stage times (from sidecar).
    // readback/wire = browser-side transport cost (the benchmark lever).
    this.timings = { detect: 0, preprocess: 0, inference: 0, decode: 0, total: 0, readback: 0, wire: 0 };
    this._pending = null;
    this._canvas = null;
    this._ctx = null;
  }

  async start() {
    const cam = await startWebcam();
    this.video = cam.video;
    this.stream = cam.stream;
    this._canvas = document.createElement('canvas');
    this._ctx = this._canvas.getContext('2d', { willReadFrequently: true });

    this.ws = new WebSocket(this.url);
    this.ws.binaryType = 'arraybuffer';
    await new Promise((resolve, reject) => {
      const fail = (e) => reject(new Error(`sidecar ws ${this.url}: ${e?.message || 'connection failed — is `npm run sidecar` running?'}`));
      this.ws.onerror = fail;
      this.ws.onclose = () => { if (!this.running) fail(); };
      this.ws.onopen = () => {
        this._readyResolve = resolve;
        this.ws.send(JSON.stringify({ type: 'config', kptThresh: this.kptThresh }));
      };
      this.ws.onmessage = (e) => this._onMessage(e.data);
    });
    this.running = true;
  }

  _onMessage(data) {
    const msg = JSON.parse(data);
    if (msg.type === 'ready') {
      this.ep = msg.ep || 'sidecar';
      this._readyResolve?.();
      this._readyResolve = null;
      return;
    }
    if (msg.type === 'pose') {
      const native = msg.timings;
      const t = performance.now();
      const round = t - this._sentAt; // full request→response
      // wire = round-trip minus the work the sidecar reported doing on it.
      this.timings = {
        detect: native.detect, preprocess: native.preprocess, inference: native.inference,
        decode: native.decode, total: native.total,
        readback: this._readbackMs, wire: Math.max(0, round - native.total)
      };
      this.ep = msg.ep || this.ep;
      this.latestFrame = msg.frame ? this._toVideoSpace(msg.frame) : null;
      const resolve = this._pending;
      this._pending = null;
      resolve?.(this.latestFrame);
    }
  }

  // k2d comes back in DOWNSCALED-frame px; scale to video px so keypoints2D match
  // the worker's contract (overlay is sized to video). k3d is normalized → as-is.
  _toVideoSpace(frame) {
    const fx = (this.video.videoWidth || 1) / this._sentW;
    const fy = (this.video.videoHeight || 1) / this._sentH;
    return {
      ...frame,
      keypoints2D: frame.keypoints2D.map((p) => ({ x: p.x * fx, y: p.y * fy, confidence: p.confidence }))
    };
  }

  async infer() {
    if (!this.running || !this.video || this.ws?.readyState !== WebSocket.OPEN) return null;
    const vidW = this.video.videoWidth;
    const vidH = this.video.videoHeight;
    if (!vidW || !vidH) return null;
    if (this._pending) return null; // one in flight (newest-only)

    // Downscale to sendMaxSide longest edge — ship only what the models need.
    const scale = Math.min(1, this.sendMaxSide / Math.max(vidW, vidH));
    const w = Math.round(vidW * scale);
    const h = Math.round(vidH * scale);
    if (this._canvas.width !== w) { this._canvas.width = w; this._canvas.height = h; }

    const rb0 = performance.now();
    this._ctx.drawImage(this.video, 0, 0, w, h);
    const rgba = this._ctx.getImageData(0, 0, w, h).data; // GPU→CPU readback (the cost)
    this._readbackMs = performance.now() - rb0;

    const buf = new Uint8Array(HEADER_BYTES + rgba.length);
    const dv = new DataView(buf.buffer);
    const ts = performance.now();
    dv.setFloat64(0, ts, true);
    dv.setUint32(8, w, true);
    dv.setUint32(12, h, true);
    dv.setUint32(16, this.detectEveryN, true);
    buf.set(rgba, HEADER_BYTES);

    this._sentW = w; this._sentH = h; this._sentAt = ts;
    return new Promise((resolve) => {
      this._pending = resolve;
      this.ws.send(buf);
    });
  }

  stop() {
    this.running = false;
    this._pending = null;
    if (this.ws) { try { this.ws.close(); } catch { /* already closed */ } this.ws = null; }
    stopWebcam(this.stream);
    this.video = null;
    this.stream = null;
  }
}
