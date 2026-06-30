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
    // Full instrumentation. Stage times: detect/preprocess/inference/decode = NATIVE
    // (sidecar). readback = browser GPU→CPU. serverTotal = whole sidecar handler
    // (recv→reply). round = send→reply-received. transport = round − serverTotal
    // (WS + the browser main-thread scheduling lag while it's busy rendering) — the
    // number to watch when "wire" looks big.
    this.timings = {
      detect: 0, preprocess: 0, inference: 0, decode: 0, total: 0,
      readback: 0, serverDecode: 0, serverTotal: 0, transport: 0, round: 0, wire: 0, bytes: 0
    };
    // Frame-flow counters — sent vs received reveals drops/desync at a glance.
    this.stats = { sent: 0, recv: 0, dropped: 0, stale: 0, timedOut: 0, sendHz: 0 };
    this.debug = false; // when true, console.logs a per-frame trace + periodic summary
    this._pending = null;
    this._pendingTimer = null;
    this._replyTimeoutMs = 3000; // a frame's reply must arrive within this or we drop it
    this._lastSentAt = 0;
    this._canvas = null;
    this._ctx = null;
  }

  // Resolve the single in-flight request exactly once, clearing its timeout. Used
  // by the reply path, the timeout, and a mid-session disconnect — so a lost reply
  // (busy box, killed sidecar) drops one frame cleanly instead of deadlocking.
  _settle(frame) {
    if (this._pendingTimer) { clearTimeout(this._pendingTimer); this._pendingTimer = null; }
    const resolve = this._pending;
    this._pending = null;
    this.latestFrame = frame;
    resolve?.(frame);
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
      this.ws.onclose = () => { if (!this.running) fail(); else this._settle(null); };
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
      // Stale-reply guard (the sidecar echoes the frame's send-timestamp): if this
      // reply is for a frame that already timed out — so we moved on and a NEWER
      // frame is now in flight — its keypoints + _sentW/H are wrong for the current
      // slot. Drop it; the in-flight frame's own reply will arrive. (Prevents the
      // timeout from cross-wiring one frame's pose onto the next → "desync".)
      if (msg.timestampMs !== this._sentAt) { this.stats.stale += 1; return; }
      const native = msg.timings;
      const round = performance.now() - this._sentAt; // full send→reply
      const serverTotal = native.serverTotal ?? native.total ?? 0;
      const transport = Math.max(0, round - serverTotal); // WS + browser scheduling lag
      this.timings = {
        detect: native.detect, preprocess: native.preprocess, inference: native.inference,
        decode: native.decode, total: native.total,
        readback: this._readbackMs, serverDecode: native.serverDecode ?? 0,
        serverTotal, transport, round, wire: transport, bytes: native.bytes ?? this._bytesSent
      };
      this.ep = msg.ep || this.ep;
      if (msg.frame) this.stats.recv += 1; else this.stats.dropped += 1;
      if (this.debug) {
        const tm = this.timings;
        console.log(`[sidecar] f#${this.stats.sent} ${msg.frame ? 'pose' : 'NULL'} | readback ${tm.readback.toFixed(1)} → wire ${(round - serverTotal).toFixed(1)} → server ${serverTotal.toFixed(1)} (decode ${(native.serverDecode ?? 0).toFixed(1)} det ${native.detect.toFixed(1)} inf ${native.inference.toFixed(1)}) | round ${round.toFixed(1)}ms | ${(tm.bytes / 1024) | 0}KB | sent ${this.stats.sent} recv ${this.stats.recv} drop ${this.stats.dropped} stale ${this.stats.stale} TO ${this.stats.timedOut}`);
      }
      this._settle(msg.frame ? this._toVideoSpace(msg.frame) : null);
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
    // Always detect every frame on the sidecar (= 1, NOT this.detectEveryN). yolo
    // is a few ms natively and runs off the renderer's device, so there's no reason
    // to reuse a stale box — which, under load, misaligns the crop and feeds rtmw a
    // half-cropped person → the intermittent garbage/glitch. (detectEveryN only
    // exists to spare the webgpu worker's GPU contention; the sidecar has none.)
    dv.setUint32(16, 1, true);
    buf.set(rgba, HEADER_BYTES);

    this._sentW = w; this._sentH = h; this._sentAt = ts;
    this._bytesSent = buf.length;
    this.stats.sent += 1;
    if (this._lastSentAt) this.stats.sendHz = 1000 / Math.max(1, ts - this._lastSentAt);
    this._lastSentAt = ts;
    return new Promise((resolve) => {
      this._pending = resolve;
      // Drop the frame if no reply lands in time — never deadlock the loop.
      this._pendingTimer = setTimeout(() => {
        this.stats.timedOut += 1;
        console.warn(`[sidecar] reply timeout (${this._replyTimeoutMs}ms) — dropping frame; total timeouts ${this.stats.timedOut}`);
        this._settle(null);
      }, this._replyTimeoutMs);
      try {
        this.ws.send(buf);
      } catch {
        this._settle(null); // send failed (socket closing) → drop, recover
      }
    });
  }

  stop() {
    this.running = false;
    this._pending = null;
    if (this._pendingTimer) { clearTimeout(this._pendingTimer); this._pendingTimer = null; }
    if (this.ws) { try { this.ws.close(); } catch { /* already closed */ } this.ws = null; }
    stopWebcam(this.stream);
    this.video = null;
    this.stream = null;
  }
}
