// Sidecar pose provider — same API as WorkerPoseProvider (start/infer/stop/
// timings/latestFrame/stats), but inference runs in the NATIVE Python sidecar
// (TensorRT/CUDA/CoreML) reached over a localhost WebSocket.
//
// The WS + frame readback live in a Web Worker (sidecar-ws-worker.js), OFF the
// render-contended main thread — that's what removed the ~57ms transport lag the
// instrumentation exposed (the reply was waiting behind a render frame). Here the
// main thread only snapshots the webcam (createImageBitmap) and consumes keypoints;
// the round-trip is handled on the worker thread. Critical on a TRT box where
// inference is ~10ms and transport would otherwise dominate.
import { startWebcam, stopWebcam } from './webcam.js';

export class SidecarPoseProvider {
  constructor({ kptThresh = 0.3, url = 'ws://127.0.0.1:8787', sendMaxSide = 1280, readback = 'bitmap' } = {}) {
    this.kptThresh = kptThresh;
    this.url = url;
    this.sendMaxSide = sendMaxSide; // longest edge of the downscaled frame shipped
    this.readback = readback; // 'bitmap' = createImageBitmap | 'videoframe' = WebCodecs VideoFrame
    this.detectEveryN = 1; // ignored by the sidecar (it always detects); kept for API parity
    this.video = null;
    this.stream = null;
    this.worker = null;
    this.running = false;
    this.latestFrame = null;
    this.ep = 'sidecar';
    // Full breakdown (filled from the worker): NATIVE stages + readback + serverTotal
    // + transport (round−server, now the TRUE WS cost off the main thread) + round.
    this.timings = {
      capture: 0, detect: 0, preprocess: 0, inference: 0, decode: 0, total: 0,
      readback: 0, serverDecode: 0, serverTotal: 0, transport: 0, round: 0, wire: 0, bytes: 0
    };
    this.stats = { sent: 0, recv: 0, dropped: 0, stale: 0, timedOut: 0, sendHz: 0 };
    this.debug = false;
    this._pending = null;
    this._pendingTimer = null;
    this._replyTimeoutMs = 3000;
    this._seq = 0;
    this._inflightSeq = -1;
    this._lastSentAt = 0;
    this._lastCaptureMs = 0; // main-thread frame snapshot cost (createImageBitmap vs new VideoFrame)
  }

  async start() {
    const cam = await startWebcam();
    this.video = cam.video;
    this.stream = cam.stream;
    this.worker = new Worker(new URL('./sidecar-ws-worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => this._onWorker(e.data);
    await new Promise((resolve, reject) => {
      this._readyResolve = resolve;
      this._readyReject = reject;
      this.worker.onerror = (err) => reject(new Error(`sidecar worker: ${err.message}`));
      this.worker.postMessage({ type: 'init', url: this.url, kptThresh: this.kptThresh, sendMaxSide: this.sendMaxSide });
    });
    this.running = true;
  }

  // Resolve the single in-flight request once, clearing its timeout — so a lost
  // reply / killed sidecar drops one frame cleanly instead of deadlocking.
  _settle(frame) {
    if (this._pendingTimer) { clearTimeout(this._pendingTimer); this._pendingTimer = null; }
    const resolve = this._pending;
    this._pending = null;
    this.latestFrame = frame;
    resolve?.(frame);
  }

  _onWorker(msg) {
    if (msg.type === 'inited') {
      this.ep = msg.ep || 'sidecar';
      this._readyResolve?.(); this._readyResolve = null; this._readyReject = null;
      return;
    }
    if (msg.type === 'error') {
      if (this._readyReject) { this._readyReject(new Error(msg.message)); this._readyReject = null; this._readyResolve = null; }
      else { console.error('[sidecar]', msg.message); }
      this._settle(null);
      return;
    }
    if (msg.type === 'closed') {
      if (this.running) this._settle(null);
      return;
    }
    if (msg.type === 'pose') {
      if (msg.seq !== this._inflightSeq) { this.stats.stale += 1; return; } // stale → ignore
      this.timings = msg.timings;
      this.timings.capture = this._lastCaptureMs; // main-thread snapshot cost (the A/B lever)
      this.ep = msg.ep || this.ep;
      if (msg.frame) this.stats.recv += 1; else this.stats.dropped += 1;
      if (this.debug) {
        const t = msg.timings;
        console.log(`[sidecar] f#${this.stats.sent} ${msg.frame ? 'pose' : 'NULL'} ${this.readback} | capture ${t.capture.toFixed(1)} readback ${t.readback.toFixed(1)} → wire ${t.transport.toFixed(1)} → server ${t.serverTotal.toFixed(1)} (decode ${t.serverDecode.toFixed(1)} det ${t.detect.toFixed(1)} inf ${t.inference.toFixed(1)}) | round ${t.round.toFixed(1)}ms | ${(t.bytes / 1024) | 0}KB | sent ${this.stats.sent} recv ${this.stats.recv} drop ${this.stats.dropped} stale ${this.stats.stale} TO ${this.stats.timedOut}`);
      }
      this._settle(msg.frame);
    }
  }

  async infer() {
    if (!this.running || !this.video || !this.worker) return null;
    const vidW = this.video.videoWidth;
    const vidH = this.video.videoHeight;
    if (!vidW || !vidH) return null;
    if (this._pending) return null; // one in flight (newest-only)

    // Frame snapshot. 'videoframe' = WebCodecs `new VideoFrame(video)`: synchronous,
    // just wraps the GPU frame (no decode/copy) — cheaper than createImageBitmap, which
    // decodes+copies on the main thread. Both are transferable + valid drawImage sources,
    // so the worker readback (drawImage→getImageData at 640) is identical either way.
    let source;
    const c0 = performance.now();
    try {
      if (this.readback === 'videoframe' && typeof VideoFrame !== 'undefined') {
        source = new VideoFrame(this.video);
      } else {
        source = await createImageBitmap(this.video);
      }
    } catch { return null; }
    this._lastCaptureMs = performance.now() - c0;
    if (!this.running || this._pending || !this.worker) { source.close?.(); return null; } // raced with stop
    const bitmap = source;

    const seq = ++this._seq;
    this._inflightSeq = seq;
    this.stats.sent += 1;
    const now = performance.now();
    if (this._lastSentAt) this.stats.sendHz = 1000 / Math.max(1, now - this._lastSentAt);
    this._lastSentAt = now;

    return new Promise((resolve) => {
      this._pending = resolve;
      this._pendingTimer = setTimeout(() => {
        this.stats.timedOut += 1;
        console.warn(`[sidecar] reply timeout (${this._replyTimeoutMs}ms) — dropping frame; total ${this.stats.timedOut}`);
        this._settle(null);
      }, this._replyTimeoutMs);
      this.worker.postMessage({ type: 'frame', bitmap, seq, sendMaxSide: Math.round(this.sendMaxSide) }, [bitmap]);
    });
  }

  stop() {
    this.running = false;
    this._pending = null;
    if (this._pendingTimer) { clearTimeout(this._pendingTimer); this._pendingTimer = null; }
    if (this.worker) { this.worker.terminate(); this.worker = null; }
    stopWebcam(this.stream);
    this.video = null;
    this.stream = null;
  }
}
