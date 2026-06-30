// Pose recording + deterministic replay (T6, §46). Capture RTMWPoseFrames, then
// replay the SAME motion through the pipeline — so retarget / smoothing / twist /
// calibration can be iterated against a fixed clip instead of live webcam.

export class PoseRecorder {
  constructor() {
    this.frames = [];
    this.recording = false;
    this.metadata = null;
  }

  start(metadata = {}) {
    this.frames = [];
    this.metadata = metadata;
    this.recording = true;
  }

  stop() {
    this.recording = false;
  }

  capture(frame) {
    if (this.recording && frame) this.frames.push(frame);
  }

  get length() {
    return this.frames.length;
  }

  // RecordedPoseSession (§46) — JSON-serializable.
  toSession() {
    return { metadata: this.metadata ?? {}, frames: this.frames };
  }

  load(session) {
    this.frames = Array.isArray(session?.frames) ? session.frames : [];
    this.metadata = session?.metadata ?? null;
    this.recording = false;
  }
}

// Plays recorded frames back by timestamp, looping. frameAt(elapsedMs) returns
// the frame to show at `elapsedMs` since playback start.
export class PosePlayer {
  constructor(frames) {
    this.frames = frames ?? [];
    this.t0 = this.frames.length ? this.frames[0].timestampMs : 0;
    this.duration = this.frames.length
      ? this.frames[this.frames.length - 1].timestampMs - this.t0
      : 0;
    this.i = 0;
  }

  reset() {
    this.i = 0;
  }

  frameAt(elapsedMs) {
    if (!this.frames.length) return null;
    const loopT = this.duration > 0 ? elapsedMs % this.duration : 0;
    const target = this.t0 + loopT;
    // index may move either direction (loop wrap) — clamp both ways
    while (this.i < this.frames.length - 1 && this.frames[this.i + 1].timestampMs <= target) this.i += 1;
    while (this.i > 0 && this.frames[this.i].timestampMs > target) this.i -= 1;
    return this.frames[this.i];
  }
}
