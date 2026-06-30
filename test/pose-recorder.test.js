import { describe, it, expect } from 'vitest';
import { PoseRecorder, PosePlayer } from '../src/pose/recorder.js';

const frame = (t) => ({ timestampMs: t, keypoints2D: [], keypoints3D: [{ x: t, y: 0, z: 0, confidence: 0.9 }] });

describe('PoseRecorder — T6', () => {
  it('captures only while recording', () => {
    const r = new PoseRecorder();
    r.capture(frame(0)); // ignored — not started
    r.start({ inputWidth: 1280 });
    r.capture(frame(1));
    r.capture(frame(2));
    r.stop();
    r.capture(frame(3)); // ignored — stopped
    expect(r.length).toBe(2);
  });

  it('round-trips through session JSON', () => {
    const r = new PoseRecorder();
    r.start({ inputWidth: 640 });
    r.capture(frame(10));
    r.capture(frame(20));
    const session = JSON.parse(JSON.stringify(r.toSession()));
    const r2 = new PoseRecorder();
    r2.load(session);
    expect(r2.length).toBe(2);
    expect(r2.metadata.inputWidth).toBe(640);
  });
});

describe('PosePlayer — replay by timestamp', () => {
  const frames = [frame(0), frame(100), frame(200), frame(300)];

  it('returns the frame at/just before elapsed time', () => {
    const p = new PosePlayer(frames);
    expect(p.frameAt(0).timestampMs).toBe(0);
    expect(p.frameAt(150).timestampMs).toBe(100);
    expect(p.frameAt(250).timestampMs).toBe(200);
  });

  it('loops (wraps past duration)', () => {
    const p = new PosePlayer(frames);
    p.frameAt(250);
    expect(p.frameAt(310).timestampMs).toBe(0); // 310 % 300 = 10 → first frame
  });

  it('empty recording → null', () => {
    expect(new PosePlayer([]).frameAt(0)).toBe(null);
  });
});
