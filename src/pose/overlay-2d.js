// 2D skeleton overlay on a canvas sized to the video — the M1 "does it track me"
// check. Draws raw keypoints2D + body/foot bones, gated by confidence. `mirror`
// flips x to match a CSS-mirrored selfie video.
import { BODY_BONES, KPT_GROUPS } from './topology.js';

export function drawOverlay(ctx, frame, vidW, vidH, { kptThresh = 0.3, mirror = true, bg = null } = {}) {
  // bg = optional preview image (native-capture mode has no <video>, so the sidecar's
  // low-res webcam JPEG is the backdrop). Mirror it to match the mirrored skeleton.
  if (bg) {
    ctx.save();
    if (mirror) { ctx.translate(vidW, 0); ctx.scale(-1, 1); }
    ctx.drawImage(bg, 0, 0, vidW, vidH);
    ctx.restore();
  } else {
    ctx.clearRect(0, 0, vidW, vidH);
  }
  if (!frame) return;

  const k = frame.keypoints2D;
  const X = (x) => (mirror ? vidW - x : x);

  ctx.lineWidth = 3;
  ctx.strokeStyle = '#00e5ff';
  for (const [a, b] of BODY_BONES) {
    const ka = k[a];
    const kb = k[b];
    if (!ka || !kb || ka.confidence < kptThresh || kb.confidence < kptThresh) continue;
    ctx.beginPath();
    ctx.moveTo(X(ka.x), ka.y);
    ctx.lineTo(X(kb.x), kb.y);
    ctx.stroke();
  }

  for (const group of KPT_GROUPS) {
    ctx.fillStyle = group.color;
    for (let j = group.lo; j < group.hi; j += 1) {
      const p = k[j];
      if (!p || p.confidence < kptThresh) continue;
      ctx.beginPath();
      ctx.arc(X(p.x), p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
