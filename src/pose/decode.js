// RTMW3D SimCC decode — pure math on typed arrays (no canvas/ort), so it
// unit-tests in node. Faithful port of the working reference (object-detect
// decode3d / posePreprocess rect). See I.reference.
import { Z_RANGE, RTMW3D_MODEL, POSE_PADDING } from './rtmw-constants.js';

// Pad a detector box to the model aspect + padding → the crop rect. Its inverse
// (sx + mx/resW*sw) maps model-input px back to source-frame px. Pure.
export function bboxToRect(box, resW = RTMW3D_MODEL.resW, resH = RTMW3D_MODEL.resH, padding = POSE_PADDING) {
  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;
  let w = box.w;
  let h = box.h;
  const aspect = resW / resH;
  if (w > h * aspect) h = w / aspect;
  else w = h * aspect;
  w *= padding;
  h *= padding;
  return { sx: cx - w / 2, sy: cy - h / 2, sw: w, sh: h };
}

// Letterbox a vidW×vidH frame into an R×R square preserving aspect (yolo wants
// aspect-correct input, not a stretched square). Returns the draw rect + the
// inverse mapping (detection px in R-space → source px).
export function letterboxRect(vidW, vidH, R) {
  const scale = R / Math.max(vidW, vidH);
  const drawW = vidW * scale;
  const drawH = vidH * scale;
  return { scale, drawW, drawH, offsetX: (R - drawW) / 2, offsetY: (R - drawH) / 2 };
}

// 3-axis SimCC argmax decode. `out` = { [outX]:{data,dims}, [outY]:..., [outZ]:... }.
// Returns k2d (source-frame px + score) for overlays and k3d (root-relative
// normalized + score) for driving. resW/resH = model input W/H.
export function decode3d(out, rect, resW = RTMW3D_MODEL.resW, resH = RTMW3D_MODEL.resH,
  outNames = [RTMW3D_MODEL.outX, RTMW3D_MODEL.outY, RTMW3D_MODEL.outZ]) {
  // outNames = the loaded session's output order [X,Y,Z]. Default = the x-export
  // names so swapping rtmw variant (whose graph node names differ) still decodes.
  const tx = out[outNames[0]];
  const ty = out[outNames[1]];
  const tz = out[outNames[2]];
  const sx = tx.data;
  const sy = ty.data;
  const sz = tz.data;
  const K = tx.dims[1];
  const Wx = tx.dims[2];
  const Hy = ty.dims[2];
  const Wz = tz.dims[2];
  const rX = Wx / resW;
  const rY = Hy / resH;
  const rZ = Wz / resW;

  const k2d = new Array(K);
  const k3d = new Array(K);
  for (let k = 0; k < K; k += 1) {
    let bx = 0;
    let vx = -1e9;
    const ox = k * Wx;
    for (let i = 0; i < Wx; i += 1) { const v = sx[ox + i]; if (v > vx) { vx = v; bx = i; } }
    let by = 0;
    let vy = -1e9;
    const oy = k * Hy;
    for (let i = 0; i < Hy; i += 1) { const v = sy[oy + i]; if (v > vy) { vy = v; by = i; } }
    let bz = 0;
    let vz = -1e9;
    const oz = k * Wz;
    for (let i = 0; i < Wz; i += 1) { const v = sz[oz + i]; if (v > vz) { vz = v; bz = i; } }

    const mx = bx / rX;
    const my = by / rY;
    const mz = bz / rZ;
    const score = 0.5 * (vx + vy); // z maxima unreliable → gate on x,y only

    k2d[k] = { x: rect.sx + (mx / resW) * rect.sw, y: rect.sy + (my / resH) * rect.sh, score };
    k3d[k] = {
      x: mx / (resH / 2),
      y: my / (resH / 2),
      z: (mz / (resW / 2) - 1) * Z_RANGE,
      score
    };
  }
  return { k2d, k3d };
}
