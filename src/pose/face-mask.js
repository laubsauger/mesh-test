// Face region mask (T27 / V27, V28): per-vertex region weights that say WHICH mesh
// vertices each face-expression scalar (I.faceExpr) moves, and how much. The mask is
// derived from MESH GEOMETRY ALONE (head-bone skin weights + anatomical height bands)
// — NEVER from the landmarks (V27: landmark space and mesh space meet only via the
// scalars). Auto-gen gives a starting guess; a hand-painter (T29) refines it later,
// and the refined result persists as a `.bin` sidecar next to the GLB (V28).
//
// Pure — plain typed arrays in, plain data out. No three/DOM → node + vitest friendly.

// Fixed region order. The scalars in I.faceExpr map onto these regions in T28's deform:
//   jaw       ← jawOpen (hinge-rotate)
//   lowerLip  ← jawOpen (translate down)
//   mouthCorner ← smile / pucker (translate out / in)
//   upperLidL/R ← blinkL/R (translate down)
//   browL/R   ← browL/R (translate up)
export const FACE_REGIONS = ['jaw', 'lowerLip', 'mouthCorner', 'upperLidL', 'upperLidR', 'browL', 'browR'];

const MAGIC = 0x464d534b; // 'FMSK'
const VERSION = 1;
// magic + ver + regionCount + vertexCount + hinge(6 f32) + headHeight(1 f32)
const HEADER_BYTES = 4 + 1 + 1 + 4 + 6 * 4 + 4;

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
// Trapezoid membership: ramps 0→1 over [lo-f,lo], 1 across [lo,hi], 1→0 over [hi,hi+f].
function band(v, lo, hi, f) {
  if (v <= lo - f || v >= hi + f) return 0;
  if (v < lo) return (v - (lo - f)) / f;
  if (v > hi) return ((hi + f) - v) / f;
  return 1;
}
const smooth = (t) => { const x = clamp01(t); return x * x * (3 - 2 * x); };

// Auto-generate region weights from geometry. `positions` (3×count), `skinIndices`
// (4×count), `skinWeights` (4×count) are the raw attribute arrays; `headBoneIndex` is
// the skeleton index of the 'Head' bone. `forwardSign` = which z hemisphere the face
// points down (+1 = +z, Mixamo default). Returns {vertexCount, regions, data, hinge}.
export function generateFaceMask({ positions, skinIndices, skinWeights, count, headBoneIndex, forwardSign = 1 }) {
  if (!Number.isInteger(headBoneIndex) || headBoneIndex < 0) {
    throw new Error(`generateFaceMask: invalid headBoneIndex ${headBoneIndex}`);
  }
  // Per-vertex head membership = summed skin weight bound to the Head bone.
  const headW = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    let w = 0;
    for (let k = 0; k < 4; k += 1) {
      if (skinIndices[i * 4 + k] === headBoneIndex) w += skinWeights[i * 4 + k];
    }
    headW[i] = w;
  }

  // Head AABB over mostly-head verts (weight > 0.5) → the anatomical reference box.
  let minX = Infinity; let maxX = -Infinity; let minY = Infinity; let maxY = -Infinity; let minZ = Infinity; let maxZ = -Infinity;
  let nHead = 0;
  for (let i = 0; i < count; i += 1) {
    if (headW[i] <= 0.5) continue;
    nHead += 1;
    const x = positions[i * 3]; const y = positions[i * 3 + 1]; const z = positions[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  if (nHead === 0) throw new Error('generateFaceMask: no head-weighted vertices (>0.5) — wrong headBoneIndex?');
  const hY = maxY - minY || 1;
  const cx = (minX + maxX) / 2;
  const cz = (minZ + maxZ) / 2;
  const hZ = (maxZ - minZ) || 1;

  const R = FACE_REGIONS.length;
  const data = new Uint8Array(R * count);
  const put = (region, i, w) => { data[region * count + i] = Math.round(clamp01(w) * 255); };

  for (let i = 0; i < count; i += 1) {
    const hw = headW[i];
    if (hw <= 0.05) continue;
    const y = positions[i * 3 + 1];
    const x = positions[i * 3];
    const z = positions[i * 3 + 2];
    const h = (y - minY) / hY;                 // 0 chin … 1 crown
    // Forwardness: face is the front z-hemisphere. Normalize signed z toward front.
    const fwd = smooth(((z - cz) * forwardSign) / (hZ * 0.5) * 0.5 + 0.5); // 0 back … 1 front
    if (fwd < 0.15) continue;                   // back of the head never deforms
    const base = hw * fwd;
    const leftness = clamp01((x - cx) / (Math.max(1e-6, maxX - cx))); // 0 center … 1 far-left(+x)
    const rightness = clamp01((cx - x) / (Math.max(1e-6, cx - minX)));
    const sideMag = Math.abs(x - cx) / Math.max(1e-6, (maxX - minX) / 2); // 0 center … 1 edge

    // jaw / chin — lower-front wedge, hinge-rotated. Strongest at the chin, fading up.
    put(0, i, base * band(h, 0.0, 0.30, 0.12) * (1 - 0.4 * sideMag));
    // lower lip — narrow band just above the chin, near center.
    put(1, i, base * band(h, 0.24, 0.34, 0.05) * (1 - smooth(sideMag)));
    // mouth corners — same mouth height, but the SIDES (smile/pucker pull).
    put(2, i, base * band(h, 0.24, 0.36, 0.06) * smooth(sideMag));
    // upper eyelids — eye height, split L/R, front-central.
    put(3, i, base * band(h, 0.50, 0.62, 0.05) * smooth(leftness));   // upperLidL (+x)
    put(4, i, base * band(h, 0.50, 0.62, 0.05) * smooth(rightness));  // upperLidR (-x)
    // brows — just above the eyes, split L/R.
    put(5, i, base * band(h, 0.62, 0.74, 0.06) * smooth(leftness));   // browL
    put(6, i, base * band(h, 0.62, 0.74, 0.06) * smooth(rightness));  // browR
  }

  // Jaw hinge: pivot at the top of the jaw wedge (h≈0.30), centered, mid-depth; axis =
  // the ear-to-ear line (mesh x) so the jaw swings open about it (T28 hinge-rotate).
  const hinge = {
    origin: [cx, minY + 0.30 * hY, cz],
    axis: [1, 0, 0]
  };
  // headHeight = the deform's scale reference (geometry units): translations (lip
  // drop, lid close, brow raise) are expressed as fractions of it so they hold across
  // Meshy exports of differing scale (T28). Hinge rotation is scale-free.
  return { vertexCount: count, regions: FACE_REGIONS, data, hinge, headHeight: hY };
}

// Serialize to the `.bin` sidecar layout (see HEADER_BYTES). Returns an ArrayBuffer.
export function encodeFaceMask(mask) {
  const { vertexCount, data, hinge, headHeight = 1 } = mask;
  const R = FACE_REGIONS.length;
  if (data.length !== R * vertexCount) {
    throw new Error(`encodeFaceMask: data length ${data.length} != regions×verts ${R * vertexCount}`);
  }
  const buf = new ArrayBuffer(HEADER_BYTES + data.length);
  const dv = new DataView(buf);
  let o = 0;
  dv.setUint32(o, MAGIC, true); o += 4;
  dv.setUint8(o, VERSION); o += 1;
  dv.setUint8(o, R); o += 1;
  dv.setUint32(o, vertexCount, true); o += 4;
  for (let k = 0; k < 3; k += 1) { dv.setFloat32(o, hinge.origin[k], true); o += 4; }
  for (let k = 0; k < 3; k += 1) { dv.setFloat32(o, hinge.axis[k], true); o += 4; }
  dv.setFloat32(o, headHeight, true); o += 4;
  new Uint8Array(buf, HEADER_BYTES).set(data);
  return buf;
}

// Parse a `.bin` sidecar. Throws on bad magic / version / region count (fail loud,
// no silent accept of a stale or corrupt file).
export function decodeFaceMask(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  if (arrayBuffer.byteLength < HEADER_BYTES) throw new Error('decodeFaceMask: buffer smaller than header');
  let o = 0;
  if (dv.getUint32(o, true) !== MAGIC) throw new Error('decodeFaceMask: bad magic (not a face-mask .bin)'); o += 4;
  const version = dv.getUint8(o); o += 1;
  if (version !== VERSION) throw new Error(`decodeFaceMask: version ${version} != ${VERSION}`);
  const R = dv.getUint8(o); o += 1;
  if (R !== FACE_REGIONS.length) throw new Error(`decodeFaceMask: region count ${R} != ${FACE_REGIONS.length}`);
  const vertexCount = dv.getUint32(o, true); o += 4;
  const origin = [dv.getFloat32(o, true), dv.getFloat32(o + 4, true), dv.getFloat32(o + 8, true)]; o += 12;
  const axis = [dv.getFloat32(o, true), dv.getFloat32(o + 4, true), dv.getFloat32(o + 8, true)]; o += 12;
  const headHeight = dv.getFloat32(o, true); o += 4;
  const expect = R * vertexCount;
  const data = new Uint8Array(arrayBuffer, HEADER_BYTES, expect);
  if (data.length !== expect) throw new Error(`decodeFaceMask: data length ${data.length} != ${expect}`);
  return { vertexCount, regions: FACE_REGIONS.slice(), data: new Uint8Array(data), hinge: { origin, axis }, headHeight };
}

// V28 guard: a mask only applies to the geometry it was built for. A vertexCount
// mismatch means a stale/wrong sidecar → THROW (never silently mis-index vertices).
export function assertMaskFits(mask, vertexCount) {
  if (mask.vertexCount !== vertexCount) {
    throw new Error(`face mask vertexCount ${mask.vertexCount} != geometry ${vertexCount} — stale sidecar, regenerate`);
  }
  return mask;
}
