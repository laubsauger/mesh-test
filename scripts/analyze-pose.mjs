// Pose-recording analyzer — PROVE what the raw pose data actually does before we
// rebuild the retarget. Reads a session saved by the app's "Save Recording" button
// (raw RTMW frames: keypoints3D w/ z + confidence + boundingBox) and reports:
//   - facing-sign stability: sign of each L↔R depth axis over time, and — crucially —
//     splits flips into TRUE reversals (sustained, large |z|) vs FRONTAL DITHER (sign
//     of a ~0 quantity = pure noise). The mesh 180° spin is dither amplified by yaw.
//   - ambiguity fraction: how much of the clip sits near-frontal where sign is junk
//   - bounding-box jitter: frame-to-frame size/center hops = the on-mesh scale jitter
//   - per-keypoint z jitter + whether flips happen at HIGH confidence (real bistability)
//
//   node scripts/analyze-pose.mjs recordings/perf1.json [assumedFps=24]
//
// Frames carry NO timestamp (native provider bug — see note), so analysis is by frame
// index; pass an fps to get a rough seconds estimate (clearly labelled "assumed").
import { readFileSync } from 'node:fs';
import { KPT } from '../src/pose/rtmw-constants.js';

const path = process.argv[2];
const FPS = Number(process.argv[3]) || 24; // assumed — frames have no timestamp
if (!path) { console.error('usage: node scripts/analyze-pose.mjs <recording.json> [fps]'); process.exit(1); }

const session = JSON.parse(readFileSync(path, 'utf8'));
const frames = session.frames ?? [];
if (!frames.length) { console.error('no frames'); process.exit(1); }
const k3 = (f) => f.keypoints3D || null;
if (!k3(frames[0])) { console.error('no keypoints3D'); process.exit(1); }

const N = frames.length;
const secs = N / FPS;

const stats = (arr) => {
  const a = arr.filter((v) => Number.isFinite(v));
  if (!a.length) return { mean: 0, std: 0, min: 0, max: 0, p95: 0 };
  const mean = a.reduce((x, y) => x + y, 0) / a.length;
  const std = Math.sqrt(a.reduce((x, y) => x + (y - mean) ** 2, 0) / a.length);
  const s = [...a].sort((x, y) => x - y);
  return { mean, std, min: s[0], max: s[s.length - 1], p95: s[Math.floor(0.95 * (s.length - 1))] };
};

// depth axis value = right.z − left.z. DEAD = magnitude below which the sign is noise
// (near-frontal). A TRUE reversal = sign change where BOTH neighbours are past DEAD
// (the body genuinely faced one way, then the other). Otherwise = frontal DITHER.
const DEAD = 0.12;
function axis(name, li, ri) {
  const v = [];
  const conf = [];
  for (const f of frames) {
    const k = k3(f); const l = k[li]; const r = k[ri];
    v.push(l && r ? r.z - l.z : NaN);
    conf.push(l && r ? Math.min(l.confidence ?? 1, r.confidence ?? 1) : 0);
  }
  const jumps = [];
  let dither = 0; let real = 0; const realConf = [];
  let lastCommitted = 0; // last sign we were confident about (past DEAD)
  for (let i = 0; i < N; i += 1) {
    if (i > 0 && Number.isFinite(v[i]) && Number.isFinite(v[i - 1])) jumps.push(Math.abs(v[i] - v[i - 1]));
    if (!Number.isFinite(v[i])) continue;
    if (Math.abs(v[i]) < DEAD) { dither += 1; continue; } // near-frontal → sign meaningless
    const s = Math.sign(v[i]);
    if (lastCommitted !== 0 && s !== lastCommitted) { real += 1; realConf.push(conf[i]); }
    lastCommitted = s;
  }
  const ambig = v.filter((x) => Number.isFinite(x) && Math.abs(x) < DEAD).length / N;
  return { name, vs: stats(v), js: stats(jumps), real, dither, ambig, realConf, v };
}

const shoulder = axis('shoulder', KPT.leftShoulder, KPT.rightShoulder);
const hip = axis('hip     ', KPT.leftHip, KPT.rightHip);
const ear = axis('ear     ', KPT.leftEar, KPT.rightEar);

console.log(`\n=== ${path} ===`);
console.log(`${N} frames  (~${secs.toFixed(1)}s at assumed ${FPS}fps — frames carry NO timestamp)  meta=${JSON.stringify(session.metadata ?? {})}`);

console.log('\n=== FACING AXES  (right.z − left.z) ===');
console.log('axis      z-range          std    | TRUE reversals | frontal-DITHER | %near-frontal(sign=noise) | Δz p95/max');
for (const a of [shoulder, hip, ear]) {
  console.log(
    `${a.name}  [${a.vs.min.toFixed(2)},${a.vs.max.toFixed(2)}]`.padEnd(28) +
    `${a.vs.std.toFixed(3)}  | ${String(a.real).padStart(13)}  | ${String(a.dither).padStart(13)}  | ${(a.ambig * 100).toFixed(0).padStart(22)}% | ${a.js.p95.toFixed(2)}/${a.js.max.toFixed(2)}`
  );
}
const rc = shoulder.realConf.length ? stats(shoulder.realConf).mean.toFixed(2) : 'n/a';
console.log(`\nshoulder TRUE-reversal mean conf = ${rc}  (high → genuine monocular bistability, not dropout)`);

console.log('\n=== BOUNDING BOX JITTER  (the on-mesh scale jump) ===');
const bb = frames.map((f) => f.boundingBox).filter(Boolean);
if (bb.length) {
  const w = bb.map((b) => b.width); const h = bb.map((b) => b.height);
  const cx = bb.map((b) => b.x + b.width / 2); const cy = bb.map((b) => b.y + b.height / 2);
  const dpct = (arr) => { const d = []; for (let i = 1; i < arr.length; i += 1) d.push(Math.abs(arr[i] - arr[i - 1]) / (Math.abs(arr[i - 1]) || 1) * 100); return stats(d); };
  const ws = stats(w); const hs = stats(h);
  console.log(`width : mean ${ws.mean.toFixed(0)}  std ${ws.std.toFixed(0)} (${(ws.std / ws.mean * 100).toFixed(0)}%)  frame-to-frame Δ% p95 ${dpct(w).p95.toFixed(1)}  max ${dpct(w).max.toFixed(1)}`);
  console.log(`height: mean ${hs.mean.toFixed(0)}  std ${hs.std.toFixed(0)} (${(hs.std / hs.mean * 100).toFixed(0)}%)  frame-to-frame Δ% p95 ${dpct(h).p95.toFixed(1)}  max ${dpct(h).max.toFixed(1)}`);
  console.log(`center Δpx p95: x ${dpct(cx).p95.toFixed(1)}%  y ${dpct(cy).p95.toFixed(1)}%   (big % = box + crop-relative overlay pulse)`);
}

console.log('\n=== SHOULDER DEPTH-AXIS HISTOGRAM (bimodality) ===');
const vals = shoulder.v.filter(Number.isFinite);
const lo = shoulder.vs.min; const hi = shoulder.vs.max; const B = 21;
const hist = new Array(B).fill(0);
for (const x of vals) hist[Math.min(B - 1, Math.max(0, Math.floor(((x - lo) / (hi - lo || 1)) * B)))] += 1;
const pk = Math.max(...hist);
for (let b = 0; b < B; b += 1) {
  const c = lo + ((b + 0.5) / B) * (hi - lo);
  console.log(`${c.toFixed(2).padStart(6)} | ${'#'.repeat(Math.round((hist[b] / pk) * 40))}${Math.abs(c) < (hi - lo) / B ? ' <- 0 frontal' : ''}`);
}

// --- Arm axial twist: the upper-arm roll comes from the bend-plane normal
// (wrist−elbow)×(elbow−shoulder); the forearm roll from the knuckle line. Both are
// derived from wrist/hand z. When the arm is near-STRAIGHT the plane is undefined →
// the twist is arbitrary; when z is noisy the normal flips → hard axial spin.
const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const cross = (a, b) => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x });
const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const len = (a) => Math.hypot(a.x, a.y, a.z);
const angBetween = (a, b) => {
  const la = len(a); const lb = len(b);
  if (la < 1e-6 || lb < 1e-6) return NaN;
  return Math.acos(Math.max(-1, Math.min(1, dot(a, b) / (la * lb)))) * 180 / Math.PI;
};

function armTwist(name, si, ei, wi, ii, pi) {
  const elbow = []; const planeJump = []; let undef = 0; let lastN = null; const foreConf = [];
  for (const f of frames) {
    const k = k3(f);
    const s = k[si]; const e = k[ei]; const w = k[wi];
    if (!s || !e || !w) continue;
    elbow.push(angBetween(sub(s, e), sub(w, e))); // 180 = straight → plane undefined
    const dir = sub(e, s); const fore = sub(w, e);
    const n = cross(fore, dir);
    if (len(n) < 0.04 * len(dir)) { undef += 1; lastN = null; continue; } // near-straight
    if (lastN) planeJump.push(angBetween(lastN, n)); // frame-to-frame twist wobble (deg)
    lastN = n;
    const ix = k[ii]; const pk = k[pi];
    if (ix && pk) foreConf.push(Math.min(ix.confidence ?? 0, pk.confidence ?? 0));
  }
  return {
    name, elbow: stats(elbow), jump: stats(planeJump),
    undefPct: undef / N, foreConf: foreConf.length ? stats(foreConf).mean : 0
  };
}
const la = armTwist('L arm', KPT.leftShoulder, KPT.leftElbow, KPT.leftWrist, KPT.leftIndexBase, KPT.leftPinkyBase);
const ra = armTwist('R arm', KPT.rightShoulder, KPT.rightElbow, KPT.rightWrist, KPT.rightIndexBase, KPT.rightPinkyBase);

console.log('\n=== ARM AXIAL TWIST  (upper-arm bend-plane wobble = the roll spin) ===');
console.log('arm     elbow-angle(180=straight)  | bend-plane Δ/frame p95/max deg | %near-straight(twist undefined) | knuckle conf');
for (const a of [la, ra]) {
  console.log(
    `${a.name}   mean ${a.elbow.mean.toFixed(0)} [${a.elbow.min.toFixed(0)},${a.elbow.max.toFixed(0)}]`.padEnd(38) +
    `| ${a.jump.p95.toFixed(0).padStart(6)} / ${a.jump.max.toFixed(0).padStart(3)}          | ${(a.undefPct * 100).toFixed(0).padStart(24)}% | ${a.foreConf.toFixed(2)}`
  );
}
console.log('  (bend-plane Δ near 180 = the normal FLIPPED that frame → upper arm snapped its axial roll)');

console.log('\n=== VERDICT ===');
const totalReal = shoulder.real + ear.real + hip.real;
const totalDither = shoulder.dither + ear.dither + hip.dither;
console.log(`TRUE facing reversals: shoulder ${shoulder.real}, ear ${ear.real}, hip ${hip.real}`);
console.log(`Frontal-dither frames: shoulder ${shoulder.dither}, ear ${ear.dither} (sign of ~0 depth = noise)`);
console.log(`Time spent near-frontal (sign ambiguous): shoulder ${(shoulder.ambig * 100).toFixed(0)}%, hip ${(hip.ambig * 100).toFixed(0)}%`);
if (shoulder.ambig > 0.4 && shoulder.real <= 3) {
  console.log('→ The 180° spin is NOT real turning — it is the SIGN of a near-zero depth axis flipping.');
  console.log('  Fix: deadzone + hysteresis on facing (hold when |z-axis| small); hips are the stable ref (fewest flips).');
} else if (totalReal > 6) {
  console.log('→ Genuine depth bistability (sustained sign reversals). Needs soft-argmax z + temporal facing lock.');
}
console.log('');
