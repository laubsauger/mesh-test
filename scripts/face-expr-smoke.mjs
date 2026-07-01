// Smoke test for FaceExpressionExtractor (T26 / V26). No test runner in this repo,
// so this is a runnable node assert (like scripts/analyze-pose.mjs). Verifies INTENT
// (V26): mouth-open → jawOpen↑, closed eye → blink↑, wide mouth → smile↑, raised brow
// → brow↑, and a held neutral face reads ~0 on every scalar.
//
//   node scripts/face-expr-smoke.mjs   → exit 0 = pass, 1 = fail
import { FaceExpressionExtractor, FACE_OFFSET } from '../src/pose/face-expression.js';

const F = (i) => i + FACE_OFFSET; // iBUG-68 id → canonical (133) index

// Neutral face landmark layout (canonical +y up; arbitrary units). Only the points
// the extractor reads are placed; iod = outer-eye span = 0.6.
function neutralJoints() {
  const j = Array.from({ length: 133 }, () => ({ x: 0, y: 0, z: 0, confidence: 1 }));
  const set = (i, x, y) => { j[i] = { x, y, z: 0, confidence: 1 }; };
  // Right eye ring (iBUG 36-41): EAR = 0.5.
  set(F(36), -0.30, 0.10); set(F(37), -0.25, 0.15); set(F(38), -0.15, 0.15);
  set(F(39), -0.10, 0.10); set(F(40), -0.15, 0.05); set(F(41), -0.25, 0.05);
  // Left eye ring (iBUG 42-47): EAR = 0.5.
  set(F(42), 0.10, 0.10); set(F(43), 0.15, 0.15); set(F(44), 0.25, 0.15);
  set(F(45), 0.30, 0.10); set(F(46), 0.25, 0.05); set(F(47), 0.15, 0.05);
  // Brow centers (iBUG 19 right, 24 left): 0.10 above eye center.
  set(F(19), -0.20, 0.20); set(F(24), 0.20, 0.20);
  // Mouth outer corners + inner-lip ring (near-closed: gap 0.04, width 0.30 → MAR 0.13).
  set(F(48), -0.20, -0.30); set(F(54), 0.20, -0.30);
  set(F(60), -0.15, -0.30); set(F(64), 0.15, -0.30);
  set(F(61), -0.05, -0.28); set(F(62), 0.00, -0.28); set(F(63), 0.05, -0.28);
  set(F(67), -0.05, -0.32); set(F(66), 0.00, -0.32); set(F(65), 0.05, -0.32);
  return j;
}

const clone = (j) => j.map((p) => ({ ...p }));

// Run: calibrate neutral (CALIB_FRAMES), then feed `settle` frames of a modified face,
// return the final scalars. Fresh extractor per case → clean One Euro state.
function run(modify, settle = 40) {
  const ex = new FaceExpressionExtractor();
  const dt = 1 / 30;
  for (let i = 0; i < 20; i += 1) ex.update({ timestampMs: 0, joints: neutralJoints() }, dt); // set baseline
  let v;
  const j = { timestampMs: 0, joints: modify(neutralJoints()) };
  for (let i = 0; i < settle; i += 1) v = ex.update(j, dt);
  return { ...v };
}

let fails = 0;
const check = (name, cond, got) => {
  const ok = !!cond;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : `  (got ${JSON.stringify(got)})`}`);
  if (!ok) fails += 1;
};

// Neutral held → everything ~0.
const rest = run((j) => j);
check('neutral: all scalars ~0', Object.values(rest).every((x) => x < 0.05), rest);

// Mouth open: widen the inner-lip vertical gap → jawOpen up.
const open = run((j) => {
  j[F(61)].y = -0.24; j[F(62)].y = -0.24; j[F(63)].y = -0.24;
  j[F(67)].y = -0.36; j[F(66)].y = -0.36; j[F(65)].y = -0.36;
  return j;
});
check('mouth open: jawOpen > 0.2', open.jawOpen > 0.2, open);

// Right eye closed: collapse the ring vertically → blinkR up, blinkL stays low.
const blink = run((j) => {
  for (const i of [37, 38, 40, 41]) j[F(i)].y = 0.10; // top/bottom lids meet the corners
  return j;
});
check('blink R: blinkR > 0.5', blink.blinkR > 0.5, blink);
check('blink R: blinkL stays low', blink.blinkL < 0.15, blink);

// Wide mouth: pull corners apart → smile up, pucker 0.
const smile = run((j) => { j[F(48)].x = -0.35; j[F(54)].x = 0.35; return j; });
check('smile: smile > 0.4', smile.smile > 0.4, smile);
check('smile: pucker == 0', smile.pucker === 0, smile);

// Narrow mouth: pull corners in → pucker up, smile 0.
const pucker = run((j) => { j[F(48)].x = -0.10; j[F(54)].x = 0.10; return j; });
check('pucker: pucker > 0.4', pucker.pucker > 0.4, pucker);
check('pucker: smile == 0', pucker.smile === 0, pucker);

// Raised brows: lift brow centers → browL/R up.
const brow = run((j) => { j[F(19)].y = 0.30; j[F(24)].y = 0.30; return j; });
check('brow raise: browR > 0.4', brow.browR > 0.4, brow);
check('brow raise: browL > 0.4', brow.browL > 0.4, brow);

if (fails) { console.error(`\n${fails} check(s) FAILED`); process.exit(1); }
console.log('\nall checks passed');
