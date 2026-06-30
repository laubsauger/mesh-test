// Preflight: verify the pose ONNX inference models are present before dev/build.
//
// These models are NOT in this repo — they're large (rtmw3d ~352MB > GitHub's
// 100MB limit) and exported by the separate `object-detect` project, not
// vendored here. A fresh clone has none. Bring them over via file transfer and
// drop them under public/inference/ (paths below).
//
// This check is WARN-ONLY (exits 0): the crowd renderer runs fine without them;
// only webcam pose driving needs them. It surfaces a clear, actionable message
// instead of an opaque "model fetch 404" the first time pose starts. Run
// directly (`npm run check:models`) or automatically via predev/prebuild.

import { existsSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, relative } from 'node:path';
import { rtmwUrl, yoloUrl } from '../src/pose/rtmw-constants.js';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const MIN_BYTES = 1_000_000; // catch truncated / empty transfers, not just absence

// Mirror the app's DEFAULT selection (src/main.js state: poseRtmwVariant / poseYoloRes).
// The variant file is the one that must be file-transferred; yolo res files are committed.
const DEFAULT_VARIANT = 'l';
const DEFAULT_YOLO_RES = 320;
const expected = [
  { label: `RTMW3D-${DEFAULT_VARIANT} 3D pose (default variant)`, url: rtmwUrl(DEFAULT_VARIANT) },
  { label: `yolo26n detector @${DEFAULT_YOLO_RES}`, url: yoloUrl(DEFAULT_YOLO_RES) }
].map((m) => ({ ...m, path: join(repoRoot, 'public', m.url) }));

const missing = expected.filter((m) => {
  if (!existsSync(m.path)) return true;
  return statSync(m.path).size < MIN_BYTES; // present but truncated counts as missing
});

if (missing.length === 0) {
  console.log('[check:models] pose inference models present ✓');
  process.exit(0);
}

const lines = [
  '',
  '  ⚠  Pose inference models missing — webcam pose driving will 404 at runtime.',
  '     (The crowd renderer still works; only pose needs these.)',
  '',
  '     These are NOT in this repo (too large for GitHub, exported by the',
  '     separate `object-detect` project). Bring them over by file transfer:',
  ''
];
for (const m of missing) {
  lines.push(`       public/${m.url}   # ${m.label}`);
}
lines.push('');
lines.push(`     Re-check with:  npm run check:models`);
lines.push('');
console.warn(lines.join('\n'));
process.exit(0); // warn-only — never block dev/build/CI
