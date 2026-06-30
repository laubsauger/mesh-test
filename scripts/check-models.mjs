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
import { RTMW3D_MODEL, YOLO_DET_MODEL } from '../src/pose/rtmw-constants.js';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const MIN_BYTES = 1_000_000; // catch truncated / empty transfers, not just absence

// The app fetches these from `public/` via assetUrl(MODEL.url) — derive the
// expected disk paths from the same constants so this never drifts.
const expected = [
  { label: 'RTMW3D-x 3D pose', url: RTMW3D_MODEL.url },
  { label: 'yolo26n detector', url: YOLO_DET_MODEL.url }
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
