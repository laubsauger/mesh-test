// EP probe (SPEC §T.22 / V23): bench the rtmw3d model on each onnxruntime-web
// EP with a synthetic input, so we can pick the EP that best coexists with the
// three WebGPU renderer. Inference ms is measured here; render-fps contention is
// observed live (run the probe with the crowd rendering). No auto-fallback —
// each EP is created explicitly and a failure is reported, not hidden.
import { ort, createPoseSession } from './ort-session.js';
import { POSE_EPS } from './ep.js';
import { benchSession } from './ep-bench.js';
import { RTMW3D_MODEL } from './rtmw-constants.js';
import { assetUrl } from '../asset-url.js';

let modelBytes = null;

async function fetchModelBytes() {
  if (modelBytes) return modelBytes;
  const url = assetUrl(RTMW3D_MODEL.url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`rtmw3d model fetch ${res.status}: ${url}`);
  modelBytes = await res.arrayBuffer();
  return modelBytes;
}

export async function probeEPs(eps = POSE_EPS, opts = {}) {
  const buf = await fetchModelBytes();
  const dims = [1, 3, RTMW3D_MODEL.resH, RTMW3D_MODEL.resW];
  const len = dims.reduce((a, b) => a * b, 1);

  const results = [];
  for (const ep of eps) {
    const result = { ep, ok: false };
    try {
      const session = await createPoseSession(buf, ep);
      const input = new ort.Tensor('float32', new Float32Array(len), dims);
      Object.assign(result, await benchSession(session, { [RTMW3D_MODEL.inputName]: input }, opts));
      input.dispose();
      await session.release?.();
      result.ok = true;
    } catch (error) {
      result.error = error.message;
    }
    logResult(result);
    results.push(result);
  }

  const winner = results.filter((r) => r.ok).sort((a, b) => a.median - b.median)[0];
  if (winner) console.info(`[ep-probe] winner: ${winner.ep} (median ${winner.median.toFixed(1)}ms)`);
  return { results, winner: winner ? winner.ep : null };
}

function logResult(r) {
  if (r.ok) {
    console.info(
      `[ep-probe] ${r.ep}: median ${r.median.toFixed(1)}ms, mean ${r.mean.toFixed(1)}ms ` +
      `(min ${r.min.toFixed(1)}, max ${r.max.toFixed(1)}, ${r.runs} runs)`
    );
  } else {
    console.warn(`[ep-probe] ${r.ep}: FAILED — ${r.error}`);
  }
}
