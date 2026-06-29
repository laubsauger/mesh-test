// ONNX execution-provider selection — pure, no ort import so it unit-tests in
// node. EP is explicit and measured (V23): the chosen EP either works or throws.
// No silent webgpu→wasm auto-fallback (project + global no-fallback rule).
export const POSE_EPS = ['webgpu', 'wasm'];

export function assertPoseEP(ep) {
  if (!POSE_EPS.includes(ep)) {
    throw new Error(`unknown pose EP "${ep}" — expected ${POSE_EPS.join(' | ')}`);
  }
  return ep;
}
