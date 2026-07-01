// Shared mutable face runtime state (T31). Both the deform (reader) and the editor
// (writer) touch these, so they live in one tiny module instead of being threaded
// through function signatures or duplicated.
//   latestFaceExpr    — the newest FaceExpression from the webcam (set by main.js each
//                       accepted pose frame; read by writeFaceExpr for pose performers).
//   debugFaceOverride — a forced expression from the editor test-drive / CDP harness
//                       (set by the editor; when non-null it overrides latestFaceExpr).
export const runtime = {
  latestFaceExpr: null,
  debugFaceOverride: null
};
