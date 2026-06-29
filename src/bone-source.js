// Per-walker bone-matrix source selection. Pure, testable. The first `poseCount`
// walkers (by spawn index) are pose-driven performers; the rest play their
// animation clip. Both write the SAME boneMatrices slot (V5) — clip + pose
// instances coexist in one batch (V20).
export const BONE_SOURCES = ['clip', 'pose'];

export function boneSourceForIndex(index, poseCount) {
  return index < poseCount ? 'pose' : 'clip';
}
