// COCO-WholeBody bone topology for overlays. Body + foot edges are enough for
// the M1 "does the skeleton track me" check. Ported from I.reference.
export const BODY_EDGES = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // shoulders + arms
  [5, 11], [6, 12], [11, 12],             // torso
  [11, 13], [13, 15], [12, 14], [14, 16], // legs
  [0, 1], [0, 2], [1, 3], [2, 4],         // nose-eyes-ears
  [3, 5], [4, 6]                          // ears-shoulders
];

export const FOOT_EDGES = [
  [15, 17], [15, 18], [15, 19], // L ankle → big/small toe, heel
  [16, 20], [16, 21], [16, 22]  // R ankle → ...
];

export const BODY_BONES = BODY_EDGES.concat(FOOT_EDGES);

// Keypoint groups → color + index range [lo,hi) for dot rendering.
export const KPT_GROUPS = [
  { key: 'body', lo: 0, hi: 23, color: '#00e5ff' },
  { key: 'face', lo: 23, hi: 91, color: '#ff5cf0' },
  { key: 'lhand', lo: 91, hi: 112, color: '#ffa726' },
  { key: 'rhand', lo: 112, hi: 133, color: '#ffd54f' }
];
