// RTMW3D whole-body pose constants (COCO-WholeBody 133, V5 topology). Mirrors
// the working reference (object-detect/web/src/main.js) so decode matches the
// exported model exactly. Pure data — safe to import anywhere (no three/DOM/ort).
//
// Keypoint layout: 0-16 body (COCO-17), 17-22 feet, 23-90 face,
//                  91-111 left hand, 112-132 right hand.

export const NUM_KPTS = 133;

// rtmw3d-x model meta (public/inference/rtmw3d-x). Input is CHW [1,3,resH,resW].
export const RTMW3D_MODEL = {
  id: 'rtmw3d-x',
  url: 'inference/rtmw3d-x/inference_model.onnx',
  resW: 288,
  resH: 384,
  inputName: 'input',
  // SimCC output tensor names (3-axis): X, Y, Z.
  outX: 'output',
  outY: '1554',
  outZ: '1556'
};

export const YOLO_DET_MODEL = {
  id: 'yolo26n',
  url: 'inference/yolo26n/inference_model_320.onnx', // 320 export — person bbox doesn't need 384/512
  res: 320,
  inputName: 'images',
  personClassId: 0
};

// ImageNet RGB normalization (pipeline.json, to_rgb=true, 0-255 scale).
export const POSE_MEAN = [123.675, 116.28, 103.53];
export const POSE_STD = [58.395, 57.12, 57.375];

// TopDownGetBboxCenterScale padding (pipeline.json).
export const POSE_PADDING = 1.25;

// RTMPose3d depth scale (rtmlib default) — converts decoded z to root-relative.
export const Z_RANGE = 2.1744869;

// Semantic COCO-WholeBody indices used downstream (no raw indices elsewhere).
export const KPT = {
  nose: 0,
  leftEye: 1, rightEye: 2, leftEar: 3, rightEar: 4,
  leftShoulder: 5, rightShoulder: 6,
  leftElbow: 7, rightElbow: 8,
  leftWrist: 9, rightWrist: 10,
  leftHip: 11, rightHip: 12,
  leftKnee: 13, rightKnee: 14,
  leftAnkle: 15, rightAnkle: 16,
  leftBigToe: 17, leftSmallToe: 18, leftHeel: 19,
  rightBigToe: 20, rightSmallToe: 21, rightHeel: 22,
  // Hand keypoints: left 91-111, right 112-132 (21 each: wrist, then 5 fingers ×4).
  // Middle-finger base (MCP) gives a stable hand-pointing direction.
  leftHandRoot: 91, leftMiddleBase: 100,
  rightHandRoot: 112, rightMiddleBase: 121,
  // Finger MCP bases (knuckle line index→pinky gives palm/forearm roll axis).
  leftIndexBase: 96, leftPinkyBase: 108,
  rightIndexBase: 117, rightPinkyBase: 129
};
