This consolidates the full implementation plan into one handoff.

# Implementation Handoff: Real-Time RTMW3D Full-Body Avatar Control in Three.js

## 1. Objective

Build a real-time browser-based system that uses RTMW3D pose output to drive a rigged biped character rendered in Three.js.

Assume the existing RTMW3D integration provides:

* 133 two-dimensional keypoints
* 133 three-dimensional keypoints
* Per-keypoint confidence values
* Frame timestamps
* Real-time webcam inference

The implementation must support:

* Direct full-body driving from 3D pose data
* Anchored in-place full-body capture
* Optional free-roaming root translation
* Squatting, leaning, lunging, knee raises, one-leg balance, slow kicks and arm gestures
* Toggleable body constraints
* Strong flicker, dropout and teleport protection
* Floor and camera calibration
* Visual indication of the inferred floor, performer position and confidence
* Runtime comparison between raw, constrained, grounded and solved modes

The system must treat RTMW3D as a measurement source, not as unquestioned final mocap output.

---

# 2. Product modes

The implementation must expose multiple drive modes through one shared pipeline.

```ts
export type PoseDriveMode =
  | "direct3D"
  | "direct3DConstrained"
  | "groundedDirect3D"
  | "anchoredSolved"
  | "hybrid";
```

## 2.1 Direct 3D

Use filtered RTMW3D positions as directly as possible.

* Convert 3D joint positions into bone rotations.
* Drive all major limbs.
* Drive pelvis and torso orientation.
* Apply confidence gating.
* Apply smoothing.
* Apply dropout protection.
* Reject only catastrophic measurements.
* Do not run the full body solver.
* Root translation is independently configurable.

Purpose:

* Establish the quality of the RTMW3D output.
* Preserve performer-specific motion and asymmetry.
* Avoid unnecessary correction.

## 2.2 Direct 3D constrained

Use direct 3D reconstruction with a lightweight safety layer.

Apply:

* Fixed bone lengths
* Joint limits
* Angular velocity limits
* Bend-plane stabilization
* Floor penetration prevention
* Last-known-good fallback
* Smooth reacquisition

The body solver must not invent the pose in this mode. It may only correct invalid or unstable results.

This should be the initial default mode.

## 2.3 Grounded direct 3D

Use direct 3D body driving plus optional support-foot stabilization.

Apply:

* Contact classification
* Floor collision
* Soft or hard foot locking
* Leg IK
* Optional root correction

This mode should preserve the source pose as much as possible while reducing skating and hovering.

## 2.4 Anchored solved

Keep the character at a fixed world position.

Allow the body and pelvis to move within a bounded local space.

Use stronger constraints for:

* Squatting
* One-leg balance
* Lunging
* Crouching
* Weight shifting
* Local forward and sideways movement

This mode should work without valid floor calibration.

## 2.5 Hybrid

Allow different body regions to use different strategies.

Example:

```text
root              calibrated root estimator
pelvis            constrained direct 3D
upper body        direct 3D
support leg       contact-constrained IK
free leg          direct 3D
feet              grounded
```

This is likely to become the highest-quality final mode.

---

# 3. Input data contract

```ts
export interface RTMWPoseFrame {
  timestampMs: number;
  inferenceDurationMs?: number;

  keypoints2D: Array<{
    x: number;
    y: number;
    confidence: number;
  }>;

  keypoints3D: Array<{
    x: number;
    y: number;
    z: number;
    confidence: number;
  }>;

  boundingBox?: {
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
  };
}
```

Do not use raw RTMW indices throughout the codebase.

Create one semantic mapping module.

```ts
export interface PoseIndexMap {
  nose: number;

  leftShoulder: number;
  rightShoulder: number;
  leftElbow: number;
  rightElbow: number;
  leftWrist: number;
  rightWrist: number;

  leftHip: number;
  rightHip: number;
  leftKnee: number;
  rightKnee: number;
  leftAnkle: number;
  rightAnkle: number;

  leftHeel?: number;
  rightHeel?: number;
  leftBigToe?: number;
  rightBigToe?: number;
  leftSmallToe?: number;
  rightSmallToe?: number;

  leftHand: number[];
  rightHand: number[];
  face: number[];
}
```

All downstream modules must use semantic joint names.

---

# 4. High-level architecture

```text
Webcam
  ↓
RTMW3D inference
  ↓
Pose observation adapter
  ↓
Coordinate conversion
  ↓
Confidence and validity checks
  ↓
Outlier rejection
  ↓
Temporal filtering
  ↓
Bone-length normalization
  ↓
Canonical skeleton
  ↓
Pose drive mode
  ├── direct 3D
  ├── constrained direct 3D
  ├── grounded direct 3D
  ├── anchored solver
  └── hybrid
  ↓
Rig retargeting
  ↓
Optional IK and grounding
  ↓
Final movement limits
  ↓
Three.js skeleton update
```

Suggested module structure:

```text
pose/
  RTMWPoseProvider
  RTMWIndexMap
  PoseObservationAdapter
  PoseFrameBuffer
  PoseValidityChecker
  PoseConfidenceFilter
  PoseRecorder

skeleton/
  CanonicalSkeleton
  PerformerCalibration
  BoneLengthNormalizer
  JointHierarchy
  PoseQualityEstimator

robustness/
  JointHistoryStore
  OutlierRejector
  DepthFlipDetector
  LeftRightSwapDetector
  DropoutManager
  RecoveryManager
  TeleportBarrier

solver/
  SupportStateEstimator
  PoseFamilyClassifier
  PelvisSolver
  AnchoredBodySolver
  JointConstraintSolver
  TwoBoneIK
  FootStabilizer

root/
  RootMotionEstimator
  FixedRootEstimator
  DirectPelvisRootEstimator
  ImageScaleRootEstimator
  FloorContactRootEstimator
  HybridRootEstimator

retargeting/
  HumanoidRigDefinition
  RigCalibration
  HumanoidRetargeter
  SwingTwistSolver

calibration/
  CameraCalibration
  MarkerCalibration
  ManualFloorCalibration
  FloorProjection
  CalibrationValidator
  CameraDriftDetector

runtime/
  CharacterPoseController
  TrackingModeStateMachine
  RuntimeConfig
  PoseDebugRenderer
  CalibrationDebugRenderer
  MetricsCollector
```

---

# 5. Coordinate conventions

Convert RTMW output into one canonical coordinate system immediately.

Recommended convention:

```text
+x = performer right
+y = up
+z = performer forward
origin = midpoint between the hips
```

The adapter must:

* Detect and correct axis direction.
* Detect mirrored video.
* Correct left-right labels where needed.
* Recenter local 3D around the pelvis midpoint.
* Convert scale into metres or normalized character units.
* Preserve raw values for debugging.
* Keep 2D and 3D confidence values.

```ts
export type CanonicalJointName =
  | "pelvis"
  | "spineLower"
  | "spineUpper"
  | "chest"
  | "neck"
  | "head"

  | "leftClavicle"
  | "leftShoulder"
  | "leftElbow"
  | "leftWrist"
  | "leftHand"

  | "rightClavicle"
  | "rightShoulder"
  | "rightElbow"
  | "rightWrist"
  | "rightHand"

  | "leftHip"
  | "leftKnee"
  | "leftAnkle"
  | "leftHeel"
  | "leftToe"

  | "rightHip"
  | "rightKnee"
  | "rightAnkle"
  | "rightHeel"
  | "rightToe";
```

```ts
export interface CanonicalPoseObservation {
  timestampMs: number;

  joints: Record<CanonicalJointName, {
    position2D: THREE.Vector2;
    position3D: THREE.Vector3;
    confidence: number;
    depthConfidence: number;
  }>;

  pelvisCenter3D: THREE.Vector3;
  shoulderCenter3D: THREE.Vector3;

  rawFrame: RTMWPoseFrame;
}
```

---

# 6. Performer calibration

Require a short neutral calibration pose.

Recommended pose:

* Full body visible
* Standing upright
* Feet slightly separated
* Arms slightly away from torso
* Facing approximately toward the camera
* Held for two to four seconds

Calculate robust median measurements.

```ts
export interface PerformerCalibration {
  bodyHeight: number;

  shoulderWidth: number;
  hipWidth: number;

  leftUpperArmLength: number;
  leftForearmLength: number;
  rightUpperArmLength: number;
  rightForearmLength: number;

  leftThighLength: number;
  leftShinLength: number;
  rightThighLength: number;
  rightShinLength: number;

  leftFootLength?: number;
  rightFootLength?: number;

  neutralPelvisHeight: number;
  neutralFootSeparation: number;

  sourceFacingRotation: THREE.Quaternion;
  sourceScale: number;
}
```

Bone lengths must not update independently every frame.

Use fixed calibrated lengths to prevent:

* Limb stretching
* Apparent body-scale breathing
* Asymmetric leg lengths
* Depth noise propagating to the avatar
* Pelvis-height flicker

---

# 7. Character hierarchy

Use three separate root transforms.

```text
StageRoot
└── BodyRoot
    └── RigRoot
        └── Skeleton
```

## StageRoot

* Represents the avatar’s world placement.
* Fixed in anchored mode.
* Moved by the root estimator in free-roaming modes.

## BodyRoot

* Represents local pelvis and body movement.
* Used for squatting, leaning and balance.
* Constrained to a local range.

## RigRoot

* Contains avatar-specific scale and orientation corrections.
* Converts canonical skeleton space into rig space.

```ts
export interface AnchoredBodyLimits {
  maxLateralOffset: number;
  maxForwardOffset: number;
  maxBackwardOffset: number;
  maxDownwardOffset: number;
  maxUpwardOffset: number;
  maxYawRadians: number;
}
```

Suggested initial values:

```ts
const limits: AnchoredBodyLimits = {
  maxLateralOffset: bodyHeight * 0.25,
  maxForwardOffset: bodyHeight * 0.18,
  maxBackwardOffset: bodyHeight * 0.12,
  maxDownwardOffset: bodyHeight * 0.50,
  maxUpwardOffset: bodyHeight * 0.08,
  maxYawRadians: THREE.MathUtils.degToRad(50),
};
```

All values must remain configurable.

---

# 8. Direct 3D pose driving

The direct path must use the 3D skeleton as the primary observation.

```text
RTMW3D 3D positions
  ↓
coordinate conversion
  ↓
validity checks
  ↓
temporal filtering
  ↓
optional bone normalization
  ↓
segment direction extraction
  ↓
bone rotations
  ↓
optional constraints
  ↓
rig retargeting
```

For each bone:

```ts
const direction = childPosition
  .clone()
  .sub(parentPosition)
  .normalize();
```

Do not independently move every rig bone to the source positions.

Use the source positions to derive rotations and constrained root translation.

---

# 9. Pelvis orientation

Calculate pelvis orientation from the hips and torso.

```ts
const hipAxis = rightHip
  .clone()
  .sub(leftHip)
  .normalize();

const torsoUp = shoulderCenter
  .clone()
  .sub(hipCenter)
  .normalize();

const pelvisForward = hipAxis
  .clone()
  .cross(torsoUp)
  .normalize();

const pelvisUp = pelvisForward
  .clone()
  .cross(hipAxis)
  .normalize();
```

Maintain the previous valid pelvis orientation when:

* The basis becomes degenerate.
* Hip confidence is poor.
* Hips overlap.
* Left and right labels appear swapped.
* The user is in an ambiguous profile view.

Filter pelvis rotation more heavily than arms or wrists.

---

# 10. Root translation and free roaming

Local pose and global root translation must remain independent.

```text
local body pose = RTMW3D pelvis-relative 3D
world root      = selected root estimator
```

```ts
export interface RootMotionEstimator {
  reset(calibration: RuntimeCalibration): void;

  update(
    observation: CanonicalPoseObservation,
    previous: RootMotionState,
    dt: number,
  ): RootMotionState;
}

export interface RootMotionState {
  measuredPosition?: THREE.Vector3;
  acceptedPosition: THREE.Vector3;
  renderedPosition: THREE.Vector3;

  rotation: THREE.Quaternion;
  velocity: THREE.Vector3;
  confidence: number;
}
```

Implement:

```ts
class FixedRootEstimator
class DirectPelvisRootEstimator
class ImageScaleRootEstimator
class FloorContactRootEstimator
class HybridRootEstimator
```

## Root estimation priority

```text
valid support-foot floor anchors
  ↓
single reliable support foot
  ↓
stable camera-relative pelvis estimate
  ↓
image-space pelvis and scale estimate
  ↓
last-known-good root
```

Do not assume that any field labelled 3D contains usable absolute world translation.

---

# 11. Constraint configuration

All correction layers must be toggleable.

```ts
export interface ConstraintWeights {
  boneLength: number;
  jointLimits: number;
  temporalContinuity: number;
  bendPlane: number;
  footContact: number;
  floorCollision: number;
  pelvisCorrection: number;
}
```

Interpretation:

```text
0.0 = disabled
1.0 = full correction
```

Recommended presets:

```ts
export const constraintPresets = {
  rawDirect: {
    boneLength: 0.0,
    jointLimits: 0.0,
    temporalContinuity: 0.2,
    bendPlane: 0.0,
    footContact: 0.0,
    floorCollision: 0.0,
    pelvisCorrection: 0.0,
  },

  safeDirect: {
    boneLength: 0.8,
    jointLimits: 0.5,
    temporalContinuity: 0.5,
    bendPlane: 0.7,
    footContact: 0.0,
    floorCollision: 0.5,
    pelvisCorrection: 0.2,
  },

  groundedDirect: {
    boneLength: 1.0,
    jointLimits: 0.8,
    temporalContinuity: 0.6,
    bendPlane: 0.9,
    footContact: 0.8,
    floorCollision: 1.0,
    pelvisCorrection: 0.6,
  },

  anchored: {
    boneLength: 1.0,
    jointLimits: 1.0,
    temporalContinuity: 0.8,
    bendPlane: 1.0,
    footContact: 1.0,
    floorCollision: 1.0,
    pelvisCorrection: 1.0,
  },
};
```

---

# 12. Flicker and teleport protection

No raw RTMW measurement may be applied directly to the avatar.

Required pipeline:

```text
raw joint measurement
  ↓
finite and confidence checks
  ↓
left-right continuity check
  ↓
bone-length validation
  ↓
motion prediction comparison
  ↓
depth-flip detection
  ↓
accept or reject
  ↓
accepted:
  update last-known-good
  filter normally

rejected:
  predict briefly
  then hold
  then fallback
  ↓
require stable recovery
  ↓
smooth reacquisition
  ↓
final movement limit
  ↓
avatar
```

---

# 13. Joint tracking state

```ts
export type JointTrackingState =
  | "stable"
  | "suspect"
  | "predicted"
  | "held"
  | "recovering"
  | "lost";

export interface JointHistoryState {
  state: JointTrackingState;

  lastRawPosition: THREE.Vector3;
  lastAcceptedPosition: THREE.Vector3;
  lastGoodPosition: THREE.Vector3;

  filteredPosition: THREE.Vector3;
  filteredRotation: THREE.Quaternion;

  velocity: THREE.Vector3;
  acceleration: THREE.Vector3;

  confidence: number;
  depthConfidence: number;

  lastGoodTimestampMs: number;
  invalidSinceMs?: number;
  recoveringSinceMs?: number;

  consecutiveGoodFrames: number;
  consecutiveBadFrames: number;
}
```

`lastGoodPosition` must only update when a measurement passes all validity checks.

---

# 14. Outlier rejection

Reject or mark measurements suspect when:

* Confidence is below threshold.
* Values are non-finite.
* A joint leaves the plausible body volume.
* Velocity exceeds the joint-specific limit.
* Acceleration exceeds the joint-specific limit.
* Bone length changes suddenly.
* Depth changes implausibly.
* Elbow or knee orientation flips.
* Left and right joints swap.
* The pelvis teleports.
* The torso basis flips.
* Body scale changes abruptly.

Use prediction-based innovation.

```ts
const predictedPosition = lastAcceptedPosition
  .clone()
  .addScaledVector(velocity, dt);

const innovation = measuredPosition
  .clone()
  .sub(predictedPosition);

const innovationDistance = innovation.length();
```

Adaptive threshold:

```ts
const allowedInnovation =
  baseTolerance +
  speedTolerance * currentSpeed +
  uncertaintyTolerance * trackingUncertainty;
```

---

# 15. Depth-flip protection

RTMW3D may produce large Z changes with little XY motion.

```ts
const xyMovement = current2D.distanceTo(previous2D);
const zMovement = Math.abs(current3D.z - previous3D.z);

const likelyDepthFlip =
  xyMovement < xyStableThreshold &&
  zMovement > zJumpThreshold;
```

When detected:

* Keep previous accepted depth.
* Continue accepting reliable XY motion.
* Mark depth as suspect.
* Require several consistent frames before accepting the new depth.
* Preserve previous limb plane and twist.

Derive a separate runtime depth confidence even if RTMW provides only one score.

---

# 16. Bone-length consistency

```ts
const measuredLength =
  childPosition.distanceTo(parentPosition);

const expectedLength =
  calibration.boneLengths[boneName];

const lengthRatio =
  measuredLength / expectedLength;
```

Initial interpretation:

```text
0.85 to 1.15    acceptable
0.70 to 1.30    suspicious
outside range   reject or reconstruct
```

When one joint is invalid:

1. Keep the valid endpoint.
2. Use the previous valid bone direction.
3. Reconstruct the missing endpoint at the calibrated length.
4. Blend toward new measurements after recovery.

---

# 17. Dropout behavior

Use staged fallback.

## Prediction

For approximately 50 to 100 ms:

* Predict from recent velocity.
* Decay velocity quickly.
* Do not predict global root movement aggressively.

## Hold

After prediction:

* Hold the last-known-good rotation or position.
* Preserve the limb hierarchy.
* Reduce tracking weight gradually.

## Fallback

For longer loss:

* Blend the affected region toward a neutral or animated fallback.
* Continue tracking unaffected regions.

## Recovery

Require:

* Several consecutive valid frames
* Plausible bone lengths
* Stable confidence
* Consistent position trend

Recommended recovery confirmation:

```text
3 to 6 consecutive frames
```

Blend smoothly from the held pose to the reacquired pose.

---

# 18. Root anti-teleport state machine

```ts
export type RootTrackingState =
  | "tracking"
  | "suspect"
  | "holding"
  | "recentering"
  | "lost";
```

## Tracking

Use accepted root measurements.

## Suspect

When a large jump appears:

* Reject the measurement.
* Continue previous velocity briefly.
* Decelerate.
* Do not move to the new position.

## Holding

If valid root data does not return:

* Stop root movement smoothly.
* Preserve the current world position.
* Continue local body tracking.

## Recentering

For large reacquisition offsets:

* Preserve the current rendered avatar position.
* Reset the tracking origin.
* Do not snap.

```ts
renderedRootPosition =
  trackingOriginOffset
    .clone()
    .add(trackedRootPosition);
```

On reacquisition:

```ts
trackingOriginOffset =
  renderedRootPosition
    .clone()
    .sub(newTrackedRootPosition);
```

---

# 19. Final teleport barrier

Even accepted target motion must have maximum application rates.

Use:

* Maximum position delta
* Maximum angular delta
* Maximum speed
* Maximum acceleration

```ts
function moveTowardsLimited(
  current: THREE.Vector3,
  target: THREE.Vector3,
  maxDistance: number,
): THREE.Vector3 {
  const delta = target.clone().sub(current);

  if (delta.length() <= maxDistance) {
    return target.clone();
  }

  return current
    .clone()
    .add(
      delta
        .normalize()
        .multiplyScalar(maxDistance),
    );
}
```

Add a hard barrier:

```ts
if (
  targetPosition.distanceTo(renderedPosition)
  > hardTeleportDistance
) {
  rejectTarget();
  enterRecoveryState();
}
```

A temporary inaccurate pose is preferable to a visible teleport.

---

# 20. Left-right swap prevention

For paired joints, compare two assignments:

```text
normal assignment:
  observed left  → previous left
  observed right → previous right

swapped assignment:
  observed left  → previous right
  observed right → previous left
```

Use continuity cost from:

* Position
* Velocity
* Depth
* Bone lengths
* Torso-side consistency

Apply to:

* Shoulders
* Elbows
* Wrists
* Hips
* Knees
* Ankles
* Feet
* Hands

---

# 21. Temporal filtering

Filtering must occur after outlier rejection.

Recommended initial approach:

* One Euro filter for positions
* Exponential smoothing for translations
* Quaternion slerp for rotations
* Explicit velocity and acceleration tracking
* State-based dropout and recovery

Use body-part-specific filter groups.

```ts
export type JointFilterGroup =
  | "root"
  | "pelvis"
  | "torso"
  | "head"
  | "arms"
  | "hands"
  | "legs"
  | "feet";
```

Responsiveness ordering:

```text
hands and wrists       fastest
head and forearms
upper arms
chest and spine
pelvis orientation
legs
pelvis translation
root translation       slowest
```

Increase smoothing when uncertainty rises.

---

# 22. Joint orientation extraction

Do not use independent Euler-angle solving.

Use:

* Segment direction alignment
* Swing-twist decomposition
* Secondary limb planes
* Previous-frame twist preservation
* Joint limits

For arms:

* Shoulder to elbow gives upper-arm direction.
* Elbow to wrist gives forearm direction.
* Shoulder-elbow-wrist plane provides the bend plane.

For legs:

* Hip to knee gives thigh direction.
* Knee to ankle gives shin direction.
* Hip-knee-ankle plane provides the knee bend plane.

When nearly straight:

* Preserve the previous bend plane.
* Prevent pole-vector inversion.

---

# 23. Joint limits

Constrain at minimum:

* Knee flexion and extension
* Elbow flexion and extension
* Hip rotation and abduction
* Shoulder elevation and twist
* Spine bending and twist
* Neck rotation
* Ankle flexion
* Wrist rotation

Prefer an anatomically valid slightly inaccurate pose over an exact invalid pose.

Allow per-rig overrides for stylized avatars.

---

# 24. Support and contact states

```ts
export type SupportState =
  | "doubleSupport"
  | "leftSupport"
  | "rightSupport"
  | "noSupport"
  | "uncertain";
```

```ts
export type FootContactPhase =
  | "airborne"
  | "contactCandidate"
  | "planted"
  | "releaseCandidate";
```

Use:

* Heel velocity
* Toe velocity
* Ankle velocity
* Relative foot height
* Depth stability
* Leg extension
* Confidence
* Previous state
* Contact duration
* Current floor calibration validity

Use hysteresis.

Do not switch support state from one noisy frame.

---

# 25. Foot grounding modes

```ts
export type FootGroundingMode =
  | "off"
  | "floorOnly"
  | "soft"
  | "locked";
```

## Off

Feet follow direct RTMW3D output.

## Floor only

Prevent floor penetration.

Do not constrain horizontal motion.

## Soft

Reduce support-foot horizontal motion while preserving source movement.

## Locked

Store a fixed support-foot anchor.

Use IK and optional root correction to maintain it.

Do not update a planted anchor from every frame.

---

# 26. Squat handling

A squat must produce:

* Pelvis lowering
* Hip flexion
* Knee flexion
* Ankle flexion
* Torso compensation
* Stable feet
* Fixed limb lengths

For double support:

1. Establish support-foot targets.
2. Estimate observed knee and hip geometry.
3. Calculate a reachable pelvis position.
4. Solve both legs with IK if enabled.
5. Apply torso lean from direct tracking.
6. Preserve performer asymmetry where possible.
7. Clamp impossible solutions.

Success means:

* Pelvis visibly lowers.
* Feet remain stable.
* Legs remain reachable.
* Knees bend consistently.
* No floor penetration occurs.

---

# 27. One-leg balance handling

For one-leg stance:

1. Detect support side.
2. Preserve the support-foot anchor.
3. Shift pelvis toward the support side.
4. Stabilize the support leg.
5. Drive the free leg from filtered RTMW3D.
6. Preserve torso and arm counterbalance.
7. Prevent support-side switching during short dropouts.

Success means:

* Correct support side.
* Stable support foot.
* Visible pelvis shift.
* Responsive raised leg.
* No depth inversion.

---

# 28. Pose families

Pose families may adjust solver weights without triggering canned animation.

```ts
export type PoseFamily =
  | "neutral"
  | "squat"
  | "leftSingleSupport"
  | "rightSingleSupport"
  | "leftLunge"
  | "rightLunge"
  | "kneeRaise"
  | "slowKick"
  | "torsoBend"
  | "uncertain";
```

Use them only in anchored and hybrid modes where useful.

Direct modes should remain as unconstrained as safely possible.

---

# 29. Calibration system

Implement a guided calibration workflow that establishes:

* Camera intrinsics
* Floor plane
* Stage origin
* Stage orientation
* Metric or normalized scale
* Performance boundary
* Camera stability
* User position relative to the floor

```ts
export type CalibrationState =
  | "uncalibrated"
  | "detecting"
  | "collecting"
  | "validating"
  | "valid"
  | "degraded"
  | "invalid";
```

World-space root translation must remain disabled when calibration is invalid.

Local body pose must continue working.

---

# 30. Calibration methods

```ts
export type CalibrationMethod =
  | "markers"
  | "manualFloorCorners";
```

## Marker mode

Preferred workflow.

Use four or more:

* AprilTags
* ArUco markers
* ChArUco markers

Each marker has:

* Known ID
* Known physical size
* Known position
* Known orientation

Use six or more where possible for redundancy.

## Manual floor mode

Allow the user to click four or more floor points.

Ask for:

* Stage width
* Stage depth
* Corner ordering
* Origin
* Forward direction

This produces a floor homography with lower confidence.

---

# 31. Camera intrinsics

```ts
export interface CameraIntrinsics {
  imageWidth: number;
  imageHeight: number;

  fx: number;
  fy: number;
  cx: number;
  cy: number;

  distortion?: number[];
}
```

Support:

```ts
export type IntrinsicsQuality =
  | "assumed"
  | "estimated"
  | "measured";
```

Measured intrinsics should use a checkerboard or ChArUco workflow.

---

# 32. Floor calibration result

```ts
export interface FloorCalibration {
  version: number;

  method: CalibrationMethod;
  state: CalibrationState;

  cameraIntrinsics: CameraIntrinsics;

  cameraToStage: THREE.Matrix4;
  stageToCamera: THREE.Matrix4;

  floorPlane: {
    normal: THREE.Vector3;
    constant: number;
  };

  stageOrigin: THREE.Vector3;
  stageForward: THREE.Vector3;
  stageRight: THREE.Vector3;
  stageUp: THREE.Vector3;

  performancePolygon: THREE.Vector3[];

  floorHomography?: number[];

  reprojectionErrorPx: number;
  floorValidationErrorM?: number;

  confidence: number;

  createdAtMs: number;
  lastValidatedAtMs: number;
}
```

Stage coordinates:

```text
+x = stage right
+y = up
+z = stage forward
floor = y = 0
```

---

# 33. Calibration UI process

## Step 1: camera placement

Show:

* Full-body framing guide
* Required visible floor area
* Camera stability instructions
* Current resolution and crop state

## Step 2: marker or corner detection

Show:

* Marker outlines
* Marker IDs
* Missing markers
* Detection stability
* Manual corner handles where applicable

## Step 3: stage configuration

Allow:

* Width
* Depth
* Origin
* Forward direction
* Avatar start point

## Step 4: geometric validation

Show:

* Detected marker corners
* Reprojected marker corners
* Pixel error vectors
* Mean error
* Maximum error
* Estimated camera height
* Estimated camera angle

## Step 5: performer validation

Ask performer to stand near:

* Centre
* Left
* Right
* Front
* Back

Display the estimated floor position.

## Step 6: save

Allow:

* Save
* Rename
* Export JSON
* Import JSON
* Revalidate
* Reset

---

# 34. Live calibration overlays

The webcam view must display:

* Marker detections
* Floor polygon
* Floor grid
* Stage origin
* Forward direction
* Performance boundary
* Warning zones
* Raw 2D skeleton
* Support-foot observations
* Foot-floor projections
* Pelvis floor projection
* Measured root
* Accepted root
* Rendered root
* Last-known-good root

The UI must distinguish:

```ts
export type SpatialMeasurementType =
  | "confirmedContact"
  | "likelyContact"
  | "projectedEstimate"
  | "heldLastKnownGood"
  | "invalid";
```

Example visual language:

```text
solid marker      confirmed
outlined marker   likely
dashed marker     estimate
faded marker      held
crossed marker    invalid
```

---

# 35. Three-dimensional calibration debug view

Provide a separate Three.js debug scene showing:

* Camera frustum
* Camera position
* Floor plane
* Stage grid
* Performance boundary
* Calibration markers
* Raw RTMW3D skeleton
* Filtered skeleton
* Canonical skeleton
* Estimated world root
* Accepted root
* Rendered root
* Foot rays
* Floor intersections
* Foot anchors
* Avatar skeleton

The operator must be able to orbit this debug scene.

---

# 36. Floor projection

```ts
export function imagePointToFloor(
  imagePoint: THREE.Vector2,
  calibration: FloorCalibration,
): THREE.Vector3 | null;
```

Return `null` when:

* Calibration is invalid.
* Ray is nearly parallel to the floor.
* Intersection lies behind the camera.
* Intersection is far outside the calibrated region.
* The source point is not believed to contact the floor.

Do not project airborne feet and treat them as measured floor positions.

---

# 37. Calibration confidence

```ts
export interface CalibrationQuality {
  overall: number;

  markerVisibility: number;
  reprojection: number;
  cameraStability: number;
  floorCoverage: number;
  currentRegionConfidence: number;
}
```

Use:

* Reprojection error
* Marker count
* Marker stability
* Camera pose consistency
* Floor validation
* Video resolution
* Crop state
* Mirrored state
* Position inside the performance region

Show component scores, not only one combined number.

---

# 38. Camera movement detection

```ts
export interface CameraDriftState {
  positionErrorM: number;
  rotationErrorDeg: number;
  reprojectionErrorPx: number;

  suspectedMovement: boolean;
  calibrationInvalid: boolean;
}
```

Small drift:

* Mark calibration degraded.
* Increase root smoothing.
* Reduce root confidence.

Large drift:

* Mark calibration invalid.
* Freeze world root translation.
* Preserve local pose tracking.
* Request recalibration.

Camera movement must never teleport the avatar.

---

# 39. Performance boundary

```ts
export type BoundaryState =
  | "inside"
  | "warning"
  | "outside";
```

Inside:

* Normal root tracking.

Warning:

* Display warning.
* Reduce confidence near edge regions.

Outside:

* Stop trusting floor projection.
* Hold or softly clamp root.
* Continue local pose.
* Do not snap to the boundary.

---

# 40. Calibration persistence

```ts
export interface CameraFingerprint {
  deviceId?: string;
  width: number;
  height: number;
  frameRate?: number;
  mirrored: boolean;
  cropSignature?: string;
}
```

On startup:

1. Load stored calibration.
2. Compare camera fingerprint.
3. Validate markers if available.
4. Recalculate reprojection error.
5. Activate only after successful validation.

Do not reuse calibration across different resolutions or mirrored states without validation.

---

# 41. Rig adapter

Start with one standardized humanoid rig.

Preferred:

* VRM
* Controlled Mixamo skeleton
* Project-specific standardized GLB

```ts
export interface HumanoidRigDefinition {
  bones: Partial<
    Record<CanonicalJointName, THREE.Bone>
  >;

  restLocalRotations: Partial<
    Record<CanonicalJointName, THREE.Quaternion>
  >;

  restDirections: Partial<
    Record<CanonicalJointName, THREE.Vector3>
  >;

  jointLimits: Partial<
    Record<CanonicalJointName, JointLimit>
  >;

  scale: number;
}
```

Provide a rig calibration tool showing:

* Bone mapping
* Rest axes
* Current source direction
* Retargeted direction
* Missing bones
* Joint limits

Do not support arbitrary unknown rigs initially.

---

# 42. Three.js update order

```ts
function updateCharacter(dt: number): void {
  const frame =
    poseFrameBuffer.getLatestUsableFrame();

  const rawObservation =
    observationAdapter.convert(frame);

  const validatedObservation =
    poseValidityChecker.evaluate(
      rawObservation,
      jointHistory,
      dt,
    );

  const robustObservation =
    dropoutManager.update(
      validatedObservation,
      jointHistory,
      dt,
    );

  const filteredObservation =
    poseFilter.update(
      robustObservation,
      dt,
    );

  const normalizedObservation =
    boneLengthNormalizer.apply(
      filteredObservation,
      performerCalibration,
    );

  const supportState =
    supportEstimator.update(
      normalizedObservation,
      dt,
    );

  const localPose =
    poseDriver.solve({
      mode: runtimeConfig.poseDriveMode,
      observation: normalizedObservation,
      supportState,
      previousPose,
      calibration: performerCalibration,
      dt,
    });

  const rootState =
    rootMotionEstimator.update(
      normalizedObservation,
      previousRootState,
      dt,
    );

  jointConstraintSolver.apply(
    localPose,
    runtimeConfig.constraintWeights,
  );

  humanoidRetargeter.apply(
    localPose,
    avatarRig,
  );

  footStabilizer.apply({
    pose: localPose,
    rig: avatarRig,
    rootState,
    supportState,
    groundingMode:
      runtimeConfig.footGroundingMode,
  });

  finalTeleportBarrier.apply({
    rig: avatarRig,
    rootState,
    dt,
  });

  avatarRig.root.updateMatrixWorld(true);
  avatarRig.skeleton.update();

  previousPose.copyFrom(localPose);
  previousRootState.copyFrom(rootState);
}
```

If an animation mixer is active, all live pose overrides occur after `mixer.update(dt)`.

---

# 43. Runtime configuration

```ts
export interface PoseDriveConfig {
  mode: PoseDriveMode;

  root: {
    enabled: boolean;
    mode:
      | "fixed"
      | "direct"
      | "estimated"
      | "floorConstrained";

    translationScale: number;
    maxSpeed: number;
    maxAcceleration: number;
    recenterEnabled: boolean;
  };

  filtering: {
    enabled: boolean;
    confidenceGating: boolean;
    adaptiveSmoothing: boolean;
    dropoutProtection: boolean;
    predictionMs: number;
  };

  normalization: {
    enforceBoneLengths: boolean;
    strength: number;
  };

  constraints: {
    enabled: boolean;
    weights: ConstraintWeights;
  };

  grounding: {
    enabled: boolean;
    mode: FootGroundingMode;
    contactDetection: boolean;
    rootCorrection: boolean;
  };

  robustness: {
    outlierRejection: boolean;
    depthFlipProtection: boolean;
    leftRightSwapProtection: boolean;
    useLastKnownGood: boolean;
    smoothRecovery: boolean;
    hardTeleportBarrier: boolean;
  };
}
```

Initial default:

```ts
const defaultConfig: PoseDriveConfig = {
  mode: "direct3DConstrained",

  root: {
    enabled: false,
    mode: "fixed",
    translationScale: 1,
    maxSpeed: 3,
    maxAcceleration: 8,
    recenterEnabled: true,
  },

  filtering: {
    enabled: true,
    confidenceGating: true,
    adaptiveSmoothing: true,
    dropoutProtection: true,
    predictionMs: 75,
  },

  normalization: {
    enforceBoneLengths: true,
    strength: 0.8,
  },

  constraints: {
    enabled: true,
    weights:
      constraintPresets.safeDirect,
  },

  grounding: {
    enabled: false,
    mode: "off",
    contactDetection: false,
    rootCorrection: false,
  },

  robustness: {
    outlierRejection: true,
    depthFlipProtection: true,
    leftRightSwapProtection: true,
    useLastKnownGood: true,
    smoothRecovery: true,
    hardTeleportBarrier: true,
  },
};
```

---

# 44. Tracking quality modes

```ts
export type TrackingMode =
  | "fullBody"
  | "upperBodyOnly"
  | "holdLastPose"
  | "recovering"
  | "lost";
```

## Full body

Use when required joints are reliable.

## Upper body only

Use when legs or feet become unreliable.

* Blend lower body toward stable neutral.
* Continue torso, head and arms.
* Freeze or reduce root motion.

## Hold last pose

Use for short dropouts.

## Recovering

Blend gradually back to tracking.

## Lost

Blend to neutral idle.

The whole character must not fail because one joint is missing.

---

# 45. Debug tooling

Required overlays:

* Raw 2D keypoints
* Raw 3D skeleton
* Accepted skeleton
* Filtered skeleton
* Bone-normalized skeleton
* Constrained skeleton
* Anchored solved skeleton
* Final avatar skeleton
* Joint confidence
* Depth confidence
* Rejected joints
* Predicted joints
* Held joints
* Recovering joints
* Support state
* Foot anchors
* Root states
* Floor calibration
* Calibration quality
* Per-stage latency

Required plots:

* Raw and accepted X/Y/Z
* Confidence
* Depth confidence
* Innovation distance
* Velocity
* Acceleration
* Bone-length ratio
* Tracking state
* Root displacement
* Foot drift

Required diagnostics:

```ts
export interface TrackingDiagnosticEvent {
  timestampMs: number;

  type:
    | "joint-outlier"
    | "depth-flip"
    | "root-teleport"
    | "left-right-swap"
    | "tracking-loss"
    | "tracking-recovered"
    | "camera-drift"
    | "calibration-invalid";

  joint?: CanonicalJointName;
  measuredDelta?: number;
  confidence?: number;
}
```

---

# 46. Recording and replay

```ts
export interface RecordedPoseSession {
  metadata: {
    createdAt: string;
    modelName: string;
    inputWidth: number;
    inputHeight: number;
    mirrored: boolean;
  };

  calibration?: PerformerCalibration;
  floorCalibration?: FloorCalibration;

  frames: Array<{
    raw: RTMWPoseFrame;
    canonical?: CanonicalPoseObservation;
    supportState?: SupportState;
    trackingMode?: TrackingMode;
    rootState?: RootMotionState;
  }>;
}
```

The same recording must be replayable through:

* Raw direct mode
* Constrained direct mode
* Grounded mode
* Anchored solved mode
* Hybrid mode

This is necessary for objective comparison.

---

# 47. Performance architecture

Requirements:

* RTMW inference runs outside the Three.js render loop.
* Use a Web Worker where possible.
* Do not queue old frames.
* Always process the newest available frame.
* Render at a separate rate from pose inference.
* Interpolate between accepted poses.

Scheduling:

```text
camera frame arrives
  ↓
if inference idle:
  process frame
else:
  replace pending frame
```

Initial targets:

* Render near 60 fps where hardware permits.
* Pose inference at 20 to 30 fps is acceptable.
* Upper-body motion-to-photon under 100 ms median.
* Lower-body and contact path under 150 ms.
* No main-thread inference stalls.
* No unbounded frame backlog.

---

# 48. Validation motions

Record and test:

1. Neutral standing for 10 seconds
2. Five shallow squats
3. Five deep squats
4. Left one-leg hold for 10 seconds
5. Right one-leg hold for 10 seconds
6. Alternating knee raises
7. Side lunges
8. Forward lunges
9. Slow forward kicks
10. Slow side kicks
11. Torso bends
12. Torso twists
13. Arm gestures during squats
14. Brief wrist occlusion
15. Brief knee occlusion
16. Brief foot occlusion
17. Full pose loss for one second
18. Reacquisition
19. Lateral movement through calibrated space
20. Forward and backward movement
21. Camera movement after calibration

---

# 49. Synthetic corruption tests

Inject into recorded data:

* One-frame wrist jump
* One-frame ankle jump
* Pelvis teleport
* Root teleport
* Left-right knee swap
* Z sign inversion
* Wrist missing for 200 ms
* Leg missing for 500 ms
* Complete pose loss
* Reacquisition at a shifted origin
* Confidence flicker around threshold
* Repeated frames
* Stale frames
* Variable frame intervals

Expected:

* No visible avatar teleport
* No root teleport
* Isolated limbs degrade independently
* Last-known-good state is preserved
* Short gaps are masked
* Long gaps blend to fallback
* Recovery is gradual
* Diagnostics identify each rejection

---

# 50. Acceptance criteria

## Stability

* No visible single-frame teleport.
* No knee or elbow inversion in supported motions.
* No uncontrolled depth flips.
* No continuous limb-length changes.
* Stable pelvis during neutral stance.

## Squats

* Pelvis lowers visibly.
* Both feet remain stable when grounding is enabled.
* Knees bend consistently.
* No leg stretching.
* No floor penetration.

## One-leg balance

* Correct support side.
* Stable support foot.
* Pelvis shifts toward support side.
* Raised leg remains responsive.
* No rapid support switching.

## Dropout handling

* Dropouts below 100 ms are visually masked.
* Medium dropouts hold or reconstruct the affected region.
* Longer loss blends to fallback.
* Recovery requires several good frames.
* Recovery does not snap.

## Root motion

* One bad root frame causes no visible movement.
* Calibration loss freezes world translation.
* Reacquisition does not teleport.
* Root movement obeys speed and acceleration limits.

## Calibration

* Floor grid aligns visually with the floor.
* Marker reprojection error remains below configured threshold.
* Camera movement is detected.
* User position is visibly represented.
* Measured, estimated, held and invalid positions are distinguishable.

## Responsiveness

* Upper body remains responsive.
* Robustness logic does not introduce unnecessary global latency.
* Pose inference does not stall rendering.

---

# 51. Implementation milestones

## Milestone 1: raw pose inspection

Deliver:

* Live RTMW3D input
* 2D overlay
* 3D skeleton view
* Confidence display
* Recording and replay
* Axis and mirroring verification

## Milestone 2: canonical skeleton

Deliver:

* Semantic joint mapping
* Canonical coordinate conversion
* Performer calibration
* Bone-length normalization
* Raw versus normalized comparison

## Milestone 3: robustness layer

Deliver:

* Outlier rejection
* Last-known-good state
* Depth-flip detection
* Left-right swap detection
* Prediction
* Hold
* Recovery
* Hard teleport barrier
* Diagnostic events

## Milestone 4: direct avatar retargeting

Deliver:

* Full-body bone rotation extraction
* Swing-twist handling
* Pelvis orientation
* One standardized rig
* Direct 3D mode
* Direct constrained mode

## Milestone 5: anchored pose mode

Deliver:

* Fixed StageRoot
* Local pelvis movement
* Squats
* One-leg balance
* Lunges
* Knee raises
* Slow kicks
* Bounded solver

## Milestone 6: calibration UI

Deliver:

* Manual floor calibration
* Marker calibration
* Floor grid
* Stage boundary
* Calibration quality
* User footprint
* Camera drift detection
* Save and load

## Milestone 7: root translation

Deliver:

* Root estimator interface
* Floor-based root
* Direct pelvis root experiment
* Root anti-teleport logic
* Origin recentering
* Free-roaming mode

## Milestone 8: grounding

Deliver:

* Contact classification
* Floor collision
* Soft grounding
* Hard foot locking
* Leg IK
* Root correction

## Milestone 9: hybrid mode

Deliver:

* Independent strategy per body region
* Support leg constrained
* Free leg direct
* Upper body direct
* Root calibrated
* Runtime mode switching

## Milestone 10: performance and compatibility

Deliver:

* Worker-based inference
* Latest-frame scheduling
* Pose interpolation
* Browser compatibility testing
* GPU profiling
* Latency report

---

# 52. Initial implementation order

Begin in this order:

1. Raw RTMW3D visualization
2. Recording and deterministic replay
3. Canonical coordinate conversion
4. Performer calibration
5. Last-known-good and anti-teleport layer
6. Direct 3D retargeting
7. Constraint toggles
8. Anchored pose mode
9. Calibration UI
10. Free-roaming root estimation
11. Grounding and foot locking
12. Hybrid mode

Do not begin with aggressive full-body solving.

First establish how much of the RTMW3D result can be used directly.

---

# 53. Out of scope for the first implementation

Do not initially support:

* Running
* Jumping with both feet airborne
* Full 360-degree turning
* Long back-facing capture
* Crawling
* Lying down
* Floor interaction
* Multi-person control
* Arbitrary unknown rigs
* Medical or biomechanical accuracy
* Highly accurate finger articulation
* Stairs or uneven terrain

---

# 54. Non-negotiable principles

## Preserve direct 3D as a first-class mode

The full body solver must remain optional.

## Reject before smoothing

A smoothing filter must not chase obviously invalid measurements.

## Never overwrite last-known-good with suspect data

Last-known-good state must remain trustworthy.

## Never teleport

If tracking becomes uncertain:

```text
use accepted measurement
  ↓
predict briefly
  ↓
hold last-known-good
  ↓
blend to fallback
```

## Separate local pose from world translation

A good local 3D pose does not automatically imply a reliable global root.

## Calibration must be visible

The user and developer must be able to see:

* What is considered floor
* Where the performer is estimated to be
* Which foot is supporting
* What root source is active
* Whether the calibration is valid

## Degrade functionality, not stability

When calibration or tracking fails:

* Freeze world translation.
* Preserve local pose where possible.
* Blend unreliable regions to fallback.
* Do not destabilize the whole avatar.

## Measure every correction

The debug system must show how much filtering, normalization, constraints, grounding and solving changed the source data.

The goal is to preserve as much valid RTMW3D motion as possible while preventing flicker, impossible poses, foot instability and teleports.
