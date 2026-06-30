# SPEC

## §G GOAL
Drive per-instance bones of GPU-skinned crowd via swappable sources (clip | pose) + spatial choreography patterns. Apex: ≥1 rig driven live by RTMW3D webcam pose, realtime.

## §C CONSTRAINTS
- WebGPU only. ⊥ WebGL fallback. `renderer.init()` ! resolve pre-render.
- `three` 0.185.0 (lockfile-pinned; `"latest"` in package.json). TSL/compute APIs version-fragile — verify before use.
- Pure ESM, no framework. Runtime controls ! via three **Inspector** (`renderer.inspector`), ⊥ custom DOM (only canvas + status line).
- Substrate = TSL compute skinning: per-instance bone matrices packed in `StorageBufferAttribute`, one `THREE.InstancedMesh` draw/batch. Modeled on `docs/webgpu_skinning_instancing_individual.html` — read before touching compute path.
- Bone-drive primitive = write 16-float mats into `boneMatrices.array` @ `walker.batchIndex*boneCount*16`. ∀ source feeds this slot.
- Pose = measurement, ⊥ unquestioned mocap.
- ⊥ convenience fallback for core fn → fix cause | throw clear err (global + project rule).
- Pose modules new under `src/pose,skeleton,robustness,solver,root,retargeting,calibration,runtime` per `docs/pose-driver.md` §4. Crowd substrate stays `src/main.js`.
- Input: RTMW3D = 133 keypoints (2D + 3D) COCO-WholeBody V5, realtime webcam. **NOT upstream/bridge — project runs inference in-browser** via `onnxruntime-web` (EP WebGPU | WASM). Top-down 2-stage: yolo26n detector .onnx → bbox affine-crop → RTMW (192×256) → 3-axis SimCC decode. Inference .onnx via file transfer (exported by external `object-detect` proj, not in-repo) → `public/inference/`; `check:models` preflight gates dev/build (warn-only). **Real ref shape (`I.reference`) diverges from `docs/pose-driver.md` §3** — adapter ! bridge.
- 3D output = root-relative normalized, ⊥ metric: `x=mx/(poseH/2)`, `y=my/(poseH/2)`, `z=(mz/(poseW/2)-1)*Z_RANGE`, `Z_RANGE=2.1744869` (rtmlib RTMPose3d). `score=0.5*(xMax+yMax)`; z maxima unreliable → derive depth confidence separately.
- Risk: `onnxruntime-web` WebGPU EP + three WebGPU renderer = 2 GPU consumers/page → contention. Renderer is WebGPU-only → decide EP upfront.
- v1 out-of-scope (`docs/pose-driver.md` §53): running, both-feet-airborne jump, 360 turn, long back-facing, crawl, lie-down, multi-person, arbitrary unknown rigs, finger articulation, uneven terrain.

## §I INTERFACES
- `I.choreo` — `state.choreography ∈ {synced, desync, ripple, wave, index, random}` → per-walker clip phase. `state.choreoDelay` = delay scale. Generalizes existing `state.animDesync`. `ripple` = radial offset from `armyBounds` center (inside→outside).
- `I.boneSource` — per-walker source ∈ `{clip, pose}`. Pose performers count = `state.posePerformerCount` (≥1).
- `I.poseInput` — target = `RTMWPoseFrame {timestampMs, inferenceDurationMs?, keypoints2D[], keypoints3D[], boundingBox?}` (`docs/pose-driver.md` §3). Semantic index map module; ⊥ raw RTMW indices downstream.
- `I.reference` — working in-browser RTMW impl @ `/Users/flo/work/code/object-detect/web/src/` (basic, imperfect). Files: `main.js` = inference (ONNX load, top-down detect→crop→RTMW, `decodeSimcc`/`decode3d` @ L1066/L1092, groups/edges @ L98); `pose3d.js` = 3D viewer only (pelvis-center mid(11,12), Y-flip, X-mirror, `depthScale` exaggeration); `bridge.js` = **unrelated** (object-detect over WS — ignore for pose). Shape: `persons[] = {kpts, kpts3d, box}`; kpt = `{x, y, z, score}` (field **`score`** ≠ `confidence`); COCO-WholeBody idx (leftHip=11, rightHip=12). `k3d` already root-relative normalized (see §C). No `timestampMs` — synthesize. Multi-person in data (poseMode all|cap3|cap2|single); v1 controls single.
- `I.canonical` — `CanonicalJointName`, `CanonicalPoseObservation` (`docs/pose-driver.md` §5). Origin = hip midpoint, +x right +y up +z forward.
- `I.poseMode` — `PoseDriveMode ∈ direct3D | direct3DConstrained | groundedDirect3D | anchoredSolved | hybrid` (`docs/pose-driver.md` §2). Default `direct3DConstrained`.
- `I.poseConfig` — `PoseDriveConfig` (`docs/pose-driver.md` §43): mode, root, filtering, normalization, constraints, grounding, robustness. `constraintPresets` rawDirect|safeDirect|groundedDirect|anchored.
- `I.inspector` — Inspector adds: choreography selector + delay, pose performer count, pose drive mode, constraint preset, EP selector (`poseEP` wasm|webgpu), debug overlays.
- `I.recording` — `RecordedPoseSession` (`docs/pose-driver.md` §46), replayable ∀ mode (objective compare).

## §V INVARIANTS

Substrate:
- V1: GPU-skinned batch draw ! `THREE.InstancedMesh`, ⊥ `Mesh`+`.count` (per-object bindings; `count===1` Mesh collapses onto shared buffer → all batches render same bind-pose).
- V2: render geometry indexed; vertex read = `instanceIndex*vertexCount+vertexIndex`; ⊥ `toNonIndexed`.
- V3: per-instance world transform baked into computed vertices buffer ∴ `InstancedMesh.instanceMatrix` = identity (else double-transform).
- V4: ∀ batch/frame → `boneMatrices.needsUpdate=true` & `instanceMatrices.needsUpdate=true` & `renderer.compute(node)` ∀ computeNode.
- V5: ∀ bone source writes 16-float mats into `boneMatrices.array` @ `walker.batchIndex*boneCount*16`; clip & pose share slot.
- V6: `selectAnimation` throws if mesh lacks chosen clip; ⊥ fallback.
- V7: choreography phase = pure fn(walker, elapsed, state) → reproducible (`hashRandom`, ⊥ `Math.random`) except `shufflePhase`.
- V8: frustum culling disabled on instanced meshes (one giant bound culls wrong).

Pose (`docs/pose-driver.md` §54):
- V9: direct3D first-class; full-body solver optional ∀ mode.
- V10: reject before smooth — filter ⊥ chase invalid measurement.
- V11: ⊥ overwrite last-known-good w/ suspect data.
- V12: ⊥ teleport. Uncertain → accepted → predict → hold → fallback.
- V13: local pose ⊥ coupled world root; good local pose ≠ reliable root.
- V14: calibration invalid → freeze world root translation, keep local pose.
- V15: ⊥ raw RTMW measurement applied direct to avatar (validity → outlier-reject → filter first, `docs/pose-driver.md` §12).
- V16: bone lengths fixed from `PerformerCalibration`, ⊥ per-frame update.
- V17: pose inference off main thread (worker), newest frame only, ⊥ frame backlog (`docs/pose-driver.md` §47).
- V18: tracking/calibration fail → degrade region, ⊥ destabilize whole avatar.
- V19: ∀ correction (filter/normalize/constraint/ground/solve) measurable in debug — show delta vs source.
- V20: pose performer = walker `boneSource=pose`; clip + pose instances coexist same batch; each snapshots skeleton independently in update loop.
- V21: adapter bridges `I.reference` → canonical: `score`→confidence, COCO-WholeBody idx (hips 11/12), Y-down→up, X un-mirror, normalized→canonical scale; ⊥ raw decode coords to rig. Single concrete index/field map module (V15-feed).
- V22: RTMW 3D space = root-relative normalized (z×`Z_RANGE`), ⊥ metric/world translation. World placement via root estimator + calibration only (reinforces V13).
- V23: RTMW ONNX EP runtime-switchable `{wasm, webgpu}` (`state.poseEP`); both impl'd, A/B measured (inference ms + render fps + contention) → pick default. ⊥ assume one wins.
- V24: drive bones from robust AGGREGATE targets (head = face-point centroid nose+eyes+ears), ⊥ single noisy keypoint. Per-region depth weight (arms full, torso/legs/head damped) — noisy monocular z ⊥ over-drive torso/knees.
- V25: visual smoothness decoupled from pose-fps via render-rate bone interpolation (slerp toward target each render frame); ⊥ snap at inference rate (stop-motion). Two layers: One Euro on canonical (pose-rate) + bone slerp (render-rate).

## §T TASKS
id|status|task|cites
T1|x|extract per-walker phase fn from `updateCrowdBatches` animOffset; add `state.choreography` + `state.choreoDelay`|V5,V7
T2|x|impl patterns synced/desync/ripple(radial inside→out via armyBounds center)/wave(axis)/index/random|V7,I.choreo
T3|x|Inspector controls for choreography + delay; fold existing animDesync into it|V7,I.choreo,I.inspector
T4|x|abstract BoneSource: clip writer (current mixer→snapshot path) vs pose writer; per-walker source select|V5,V20,I.boneSource
T5|~|RTMWPoseProvider (after T22): port `I.reference` in-browser inference (onnxruntime-web, yolo26n detect → affine crop → RTMW → `decode3d`); emit RTMWPoseFrame; raw 2D+3D overlay, conf display, axis/mirror verify (M1)|V15,V21,V22,I.poseInput,I.reference
T6|x|PoseRecorder + deterministic replay (`RecordedPoseSession`)|I.recording
T7|~|PoseObservationAdapter: bridge `I.reference` `decode3d` normalized space → canonical (score→conf, Y-flip, X un-mirror, pelvis-recenter mid(11,12)), L-R correct (M2)|V13,V21,V22,I.canonical,I.reference
T8|~|Calibration: guided neutral-hold (~2s) → median bone lengths → bone-length plausibility GATE (reject bad joints, §16→§14). Note: direction-based retarget needs no length-normalize. Pending: scale/floor capture for future|V16,I.canonical
T9|.|robustness: outlier reject, last-known-good, depth-flip, L-R swap, predict/hold/recover, hard teleport barrier, diagnostics (M3)|V10,V11,V12,V15
T10|~|temporal filter (One Euro pos, slerp rot, per-part groups) after rejection|V10,V16
T11|x|HumanoidRigDefinition + bone map `CanonicalJointName`→Meshy biped bone (? per-rig calibration)|V9,I.canonical
T12|~|HumanoidRetargeter: segment-dir + swing-twist → bone rotations onto shared skeleton; pose writer snapshots performer slice (M4)|V5,V9,V20
T13|~|joint limits (clamp bone rotation from rest, §23) done; direct3D modes + `PoseDriveConfig` presets pending|V9,I.poseMode,I.poseConfig
T14|~|**APEX**: drive ≥1 crowd rig live from webcam; mixed clip+pose crowd renders one frame|V12,V18,V20,I.boneSource
T15|.|anchoredSolved: fixed StageRoot, bounded BodyRoot, squat/one-leg/lunge/kneeRaise/kick (M5)|V9,V13,I.poseMode
T16|.|calibration UI: manual floor + markers, floor grid, boundary, quality, drift detect, save/load (M6)|V14
T17|.|root translation: estimator iface, floor-based root, anti-teleport state machine, recenter, free-roam (M7)|V12,V13,V14
T18|~|grounding: vertical foot-anchor done (offset Hips so lowest foot stays at rest floor → squats lower body, feet planted). Pending: contact hysteresis, horizontal lock, leg IK (M8)|V13
T19|.|hybrid mode: per-region strategy, runtime switch (M9)|V9,I.poseMode
T20|x|perf: RTMW in a Worker (own GPUDevice → no renderer contention; inference stable 71ms vs climbing 182), newest-frame, render-rate interpolation, stage-timing panel (M10)|V17,V23
T21|.|synthetic corruption + validation-motion tests (`docs/pose-driver.md` §48,§49)|V10,V11,V12
T22|~|**EP probe (before T5)**: impl both onnxruntime-web EPs (wasm + webgpu) behind `state.poseEP` selector; bench inference ms + render fps + contention w/ three WebGPU renderer; set winning default|V23,I.inspector,I.reference
T23|.|correction-measurement + diagnostics debug (§45, §2488): per-stage deltas (filter/clamp/depth/smooth changed-vs-source), `TrackingDiagnosticEvent` log, optional 3D debug skeleton|V19
T24|.|tracking-quality modes (§44): fullBody|upperBodyOnly|holdLastPose|recovering|lost; degrade region ⊥ whole avatar|V18
T25|.|pose families (§28): neutral/squat/single-support/lunge/kneeRaise → adjust solver weights, ⊥ canned anim (anchored/hybrid only)|V9

## §B BUGS
id|date|cause|fix
B1|2026-06-29|ONNX webgpu EP (jsep) contends w/ three WebGPU renderer device → `table index out of bounds` + driven mesh vanishes (GPU corruption). Reference dodged it (WebGL viewer). poseEP default was `webgpu`|V23; default poseEP→`wasm` (isolated). webgpu EP needs own adapter to be viable
B2|2026-06-29|`skeleton.pose()` rebuilds bind from boneInverses in pre-`normalizeModel` space → residual ~0.01 scale on normalized source → pose-driven mesh collapses to a point (vanishes). Clip path unaffected (no pose())|retargeter restores rest from load-captured normalized rest quats; ⊥ `skeleton.pose()` in driving path
B3|2026-06-29|Meshy `Spine` bone child sits below it (`Spine→Spine01 = -y`); aiming Spine up = ~180° flip → degenerate shortest-arc → constant dramatic chest roll (one shoulder up, one down)|drop `spine` segment; drive torso lean via `Hips` (`Hips→Spine = +y`). symmetry guard test
B4|2026-06-29|adapter copied reference VIEWER's all-axes negate (point reflection) → with `mirrorX` = double-flip → crossed/side-swapped arms|clean canonical frame matched to measured rig (+x image-right=rig left, +y up, +z depth); mirrorX default off
B6|2026-06-30|grounding wrote world-space foot delta onto Hips LOCAL position under normalizeModel's ~0.01 scale → ~100× too small → squat didn't lower body (legs raised instead)|convert world delta → hips-local via parent world scale (`/hipsParentScaleY`)
B5|2026-06-30|wasm EP OOM-crashes tab: app holds ~206MB GLB chars in renderer + wasm loads 369MB rtmw3d into wasm heap (+ArrayBuffer copy) → memory blowout. Ref ran wasm OK (no heavy scene)|wasm not viable for this model here → webgpu EP only; ease its renderer-GPU contention via per-inference rAF yield; real fix = worker (T20)
B7|2026-06-30|grounding read stale foot `matrixWorld`: feet are world-updated only by their SEGMENTS aim, SKIPPED on low foot/toe conf → foot pos lagged the moved upper-leg → hips bobbed + body sank through floor. Also EMA lagged lift dir → transient clip|`hips.updateWorldMatrix(false,true)` before reading feet (full subtree refresh); clamp `groundOffsetY ≥ targetDelta` (V13: lowest foot never below rest floor — instant lift, smooth drop)
