# SPEC

## §G GOAL
Drive per-instance bones of GPU-skinned crowd via swappable sources (clip | pose) + spatial choreography patterns. Apex: ≥1 rig driven live by RTMW3D webcam pose, realtime.
Extension (face): drive facial deform (jaw/chin, eyes, mouth) of face-less Meshy rigs from RTMW3D face landmarks (idx 23-90) via region-masked vertex displacement in the TSL positionNode compute path. Hero-performer scoped (⊥ 65-army).

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
- `I.faceExpr` — `FaceExpression {jawOpen, smile, pucker, blinkL, blinkR, browL, browR}` = 0..1 scalars, per pose-performer (`boneSource=pose`, V20/I.boneSource — face rides the SAME performer slot as body pose, ⊥ separate concept). Derived deterministic from landmarks 23-90 (2D x/y ratios: MAR=jawOpen, corner-width=smile, EAR=blink, brow-eye gap=brow), neutral-calibrated, OneEuro-smoothed. ⊥ model inference (code answers). z ignored (monocular garbage).
- `I.editorUI` — React+Tailwind editor mode (T30), DOM overlay in the VJ app (§C exception: custom DOM allowed for the editor ONLY; the runtime VJ layer stays canvas+status). Glue = framework-agnostic `editor-store` (state + pub/sub + action registry); main.js owns three & registers actions, React renders panels + drives them. Deps: react, react-dom, @vitejs/plugin-react, tailwindcss v4 (@tailwindcss/vite).
- `I.faceMask` — per-model region-weight sidecar `public/models/<id>.face-mask.bin`. Header `{vertexCount, regions[], hinge/anchor params}` + per-vertex uint8 weight × region. Regions: `jaw, lowerLip, mouthCorner, upperLidL, upperLidR, browL, browR`. Committed artifact (sibling to GLB, like the manifest). Auto-gen once/model from mesh geometry; optional hand-paint fixup.

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

Face:
- V26: face expression = deterministic 2D-geometry fn(landmarks 23-90) → scalars, ⊥ model inference (V5-global "code answers, ⊥ model for deterministic transform"). Metrics = ratios (MAR/EAR/corner-width), scale+mostly-rotation-invariant; neutral-calibrated (reuse `recalibrateFacing` pattern); OneEuro-smoothed (mirror body pipeline V25). z ⊥ used (monocular unreliable, §C).
- V27: landmark space (expr params) ⊥ coupled to mesh space (mask); they meet ONLY via the ~7 scalars. ⊥ landmark→mesh projection/alignment. Mask seeded from mesh geometry alone; params measured in landmark space alone.
- V28: face mask = AUTO-GEN from mesh geometry (primary, V27); committed `.bin` sidecar is an OPTIONAL OVERRIDE (editor output, T29) preferred when present + valid. Sidecar `mask.vertexCount === geometry.vertexCount` else THROW (⊥ silent mismatch → stale-mask mis-index). Indexed geometry ∴ mask indexes 1:1 to vertices (V2). Missing/invalid sidecar → auto-gen, ⊥ crash the crowd. (Amended: was "missing = face off"; auto-gen is deterministic+mesh-space so it's the sane default, sidecar just persists editor fixes.)
- V29: face deform lives in `positionNode` compute path, ⊥ `morphTargetInfluences` (bypasses the custom compute draw, V1/V2). Per-instance expr = instanced attribute (same per-instance machinery as `boneMatrices`, V5 — cheap: 7 floats/instance vs boneCount×16). Face drives the SAME pose-performer instances (`boneSource=pose`), ⊥ new performer type. Jaw = hinge-rotate verts about a head-local axis, ⊥ linear translate (linear looks wrong).
- V30: canonical mirror (`MIRROR_INDEX`) swaps ALL paired landmarks incl face (23-90), ⊥ body/hands only — else mirrored performer scrambles the face. (observation-adapter.js gap.)

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
T26|x|**face phase 1 — extract**: `FaceExpression` from landmarks 23-90 → 7 scalars (MAR jawOpen, corner-width smile, pucker, EAR blinkL/R, brow-eye browL/R); neutral calib + OneEuro; Inspector live plot. Also fix `MIRROR_INDEX` face gap (add 23-90 L/R pairs)|V26,V30,I.faceExpr
T27|x|**face phase 2 — mask**: auto-gen region weights from mesh geometry (Y-bands + symmetry + anchors, ⊥ landmarks); write `public/models/<id>.face-mask.bin` (header + per-vtx uint8/region); loader w/ vertexCount THROW guard; vertex-color debug overlay|V27,V28,I.faceMask
T28|x|**face phase 3 — deform**: `positionNode` reads mask + per-instance expr instanced attr; hinge-rotate jaw + translate lowerLip/mouthCorner/eyelid/brow. Scope jaw+mouth FIRST, eyes after. Normals skipped (note; fix if lighting reads wrong). ✅ hero performer opens mouth/blinks in webcam sync|V29,I.faceMask,I.faceExpr
T29|x|**face phase 4 (gated on T28 quality)**: in-app mask painter — raycast brush per region, edit hinge/anchor params, save `.bin`. Skip if auto-gen good enough|I.faceMask,I.inspector
T30|x|**face phase 5 — proper editor UI**: React+Tailwind in-app editor mode (DOM overlay, NOT the three Inspector — §C exception, editor-only). Panels: tools/brush (region+mode+radius+strength+erase+symmetric), region/bone list w/ color chips + visibility, view/overlay toggles (crowd mesh, edit-head, wireframe, mask overlay, camera presets), expression TEST-DRIVE sliders (7 scalars → live deform). Glue via framework-agnostic editor store (pub/sub); main.js owns three + registers actions, React reads/writes store. Reuses T29 edit-head/paint/mask/deform|I.editorUI,I.faceMask,I.faceExpr

## §B BUGS
id|date|cause|fix
B1|2026-06-29|ONNX webgpu EP (jsep) contends w/ three WebGPU renderer device → `table index out of bounds` + driven mesh vanishes (GPU corruption). Reference dodged it (WebGL viewer). poseEP default was `webgpu`|V23; default poseEP→`wasm` (isolated). webgpu EP needs own adapter to be viable
B2|2026-06-29|`skeleton.pose()` rebuilds bind from boneInverses in pre-`normalizeModel` space → residual ~0.01 scale on normalized source → pose-driven mesh collapses to a point (vanishes). Clip path unaffected (no pose())|retargeter restores rest from load-captured normalized rest quats; ⊥ `skeleton.pose()` in driving path
B3|2026-06-29|Meshy `Spine` bone child sits below it (`Spine→Spine01 = -y`); aiming Spine up = ~180° flip → degenerate shortest-arc → constant dramatic chest roll (one shoulder up, one down)|drop `spine` segment; drive torso lean via `Hips` (`Hips→Spine = +y`). symmetry guard test
B4|2026-06-29|adapter copied reference VIEWER's all-axes negate (point reflection) → with `mirrorX` = double-flip → crossed/side-swapped arms|clean canonical frame matched to measured rig (+x image-right=rig left, +y up, +z depth); mirrorX default off
B6|2026-06-30|grounding wrote world-space foot delta onto Hips LOCAL position under normalizeModel's ~0.01 scale → ~100× too small → squat didn't lower body (legs raised instead)|convert world delta → hips-local via parent world scale (`/hipsParentScaleY`)
B5|2026-06-30|wasm EP OOM-crashes tab: app holds ~206MB GLB chars in renderer + wasm loads 369MB rtmw3d into wasm heap (+ArrayBuffer copy) → memory blowout. Ref ran wasm OK (no heavy scene)|wasm not viable for this model here → webgpu EP only; ease its renderer-GPU contention via per-inference rAF yield; real fix = worker (T20)
B7|2026-06-30|grounding read stale foot `matrixWorld`: feet are world-updated only by their SEGMENTS aim, SKIPPED on low foot/toe conf → foot pos lagged the moved upper-leg → hips bobbed + body sank through floor|`hips.updateWorldMatrix(false,true)` before reading feet (full subtree refresh). NOTE: an asymmetric "instant-lift" floor clamp (`groundOffsetY ≥ targetDelta`) was tried for V13-no-clip but turned foot-Y noise into a sawtooth bob → reverted to symmetric EMA. True no-clip needs denoised foot-contact (T18), not a per-frame clamp
B10|2026-07-01|hard freeze past a fixed crowd count (e.g. 418 of a ~10k-vert mesh, 417 ok). Crowd skinning packs ALL instances' verts into ONE `attributeArray` (walkers×vertexCount×2 vec4 = ×32B); WebGPURenderer requests a 'compatibility' adapter that caps maxStorageBufferBindingSize at 128MiB → binding overflows at 418×10044×32 ≈ 134MiB, device rejects, loop hangs. (three already 2D-tiles the dispatch so maxComputeWorkgroupsPerDimension is NOT the wall)|request a CORE adapter+device at its real max binding/buffer size, pass `device` to WebGPURenderer (compatibility caps gone, MSAA now active too). + fail-loud guard in createComputedSkinnedMesh: throw if a batch's vert buffer > device cap (was a silent hang). True unbounded scaling needs per-batch chunking (split a batch's walkers across multiple buffers/draws)
B9|2026-07-01|retarget jumpy + "doesn't follow" for neck/head/torso. Causes: (1) binary conf gate `conf<thresh→return` froze a bone then POPPED on reacquire; (2) monocular z mixed into every aim tilted short-baseline bones (neck) + long torso lean from noise; (3) torso = ONE rigid Hips bone (no spine, B3) → real torso flex had nowhere to go; (4) shoulders not mapped at all|(1) soft gate `confGate`: FULL drive at/above thresh, fade to REST only in a band BELOW thresh (first cut faded 0→1 ABOVE thresh → confident arms sat ~30% toward rest = "pinned"/didn't follow; corrected); (2) z-damp HIPS_DEPTH 0.3→0.12, neck NECK_DEPTH 0.85→0.3; (3) `_aimTorso` distributes ONE world delta Q across SPINE_CHAIN (hips.15/spine.28/spine01.28/spine02.29 cumulative→full at top) as `Q^f∘bindWorld` (world-delta = bone-rest-dir-agnostic, dodges B3 -y flip); pelvis barely tilts (legs grounded), curl lives in spine; (4) `_aimClavicles` shrug/protraction at 0.6 gain
B11|2026-07-01|mesh shape clearly ≠ detected pose (gross mismatch, not noise). Cause (user-diagnosed): bones aimed independently in world + TOTAL-ANGLE limit clamped on top → a bit of arbitrary shortest-arc ROLL inflated the delta past the cap → clamp yanked the bone toward rest = "positioned separately then limits go weird". Limits were bounding roll+reach together|FK step 1: swing/twist limits. `_setBoneFromWorld` splits the local articulation into TWIST (axial roll, about the bone's captured `localAim`) + SWING (reach/bend), caps twist tight (90°) and swing at the slider limit, recomposes. Roll can't eat the reach budget; roll bounded everywhere (kills wild twist). Spine (no localAim) keeps total-angle. NEXT: two-frame torso (pelvis+chest → real torso twist) + hierarchical facing for turns
B8b|2026-07-01|even after §B8 (hold/deadzone/degeneracy guard) depth-yaw STILL blew stable 30fps input into violent whole-body spin/flip-flop. ROOT CAUSE (found): the yaw hack extracted a 1D depth scalar `(rh.z−lh.z)`, AMPLIFIED it ×yawGain(2.5), deadzoned + sign-held it. Amplifying noisy monocular z made it dominate the stable horizontal hip separation → hip-axis sign flips → spin. Pile of piecemeal logic fighting the data|rewrite `_aimTorso` to build the torso frame from the RAW pelvis-aligned 3D directly: up=hip→shoulder, right=hip-axis, orthonormalize → basisQuat → distribute across spine. NO amplify/deadzone/hold/degeneracy-guard. Horizontal hip sep dominates the sign (stable near frontal); real turn rotates the axis into z on its own; noise handled by input OneEuro + per-bone slerp (smooth the ROTATION, not a scalar). `poseBodyYaw` back ON — turn/tilt/pivot come "for free" from the 3D. (`smoothAxis` 'hold' mode now unused; kept as util). EXTENDED to all bones: `_aimHead` rebuilt as a 3D frame too (up=shoulder→head, right=ear→ear → pitch+yaw+ROLL for free; dropped nose-ear scalar + yawGain). All per-bone `depth` damping removed → ONE global `depthScale` knob applied to every aim's z (1=raw 3D). Piecemeal per-bone hacks gone
B8|2026-06-30|whole mesh snapped 180° on turn. Torso=Hips bone (no spine seg, B3) so hip/torso/shoulder flip together = Hips root flip. Hips yaw "across" axis sign comes ONLY from `rh.z−lh.z` (monocular depth) → near-frontal that z is noise; sign flip → `basisQuat` = exact 180° about up. `smoothAxis` lock=false only held on EXACT antiparallel; a "mostly flipped" axis still swept 180° in ~3 frames|smoothAxis sign policy → `'hold'` mode for facing axes: REJECT any backward candidate (`dot<0`) = hold facing, only same-hemisphere blends (kills depth-sign flips; genuine 180 turn is out-of-scope §53). + basisQuat degeneracy guard: skip yaw when across∥up (`|dot|≥0.94`) → lean-only. Diagnosis: yaw-from-monocular-depth is inherently sign-ambiguous — this is damage control, not a true turn estimator (T17)
