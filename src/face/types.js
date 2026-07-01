// Central JSDoc type shapes for the face pipeline (T33 / V35 — TypeScript readiness,
// NOT a conversion). Reference these from other modules via
//   `@typedef {import('./types.js').FaceMask} FaceMask`
// or `@param {import('./types.js').RegionConfig} c`. When the codebase migrates to TS
// (order below), these become real `interface`s and this file becomes types.ts.
//
// .js → .ts migration order (V35): data typedefs (this file) → pure modules
// (face-mask.js, face-expression.js) → deform.js → editor.js → editor UI → main.js last.

/**
 * One face-expression scalar set, 0..1 (I.faceExpr). Drives the deform.
 * @typedef {Object} FaceExpression
 * @property {number} jawOpen
 * @property {number} smile
 * @property {number} pucker
 * @property {number} blinkL
 * @property {number} blinkR
 * @property {number} browL
 * @property {number} browR
 */

/**
 * Per-region deform config (T30 configurable regions). `driver` indexes EXPR_KEYS;
 * `type` indexes DEFORM_TYPES (0 translate, 1 hinge); `amount` = radians (hinge) or
 * fraction-of-headHeight (translate); `dir` = local translate direction; `mirrorX`
 * flips dir.x by the vertex side; hinge origin/axis in geometry-local space.
 * @typedef {Object} RegionConfig
 * @property {number} driver
 * @property {number} type
 * @property {number} amount
 * @property {number[]} dir
 * @property {boolean} mirrorX
 * @property {number[]} hingeOrigin
 * @property {number[]} hingeAxis
 */

/**
 * A model's face mask: per-vertex region weights (uint8, region-major) + per-region
 * deform config + the auto-gen hinge + head scale. Persisted as the `.bin` sidecar.
 * @typedef {Object} FaceMask
 * @property {number} vertexCount
 * @property {string[]} regions
 * @property {Uint8Array} data          region-major weights (regions.length × vertexCount)
 * @property {{ origin: number[], axis: number[] }} hinge
 * @property {number} headHeight        geometry-units scale reference for translations
 * @property {RegionConfig[]} config    one per region
 */

export {}; // keep this a module (typedefs only)
