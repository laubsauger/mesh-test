// Editor store (T30 / I.editorUI): framework-agnostic glue between the React editor
// UI and the vanilla three app (main.js). main.js owns the scene + registers actions;
// React reads `editorState` (via useEditorState) and calls `actions.*`. Pub/sub keeps
// the two in sync without either importing the other's internals.

/**
 * The reactive editor state React renders (T33 / V35).
 * @typedef {Object} EditorState
 * @property {boolean} open
 * @property {{ index: number, name: string }[]} models
 * @property {number} modelIndex
 * @property {string} region
 * @property {'camera'|'brush'} tool
 * @property {'paint'|'erase'|'hinge'} mode
 * @property {number} radius
 * @property {number} strength
 * @property {boolean} symmetric
 * @property {{ editHead: boolean, wireframe: boolean, maskCloud: boolean, crowd: boolean, texAlpha: number }} overlays
 * @property {boolean} canUndo
 * @property {boolean} canRedo
 * @property {import('../face/types.js').FaceExpression} expr
 * @property {import('../face/types.js').RegionConfig} regionConfig
 * @property {string} status
 */

/**
 * The action registry main.js fills (registerActions) + React calls (I.editorUI).
 * @typedef {Object} EditorActions
 * @property {(v: boolean) => void} setOpen
 * @property {(i: number) => void} setModel
 * @property {(r: string) => void} setRegion
 * @property {(t: 'camera'|'brush') => void} setTool
 * @property {(m: 'paint'|'erase'|'hinge') => void} setMode
 * @property {(v: number) => void} setRadius
 * @property {(v: number) => void} setStrength
 * @property {(v: boolean) => void} setSymmetric
 * @property {(patch: Partial<import('../face/types.js').RegionConfig>) => void} setRegionConfig
 * @property {(flip: boolean) => void} reseed
 * @property {() => void} clearRegion
 * @property {() => void} undo
 * @property {() => void} redo
 * @property {() => void} save
 * @property {(buf: ArrayBuffer) => void} loadMask
 * @property {(expr: import('../face/types.js').FaceExpression) => void} setExpr
 * @property {() => void} resetExpr
 * @property {(v: string) => void} frameView
 * @property {(v: number) => void} setTexAlpha
 * @property {(k: string, v: boolean) => void} setOverlay
 */

/** @type {EditorState} */
export const editorState = {
  open: false,
  models: [],            // [{ index, name }]
  modelIndex: 0,
  region: 'jaw',
  tool: 'brush',         // 'camera' (orbit) | 'brush' (paint)
  mode: 'paint',         // brush sub-mode: 'paint' | 'erase' | 'hinge'
  radius: 0.04,
  strength: 0.5,
  symmetric: true,
  overlays: { editHead: true, wireframe: false, maskCloud: false, crowd: true, texAlpha: 0 },
  canUndo: false, canRedo: false,
  expr: { jawOpen: 0, smile: 0, pucker: 0, blinkL: 0, blinkR: 0, browL: 0, browR: 0 },
  // Current region's deform config (T30 configurable regions), mirrored from main.js.
  regionConfig: { driver: 0, type: 0, amount: 0, dir: [0, -1, 0], mirrorX: false, hingeOrigin: [0, 0, 0], hingeAxis: [1, 0, 0] },
  status: ''
};

// Registered by main.js. React calls these; it never touches three directly.
/** @type {Partial<EditorActions>} */
export const actions = {};

const listeners = new Set();
let version = 0;
export const getVersion = () => version; // monotonic snapshot for useSyncExternalStore
export function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
export function emit() { version += 1; for (const fn of listeners) fn(); }

// Shallow-merge a patch into editorState and notify (nested objects should be passed
// whole, e.g. setState({ expr: { ...editorState.expr, jawOpen: v } })).
export function setEditorState(patch) { Object.assign(editorState, patch); emit(); }
export function registerActions(a) { Object.assign(actions, a); }
