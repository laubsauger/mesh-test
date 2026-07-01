// Editor store (T30 / I.editorUI): framework-agnostic glue between the React editor
// UI and the vanilla three app (main.js). main.js owns the scene + registers actions;
// React reads `editorState` (via useEditorState) and calls `actions.*`. Pub/sub keeps
// the two in sync without either importing the other's internals.
export const editorState = {
  open: false,
  models: [],            // [{ index, name }]
  modelIndex: 0,
  region: 'jaw',
  mode: 'paint',         // 'paint' | 'erase' | 'hinge'
  radius: 0.04,
  strength: 0.5,
  symmetric: true,
  overlays: { editHead: true, wireframe: false, maskCloud: false, crowd: true },
  expr: { jawOpen: 0, smile: 0, pucker: 0, blinkL: 0, blinkR: 0, browL: 0, browR: 0 },
  regionSums: {},        // region → painted-weight sum (for the region list badges)
  status: ''
};

// Registered by main.js. React calls these; it never touches three directly.
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
