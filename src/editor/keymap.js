// Editor hotkeys (single source of truth). Both the keyboard handler (editor.js) and
// the UI (Btn `kbd` chips) read these, so a shortcut and its on-screen label can never
// drift. `label` is what the button shows; `key`/mods are what the handler matches.
// meta = ⌘ on mac / Ctrl elsewhere. Photoshop-familiar where it helps ([ ] brush size).
export const HOTKEYS = {
  camera: { key: 'v', label: 'V' },
  brush: { key: 'b', label: 'B' },
  paint: { key: 'p', label: 'P' },
  erase: { key: 'e', label: 'E' },
  hinge: { key: 'h', label: 'H' },
  symmetric: { key: 'x', label: 'X' },
  radiusDown: { key: '[', label: '[' },
  radiusUp: { key: ']', label: ']' },
  reseed: { key: 'r', label: 'R' },
  flip: { key: 'r', shift: true, label: '⇧R' },
  clear: { key: 'backspace', label: '⌫' },
  save: { key: 's', meta: true, label: '⌘S' },
  undo: { key: 'z', meta: true, label: '⌘Z' },
  redo: { key: 'z', meta: true, shift: true, label: '⇧⌘Z' }
};

// The 7 regions map to number keys 1..7 (shown as a badge on each region chip).
export const REGION_KEYS = ['1', '2', '3', '4', '5', '6', '7'];

// Does a keydown event match a HOTKEYS entry? (meta = metaKey||ctrlKey.)
export function matchesHotkey(e, hk) {
  if (e.key.toLowerCase() !== hk.key) return false;
  const meta = e.metaKey || e.ctrlKey;
  if (!!hk.meta !== meta) return false;
  if (!!hk.shift !== e.shiftKey) return false;
  return true;
}
