// Brush controls (T32): radius, strength, symmetric + auto-gen fixes (re-seed / flip
// forward / clear region).
import { actions } from '../editor-store.js';
import { HOTKEYS } from '../keymap.js';
import { Section, Slider, Btn, Kbd } from './primitives.jsx';

export default function BrushPanel({ s }) {
  return (
    <>
      <Section title="Brush">
        <Slider label="Radius" value={s.radius} min={0.005} max={0.3} step={0.005} onChange={actions.setRadius} />
        <div className="text-[10px] text-slate-400 -mt-0.5 mb-1.5 ml-16">resize <Kbd>[</Kbd> <Kbd>]</Kbd></div>
        <Slider label="Strength" value={s.strength} min={0.05} max={1} step={0.05} onChange={actions.setStrength} />
        <label className="flex items-center gap-2 text-xs text-slate-300 mt-1">
          <input type="checkbox" checked={s.symmetric} onChange={(e) => actions.setSymmetric(e.target.checked)}
            className="accent-cyan-400" />
          Symmetric (mirror X) <Kbd>{HOTKEYS.symmetric.label}</Kbd>
        </label>
      </Section>
      <Section title="Auto-gen fixes">
        <div className="flex flex-wrap gap-1.5">
          <Btn onClick={() => actions.reseed(false)} kbd={HOTKEYS.reseed.label} title="Re-run auto-gen">Re-seed</Btn>
          <Btn onClick={() => actions.reseed(true)} kbd={HOTKEYS.flip.label} title="Flip assumed forward axis">Flip Forward</Btn>
          <Btn onClick={() => actions.clearRegion()} kbd={HOTKEYS.clear.label} title="Zero the current region">Clear</Btn>
        </div>
      </Section>
    </>
  );
}
