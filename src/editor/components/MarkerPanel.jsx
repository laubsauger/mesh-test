// Per-region deform config (T32 / T30 configurable regions): driver, type, amount,
// direction (+mirrorX) or the hinge hint.
import { actions } from '../editor-store.js';
import { EXPR_KEYS, DEFORM_TYPES } from '../../pose/face-mask.js';
import { Section, Slider, Btn } from './primitives.jsx';

const DIRS = [['+X', [1, 0, 0]], ['-X', [-1, 0, 0]], ['+Y', [0, 1, 0]], ['-Y', [0, -1, 0]], ['+Z', [0, 0, 1]], ['-Z', [0, 0, -1]]];
const sameDir = (a, b) => a && b && a[0] === b[0] && a[1] === b[1] && a[2] === b[2];

export default function MarkerPanel({ s, rc }) {
  return (
    <Section title={`Marker: ${s.region}`}>
      <label className="flex items-center gap-2 text-xs text-slate-300 mb-1.5 min-w-0">
        <span className="w-14 shrink-0 text-slate-400">Driver</span>
        <select value={rc.driver} onChange={(e) => actions.setRegionConfig({ driver: +e.target.value })}
          className="flex-1 min-w-0 bg-white/5 rounded px-1.5 py-0.5 border border-white/10">
          {EXPR_KEYS.map((k, i) => <option key={k} value={i}>{k}</option>)}
        </select>
      </label>
      <div className="flex gap-1.5 mb-1.5">
        {DEFORM_TYPES.map((t, i) => (
          <Btn key={t} active={rc.type === i} onClick={() => actions.setRegionConfig({ type: i })}>{t}</Btn>
        ))}
      </div>
      <Slider label="Amount" value={rc.amount} min={0} max={1.2} step={0.01} onChange={(v) => actions.setRegionConfig({ amount: v })} />
      {rc.type === 0 ? (
        <>
          <div className="text-[10px] text-slate-400 mt-1 mb-1">Direction</div>
          <div className="grid grid-cols-6 gap-1">
            {DIRS.map(([lbl, d]) => (
              <Btn key={lbl} active={sameDir(rc.dir, d)} onClick={() => actions.setRegionConfig({ dir: d })}>{lbl}</Btn>
            ))}
          </div>
          <label className="flex items-center gap-2 text-xs text-slate-300 mt-1.5">
            <input type="checkbox" checked={rc.mirrorX} onChange={(e) => actions.setRegionConfig({ mirrorX: e.target.checked })}
              className="accent-cyan-400" />
            Mirror X by side (spread)
          </label>
        </>
      ) : (
        <div className="text-[10px] text-cyan-300/80 mt-1 leading-relaxed">
          Hinge: use the <b>hinge</b> tool → click the head for origin, then a 2nd point for the axis. Amount = radians.
        </div>
      )}
    </Section>
  );
}
