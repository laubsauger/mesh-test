// Expression test-drive bar (T32): 7 sliders → live deform on the edit-head.
import { actions } from '../editor-store.js';
import { MiniSlider, Btn } from './primitives.jsx';

const EXPR = [
  ['jawOpen', 'Jaw'], ['smile', 'Smile'], ['pucker', 'Pucker'],
  ['blinkL', 'Blink L'], ['blinkR', 'Blink R'], ['browL', 'Brow L'], ['browR', 'Brow R']
];

export default function TestDrivePanel({ s }) {
  return (
    <div className="absolute bottom-3 left-3 right-3 md:left-1/2 md:right-auto md:-translate-x-1/2 md:w-[680px]
      pointer-events-auto bg-black/70 backdrop-blur rounded-lg px-4 py-3 border border-white/10">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-cyan-300/70">
          Test-drive (live deform)</span>
        <Btn onClick={() => actions.resetExpr()}>Reset</Btn>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-5 gap-y-2">
        {EXPR.map(([k, label]) => (
          <MiniSlider key={k} label={label} value={s.expr[k]} onChange={(v) => actions.setExpr({ ...s.expr, [k]: v })} />
        ))}
      </div>
    </div>
  );
}
