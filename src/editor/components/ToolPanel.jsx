// Tool select (T32): Move (orbit) vs Brush (paint), + brush sub-modes.
import { actions } from '../editor-store.js';
import { HOTKEYS } from '../keymap.js';
import { Section, Btn } from './primitives.jsx';

const MODE_KEY = { paint: HOTKEYS.paint.label, erase: HOTKEYS.erase.label, hinge: HOTKEYS.hinge.label };

export default function ToolPanel({ s }) {
  return (
    <Section title="Tool">
      <div className="flex gap-1.5 mb-1.5">
        <Btn active={s.tool === 'camera'} kbd={HOTKEYS.camera.label} onClick={() => actions.setTool('camera')} title="Orbit the camera">✋ Move</Btn>
        <Btn active={s.tool === 'brush'} kbd={HOTKEYS.brush.label} onClick={() => actions.setTool('brush')} title="Paint on the head">🖌 Brush</Btn>
      </div>
      {s.tool === 'brush' ? (
        <>
          <div className="flex gap-1.5">
            {['paint', 'erase', 'hinge'].map((m) => (
              <Btn key={m} active={s.mode === m} kbd={MODE_KEY[m]} onClick={() => actions.setMode(m)}>{m}</Btn>
            ))}
          </div>
          <div className="text-[10px] text-slate-400 mt-1.5">Hold <b>Ctrl</b> to orbit while painting.</div>
        </>
      ) : <div className="text-[10px] text-slate-400">Drag to orbit · scroll to zoom.</div>}
    </Section>
  );
}
