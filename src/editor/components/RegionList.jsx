// Region/marker picker (T32): color-coded chips, click to select the active region.
import { actions } from '../editor-store.js';
import { FACE_REGIONS } from '../../pose/face-mask.js';
import { REGION_KEYS } from '../keymap.js';
import { Section, Kbd } from './primitives.jsx';

const REGION_COLOR = {
  jaw: '#ff5c5c', lowerLip: '#ff9f43', mouthCorner: '#feca57',
  upperLidL: '#54a0ff', upperLidR: '#5f27cd', browL: '#1dd1a1', browR: '#00d2d3'
};

export default function RegionList({ s }) {
  return (
    <Section title="Region">
      <div className="grid grid-cols-2 gap-1.5">
        {FACE_REGIONS.map((r, i) => (
          <button key={r} onClick={() => actions.setRegion(r)}
            className={`flex items-center gap-1.5 px-2 py-1 rounded text-[11px] min-w-0 transition-all ${s.region === r
              ? 'bg-white/15 ring-1 ring-cyan-400' : 'bg-white/5 hover:bg-white/10'}`}>
            <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: REGION_COLOR[r] }} />
            <span className="truncate flex-1 text-left">{r}</span>
            {REGION_KEYS[i] ? <Kbd>{REGION_KEYS[i]}</Kbd> : null}
          </button>
        ))}
      </div>
    </Section>
  );
}
