// Face-mask editor UI (T30 / I.editorUI), composed from panels (T32). React + Tailwind
// overlay on the VJ app. Reads the shared editor store; every mutation goes through
// `actions.*` (registered by main.js) so React never touches three directly (V32). The
// root is pointer-events-none so clicks in the empty center reach the canvas (painting);
// panels re-enable pointers.
import { useEditor } from './components/primitives.jsx';
import Header from './components/Header.jsx';
import ToolPanel from './components/ToolPanel.jsx';
import RegionList from './components/RegionList.jsx';
import MarkerPanel from './components/MarkerPanel.jsx';
import BrushPanel from './components/BrushPanel.jsx';
import ViewPanel from './components/ViewPanel.jsx';
import TestDrivePanel from './components/TestDrivePanel.jsx';

const DEFAULT_RC = { driver: 0, type: 0, amount: 0, dir: [0, -1, 0], mirrorX: false };

export default function EditorUI() {
  const s = useEditor();
  if (!s.open) return null;
  const rc = s.regionConfig || DEFAULT_RC;

  return (
    <div className="fixed inset-0 z-50 pointer-events-none select-none font-sans text-slate-100">
      <Header s={s} />

      <div className="absolute top-16 left-3 w-64 pointer-events-auto max-h-[calc(100vh-6rem)]
        overflow-y-auto overflow-x-hidden bg-black/70 backdrop-blur rounded-lg p-3 border border-white/10">
        <ToolPanel s={s} />
        <RegionList s={s} />
        <MarkerPanel s={s} rc={rc} />
        <BrushPanel s={s} />
        <ViewPanel s={s} />
      </div>

      <TestDrivePanel s={s} />

      {s.status ? (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 pointer-events-none
          bg-cyan-500/90 text-black text-xs font-medium px-3 py-1.5 rounded-full">{s.status}</div>
      ) : null}
    </div>
  );
}
