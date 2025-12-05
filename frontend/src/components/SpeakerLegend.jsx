import { SPEAKER_COLORS } from "../utils/constants";

const SpeakerLegend = ({ segments }) => {
  if (segments.length === 0) return null;

  return (
    <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-800/50">
      <div className="flex flex-wrap gap-4 justify-center text-sm">
        {[0, 1, 2].map((id) => (
          <div
            key={id}
            className="flex items-center gap-2 px-3 py-1.5 bg-slate-900 rounded-full border border-slate-800"
          >
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: SPEAKER_COLORS[id] }}
            />
            <span className="text-slate-300 font-medium">Speaker {id}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SpeakerLegend;
