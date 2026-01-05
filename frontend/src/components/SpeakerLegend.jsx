import { getSpeakerColor } from "../utils/functions";
import { Download } from "lucide-react";

const SpeakerLegend = ({ segments, onDownload }) => {
  if (segments.length === 0) return null;

  // Get unique speakers with their cluster_id and name
  const speakerMap = new Map();
  segments.forEach((s) => {
    if (!speakerMap.has(s.speaker)) {
      speakerMap.set(s.speaker, s.cluster_id);
    }
  });
  
  // Sort by cluster_id for consistent ordering
  const uniqueSpeakers = Array.from(speakerMap.entries()).sort((a, b) => a[1] - b[1]);

  return (
    <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-800/50 flex flex-col md:flex-row items-center justify-between gap-4">
      <div className="flex flex-wrap gap-4 justify-center text-sm">
        {uniqueSpeakers.map(([speakerName, clusterId]) => (
          <div
            key={speakerName}
            className="flex items-center gap-2 px-3 py-1.5 bg-slate-900 rounded-full border border-slate-800"
          >
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: getSpeakerColor(clusterId) }}
            />
            <span className="text-slate-300 font-medium">{speakerName}</span>
          </div>
        ))}
      </div>

      {onDownload && (
        <button
          onClick={onDownload}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm font-medium transition-colors border border-slate-700 shadow-sm"
          title="Download raw analysis results"
        >
          <Download size={16} />
          Download JSON
        </button>
      )}
    </div>
  );
};

export default SpeakerLegend;
