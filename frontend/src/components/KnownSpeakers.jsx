import { useState, useEffect } from "react";
import { Users, Check } from "lucide-react";

const KnownSpeakers = ({ refreshTrigger }) => {
  const [speakers, setSpeakers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSpeakers();
  }, [refreshTrigger]);

  const fetchSpeakers = async () => {
    try {
      const response = await fetch("/status");
      const data = await response.json();
      setSpeakers(data.known_speakers || []);
    } catch (error) {
      console.error("Failed to fetch speakers:", error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return null;

  return (
    <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-800/50">
      <div className="flex items-center gap-2 mb-3">
        <Users size={16} className="text-slate-400" />
        <h3 className="text-sm font-medium text-slate-300">Enrolled Speakers</h3>
      </div>
      {speakers.length === 0 ? (
        <p className="text-xs text-slate-500">No speakers enrolled yet</p>
      ) : (
        <div className="flex flex-wrap gap-2">
          {speakers.map((name) => (
            <div
              key={name}
              className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-800 rounded-full text-xs text-slate-300"
            >
              <Check size={12} className="text-green-400" />
              {name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default KnownSpeakers;
