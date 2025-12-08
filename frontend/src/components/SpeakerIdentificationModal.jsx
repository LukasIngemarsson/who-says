import { useState, useEffect } from "react";
import { X, RefreshCw, Check, AlertCircle, UserPlus, UserCheck } from "lucide-react";

const SpeakerIdentificationModal = ({ isOpen, onClose, audioData, speakerInfo, knownSpeakers, onRefresh }) => {
  const [isNewSpeaker, setIsNewSpeaker] = useState(null);
  const [selectedSpeaker, setSelectedSpeaker] = useState("");
  const [newSpeakerName, setNewSpeakerName] = useState("");
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!isOpen) {
      setIsNewSpeaker(null);
      setSelectedSpeaker("");
      setNewSpeakerName("");
      setStatus("idle");
      setError("");
    }
  }, [isOpen]);

  const handleSubmit = async () => {
    if (isNewSpeaker === null) return;
    
    if (isNewSpeaker && !newSpeakerName.trim()) {
      setError("Please enter a speaker name");
      return;
    }
    
    if (!isNewSpeaker && !selectedSpeaker) {
      setError("Please select a speaker");
      return;
    }

    if (!audioData || !audioData.audioData) {
      setError("Audio data is missing");
      return;
    }

    setStatus("saving");
    setError("");

    try {
      const response = await fetch("/correct_speaker", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          audio_data: audioData.audioData,
          sample_rate: audioData.sampleRate || "16000",
          is_new_speaker: isNewSpeaker ? "true" : "false",
          speaker_name: isNewSpeaker ? newSpeakerName.trim() : selectedSpeaker,
          session_id: audioData.sessionId || "default",
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to save");

      setStatus("success");
      if (onRefresh) onRefresh();
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  };

  if (!isOpen) return null;

  const isDisabled = status === "saving" || status === "success";
  const detectedSpeaker = speakerInfo?.detectedSpeaker || "Unknown";
  const confidence = speakerInfo?.confidence || 0;
  const isGeneric = speakerInfo?.isGeneric || false;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-md p-6 relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-500 hover:text-slate-300"
          disabled={isDisabled}
        >
          <X size={20} />
        </button>

        <h2 className="text-xl font-bold text-white mb-1">Speaker Identification</h2>
        <p className="text-slate-400 text-sm mb-2">
          {isGeneric 
            ? `Detected as generic speaker: ${detectedSpeaker}`
            : `Uncertain identification (confidence: ${(confidence * 100).toFixed(1)}%)`
          }
        </p>
        {speakerInfo?.consecutiveFailures > 0 && (
          <p className="text-slate-500 text-xs mb-2">
            Failed to identify {speakerInfo.consecutiveFailures} time(s) in a row
          </p>
        )}
        <p className="text-slate-500 text-xs mb-6">
          Who was speaking?
        </p>

        <div className="space-y-4">
          <div className="flex gap-3">
            <button
              onClick={() => setIsNewSpeaker(false)}
              disabled={isDisabled}
              className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
                isNewSpeaker === false
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              } ${isDisabled ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <UserCheck size={18} className="inline mr-2" />
              Existing Speaker
            </button>
            <button
              onClick={() => setIsNewSpeaker(true)}
              disabled={isDisabled}
              className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
                isNewSpeaker === true
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              } ${isDisabled ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <UserPlus size={18} className="inline mr-2" />
              New Speaker
            </button>
          </div>

          {isNewSpeaker === false && (
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1 uppercase tracking-wider">
                Select Speaker
              </label>
              <select
                value={selectedSpeaker}
                onChange={(e) => setSelectedSpeaker(e.target.value)}
                disabled={isDisabled}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500 transition-colors"
              >
                <option value="">Choose a speaker...</option>
                {knownSpeakers.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {isNewSpeaker === true && (
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1 uppercase tracking-wider">
                Speaker Name
              </label>
              <input
                type="text"
                value={newSpeakerName}
                onChange={(e) => setNewSpeakerName(e.target.value)}
                placeholder="e.g. John Doe"
                disabled={isDisabled}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>
          )}

          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm flex items-center gap-2">
              <AlertCircle size={14} /> {error}
            </div>
          )}

          {status === "success" && (
            <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg text-green-400 text-sm flex items-center gap-2">
              <Check size={14} /> Speaker {isNewSpeaker ? "enrolled" : "updated"} successfully!
            </div>
          )}

          {isNewSpeaker !== null && (
            <button
              onClick={handleSubmit}
              disabled={isDisabled || (isNewSpeaker && !newSpeakerName.trim()) || (!isNewSpeaker && !selectedSpeaker)}
              className={`w-full py-2.5 rounded-lg font-medium flex items-center justify-center gap-2 transition-all mt-2 ${
                isDisabled || (isNewSpeaker && !newSpeakerName.trim()) || (!isNewSpeaker && !selectedSpeaker)
                  ? "bg-slate-800 text-slate-600 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20"
              }`}
            >
              {status === "saving" ? (
                <>
                  <RefreshCw className="animate-spin" size={18} />
                  Saving...
                </>
              ) : (
                <>
                  <Check size={18} />
                  {isNewSpeaker ? "Add New Speaker" : "Update Speaker"}
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default SpeakerIdentificationModal;
