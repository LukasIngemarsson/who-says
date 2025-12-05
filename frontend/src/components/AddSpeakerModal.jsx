import { useState, useRef, useEffect } from "react";
import {
  X,
  Mic,
  Square,
  Save,
  RefreshCw,
  Check,
  AlertCircle,
} from "lucide-react";

const AddSpeakerModal = ({ isOpen, onClose }) => {
  const [name, setName] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [status, setStatus] = useState("idle"); // idle, recording, recorded, saving, success, error
  const [error, setError] = useState("");

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  useEffect(() => {
    if (!isOpen) {
      // Reset state when closed
      setName("");
      setAudioBlob(null);
      setStatus("idle");
      setError("");
    }
  }, [isOpen]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        setStatus("recorded");
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setStatus("recording");
    } catch (err) {
      setError(`Mic access denied or error: ${err.message}`);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleSave = async () => {
    if (!name || !audioBlob) return;

    setStatus("saving");
    const formData = new FormData();
    formData.append("file", audioBlob, "reference.webm");
    formData.append("name", name);

    try {
      const response = await fetch("/upload_embeddings", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to save");

      setStatus("success");
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-md p-6 relative animate-in fade-in zoom-in duration-200">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-500 hover:text-slate-300"
        >
          <X size={20} />
        </button>

        <h2 className="text-xl font-bold text-white mb-1">Add New Speaker</h2>
        <p className="text-slate-400 text-sm mb-6">
          Record a short voice snippet (5-10s) to create a reference for this
          speaker.
        </p>

        <div className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1 uppercase tracking-wider">
              Speaker Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. John Doe"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500 transition-colors"
            />
          </div>

          <div className="border-t border-slate-800 pt-4">
            <label className="block text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">
              Voice Reference
            </label>

            <div className="flex items-center justify-between bg-slate-950 rounded-lg p-3 border border-slate-800">
              {status === "recording" ? (
                <div className="flex items-center gap-3 text-red-400 animate-pulse">
                  <div className="w-3 h-3 bg-red-500 rounded-full" />
                  <span className="font-mono text-sm">Recording...</span>
                </div>
              ) : status === "recorded" ||
                status === "saving" ||
                status === "success" ? (
                <div className="flex items-center gap-2 text-green-400">
                  <Check size={16} />
                  <span className="text-sm">Audio captured</span>
                </div>
              ) : (
                <span className="text-slate-600 text-sm italic">
                  No audio recorded
                </span>
              )}

              {status === "recording" ? (
                <button
                  onClick={stopRecording}
                  className="p-2 bg-red-500/20 text-red-400 hover:bg-red-500/30 rounded-full transition-colors"
                >
                  <Square size={18} fill="currentColor" />
                </button>
              ) : (
                <button
                  onClick={startRecording}
                  className="p-2 bg-slate-800 text-slate-300 hover:bg-slate-700 rounded-full transition-colors"
                >
                  <Mic size={18} />
                </button>
              )}
            </div>
          </div>

          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm flex items-center gap-2">
              <AlertCircle size={14} /> {error}
            </div>
          )}

          {status === "success" && (
            <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg text-green-400 text-sm flex items-center gap-2">
              <Check size={14} /> Speaker enrolled successfully!
            </div>
          )}

          <button
            onClick={handleSave}
            disabled={
              !name || !audioBlob || status === "saving" || status === "success"
            }
            className={`w-full py-2.5 rounded-lg font-medium flex items-center justify-center gap-2 transition-all mt-2 ${
              !name || !audioBlob
                ? "bg-slate-800 text-slate-600 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20"
            }`}
          >
            {status === "saving" ? (
              <RefreshCw className="animate-spin" size={18} />
            ) : (
              <Save size={18} />
            )}
            {status === "saving" ? "Saving..." : "Save Speaker"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AddSpeakerModal;
