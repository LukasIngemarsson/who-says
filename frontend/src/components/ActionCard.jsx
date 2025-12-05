import { Upload, Mic, Square, Users } from "lucide-react";
import { formatTime } from "../utils/functions.js";

const ActionCard = ({
  mode,
  isRecording,
  recordingTime,
  handleFileUpload,
  startRecording,
  stopRecording,
  numSpeakers,
  setNumSpeakers,
  isProcessing,
}) => {
  return (
    <div className="md:col-span-3 bg-slate-900 rounded-xl border border-slate-800 p-6 flex flex-col items-center justify-center min-h-[200px] relative overflow-hidden transition-all duration-300 hover:border-slate-700">
      {/* Settings Bar */}
      <div className="absolute top-4 right-4 flex items-center gap-3 z-20">
        <div className="flex items-center gap-2 bg-slate-950/50 px-3 py-1.5 rounded-full border border-slate-800">
          <Users size={14} className="text-slate-400" />
          <span className="text-xs text-slate-400 font-medium">Speakers:</span>
          <input
            type="number"
            min="1"
            max="10"
            value={numSpeakers}
            onChange={(e) => setNumSpeakers(parseInt(e.target.value) || 2)}
            className="w-8 bg-transparent text-center text-sm font-bold text-white focus:outline-none appearance-none"
          />
        </div>
      </div>

      {mode === "upload" ? (
        <div className="text-center space-y-4 z-10 animate-in fade-in zoom-in duration-300">
          <div className="w-16 h-16 bg-blue-500/10 rounded-full flex items-center justify-center mx-auto mb-2">
            <Upload className="text-blue-400 w-8 h-8" />
          </div>
          <div className="relative group">
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileUpload}
              disabled={isProcessing}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
            />
            <button
              disabled={isProcessing}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg font-medium transition-colors shadow-lg shadow-blue-900/20"
            >
              {isProcessing ? "Processing..." : "Select Audio File"}
            </button>
          </div>
          <p className="text-slate-500 text-sm">Supports MP3, WAV, M4A</p>
        </div>
      ) : (
        <div className="text-center space-y-4 z-10 animate-in fade-in zoom-in duration-300">
          {isRecording ? (
            <div className="space-y-4 animate-pulse">
              <div className="text-5xl font-mono text-red-500 font-bold">
                {formatTime(recordingTime)}
              </div>
              <button
                onClick={stopRecording}
                className="px-8 py-3 bg-red-500 hover:bg-red-600 text-white rounded-full font-bold flex items-center gap-2 mx-auto transition-transform active:scale-95"
              >
                <Square fill="currentColor" size={16} /> Stop Recording
              </button>
            </div>
          ) : (
            <>
              <div className="w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-2">
                <Mic className="text-red-400 w-8 h-8" />
              </div>
              <button
                onClick={startRecording}
                disabled={isProcessing}
                className="px-6 py-2.5 bg-red-600 hover:bg-red-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg font-medium transition-colors shadow-lg shadow-red-900/20"
              >
                Start Recording
              </button>
              <p className="text-slate-500 text-sm">
                Click to start capturing audio
              </p>
            </>
          )}
        </div>
      )}
      {/* Background decoration */}
      <div className="absolute inset-0 bg-linear-to-b from-transparent to-slate-900/50 pointer-events-none" />
    </div>
  );
};

export default ActionCard;
