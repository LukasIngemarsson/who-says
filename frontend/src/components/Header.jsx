import { Upload, Mic, UserPlus } from "lucide-react";

const Header = ({ mode, setMode, onAddSpeaker }) => (
  <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
    <div>
      <h1 className="text-3xl font-bold bg-linear-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
        WhoSays
      </h1>
      <p className="text-slate-400 mt-1">
        Speaker Diarization & Transcription Workspace
      </p>
    </div>

    <div className="flex gap-3">
      <button
        onClick={onAddSpeaker}
        className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm font-medium transition-colors border border-slate-700"
      >
        <UserPlus size={16} /> Add Speaker
      </button>

      <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-800">
        <button
          onClick={() => setMode("upload")}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
            mode === "upload"
              ? "bg-slate-800 text-white shadow-sm"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Upload size={16} /> Upload
        </button>
        <button
          onClick={() => setMode("record")}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
            mode === "record"
              ? "bg-slate-800 text-white shadow-sm"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Mic size={16} /> Record
        </button>
      </div>
    </div>
  </header>
);

export default Header;
