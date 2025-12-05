import { Upload, Mic } from "lucide-react";

const Header = ({ mode, setMode }) => (
  <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
    <div>
      <h1 className="text-3xl font-bold bg-linear-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
        Who Says?
      </h1>
      <p className="text-slate-400 mt-1">
        Speaker diarization made easy with AI
      </p>
    </div>

    <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-800">
      <button
        onClick={() => setMode("upload")}
        className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
          mode === "upload"
            ? "bg-slate-800 text-white shadow-sm"
            : "text-slate-400 hover:text-slate-200"
        }`}
      >
        <Upload size={16} /> Upload File
      </button>
      <button
        onClick={() => setMode("record")}
        className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
          mode === "record"
            ? "bg-slate-800 text-white shadow-sm"
            : "text-slate-400 hover:text-slate-200"
        }`}
      >
        <Mic size={16} /> Microphone
      </button>
    </div>
  </header>
);

export default Header;
