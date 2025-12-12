import { Play, Pause } from "lucide-react";

const PlayerControls = ({ isPlaying, togglePlay, disabled }) => (
  <div className="flex justify-center pt-2">
    <button
      onClick={togglePlay}
      disabled={disabled}
      className={`w-14 h-14 flex items-center justify-center rounded-full transition-all duration-300 ${
        disabled
          ? "bg-slate-800 text-slate-600 scale-95 cursor-not-allowed"
          : "bg-blue-500 hover:bg-blue-400 text-white shadow-xl shadow-blue-500/20 hover:scale-105 active:scale-95"
      }`}
    >
      {isPlaying ? (
        <Pause size={24} fill="currentColor" />
      ) : (
        <Play size={24} fill="currentColor" className="ml-1" />
      )}
    </button>
  </div>
);

export default PlayerControls;
