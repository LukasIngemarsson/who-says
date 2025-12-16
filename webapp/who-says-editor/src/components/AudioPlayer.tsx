import React, {
  forwardRef,
  useState,
  useEffect,
  useRef,
  useImperativeHandle,
} from "react";
import { formatTime } from "../utils/time";

type Props = { className?: string };
export const AudioPlayer = forwardRef<HTMLAudioElement, Props>(
  ({ className }, ref) => {
    const audioRef = useRef<HTMLAudioElement>(null);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    // Expose the audio element to parent components
    useImperativeHandle(ref, () => audioRef.current as HTMLAudioElement);

    useEffect(() => {
      const audio = audioRef.current;
      if (!audio) return;

      const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
      const handleDurationChange = () => setDuration(audio.duration);
      const handlePlay = () => setIsPlaying(true);
      const handlePause = () => setIsPlaying(false);

      audio.addEventListener("timeupdate", handleTimeUpdate);
      audio.addEventListener("durationchange", handleDurationChange);
      audio.addEventListener("play", handlePlay);
      audio.addEventListener("pause", handlePause);

      return () => {
        audio.removeEventListener("timeupdate", handleTimeUpdate);
        audio.removeEventListener("durationchange", handleDurationChange);
        audio.removeEventListener("play", handlePlay);
        audio.removeEventListener("pause", handlePause);
      };
    }, []);

    const handlePlayPause = () => {
      const audio = audioRef.current;
      if (!audio) return;
      if (isPlaying) {
        audio.pause();
      } else {
        audio.play();
      }
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
      const audio = audioRef.current;
      if (!audio) return;
      const newTime = parseFloat(e.target.value);
      audio.currentTime = newTime;
      setCurrentTime(newTime);
    };

    return (
      <div className={className}>
        <audio ref={audioRef} preload="metadata" className="hidden" />
        <div className="flex items-center gap-3 bg-slate-50 rounded p-3">
          <button
            onClick={handlePlayPause}
            className="rounded border px-3 py-1 bg-white hover:bg-slate-100"
            disabled={!duration}
          >
            {isPlaying ? "⏸" : "▶"}
          </button>
          <input
            type="range"
            min="0"
            max={duration || 0}
            step="0.001"
            value={currentTime}
            onChange={handleSeek}
            className="flex-1"
            disabled={!duration}
          />
          <div className="text-sm font-mono whitespace-nowrap">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
        </div>
      </div>
    );
  }
);
AudioPlayer.displayName = "AudioPlayer";
