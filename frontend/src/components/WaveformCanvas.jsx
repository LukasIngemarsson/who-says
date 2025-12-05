import { useRef, useEffect, useState } from "react";
import { FileAudio, RefreshCw, AlertCircle } from "lucide-react";
import { SPEAKER_COLORS, formatTime } from "../utils/constants.js";

const WaveformCanvas = ({
  audioBuffer,
  segments,
  duration,
  currentTime,
  isRecording,
  processing,
  onSeek,
}) => {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const animationRef = useRef(null);
  const [hoveredSegment, setHoveredSegment] = useState(null);

  // Handle Seek Click
  const handleClick = (e) => {
    if (!containerRef.current || !duration) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = rect.width;
    const seekTime = (x / width) * duration;
    onSeek(seekTime);
  };

  // Handle Hover Logic
  const handleMouseMove = (e) => {
    if (!containerRef.current || segments.length === 0) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = rect.width;
    const hoverTime = (x / width) * duration;
    const segment = segments.find(
      (seg) => hoverTime >= seg.start && hoverTime <= seg.end
    );
    setHoveredSegment(segment || null);
  };

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas || !containerRef.current) return;

    const ctx = canvas.getContext("2d");
    const width = containerRef.current.offsetWidth;
    const height = 240;

    // Handle high DPI
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    ctx.clearRect(0, 0, width, height);
    const centerY = height / 2;

    // 1. Draw Waveform
    if (audioBuffer) {
      const rawData = audioBuffer.getChannelData(0);
      const samples = 800;
      const blockSize = Math.floor(rawData.length / samples);
      const barWidth = width / samples;

      for (let i = 0; i < samples; i++) {
        const start = i * blockSize;
        let sum = 0;
        for (let j = 0; j < blockSize; j++) {
          sum += Math.abs(rawData[start + j]);
        }
        const avg = sum / blockSize;
        const barHeight = Math.max(2, avg * height * 2.5);
        const x = i * barWidth;
        const y = centerY - barHeight / 2;

        // Color based on playback
        const isPlayed = i / samples < currentTime / duration;
        ctx.fillStyle = isPlayed ? "#3b82f6" : "#475569";
        ctx.fillRect(x, y, barWidth - 0.5, barHeight);
      }
    } else if (isRecording) {
      // Fake recording visualizer
      ctx.fillStyle = "#ef4444";
      const time = Date.now() / 200;
      for (let i = 0; i < 50; i++) {
        const h = Math.sin(time + i * 0.2) * 20 + 30;
        ctx.fillRect(width / 2 + i * 6 - 150, centerY - h / 2, 4, h);
      }
      ctx.font = "14px sans-serif";
      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.fillText("Recording Audio...", width / 2, centerY + 60);
    } else {
      // Placeholder line
      ctx.beginPath();
      ctx.strokeStyle = "#334155";
      ctx.lineWidth = 2;
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();
    }

    // 2. Draw Segments
    if (segments.length > 0 && !isRecording) {
      segments.forEach((seg) => {
        const x = (seg.start / duration) * width;
        const w = (seg.duration / duration) * width;
        const color = SPEAKER_COLORS[seg.cluster_id] || SPEAKER_COLORS.default;

        ctx.fillStyle = color + "15";
        ctx.fillRect(x, 0, w, height);
        ctx.fillStyle = color;
        ctx.fillRect(x, 0, w, 4);
      });
    }

    // 3. Playhead
    if (!isRecording && duration > 0) {
      const playheadX = (currentTime / duration) * width;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.shadowColor = "rgba(0,0,0,0.5)";
      ctx.shadowBlur = 4;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  };

  useEffect(() => {
    const renderLoop = () => {
      draw();
      animationRef.current = requestAnimationFrame(renderLoop);
    };
    renderLoop();
    return () => cancelAnimationFrame(animationRef.current);
  }, [audioBuffer, currentTime, duration, segments, isRecording]);

  return (
    <div className="space-y-4">
      {/* Time Header */}
      <div className="flex items-center justify-between text-sm text-slate-400 px-1">
        <span className="font-mono">{formatTime(currentTime)}</span>
        <div className="flex items-center gap-2">
          {segments.length === 0 && duration > 0 && !processing && (
            <span className="flex items-center gap-1.5 text-amber-500 bg-amber-500/10 px-2 py-1 rounded text-xs">
              <AlertCircle size={12} /> Live audio - No transcription
            </span>
          )}
          <span className="font-mono">{formatTime(duration)}</span>
        </div>
      </div>

      {/* Main Canvas Container */}
      <div
        ref={containerRef}
        className="relative w-full h-[240px] bg-slate-900 rounded-xl border border-slate-800 shadow-inner overflow-hidden cursor-pointer group"
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredSegment(null)}
      >
        {processing && (
          <div className="absolute inset-0 z-20 bg-slate-950/80 flex items-center justify-center backdrop-blur-sm">
            <div className="flex flex-col items-center gap-3">
              <RefreshCw className="animate-spin text-blue-500 w-8 h-8" />
              <span className="text-blue-400 font-medium">
                Processing Audio...
              </span>
            </div>
          </div>
        )}

        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full z-10"
        />

        {hoveredSegment && !isRecording && (
          <div className="absolute z-30 bottom-4 left-4 right-4 animate-in fade-in slide-in-from-bottom-2 duration-200 pointer-events-none">
            <div className="bg-slate-800/90 border border-slate-700 p-4 rounded-lg shadow-2xl backdrop-blur-md">
              <div className="flex items-center gap-2 mb-2">
                <span
                  className="w-2.5 h-2.5 rounded-full ring-2 ring-slate-900"
                  style={{
                    backgroundColor: SPEAKER_COLORS[hoveredSegment.cluster_id],
                  }}
                />
                <span className="text-xs font-bold text-slate-300 tracking-wider">
                  {hoveredSegment.speaker}
                </span>
                <span className="text-xs font-mono text-slate-500 ml-auto border border-slate-700 px-1.5 py-0.5 rounded">
                  {hoveredSegment.start.toFixed(1)}s -{" "}
                  {hoveredSegment.end.toFixed(1)}s
                </span>
              </div>
              <p className="text-slate-100 text-sm leading-relaxed font-medium">
                "{hoveredSegment.text}"
              </p>
            </div>
          </div>
        )}

        {!audioBuffer && !isRecording && !processing && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-30">
            <div className="text-center">
              <FileAudio className="w-12 h-12 mx-auto mb-2 text-slate-500" />
              <p>No audio loaded</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WaveformCanvas;
