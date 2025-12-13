import { useState, useRef, useEffect } from "react";
import { useTranscriptAccumulator } from "../utils/useTranscriptAccumulator";
import { useSpeakerDetection } from "../utils/useSpeakerDetection";

/**
 * Demo Mode: Stream uploaded audio file as if it were live recording.
 * Useful for testing ASR + diarisation without a microphone.
 */
const DemoMode = ({
  disabled = false,
  sessionIdRef,
  onError,
  onDemoStateChange,
  maxWordsPerBubble = 18,
}) => {
  // Demo state
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [demoFile, setDemoFile] = useState(null);
  const [demoChunkSec, setDemoChunkSec] = useState(0.5);
  const [demoRealtime, setDemoRealtime] = useState(true);
  const [demoPlayAudio, setDemoPlayAudio] = useState(true);
  const [demoProgress, setDemoProgress] = useState(0);
  const [demoDuration, setDemoDuration] = useState(0);
  const [recordingTime, setRecordingTime] = useState(0);
  const [demoAudioUrl, setDemoAudioUrl] = useState(null);

  // Refs
  const demoAbortRef = useRef(null);
  const demoAudioRef = useRef(null);
  const progressThrottleRef = useRef(0);

  // Hooks
  const { messages, appendFromResponse, clear: clearMessages } =
    useTranscriptAccumulator({ maxWordsPerBubble });

  const {
    displayedSpeaker,
    displayedOverlap,
    displayedSpeakers,
    hasSpeech,
    processResponse,
    reset: resetSpeaker,
  } = useSpeakerDetection();

  // Ref for messages container auto-scroll
  const messagesRef = useRef(null);

  // Auto-scroll messages
  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages]);

  // Notify parent of demo state changes
  useEffect(() => {
    onDemoStateChange?.(isDemoMode);
  }, [isDemoMode, onDemoStateChange]);

  // Cleanup demo audio URL on unmount
  useEffect(() => {
    const currentUrl = demoAudioUrl;
    return () => {
      if (currentUrl) {
        URL.revokeObjectURL(currentUrl);
      }
    };
  }, [demoAudioUrl]);

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      if (demoAbortRef.current) {
        demoAbortRef.current.abort();
      }
    };
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (demoAudioUrl) {
        URL.revokeObjectURL(demoAudioUrl);
      }
      setDemoFile(file);
      setDemoAudioUrl(URL.createObjectURL(file));
      onError?.("");
    }
  };

  // Normalize audio to have consistent levels
  const normalizeAudio = (samples) => {
    let maxAbs = 0;
    for (let i = 0; i < samples.length; i++) {
      const abs = Math.abs(samples[i]);
      if (abs > maxAbs) maxAbs = abs;
    }
    if (maxAbs < 0.001) return samples;
    const scale = 0.9 / maxAbs;
    const normalized = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      normalized[i] = samples[i] * scale;
    }
    return normalized;
  };

  const decodeWavToFloat32_16k = async (arrayBuffer) => {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    const srcRate = decoded.sampleRate;
    let srcData = decoded.getChannelData(0);

    // Mix down to mono if needed
    if (decoded.numberOfChannels > 1) {
      const mono = new Float32Array(decoded.length);
      for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
        const channelData = decoded.getChannelData(ch);
        for (let i = 0; i < decoded.length; i++) {
          mono[i] += channelData[i] / decoded.numberOfChannels;
        }
      }
      srcData = mono;
    }

    let output;
    if (srcRate === 16000) {
      output = new Float32Array(srcData);
    } else {
      // Resample using OfflineAudioContext
      const durationSec = decoded.duration;
      const targetLength = Math.max(1, Math.round(durationSec * 16000));
      const offline = new OfflineAudioContext(1, targetLength, 16000);
      const buffer = offline.createBuffer(1, srcData.length, srcRate);
      buffer.copyToChannel(srcData, 0);
      const source = offline.createBufferSource();
      source.buffer = buffer;
      source.connect(offline.destination);
      source.start();
      const rendered = await offline.startRendering();
      output = new Float32Array(rendered.getChannelData(0));
    }

    return normalizeAudio(output);
  };

  const float32ToBase64 = (float32) => {
    const float32Copy = new Float32Array(float32.length);
    float32Copy.set(float32);
    const bytes = new Uint8Array(float32Copy.buffer);
    let binary = "";
    for (let i = 0; i < bytes.length; i++)
      binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  };

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const startDemo = async () => {
    if (!demoFile) {
      onError?.("Please select an audio file for demo mode");
      return;
    }

    setIsDemoMode(true);
    clearMessages();
    resetSpeaker();
    setRecordingTime(0);
    setDemoProgress(0);
    onError?.("");

    const sessionId = `demo_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    if (sessionIdRef) {
      sessionIdRef.current = sessionId;
    }

    demoAbortRef.current = new AbortController();

    try {
      // Reset backend session state
      await fetch("/reset_session", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          session_id: sessionId,
          reset_global: "1",
        }),
      }).catch((err) => console.warn("Failed to reset session:", err));

      const arrayBuffer = await demoFile.arrayBuffer();
      const audio16k = await decodeWavToFloat32_16k(arrayBuffer);

      const totalDurationSec = audio16k.length / 16000;
      setDemoDuration(totalDurationSec);

      const chunkSamples = Math.max(256, Math.round(demoChunkSec * 16000));
      const totalChunks = Math.ceil(audio16k.length / chunkSamples);

      console.log(
        `Demo: streaming ${totalDurationSec.toFixed(1)}s audio in ${totalChunks} chunks`
      );

      // Start audio playback if enabled
      if (demoPlayAudio && demoRealtime && demoAudioRef.current) {
        demoAudioRef.current.currentTime = 0;
        demoAudioRef.current
          .play()
          .catch((err) => console.warn("Audio playback failed:", err));
      }

      for (let i = 0; i < audio16k.length; i += chunkSamples) {
        if (demoAbortRef.current?.signal.aborted) break;

        const chunkStartTime = Date.now();
        const chunk = audio16k.slice(
          i,
          Math.min(i + chunkSamples, audio16k.length)
        );
        const base64Audio = float32ToBase64(chunk);

        // Throttle progress updates to reduce re-renders
        const currentTimeSec = i / 16000;
        const now = Date.now();
        if (now - progressThrottleRef.current > 250) {
          setRecordingTime(Math.floor(currentTimeSec));
          setDemoProgress((currentTimeSec / totalDurationSec) * 100);
          progressThrottleRef.current = now;
        }

        const data = await fetch("/identify_speaker", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: new URLSearchParams({
            audio_data: base64Audio,
            sample_rate: "16000",
            session_id: sessionId,
          }),
          signal: demoAbortRef.current?.signal,
        }).then((r) => r.json());

        // Process speaker detection
        processResponse(data);

        // Append transcript
        appendFromResponse(data);

        // Sync with real-time if enabled
        if (demoRealtime) {
          const elapsed = Date.now() - chunkStartTime;
          const remainingSleep = Math.max(0, demoChunkSec * 1000 - elapsed);
          if (remainingSleep > 0) {
            await sleep(remainingSleep);
          }
        }
      }

      setDemoProgress(100);
      console.log("Demo stream completed");
    } catch (e) {
      if (e.name !== "AbortError") {
        console.error("Demo stream error:", e);
        onError?.(e.message || "Demo stream failed");
      }
    } finally {
      if (demoAudioRef.current) {
        demoAudioRef.current.pause();
        demoAudioRef.current.currentTime = 0;
      }
      setIsDemoMode(false);
      setRecordingTime(0);
      setDemoProgress(0);
    }
  };

  const stopDemo = () => {
    if (demoAbortRef.current) {
      demoAbortRef.current.abort();
    }
    if (demoAudioRef.current) {
      demoAudioRef.current.pause();
      demoAudioRef.current.currentTime = 0;
    }
    setIsDemoMode(false);
    setRecordingTime(0);
    setDemoProgress(0);
    resetSpeaker();
  };

  return (
    <>
      {/* Demo Mode Controls */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div>
              <h3 className="text-lg font-semibold text-slate-100">
                Demo Mode
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Upload an audio file and stream it as if it were live microphone
                input
              </p>
            </div>
            {isDemoMode && (
              <div className="flex items-center gap-2 text-sm text-slate-300">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span>
                  {recordingTime}s / {Math.floor(demoDuration)}s
                </span>
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2">
              <span className="text-sm text-slate-300">Audio file:</span>
              <input
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                disabled={isDemoMode || disabled}
                className="text-sm text-slate-300 file:mr-2 file:py-1 file:px-3 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700 disabled:opacity-50"
              />
            </label>
            {demoFile && (
              <span
                className="text-xs text-slate-400 truncate max-w-48"
                title={demoFile.name}
              >
                {demoFile.name}
              </span>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <span>Chunk (s):</span>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="2.0"
                className="w-16 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-right text-sm"
                value={demoChunkSec}
                onChange={(e) =>
                  setDemoChunkSec(
                    parseFloat(e.target.value.replace(",", ".")) || 0.5
                  )
                }
                disabled={isDemoMode}
              />
            </label>

            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={demoRealtime}
                onChange={(e) => setDemoRealtime(e.target.checked)}
                disabled={isDemoMode}
                className="accent-indigo-500"
              />
              <span>Real-time</span>
            </label>

            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={demoPlayAudio}
                onChange={(e) => setDemoPlayAudio(e.target.checked)}
                disabled={isDemoMode || !demoRealtime}
                className="accent-green-500"
              />
              <span>Play audio</span>
            </label>

            {!isDemoMode ? (
              <button
                onClick={startDemo}
                disabled={!demoFile || disabled}
                className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg text-sm font-semibold transition-colors"
              >
                Start Demo
              </button>
            ) : (
              <button
                onClick={stopDemo}
                className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-semibold transition-colors"
              >
                Stop Demo
              </button>
            )}
          </div>

          {/* Progress bar */}
          {isDemoMode && (
            <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              <div
                className="bg-green-500 h-full transition-all duration-300"
                style={{ width: `${demoProgress}%` }}
              />
            </div>
          )}
        </div>
      </div>

      {/* Demo Mode Live Display */}
      {isDemoMode && (
        <div className="bg-slate-900 border border-green-700/50 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-slate-100 mb-2 flex items-center gap-2">
                <span className="px-2 py-0.5 bg-green-600/20 text-green-400 text-xs rounded-full">
                  DEMO
                </span>
                Streaming: {demoFile?.name}
              </h3>

              {/* Speaker display */}
              {displayedSpeaker === "Unknown" ? (
                <div className="flex items-center gap-3">
                  <span className="text-slate-400">Speech detected:</span>
                  <span className="text-xl font-semibold text-yellow-400">
                    Unknown Speaker
                  </span>
                  <span className="text-slate-500 text-sm">(Not enrolled)</span>
                </div>
              ) : displayedSpeaker ? (
                <div className="flex items-center gap-3">
                  <span className="text-slate-400">Current Speaker:</span>
                  <span className="text-2xl font-bold text-blue-400">
                    {displayedSpeaker}
                  </span>
                </div>
              ) : hasSpeech ? (
                <div className="flex items-center gap-3">
                  <span className="text-slate-400">Identifying speaker...</span>
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                </div>
              ) : (
                <div className="flex items-center gap-3">
                  <span className="text-slate-400">Processing audio...</span>
                  <span className="text-slate-500">(No speech detected)</span>
                </div>
              )}

              {/* Overlap indicator */}
              {displayedOverlap && (
                <div className="mt-3 p-3 bg-orange-500/20 border border-orange-500/40 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
                    <span className="text-orange-400 font-semibold text-sm">
                      Speaker Overlap Detected
                    </span>
                  </div>
                  {displayedSpeakers.length > 0 && (
                    <div className="mt-1 text-xs text-orange-300">
                      Overlapping speakers: {displayedSpeakers.join(", ")}
                    </div>
                  )}
                </div>
              )}

              {/* Live transcript messages */}
              {messages.length > 0 && (
                <div
                  ref={messagesRef}
                  className="mt-4 bg-slate-950/70 border border-slate-800 rounded-lg p-3 max-h-80 overflow-y-auto"
                >
                  <div className="space-y-3">
                    {messages.map((msg) => (
                      <div
                        key={msg.id}
                        className="flex flex-col items-start gap-1"
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-semibold text-slate-400 tracking-wide uppercase">
                            {msg.speaker || "Identifying..."}
                          </span>
                        </div>
                        <div className="bg-slate-800/80 rounded-2xl px-3 py-2 max-w-full">
                          <p className="text-sm text-slate-100 leading-snug">
                            {msg.text}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2 ml-4">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-slate-400 font-mono">{recordingTime}s</span>
            </div>
          </div>
        </div>
      )}

      {/* Hidden audio element for playback */}
      {demoAudioUrl && (
        <audio
          ref={demoAudioRef}
          src={demoAudioUrl}
          onEnded={() => console.log("Demo audio playback ended")}
        />
      )}
    </>
  );
};

export default DemoMode;
