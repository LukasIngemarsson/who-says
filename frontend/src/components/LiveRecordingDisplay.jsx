import { useRef, useEffect } from "react";

/**
 * Test phrases to speak when tuning live recording parameters.
 */
const TEST_PHRASES = [
  "Now, interesting. Interesting. Are you working?",
  "Why aren't you working now? That's super interesting.",
  "Okay, okay, okay, let's try this again.",
  "Are you doing that? Are you there? Are you working?",
  "This is so weird. This is really, really weird.",
];

/**
 * Display component for live recording state.
 * Shows current speaker, overlap detection, and transcript messages.
 */
const LiveRecordingDisplay = ({
  isRecording,
  doneRecording,
  recordingTime,
  displayedSpeaker,
  displayedOverlap,
  displayedSpeakers,
  hasSpeech = false,
  messages,
  showTestPhrases = true,
}) => {
  const messagesRef = useRef(null);

  // Auto-scroll messages to bottom
  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages]);

  if (!doneRecording && !isRecording) return null;

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-slate-100 mb-2">
            Live Recording
          </h3>

          {/* Speaker display */}
          {!doneRecording && displayedSpeaker === "Unknown" ? (
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
              <span className="text-slate-400">Listening...</span>
              <span className="text-slate-500">(No speech detected)</span>
            </div>
          )}

          {/* Overlap Detection Indicator */}
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

          {/* Test phrases for tuning */}
          {showTestPhrases && (
            <div className="mt-4 bg-slate-950/70 border border-slate-800 rounded-lg p-3">
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
                Test phrases for tuning
              </h4>
              <ul className="space-y-1 text-sm text-slate-300">
                {TEST_PHRASES.map((p, idx) => (
                  <li key={idx} className="flex gap-2">
                    <span className="text-slate-500">{idx + 1}.</span>
                    <span>{p}</span>
                  </li>
                ))}
              </ul>
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
                  <div key={msg.id} className="flex flex-col items-start gap-1">
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

        {/* Recording indicator */}
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
          <span className="text-slate-400">{recordingTime}s</span>
        </div>
      </div>
    </div>
  );
};

export default LiveRecordingDisplay;
