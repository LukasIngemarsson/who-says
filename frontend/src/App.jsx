import { useState, useRef, useEffect } from "react";
import { AlertCircle } from "lucide-react";

import Header from "./components/Header.jsx";
import ActionCard from "./components/ActionCard.jsx";
import WaveformCanvas from "./components/WaveformCanvas.jsx";
import PlayerControls from "./components/PlayerControls.jsx";
import SpeakerLegend from "./components/SpeakerLegend.jsx";
import AddSpeakerModal from "./components/AddSpeakerModal.jsx";
import KnownSpeakers from "./components/KnownSpeakers.jsx";
import SpeakerIdentificationModal from "./components/SpeakerIdentificationModal.jsx";

const App = () => {
  const [mode, setMode] = useState("upload");
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [segments, setSegments] = useState([]);
  const [fullTranscriptionResult, setFullTranscriptionResult] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [numSpeakers, setNumSpeakers] = useState(2);
  const [errorMsg, setErrorMsg] = useState("");
  const [liveMessages, setLiveMessages] = useState([]);

  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [currentSpeaker, setCurrentSpeaker] = useState(null); // Current speaker during recording

  const [isAddSpeakerModalOpen, setIsAddSpeakerModalOpen] = useState(false);
  const [speakerRefreshTrigger, setSpeakerRefreshTrigger] = useState(0);
  const [isSpeakerModalOpen, setIsSpeakerModalOpen] = useState(false);
  const [pendingAudioData, setPendingAudioData] = useState(null);
  const [pendingSpeakerInfo, setPendingSpeakerInfo] = useState(null);
  const [knownSpeakersList, setKnownSpeakersList] = useState([]);
  const sessionIdRef = useRef(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

  const audioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const recordingTimerRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const audioBufferAccumulatorRef = useRef([]);
  const lastProcessTimeRef = useRef(0);
  const lastSnippetRef = useRef("");
  const speakerHistoryRef = useRef([]);
  const liveMessagesRef = useRef(null);

  const fetchKnownSpeakers = async () => {
    try {
      const response = await fetch("/status");
      const data = await response.json();
      setKnownSpeakersList(data.known_speakers || []);
    } catch (error) {
      console.error("Failed to fetch speakers:", error);
    }
  };

  useEffect(() => {
    fetchKnownSpeakers();
  }, [speakerRefreshTrigger]);

  // Always keep the live messages scrolled to the bottom while recording
  useEffect(() => {
    if (liveMessagesRef.current) {
      liveMessagesRef.current.scrollTop = liveMessagesRef.current.scrollHeight;
    }
  }, [liveMessages]);

  const decodeAudioForVisualization = async (arrayBuffer) => {
    const audioContext = new (window.AudioContext ||
      window.webkitAudioContext)();
    try {
      const decodedBuffer = await audioContext.decodeAudioData(arrayBuffer);
      setAudioBuffer(decodedBuffer);
      setDuration(decodedBuffer.duration);
    } catch (error) {
      console.error("Error decoding audio data for visualization", error);
    }
  };

  const handleReset = () => {
    if (audioRef.current) audioRef.current.pause();
    setIsPlaying(false);
    setCurrentTime(0);
    setAudioUrl(null);
    setAudioBuffer(null);
    setSegments([]);
    setFullTranscriptionResult(null);
    setDuration(0);
    setErrorMsg("");
    setLiveMessages([]);
    lastSnippetRef.current = "";
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    handleReset();
    setProcessing(true);
    setErrorMsg("");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("num_speakers", numSpeakers);

    const url = URL.createObjectURL(file);
    setAudioUrl(url);

    const fileBufferForWaveform = await file.arrayBuffer();
    await decodeAudioForVisualization(fileBufferForWaveform);

    try {
      const response = await fetch("/process", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Processing failed");
      }

      if (data.segments) {
        setSegments(data.segments);
        setFullTranscriptionResult(data);
      } else {
        console.warn("No segments found in response", data);
      }
    } catch (error) {
      console.error("Error:", error);
      setErrorMsg(error.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleDownloadJson = () => {
    if (!fullTranscriptionResult) return;
    const dataStr =
      "data:text/json;charset=utf-8," +
      encodeURIComponent(JSON.stringify(fullTranscriptionResult, null, 2));
    const downloadAnchorNode = document.createElement("a");
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "transcription_result.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const startRecording = async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000
        } 
      });
      
      streamRef.current = stream;
      chunksRef.current = [];
      audioBufferAccumulatorRef.current = [];
      lastProcessTimeRef.current = Date.now();
      setLiveMessages([]);
      lastSnippetRef.current = "";

      // Set up Web Audio API for raw audio capture
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      
      // --- ADD THIS ABOVE onaudioprocess ---
      let bufferAccumulator = [];  // <-- put this right before processor.onaudioprocess
      // -------------------------------------


      // --- REPLACE your entire onaudioprocess with this ---
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);

        // append samples (browser may give 2048 even if you requested 4096)
        bufferAccumulator.push(...inputData);

        // once we have >= 4096 samples (~256ms), send exactly that
        const TARGET_SIZE = 4096;

        while (bufferAccumulator.length >= TARGET_SIZE) {

          const slice = bufferAccumulator.slice(0, TARGET_SIZE);
          bufferAccumulator = bufferAccumulator.slice(TARGET_SIZE);

          // Convert Float32Array → Uint8Array → Base64
          const floatArray = new Float32Array(slice);
          const audioBytes = new Uint8Array(floatArray.buffer);

          let binary = '';
          for (let i = 0; i < audioBytes.length; i++) {
            binary += String.fromCharCode(audioBytes[i]);
          }
          const base64Audio = btoa(binary);

          // Always send request for speaker identification (and live transcript)
          fetch("/identify_speaker", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({
              audio_data: base64Audio,
              sample_rate: "16000",
              session_id: sessionIdRef.current
            })
          })
            .then(r => r.json())
            .then(data => {
              console.log("Speaker identification response:", data);

              // --- Update fast "who is speaking now" indicator ---
              if (data.has_speech) {
                setCurrentSpeaker(data.speaker || "Unknown");
              } else {
                setCurrentSpeaker(null);
              }

              // --- Maintain a short history of detected speakers for labeling transcripts ---
              const detectedSpeaker = data.speaker || "Unknown";
              {
                const history = speakerHistoryRef.current.slice();
                history.push(detectedSpeaker);
                // Keep last ~20 detections (a few seconds of context)
                if (history.length > 10) history.shift();
                speakerHistoryRef.current = history;
              }

              // Live, incremental transcription coming from the same endpoint
              if (typeof data.transcript === "string") {
                let snippet = data.transcript.trim();
                if (!snippet) return;

                setLiveMessages(prev => {
                  const lastMsg = prev[prev.length - 1];
                  const lastText = lastMsg?.text || "";
                  const lastSpeaker = lastMsg?.speaker || "";

                  // Choose a robust speaker label for this snippet:
                  // - Prefer an explicit transcript_speaker from the backend (if provided)
                  // - Otherwise, use the majority speaker over the recent history
                  // - Fall back to the last bubble's speaker, then to the raw detected speaker
                  let majoritySpeaker = lastSpeaker;
                  if (!data.transcript_speaker) {
                    const counts = {};
                    for (const s of speakerHistoryRef.current) {
                      if (!s) continue;
                      counts[s] = (counts[s] || 0) + 1;
                    }
                    let best = null;
                    let bestCount = 0;
                    for (const [s, c] of Object.entries(counts)) {
                      if (c > bestCount) {
                        best = s;
                        bestCount = c;
                      }
                    }
                    if (best) {
                      majoritySpeaker = best;
                    }
                  }

                  const transcriptSpeaker =
                    data.transcript_speaker ||
                    majoritySpeaker ||
                    detectedSpeaker ||
                    "Unknown";

                  const updated = [...prev];

                  // If there is no previous message, just add the first bubble.
                  if (!lastMsg) {
                    const firstMsg = {
                      id: `${Date.now()}-0`,
                      text: snippet,
                      speaker: transcriptSpeaker,
                    };
                    lastSnippetRef.current = snippet;
                    return [firstMsg];
                  }

                  // Detect a change based on the *raw* detected speaker, since
                  // transcriptSpeaker can lag due to majority voting.
                  const newRawSpeaker =
                    detectedSpeaker && detectedSpeaker !== "Unknown"
                      ? detectedSpeaker
                      : transcriptSpeaker;
                  const speakerChanged =
                    lastSpeaker && newRawSpeaker && newRawSpeaker !== lastSpeaker;

                  // If the same person is still speaking, append new words to the
                  // existing bubble instead of replacing it. We try to find an
                  // overlap between the end of the previous text and the start
                  // of the new snippet so we don't double‑append words, but we
                  // never shrink or delete what was already shown.
                  if (!speakerChanged && transcriptSpeaker === lastSpeaker) {
                    if (snippet === lastText) {
                      lastSnippetRef.current = snippet;
                      return prev;
                    }

                    const prevWords = lastText.split(/\s+/).filter(Boolean);
                    const newWords = snippet.split(/\s+/).filter(Boolean);

                    let overlap = 0;
                    const maxOverlap = Math.min(prevWords.length, newWords.length);
                    for (let k = maxOverlap; k > 0; k--) {
                      const prevSuffix = prevWords.slice(prevWords.length - k).join(" ");
                      const newPrefix = newWords.slice(0, k).join(" ");
                      if (prevSuffix === newPrefix) {
                        overlap = k;
                        break;
                      }
                    }

                    const deltaWords = overlap > 0 ? newWords.slice(overlap) : newWords;
                    const delta = deltaWords.join(" ").trim();

                    if (!delta) {
                      lastSnippetRef.current = snippet;
                      return prev;
                    }

                    const appended = lastText ? `${lastText} ${delta}` : delta;
                    updated[updated.length - 1] = {
                      ...lastMsg,
                      text: appended,
                    };
                    lastSnippetRef.current = snippet;
                    return updated;
                  }

                  // Speaker changed (or we don't have a clear previous speaker): start a new bubble
                  const newMsg = {
                    id: `${Date.now()}-${updated.length}`,
                    text: snippet,
                    speaker: newRawSpeaker || transcriptSpeaker || "Unknown",
                  };
                  lastSnippetRef.current = snippet;
                  return [...updated, newMsg];
                });
              }

              // Background, more stable per-speaker segments from SCD+ASR.
              if (Array.isArray(data.turn_segments) && data.turn_segments.length > 0) {
                setLiveMessages(prev => {
                  const updated = [...prev];
                  for (const seg of data.turn_segments) {
                    const text = (seg.text || "").trim();
                    if (!text) continue;
                    updated.push({
                      id: `seg-${seg.start}-${seg.end}-${updated.length}`,
                      text,
                      speaker: seg.speaker || "Unknown",
                    });
                  }
                  return updated;
                });
              }

            })
            .catch(err => {
              console.error("Error identifying speaker:", err);
              setCurrentSpeaker(null);
            });

        }
      };

      
      
      source.connect(processor);
      processor.connect(audioContext.destination);

      // Determine supported mime type
      let mimeType = "audio/webm";
      if (!MediaRecorder.isTypeSupported("audio/webm")) {
        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
          mimeType = "audio/webm;codecs=opus";
        } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
          mimeType = "audio/ogg;codecs=opus";
        } else {
          mimeType = ""; // Use browser default
        }
      }

      const options = mimeType ? { mimeType } : {};
      const recorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.onerror = (e) => {
        console.error("MediaRecorder error:", e);
        setErrorMsg("Recording error occurred");
        stopRecording();
      };

      recorder.onstop = async () => {
        try {
          // Determine blob type from mimeType
          const blobType = mimeType || "audio/webm";
          const blob = new Blob(chunksRef.current, { type: blobType });
          const file = new File([blob], `recording.${blobType.includes('ogg') ? 'ogg' : 'webm'}`, { 
            type: blobType 
          });
          const fakeEvent = { target: { files: [file] } };
          await handleFileUpload(fakeEvent);
        } catch (error) {
          console.error("Error processing recording:", error);
          setErrorMsg("Failed to process recording: " + error.message);
        } finally {
          // Clean up stream
          if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => {
              track.stop();
            });
            streamRef.current = null;
          }
        }
      };

      // Start recording with timeslice to collect data periodically
      recorder.start(1000); // Collect data every second
      setIsRecording(true);
      handleReset();
      setRecordingTime(0);
      setCurrentSpeaker(null); // Reset current speaker
      setErrorMsg("");
      // Generate new session ID for next recording
      sessionIdRef.current = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      let errorMessage = "Microphone access failed.";
      if (err.name === "NotAllowedError") {
        errorMessage = "Microphone access denied. Please allow microphone access and try again.";
      } else if (err.name === "NotFoundError") {
        errorMessage = "No microphone found. Please connect a microphone and try again.";
      } else if (err.name === "NotReadableError") {
        errorMessage = "Microphone is already in use by another application.";
      }
      setErrorMsg(errorMessage);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      try {
        // Disconnect audio processor
        if (processorRef.current) {
          processorRef.current.disconnect();
          processorRef.current = null;
        }
        
        if (mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop();
        }
        setIsRecording(false);
        
        if (recordingTimerRef.current) {
          clearInterval(recordingTimerRef.current);
          recordingTimerRef.current = null;
        }
      } catch (error) {
        console.error("Error stopping recording:", error);
        setErrorMsg("Error stopping recording: " + error.message);
        setIsRecording(false);
      }
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      if (processorRef.current) {
        processorRef.current.disconnect();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) audioRef.current.pause();
      else audioRef.current.play();
      setIsPlaying(!isPlaying);
    }
  };

  const handleSeek = (newTime) => {
    if (audioRef.current) {
      audioRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-sans selection:bg-blue-500/30">
      <div className="max-w-5xl mx-auto space-y-8">
        <Header
          mode={mode}
          setMode={setMode}
          onAddSpeaker={() => setIsAddSpeakerModalOpen(true)}
        />

        <AddSpeakerModal
          isOpen={isAddSpeakerModalOpen}
          onClose={() => {
            setIsAddSpeakerModalOpen(false);
            setSpeakerRefreshTrigger(prev => prev + 1); // Trigger refresh
          }}
        />

        <SpeakerIdentificationModal
          isOpen={isSpeakerModalOpen}
          onClose={() => {
            setIsSpeakerModalOpen(false);
            setPendingAudioData(null);
            setPendingSpeakerInfo(null);
            setSpeakerRefreshTrigger(prev => prev + 1);
          }}
          audioData={pendingAudioData}
          speakerInfo={pendingSpeakerInfo}
          knownSpeakers={knownSpeakersList}
          onRefresh={() => {
            fetchKnownSpeakers();
            setSpeakerRefreshTrigger(prev => prev + 1);
          }}
        />

        <KnownSpeakers refreshTrigger={speakerRefreshTrigger} />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <ActionCard
            mode={mode}
            isRecording={isRecording}
            recordingTime={recordingTime}
            handleFileUpload={handleFileUpload}
            startRecording={startRecording}
            stopRecording={stopRecording}
            numSpeakers={numSpeakers}
            setNumSpeakers={setNumSpeakers}
            isProcessing={processing}
          />
        </div>

        {/* Current speaker indicator during recording */}
        {isRecording && (
          <div className="bg-slate-900 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-slate-100 mb-2">Live Recording</h3>
                {currentSpeaker ? (
                  <div className="flex items-center gap-3">
                    <span className="text-slate-400">Current Speaker:</span>
                    <span className="text-2xl font-bold text-blue-400">{currentSpeaker}</span>
                  </div>
                ) : currentSpeaker === "Unknown" ? (
                  <div className="flex items-center gap-3">
                    <span className="text-slate-400">Speech detected:</span>
                    <span className="text-xl font-semibold text-yellow-400">Unknown Speaker</span>
                    <span className="text-slate-500 text-sm">(Not enrolled)</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-3">
                    <span className="text-slate-400">Listening...</span>
                    <span className="text-slate-500">(No speech detected)</span>
                  </div>
                )}

                {liveMessages.length > 0 && (
                  <div
                    ref={liveMessagesRef}
                    className="mt-4 bg-slate-950/70 border border-slate-800 rounded-lg p-3 max-h-80 overflow-y-auto"
                  >
                    <div className="space-y-3">
                      {liveMessages.map((msg) => (
                        <div key={msg.id} className="flex flex-col items-start gap-1">
                          <span className="text-xs font-semibold text-slate-400 tracking-wide uppercase">
                            {msg.speaker || "Unknown"}
                          </span>
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
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                <span className="text-slate-400">{recordingTime}s</span>
              </div>
            </div>
          </div>
        )}

        {errorMsg && (
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 flex items-center gap-2">
            <AlertCircle size={20} />
            <span>{errorMsg}</span>
          </div>
        )}

        <WaveformCanvas
          audioBuffer={audioBuffer}
          segments={segments}
          duration={duration}
          currentTime={currentTime}
          isRecording={isRecording}
          processing={processing}
          onSeek={handleSeek}
        />

        <PlayerControls
          isPlaying={isPlaying}
          togglePlay={togglePlay}
          disabled={!audioBuffer || isRecording}
        />

        <SpeakerLegend
          segments={segments}
          onDownload={fullTranscriptionResult ? handleDownloadJson : null}
        />
      </div>

      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={() => {
          if (audioRef.current) setCurrentTime(audioRef.current.currentTime);
        }}
        onEnded={() => setIsPlaying(false)}
        onLoadedMetadata={() => {
          if (audioRef.current && segments.length === 0) {
            if (duration === 0) setDuration(audioRef.current.duration);
          }
        }}
      />
    </div>
  );
};

export default App;
