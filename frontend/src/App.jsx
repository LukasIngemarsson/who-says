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

      // Set up Web Audio API for raw audio capture
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioBufferAccumulatorRef.current.push(new Float32Array(inputData));
        
        const now = Date.now();
        // Increased from 3s to 5s to collect more audio before processing
        if (now - lastProcessTimeRef.current >= 5000) {
          // Combine accumulated samples
          const totalLength = audioBufferAccumulatorRef.current.reduce((sum, arr) => sum + arr.length, 0);
          
          if (totalLength === 0) {
            console.warn("No audio samples accumulated");
            audioBufferAccumulatorRef.current = [];
            lastProcessTimeRef.current = now;
            return;
          }
          
          const combined = new Float32Array(totalLength);
          let offset = 0;
          for (const chunk of audioBufferAccumulatorRef.current) {
            combined.set(chunk, offset);
            offset += chunk.length;
          }
          
          // Check if audio has any significant amplitude
          const maxAmplitude = Math.max(...Array.from(combined).map(Math.abs));
          const avgAmplitude = Array.from(combined).reduce((sum, val) => sum + Math.abs(val), 0) / combined.length;
          
          console.log(`Sending audio chunk: ${totalLength} samples, max amplitude: ${maxAmplitude.toFixed(4)}, avg: ${avgAmplitude.toFixed(4)}`);
          
          // Convert to base64
          const audioBytes = new Uint8Array(combined.buffer);
          let binary = '';
          const len = audioBytes.byteLength;
          for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(audioBytes[i]);
          }
          const base64Audio = btoa(binary);
          
          // Send to server
          fetch("/identify_speaker", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({
              audio_data: base64Audio,
              sample_rate: "16000",
              session_id: sessionIdRef.current
            })
          })
          .then(response => response.json())
          .then(data => {
            console.log("Speaker identification response:", data);
            if (data.has_speech) {
              if (data.speaker) {
                setCurrentSpeaker(data.speaker);
              } else {
                // Speech detected but no matching speaker
                setCurrentSpeaker("Unknown");
              }
              
              // No modal during recording - just display who's talking
            } else {
              setCurrentSpeaker(null);
            }
          })
          .catch(error => {
            console.error("Error identifying speaker:", error);
            setCurrentSpeaker(null);
          });
          
          audioBufferAccumulatorRef.current = [];
          lastProcessTimeRef.current = now;
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
