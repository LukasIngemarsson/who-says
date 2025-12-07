import { useState, useRef } from "react";
import { AlertCircle } from "lucide-react";

import Header from "./components/Header.jsx";
import ActionCard from "./components/ActionCard.jsx";
import WaveformCanvas from "./components/WaveformCanvas.jsx";
import PlayerControls from "./components/PlayerControls.jsx";
import SpeakerLegend from "./components/SpeakerLegend.jsx";
import AddSpeakerModal from "./components/AddSpeakerModal.jsx";
import KnownSpeakers from "./components/KnownSpeakers.jsx";

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

  const [isAddSpeakerModalOpen, setIsAddSpeakerModalOpen] = useState(false);
  const [speakerRefreshTrigger, setSpeakerRefreshTrigger] = useState(0);

  const audioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const recordingTimerRef = useRef(null);

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
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        const fakeEvent = { target: { files: [file] } };
        await handleFileUpload(fakeEvent);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      handleReset();
      setRecordingTime(0);
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Microphone access failed.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(recordingTimerRef.current);
    }
  };

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
