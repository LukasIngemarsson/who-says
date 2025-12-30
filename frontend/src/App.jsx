import { useState, useRef, useEffect } from "react";
import { AlertCircle, SlidersHorizontal } from "lucide-react";

import Header from "./components/Header.jsx";
import ActionCard from "./components/ActionCard.jsx";
import WaveformCanvas from "./components/WaveformCanvas.jsx";
import PlayerControls from "./components/PlayerControls.jsx";
import SpeakerLegend from "./components/SpeakerLegend.jsx";
import AddSpeakerModal from "./components/AddSpeakerModal.jsx";
import KnownSpeakers from "./components/KnownSpeakers.jsx";
import SpeakerIdentificationModal from "./components/SpeakerIdentificationModal.jsx";
import DemoMode from "./components/DemoMode.jsx";
import LiveRecordingDisplay from "./components/LiveRecordingDisplay.jsx";
import { useTranscriptAccumulator } from "./utils/useTranscriptAccumulator.js";
import { useSpeakerDetection } from "./utils/useSpeakerDetection.js";

// -------------------------------------------------------------------
// TUNING + TESTING HELPERS
// -------------------------------------------------------------------

// How many words to allow in a single live bubble before starting a
// new one. Try values between 12 and 24.
const MAX_WORDS_PER_BUBBLE = 18; // examples: 12, 18, 24

// UI-only: how long to keep showing the last speaker after speech stops.
// This is intentionally decoupled from speaker detection logic.
const CURRENT_SPEAKER_DISPLAY_HOLD_MS = 1500;

// UI-only: how long to keep showing overlap indicator after overlap ends.
const OVERLAP_DISPLAY_HOLD_MS = 2000;

const App = () => {
  const [mode, setMode] = useState("upload");
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [segments, setSegments] = useState([]);
  const [fullTranscriptionResult, setFullTranscriptionResult] = useState(null);
  const [testRunJson, setTestRunJson] = useState(null);
  const [testJsonExpanded, setTestJsonExpanded] = useState(false);
  const [testJsonCopied, setTestJsonCopied] = useState(false);
  const [testRunning, setTestRunning] = useState(false);
  const [testChunkSec, setTestChunkSec] = useState(1.0);
  const [testSleepMs, setTestSleepMs] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [numSpeakers, setNumSpeakers] = useState(2);
  const [errorMsg, setErrorMsg] = useState("");
  const [tuning, setTuning] = useState(null);
  const [tuningSaving, setTuningSaving] = useState(false);
  const [tuningError, setTuningError] = useState("");
  const [availablePresets, setAvailablePresets] = useState([]);
  const [presetDetails, setPresetDetails] = useState({});
  const [selectedPreset, setSelectedPreset] = useState("default");

  // Demo mode active state (controlled by DemoMode component)
  const [isDemoMode, setIsDemoMode] = useState(false);

  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  // Use shared hooks for speaker detection and transcript accumulation
  const speakerDetection = useSpeakerDetection({
    speakerHoldMs: CURRENT_SPEAKER_DISPLAY_HOLD_MS,
    overlapHoldMs: OVERLAP_DISPLAY_HOLD_MS,
  });

  const transcriptAccumulator = useTranscriptAccumulator({
    maxWordsPerBubble: MAX_WORDS_PER_BUBBLE,
  });

  const [isAddSpeakerModalOpen, setIsAddSpeakerModalOpen] = useState(false);
  const [speakerRefreshTrigger, setSpeakerRefreshTrigger] = useState(0);
  const [isSpeakerModalOpen, setIsSpeakerModalOpen] = useState(false);
  const [pendingAudioData, setPendingAudioData] = useState(null);
  const [pendingSpeakerInfo, setPendingSpeakerInfo] = useState(null);
  const [knownSpeakersList, setKnownSpeakersList] = useState([]);
  const [asrBackend, setAsrBackend] = useState(null);
  const sessionIdRef = useRef(
    `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  );

  const audioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const recordingTimerRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);

  const fetchKnownSpeakers = async () => {
    try {
      const response = await fetch("/status");
      const data = await response.json();
      setKnownSpeakersList(data.known_speakers || []);
      setAsrBackend(data.asr_backend || null);
    } catch (error) {
      console.error("Failed to fetch speakers:", error);
    }
  };

  useEffect(() => {
    fetchKnownSpeakers();
  }, [speakerRefreshTrigger]);

  // Fetch initial tuning snapshot and available presets
  useEffect(() => {
    const loadTuning = async () => {
      try {
        const res = await fetch("/tuning");
        if (!res.ok) return;
        const data = await res.json();
        setTuning(data);
        if (data.available_presets) {
          setAvailablePresets(data.available_presets);
        }
      } catch (e) {
        console.warn("Failed to load tuning snapshot:", e);
      }
    };

    const loadPresets = async () => {
      try {
        const res = await fetch("/tuning/presets");
        if (!res.ok) return;
        const data = await res.json();
        if (data.presets) {
          setAvailablePresets(data.presets);
        }
        if (data.details) {
          setPresetDetails(data.details);
        }
      } catch (e) {
        console.warn("Failed to load presets:", e);
      }
    };

    loadTuning();
    loadPresets();
  }, []);

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
    transcriptAccumulator.clear();
    speakerDetection.reset();
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

  const handleDownloadTestJson = () => {
    if (!testRunJson) return;
    const dataStr =
      "data:text/json;charset=utf-8," +
      encodeURIComponent(JSON.stringify(testRunJson, null, 2));
    const downloadAnchorNode = document.createElement("a");
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute(
      "download",
      `testsound_run_${testRunJson?.meta?.session_id || "session"}.json`
    );
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const handleCopyTestJson = async () => {
    if (!testRunJson) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(testRunJson, null, 2));
      setTestJsonCopied(true);
      setTimeout(() => setTestJsonCopied(false), 1200);
    } catch (e) {
      console.error("Failed to copy JSON:", e);
      // fallback: still expand so user can manual copy
      setTestJsonExpanded(true);
    }
  };

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // Normalize audio to have consistent levels (peak normalization to 0.9)
  const normalizeAudio = (samples) => {
    let maxAbs = 0;
    for (let i = 0; i < samples.length; i++) {
      const abs = Math.abs(samples[i]);
      if (abs > maxAbs) maxAbs = abs;
    }
    if (maxAbs < 0.001) {
      // Audio is essentially silent
      return samples;
    }
    const scale = 0.9 / maxAbs;
    const normalized = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      normalized[i] = samples[i] * scale;
    }
    return normalized;
  };

  const decodeWavToFloat32_16k = async (
    arrayBuffer,
    shouldNormalize = true
  ) => {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    const srcRate = decoded.sampleRate;
    let srcData = decoded.getChannelData(0);

    // If stereo or multi-channel, mix down to mono
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

    // Normalize audio levels for better recognition
    if (shouldNormalize) {
      output = normalizeAudio(output);
    }

    console.log(
      `Audio decoded: ${(output.length / 16000).toFixed(
        1
      )}s, ${srcRate}Hz -> 16kHz, normalized=${shouldNormalize}`
    );
    return output;
  };

  const float32ToBase64 = (float32) => {
    // IMPORTANT: When a Float32Array is sliced, .buffer still points to the original
    // full buffer. We need to create a new ArrayBuffer with just this slice's data.
    const float32Copy = new Float32Array(float32.length);
    float32Copy.set(float32);
    const bytes = new Uint8Array(float32Copy.buffer);
    let binary = "";
    for (let i = 0; i < bytes.length; i++)
      binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  };

  const runTestSound = async () => {
    if (testRunning) return;
    setErrorMsg("");
    setTestRunning(true);
    setTestJsonExpanded(false);
    transcriptAccumulator.clear();
    speakerDetection.reset();

    const sessionId = `testsound_${Date.now()}`;
    sessionIdRef.current = sessionId;

    try {
      // Reset backend session state before starting test
      await fetch("/reset_session", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          session_id: sessionId,
          reset_global: "1",
        }),
      }).catch((err) => console.warn("Failed to reset session:", err));

      const tuningSnap = await fetch("/tuning")
        .then((r) => r.json())
        .catch(() => null);
      const wavBuf = await fetch("/testaudio/thetestsound.wav").then((r) => {
        if (!r.ok) throw new Error("Could not fetch testsound from server");
        return r.arrayBuffer();
      });

      const audio16k = await decodeWavToFloat32_16k(wavBuf);
      const chunkSamples = Math.max(256, Math.round(testChunkSec * 16000));

      const responses = [];
      for (let i = 0; i < audio16k.length; i += chunkSamples) {
        const chunk = audio16k.slice(
          i,
          Math.min(i + chunkSamples, audio16k.length)
        );
        const base64Audio = float32ToBase64(chunk);

        const data = await fetch("/identify_speaker", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: new URLSearchParams({
            audio_data: base64Audio,
            sample_rate: "16000",
            session_id: sessionId,
          }),
        }).then((r) => r.json());

        responses.push({
          t_client: Date.now(),
          chunk_index: Math.floor(i / chunkSamples),
          chunk_samples: chunk.length,
          ...data,
        });

        // Use appendSimple for testsound (simpler rendering)
        const segments =
          Array.isArray(data.transcript_segments) &&
          data.transcript_segments.length > 0
            ? data.transcript_segments
            : typeof data.transcript === "string" && data.transcript.trim()
            ? [{ speaker: data.speaker, text: data.transcript.trim() }]
            : [];
        for (const seg of segments) {
          const snippet = (seg.text || "").trim();
          if (!snippet) continue;
          const segSpeaker =
            seg.speaker || data.transcript_speaker || data.speaker;
          transcriptAccumulator.appendSimple(segSpeaker, snippet);
        }

        if (testSleepMs > 0) {
          await sleep(testSleepMs);
        }
      }

      setTestRunJson({
        meta: {
          session_id: sessionId,
          chunk_sec: testChunkSec,
          sleep_ms: testSleepMs,
          created_at: new Date().toISOString(),
        },
        tuning: tuningSnap,
        responses,
      });
      setTestJsonExpanded(true);
    } catch (e) {
      console.error(e);
      setErrorMsg(e?.message || "Failed to run testsound");
    } finally {
      setTestRunning(false);
    }
  };

  const handleTuningFieldChange =
    (section, field, parser = (v) => v) =>
    (e) => {
      // Normalise decimal separator so locales using "," still work.
      const raw = e.target.value;
      const normalised = typeof raw === "string" ? raw.replace(",", ".") : raw;
      const value = parser(normalised);
      setTuning((prev) => ({
        ...(prev || {}),
        [section]: {
          ...(prev?.[section] || {}),
          [field]: value,
        },
      }));
    };

  const handleApplyTuning = async () => {
    if (!tuning) return;
    setTuningSaving(true);
    setTuningError("");
    try {
      const payload = {
        // streaming (whisper.cpp-first)
        asr_min_new_sec: tuning.streaming?.asr_min_new_sec,
        min_speech_sec: tuning.streaming?.min_speech_sec,
        min_asr_interval_sec: tuning.streaming?.min_asr_interval_sec,
        max_asr_buffer_sec: tuning.streaming?.max_asr_buffer_sec,
        wcpp_context_sec: tuning.streaming?.wcpp_context_sec,
        cross_tail_dup_pad_sec: tuning.streaming?.cross_tail_dup_pad_sec,
        use_initial_prompt: tuning.streaming?.use_initial_prompt,
        asr_prompt_max_chars: tuning.streaming?.asr_prompt_max_chars,
        // ASR
        beam_size: tuning.asr?.beam_size,
        best_of: tuning.asr?.best_of,
        // VAD
        vad_threshold: tuning.vad?.threshold,
        vad_min_speech_ms: tuning.vad?.min_speech_duration_ms,
        vad_min_silence_ms: tuning.vad?.min_silence_duration_ms,
        vad_speech_pad_ms: tuning.vad?.speech_pad_ms,
        // Overlap detection
        overlap_detection_enabled: tuning.overlap?.enabled,
        overlap_buffer_sec: tuning.overlap?.buffer_sec,
        overlap_detection_interval_sec: tuning.overlap?.detection_interval_sec,
        overlap_min_duration_sec: tuning.overlap?.min_duration_sec,
      };

      const res = await fetch("/tuning", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || "Failed to apply tuning");
      }
      const updated = await res.json();
      setTuning(updated);
    } catch (e) {
      console.error("Failed to apply tuning:", e);
      setTuningError(e.message || "Failed to apply tuning");
    } finally {
      setTuningSaving(false);
    }
  };

  const applyPreset = async (presetName) => {
    setTuningSaving(true);
    setTuningError("");
    try {
      const res = await fetch("/tuning", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset: presetName }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Failed to apply preset '${presetName}'`);
      }
      const updated = await res.json();
      // backend responds {message, settings}; fetch snapshot so UI reflects all values
      const snap = await fetch("/tuning").then((r) => r.json());
      setTuning(snap);
      setSelectedPreset(presetName);
      console.log(updated?.message || `Applied preset ${presetName}`);
    } catch (e) {
      console.error(e);
      setTuningError(e.message || "Failed to apply preset");
    } finally {
      setTuningSaving(false);
    }
  };

  const handlePresetChange = (e) => {
    const presetName = e.target.value;
    setSelectedPreset(presetName);
  };

  const startRecording = async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });

      streamRef.current = stream;
      chunksRef.current = [];
      transcriptAccumulator.clear();
      speakerDetection.reset();

      // Set up Web Audio API for raw audio capture
      // Note: ScriptProcessorNode is deprecated but AudioWorklet requires more setup
      const audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
      audioContextRef.current = audioContext;

      const nativeSampleRate = audioContext.sampleRate;
      console.log(`AudioContext sample rate: ${nativeSampleRate}Hz`);

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      // Resample from native rate to 16kHz
      const resampleTo16k = (inputSamples, fromRate) => {
        if (fromRate === 16000) return inputSamples;
        const ratio = fromRate / 16000;
        const outputLength = Math.floor(inputSamples.length / ratio);
        const output = new Float32Array(outputLength);
        for (let i = 0; i < outputLength; i++) {
          output[i] = inputSamples[Math.floor(i * ratio)];
        }
        return output;
      };

      // Accumulate samples and call /identify_speaker regularly
      let bufferAccumulator = [];

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        bufferAccumulator.push(...inputData);

        // Calculate target size at native sample rate that corresponds to desired 16kHz chunk
        // Larger chunks = more stable speaker detection, less flickering
        const TARGET_16K_SIZE = 8000; // 0.5 seconds at 16kHz
        const TARGET_SIZE = Math.ceil(
          TARGET_16K_SIZE * (nativeSampleRate / 16000)
        );

        while (bufferAccumulator.length >= TARGET_SIZE) {
          const slice = bufferAccumulator.slice(0, TARGET_SIZE);
          bufferAccumulator = bufferAccumulator.slice(TARGET_SIZE);

          // Resample to 16kHz before sending
          const resampled = resampleTo16k(
            new Float32Array(slice),
            nativeSampleRate
          );
          const audioBytes = new Uint8Array(resampled.buffer);

          let binary = "";
          for (let i = 0; i < audioBytes.length; i++) {
            binary += String.fromCharCode(audioBytes[i]);
          }
          const base64Audio = btoa(binary);

          fetch("/identify_speaker", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({
              audio_data: base64Audio,
              sample_rate: "16000",
              session_id: sessionIdRef.current,
            }),
          })
            .then((r) => r.json())
            .then((data) => {
              // Use shared hooks for speaker detection and transcript accumulation
              speakerDetection.processResponse(data);
              transcriptAccumulator.appendFromResponse(data);
            })
            .catch((err) => {
              console.error("Error identifying speaker:", err);
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
          mimeType = "";
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
          const blobType = mimeType || "audio/webm";
          const blob = new Blob(chunksRef.current, { type: blobType });
          const file = new File(
            [blob],
            `recording.${blobType.includes("ogg") ? "ogg" : "webm"}`,
            {
              type: blobType,
            }
          );
          const fakeEvent = { target: { files: [file] } };
          await handleFileUpload(fakeEvent);
        } catch (error) {
          console.error("Error processing recording:", error);
          setErrorMsg("Failed to process recording: " + error.message);
        } finally {
          if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => {
              track.stop();
            });
            streamRef.current = null;
          }
        }
      };

      recorder.start(1000);
      setIsRecording(true);
      handleReset();
      setRecordingTime(0);
      speakerDetection.reset();

      setErrorMsg("");
      sessionIdRef.current = `session_${Date.now()}_${Math.random()
        .toString(36)
        .substring(2, 11)}`;

      // Reset backend session state before starting recording
      fetch("/reset_session", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          session_id: sessionIdRef.current,
          reset_global: "1",
        }),
      }).catch((err) => console.warn("Failed to reset session:", err));

      recordingTimerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      let errorMessage = "Microphone access failed.";
      if (err.name === "NotAllowedError") {
        errorMessage =
          "Microphone access denied. Please allow microphone access and try again.";
      } else if (err.name === "NotFoundError") {
        errorMessage =
          "No microphone found. Please connect a microphone and try again.";
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
        if (processorRef.current) {
          processorRef.current.disconnect();
          processorRef.current = null;
        }

        if (mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop();
        }
        setIsRecording(false);
        speakerDetection.reset();

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
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== "inactive"
      ) {
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
            setSpeakerRefreshTrigger((prev) => prev + 1); // Trigger refresh
          }}
        />

        <SpeakerIdentificationModal
          isOpen={isSpeakerModalOpen}
          onClose={() => {
            setIsSpeakerModalOpen(false);
            setPendingAudioData(null);
            setPendingSpeakerInfo(null);
            setSpeakerRefreshTrigger((prev) => prev + 1);
          }}
          audioData={pendingAudioData}
          speakerInfo={pendingSpeakerInfo}
          knownSpeakers={knownSpeakersList}
          onRefresh={() => {
            fetchKnownSpeakers();
            setSpeakerRefreshTrigger((prev) => prev + 1);
          }}
        />

        <KnownSpeakers refreshTrigger={speakerRefreshTrigger} />

        {/* Tuning panel for live experimentation - driven by presets from tuning_presets.json */}
        {(tuning || availablePresets.length > 0) && (
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <SlidersHorizontal size={16} className="text-slate-400" />
                <h3 className="text-sm font-semibold text-slate-100">
                  Tuning (advanced)
                </h3>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                {/* Preset selector dropdown */}
                <select
                  value={selectedPreset}
                  onChange={handlePresetChange}
                  className="text-xs px-2 py-1 rounded bg-slate-950 border border-slate-700 text-slate-200"
                >
                  {availablePresets.map((preset) => (
                    <option key={preset} value={preset}>
                      {preset.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => applyPreset(selectedPreset)}
                  disabled={tuningSaving}
                  className="text-xs px-3 py-1 rounded-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium"
                  title={`Apply "${selectedPreset}" preset`}
                >
                  {tuningSaving ? "Applying..." : "Apply Preset"}
                </button>
                <button
                  onClick={handleApplyTuning}
                  disabled={tuningSaving}
                  className="text-xs px-3 py-1 rounded-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium"
                  title="Apply custom values (edited below)"
                >
                  Apply Custom
                </button>
              </div>
            </div>

            {tuningError && (
              <div className="text-xs text-red-400 flex items-center gap-1">
                <AlertCircle size={12} /> {tuningError}
              </div>
            )}

            {/* Show preset description if available */}
            {presetDetails[selectedPreset]?._comment && (
              <div className="text-xs text-slate-400 italic">
                {presetDetails[selectedPreset]._comment}
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-300">
              {/* Streaming */}
              <div className="space-y-1">
                <div className="font-semibold text-slate-200">Streaming</div>
                <label className="flex items-center justify-between gap-2">
                  <span>Min new sec</span>
                  <input
                    type="number"
                    step="0.05"
                    className="w-16 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.streaming?.asr_min_new_sec ??
                      presetDetails[selectedPreset]?.asr_min_new_sec ??
                      ""
                    }
                    onChange={handleTuningFieldChange(
                      "streaming",
                      "asr_min_new_sec",
                      (v) => parseFloat(v || "0")
                    )}
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Cross-tail pad (s)</span>
                  <input
                    type="number"
                    step="0.01"
                    className="w-20 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.streaming?.cross_tail_dup_pad_sec ??
                      presetDetails[selectedPreset]?.cross_tail_dup_pad_sec ??
                      ""
                    }
                    onChange={handleTuningFieldChange(
                      "streaming",
                      "cross_tail_dup_pad_sec",
                      (v) => parseFloat(v || "0")
                    )}
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Min ASR interval (s)</span>
                  <input
                    type="number"
                    step="0.05"
                    className="w-20 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.streaming?.min_asr_interval_sec ??
                      presetDetails[selectedPreset]?.min_asr_interval_sec ??
                      ""
                    }
                    onChange={handleTuningFieldChange(
                      "streaming",
                      "min_asr_interval_sec",
                      (v) => parseFloat(v || "0")
                    )}
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Max ASR buffer (s)</span>
                  <input
                    type="number"
                    step="0.5"
                    className="w-20 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.streaming?.max_asr_buffer_sec ??
                      presetDetails[selectedPreset]?.max_asr_buffer_sec ??
                      ""
                    }
                    onChange={handleTuningFieldChange(
                      "streaming",
                      "max_asr_buffer_sec",
                      (v) => parseFloat(v || "0")
                    )}
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Context (s)</span>
                  <input
                    type="number"
                    step="0.25"
                    min="0"
                    className="w-20 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.streaming?.wcpp_context_sec ??
                      presetDetails[selectedPreset]?.wcpp_context_sec ??
                      ""
                    }
                    onChange={handleTuningFieldChange(
                      "streaming",
                      "wcpp_context_sec",
                      (v) => parseFloat(v || "0")
                    )}
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Use initial prompt</span>
                  <input
                    type="checkbox"
                    className="accent-indigo-500"
                    checked={
                      tuning?.streaming?.use_initial_prompt ??
                      presetDetails[selectedPreset]?.use_initial_prompt ??
                      true
                    }
                    onChange={(e) =>
                      setTuning((prev) => ({
                        ...(prev || {}),
                        streaming: {
                          ...(prev?.streaming || {}),
                          use_initial_prompt: e.target.checked,
                        },
                      }))
                    }
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Prompt max chars</span>
                  <input
                    type="number"
                    step="20"
                    min="0"
                    className="w-20 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.streaming?.asr_prompt_max_chars ??
                      presetDetails[selectedPreset]?.asr_prompt_max_chars ??
                      ""
                    }
                    onChange={handleTuningFieldChange(
                      "streaming",
                      "asr_prompt_max_chars",
                      (v) => parseInt(v || "0", 10)
                    )}
                  />
                </label>
              </div>

              {/* ASR */}
              <div className="space-y-1">
                <div className="font-semibold text-slate-200">ASR</div>
                <label className="flex items-center justify-between gap-2">
                  <span>Beam size</span>
                  <input
                    type="number"
                    className="w-16 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.asr?.beam_size ??
                      presetDetails[selectedPreset]?.beam_size ??
                      ""
                    }
                    onChange={handleTuningFieldChange("asr", "beam_size", (v) =>
                      parseInt(v || "0", 10)
                    )}
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Best of</span>
                  <input
                    type="number"
                    className="w-16 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.asr?.best_of ??
                      presetDetails[selectedPreset]?.best_of ??
                      ""
                    }
                    onChange={handleTuningFieldChange("asr", "best_of", (v) =>
                      parseInt(v || "0", 10)
                    )}
                  />
                </label>
              </div>

              {/* VAD */}
              <div className="space-y-1">
                <div className="font-semibold text-slate-200">VAD</div>
                <label className="flex items-center justify-between gap-2">
                  <span>Threshold</span>
                  <input
                    type="number"
                    step="0.05"
                    className="w-20 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.vad?.threshold ??
                      presetDetails[selectedPreset]?.vad_threshold ??
                      ""
                    }
                    onChange={handleTuningFieldChange("vad", "threshold", (v) =>
                      parseFloat(v || "0")
                    )}
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Min speech (ms)</span>
                  <input
                    type="number"
                    className="w-20 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.vad?.min_speech_duration_ms ??
                      presetDetails[selectedPreset]?.vad_min_speech_ms ??
                      ""
                    }
                    onChange={handleTuningFieldChange(
                      "vad",
                      "min_speech_duration_ms",
                      (v) => parseInt(v || "0", 10)
                    )}
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Min silence (ms)</span>
                  <input
                    type="number"
                    className="w-20 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.vad?.min_silence_duration_ms ??
                      presetDetails[selectedPreset]?.vad_min_silence_ms ??
                      ""
                    }
                    onChange={handleTuningFieldChange(
                      "vad",
                      "min_silence_duration_ms",
                      (v) => parseInt(v || "0", 10)
                    )}
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Speech pad (ms)</span>
                  <input
                    type="number"
                    className="w-20 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.vad?.speech_pad_ms ??
                      presetDetails[selectedPreset]?.vad_speech_pad_ms ??
                      ""
                    }
                    onChange={handleTuningFieldChange(
                      "vad",
                      "speech_pad_ms",
                      (v) => parseInt(v || "0", 10)
                    )}
                  />
                </label>
              </div>
            </div>

            {/* Overlap Detection Settings */}
            <div className="border-t border-slate-800 pt-3 mt-3">
              <div className="font-semibold text-slate-200 text-xs mb-2">
                Overlap Detection
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs text-slate-300">
                <label className="flex items-center justify-between gap-2">
                  <span>Enabled</span>
                  <input
                    type="checkbox"
                    className="accent-indigo-500"
                    checked={
                      tuning?.overlap?.enabled ??
                      presetDetails[selectedPreset]
                        ?.overlap_detection_enabled ??
                      true
                    }
                    onChange={(e) =>
                      setTuning((prev) => ({
                        ...(prev || {}),
                        overlap: {
                          ...(prev?.overlap || {}),
                          enabled: e.target.checked,
                        },
                      }))
                    }
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Buffer (s)</span>
                  <input
                    type="number"
                    step="0.1"
                    className="w-16 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.overlap?.buffer_sec ??
                      presetDetails[selectedPreset]?.overlap_buffer_sec ??
                      ""
                    }
                    onChange={(e) =>
                      setTuning((prev) => ({
                        ...(prev || {}),
                        overlap: {
                          ...(prev?.overlap || {}),
                          buffer_sec: parseFloat(e.target.value || "0"),
                        },
                      }))
                    }
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Interval (s)</span>
                  <input
                    type="number"
                    step="0.5"
                    className="w-16 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.overlap?.detection_interval_sec ??
                      presetDetails[selectedPreset]
                        ?.overlap_detection_interval_sec ??
                      ""
                    }
                    onChange={(e) =>
                      setTuning((prev) => ({
                        ...(prev || {}),
                        overlap: {
                          ...(prev?.overlap || {}),
                          detection_interval_sec: parseFloat(
                            e.target.value || "0"
                          ),
                        },
                      }))
                    }
                  />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Min dur (s)</span>
                  <input
                    type="number"
                    step="0.05"
                    className="w-16 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-right"
                    value={
                      tuning?.overlap?.min_duration_sec ??
                      presetDetails[selectedPreset]?.overlap_min_duration_sec ??
                      ""
                    }
                    onChange={(e) =>
                      setTuning((prev) => ({
                        ...(prev || {}),
                        overlap: {
                          ...(prev?.overlap || {}),
                          min_duration_sec: parseFloat(e.target.value || "0"),
                        },
                      }))
                    }
                  />
                </label>
              </div>
            </div>
          </div>
        )}

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
            isDemoMode={isDemoMode}
          />
        </div>

        {/* Current speaker indicator during recording */}
        <LiveRecordingDisplay
          isRecording={isRecording}
          recordingTime={recordingTime}
          displayedSpeaker={speakerDetection.displayedSpeaker}
          displayedOverlap={speakerDetection.displayedOverlap}
          displayedSpeakers={speakerDetection.displayedSpeakers}
          hasSpeech={speakerDetection.hasSpeech}
          messages={transcriptAccumulator.messages}
        />

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

        {/* Demo Mode: Stream uploaded audio as live recording */}
        <DemoMode
          disabled={isRecording || testRunning}
          sessionIdRef={sessionIdRef}
          onError={setErrorMsg}
          onDemoStateChange={setIsDemoMode}
          maxWordsPerBubble={MAX_WORDS_PER_BUBBLE}
        />

        {/* Debug tools */}
        <div className="mt-4 bg-slate-900/40 border border-slate-800/60 rounded-lg p-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div>
              <div className="text-slate-200 font-semibold">
                Debug: testsound runner
              </div>
              <div className="text-xs text-slate-400">
                Streams `thetestsound.wav` through `/identify_speaker` and lets
                you download the raw JSON.
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <span>Chunk (s)</span>
                <input
                  type="number"
                  step="0.1"
                  min="0.2"
                  className="w-20 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-right"
                  value={testChunkSec}
                  onChange={(e) =>
                    setTestChunkSec(
                      parseFloat((e.target.value || "1").replace(",", ".")) ||
                        1.0
                    )
                  }
                  disabled={testRunning}
                />
              </label>
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <span>Sleep (ms)</span>
                <input
                  type="number"
                  step="10"
                  min="0"
                  className="w-24 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-right"
                  value={testSleepMs}
                  onChange={(e) =>
                    setTestSleepMs(parseInt(e.target.value || "0", 10) || 0)
                  }
                  disabled={testRunning}
                />
              </label>
              <button
                onClick={runTestSound}
                disabled={testRunning || isRecording || isDemoMode}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-sm font-semibold"
              >
                {testRunning
                  ? "Running…"
                  : isDemoMode
                  ? "Demo running…"
                  : "Run testsound"}
              </button>
              <button
                onClick={handleDownloadTestJson}
                disabled={!testRunJson}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed text-slate-200 rounded-lg text-sm font-semibold border border-slate-700"
              >
                Download test JSON
              </button>
              <button
                onClick={handleCopyTestJson}
                disabled={!testRunJson}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed text-slate-200 rounded-lg text-sm font-semibold border border-slate-700"
                title="Copy full testsound JSON to clipboard"
              >
                {testJsonCopied ? "Copied" : "Copy JSON"}
              </button>
              <button
                onClick={() => setTestJsonExpanded((v) => !v)}
                disabled={!testRunJson}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed text-slate-200 rounded-lg text-sm font-semibold border border-slate-700"
              >
                {testJsonExpanded ? "Hide JSON" : "Show JSON"}
              </button>
            </div>
          </div>

          {testRunJson && testJsonExpanded && (
            <div className="mt-3">
              <div className="text-xs text-slate-400 mb-2">
                Raw JSON (responses + tuning snapshot):
              </div>
              <pre className="text-xs bg-slate-950/70 border border-slate-800 rounded-lg p-3 max-h-80 overflow-auto text-slate-100">
                {JSON.stringify(testRunJson, null, 2)}
              </pre>
            </div>
          )}
        </div>

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
