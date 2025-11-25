import React, { useRef, useState, useMemo } from "react";
import { SegmentTable } from "../components/SegmentTable";
import { SegmentEditor } from "../components/SegmentEditor";
import { AudioPlayer } from "../components/AudioPlayer";
import { downloadJson } from "../utils/download";
import { useUndoRedo } from "../hooks/useUndoRedo";
import type { WhisperDoc, Segment } from "../types/whisperx";
import { fromTxtToWhisperDoc } from "../utils/convert";

// helper: normalize any WhisperX-style object with segments[] to our WhisperDoc
function toWhisperDoc(raw: any): WhisperDoc {
  if (!raw?.segments || !Array.isArray(raw.segments)) {
    throw new Error("Invalid JSON: missing segments");
  }
  const segments: Segment[] = raw.segments.map((s: any, idx: number) => ({
    id: s.id ?? idx,
    start: Number(s.start ?? 0),
    end: Number(s.end ?? 0),
    text: String(s.text ?? ""),
    speaker: s.speaker ?? null,
  }));
  return { segments };
}

export function Home() {
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const [projectName, setProjectName] = useState("who-says-edit");
  const { state: data, set: setData, undo, redo } =
    useUndoRedo<WhisperDoc>({ segments: [] });
  const [selected, setSelected] = useState(0);
  const segs = data.segments;
  const [speakers, setSpeakers] = useState<string[]>([
    "SPEAKER_00",
  ]);

  // track chosen files & annotation state
  const [transcriptFileName, setTranscriptFileName] = useState("No file chosen");
  const [audioFileName, setAudioFileName] = useState("No file chosen");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [annotating, setAnnotating] = useState(false);

  function setSegments(next: Segment[]) {
    setData({ segments: next });
  }

  const validation = useMemo(() => {
    const issues: Array<{ i: number; msg: string }> = [];
    for (let i = 0; i < segs.length; i++) {
      const a = segs[i];
      if (a.end < a.start) issues.push({ i, msg: "end < start" });
      if (i > 0 && a.start < segs[i - 1].end)
        issues.push({ i, msg: `overlaps with #${i - 1}` });
    }
    const dur = audioRef.current?.duration;
    if (dur && Number.isFinite(dur)) {
      segs.forEach((s, i) => {
        if (s.end > dur + 0.01) issues.push({ i, msg: "end beyond audio" });
      });
    }
    return issues;
  }, [segs]);

  async function loadJsonFile(file: File) {
    const text = await file.text();
    const raw = JSON.parse(text);
    const doc = toWhisperDoc(raw);
    setData(doc);
    setSelected(0);
    const uniq = Array.from(
      new Set(doc.segments.map((s) => s.speaker).filter(Boolean))
    ) as string[];
    if (uniq.length)
      setSpeakers((prev) => Array.from(new Set([...uniq, ...prev])));
  }

  // separate handlers so we know which file was chosen
  function handleTranscriptFile(files: FileList | null) {
    if (!files?.length) return;
    const file = files[0];
    setTranscriptFileName(file.name);

    if (file.name.toLowerCase().endsWith(".json")) {
      loadJsonFile(file).catch((e) => alert(e.message));
    } else if (file.name.toLowerCase().endsWith(".txt")) {
      file.text().then((raw) => {
        try {
          const doc = fromTxtToWhisperDoc(raw, file.name);
          setData(doc);
          setSelected(0);
          const uniq = Array.from(
            new Set(doc.segments.map((s) => s.speaker).filter(Boolean))
          ) as string[];
          if (uniq.length)
            setSpeakers((prev) => Array.from(new Set([...uniq, ...prev])));
        } catch (err: any) {
          alert("Failed to parse .txt: " + err?.message);
        }
      });
    } else {
      alert("Unsupported transcript file type. Use .json or .txt.");
    }
  }

  function handleAudioFile(files: FileList | null) {
    if (!files?.length) return;
    const file = files[0];
    setAudioFile(file);
    setAudioFileName(file.name);

    if (audioRef.current) {
      const url = URL.createObjectURL(file);
      audioRef.current.src = url;
      audioRef.current.load();
    }
  }

  // call backend /annotate with current audio file
  async function annotateAudio() {
    if (!audioFile) {
      alert("Please upload an audio file first.");
      return;
    }
    setAnnotating(true);
    try {
      const form = new FormData();
      form.append("audio", audioFile);

      const resp = await fetch("http://localhost:8000/annotate", {
        method: "POST",
        body: form,
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Backend error (${resp.status}): ${text}`);
      }

      const result = await resp.json();
      const doc = toWhisperDoc(result); // normalize backend output
      setData(doc);
      setSelected(0);
      const uniq = Array.from(
        new Set(doc.segments.map((s: Segment) => s.speaker).filter(Boolean))
      ) as string[];
      if (uniq.length)
        setSpeakers((prev) => Array.from(new Set([...uniq, ...prev])));
    } catch (err: any) {
      alert("Annotation failed: " + (err?.message ?? String(err)));
    } finally {
      setAnnotating(false);
    }
  }

  function exportJson() {
    downloadJson(`${projectName || "edited-transcript"}.json`, {
      meta: {
        exported_at: new Date().toISOString(),
        app: "who-says-editor",
        version: 1,
        segments_count: segs.length,
      },
      segments: segs,
    });
  }

  function updateSeg(i: number, patch: Partial<Segment>) {
    const next = segs.map((s, idx) => (idx === i ? { ...s, ...patch } : s));
    setSegments(next);
  }

  function insertAfter(i: number) {
    const here = segs[i];
    const newSeg: Segment = {
      id: i + 1,
      start: here?.end ?? 0,
      end: (here?.end ?? 0) + 1,
      text: "",
      speaker: here?.speaker ?? null
    };
    const next = [...segs.slice(0, i + 1), newSeg, ...segs.slice(i + 1)].map(
      (s, idx) => ({ ...s, id: idx })
    );
    setSegments(next);
    setSelected(i + 1);
  }

  function removeSeg(i: number) {
    const next = segs
      .filter((_, idx) => idx !== i)
      .map((s, idx) => ({ ...s, id: idx }));
    setSegments(next);
    setSelected((i) => Math.max(0, Math.min(i, next.length - 1)));
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border bg-white p-4">
          <h2 className="font-medium mb-2">Load files</h2>

          {/* nice-looking upload controls with file names + annotation button */}
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:gap-6">
            {/* Transcript upload */}
            <div className="flex flex-col">
              <span className="text-sm text-slate-600 mb-1">Transcript</span>
              <label className="inline-flex items-center">
                <span className="rounded bg-blue-100 border px-3 py-1.5 text-sm text-slate-700 cursor-pointer hover:bg-blue-200">
                  Select file
                </span>
                <input
                  type="file"
                  accept=".json,.txt"
                  onChange={(e) => handleTranscriptFile(e.target.files)}
                  className="hidden"
                />
              </label>
              <span className="mt-1 text-xs text-slate-500 truncate max-w-xs">
                {transcriptFileName}
              </span>
            </div>

            {/* Audio upload + annotate */}
            <div className="flex flex-col">
              <span className="text-sm text-slate-600 mb-1">Audio</span>
              <div className="flex items-center gap-2">
                <label className="inline-flex items-center">
                  <span className="rounded bg-blue-100 border px-3 py-1.5 text-sm text-slate-700 cursor-pointer hover:bg-blue-200">
                    Select file
                  </span>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(e) => handleAudioFile(e.target.files)}
                    className="hidden"
                  />
                </label>
                <button
                  className="rounded bg-slate-900 px-3 py-1.5 text-sm text-white disabled:bg-slate-400"
                  onClick={annotateAudio}
                  disabled={!audioFile || annotating}
                >
                  {annotating ? "Annotating..." : "Create annotation"}
                </button>
              </div>
              <span className="mt-1 text-xs text-slate-500 truncate max-w-xs">
                {audioFileName}
              </span>
            </div>
          </div>

          <AudioPlayer ref={audioRef} className="mt-3" />
        </div>

        <div className="rounded-xl border bg-white p-4">
          <h2 className="font-medium mb-2">Project</h2>
          <label className="text-xs text-slate-500">Project name</label>
          <input
            className="mt-1 w-full rounded border px-2 py-1"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
          />
          <div className="mt-3 flex gap-2">
            <button
              className="rounded bg-slate-900 px-3 py-1.5 text-white"
              onClick={exportJson}
            >
              Export JSON
            </button>
            <button className="rounded border px-3 py-1.5" onClick={undo}>
              Undo
            </button>
            <button className="rounded border px-3 py-1.5" onClick={redo}>
              Redo
            </button>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="md:col-span-1 rounded-xl border bg-white overflow-hidden">
          <SegmentTable
            segments={segs}
            selected={selected}
            onSelect={setSelected}
            onChange={(i, patch) => updateSeg(i, patch)}
            onInsertAfter={insertAfter}
            onRemove={removeSeg}
            audioRef={audioRef}
          />
        </div>
        <div className="md:col-span-1 rounded-xl border bg-white p-4">
          <SegmentEditor
            segment={segs[selected]}
            speakers={speakers}
            onChange={(patch) => updateSeg(selected, patch)}
          />
          <div className="mt-3">
            {validation.length ? (
              <ul className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-md p-3 list-disc list-inside">
                {validation.map((v, idx) => (
                  <li key={idx}>
                    Segment #{v.i}: {v.msg}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-md p-3">
                No validation issues
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
