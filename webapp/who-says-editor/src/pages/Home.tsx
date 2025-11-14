import React, { useRef, useState, useMemo } from "react";
import { SegmentTable } from "../components/SegmentTable";
import { SegmentEditor } from "../components/SegmentEditor";
import { AudioPlayer } from "../components/AudioPlayer";
import { downloadJson } from "../utils/download";
import { useUndoRedo } from "../hooks/useUndoRedo";
import type { WhisperDoc, Segment } from "../types/whisperx";
import { fromTxtToWhisperDoc } from "../utils/convert";

export function Home() {
  function setAudioTime(seconds: number) {
    if (audioRef.current) {
      audioRef.current.currentTime = seconds;
    }
  }
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [projectName, setProjectName] = useState("who-says-edit");
  const { state: data, set: setData, undo, redo } = useUndoRedo<WhisperDoc>({ segments: [] });
  const [selected, setSelected] = useState(0);
  const segs = data.segments;
  const [speakers, setSpeakers] = useState<string[]>(["SPEAKER_00", "SPEAKER_01"]);

  function setSegments(next: Segment[]) { setData({ segments: next }); }

  const validation = useMemo(() => {
    const issues: Array<{ i: number; msg: string }> = [];
    for (let i = 0; i < segs.length; i++) {
      const a = segs[i];
      if (a.end < a.start) issues.push({ i, msg: "end < start" });
      if (i > 0 && a.start < segs[i - 1].end) issues.push({ i, msg: `overlaps with #${i - 1}` });
    }
    const dur = audioRef.current?.duration;
    if (dur && Number.isFinite(dur)) {
      segs.forEach((s, i) => { if (s.end > dur + 0.01) issues.push({ i, msg: "end beyond audio" }); });
    }
    return issues;
  }, [segs]);

  async function loadJsonFile(file: File) {
    const text = await file.text();
    const raw = JSON.parse(text);
    if (!raw?.segments || !Array.isArray(raw.segments)) throw new Error("Invalid JSON: missing segments");
    const normalized: Segment[] = raw.segments.map((s: any, idx: number) => ({
      id: s.id ?? idx,
      start: Number(s.start ?? 0),
      end: Number(s.end ?? 0),
      text: String(s.text ?? ""),
      speaker: s.speaker ?? null,
      words: Array.isArray(s.words) ? s.words.map((w: any) => ({ start: +w.start || 0, end: +w.end || 0, word: String(w.word || "") })) : []
    }));
    setData({ segments: normalized });
    setSelected(0);
    const uniq = Array.from(new Set(normalized.map(s => s.speaker).filter(Boolean))) as string[];
    if (uniq.length) setSpeakers(prev => Array.from(new Set([...uniq, ...prev])));
  }

  function handleFileList(files: FileList | null) {
    if (!files?.length) return;
    const list = Array.from(files);
    const json = list.find(f => f.name.toLowerCase().endsWith(".json"));
    const txt = list.find(f => f.name.toLowerCase().endsWith(".txt"));
    const audio = list.find(f => /(mp3|wav|m4a|ogg|flac)$/i.test(f.name));

    if (json) loadJsonFile(json).catch(e => alert(e.message));

    if (txt) {
      txt.text().then(raw => {
        try {
          const doc = fromTxtToWhisperDoc(raw, txt.name);
          setData(doc);
          setSelected(0);
          const uniq = Array.from(new Set(doc.segments.map(s => s.speaker).filter(Boolean))) as string[];
          if (uniq.length) setSpeakers(prev => Array.from(new Set([...uniq, ...prev])));
        } catch (err: any) {
          alert("Failed to parse .txt: " + err?.message);
        }
      });
    }

    if (audio && audioRef.current) {
      const url = URL.createObjectURL(audio);
      audioRef.current.src = url;
      audioRef.current.load();
    }
  }

  function exportJson() {
    downloadJson(`${projectName || "who-says-edit"}.clean.json`, {
      meta: { exported_at: new Date().toISOString(), app: "who-says-editor", version: 1, segments_count: segs.length },
      segments: segs
    });
  }

  function updateSeg(i: number, patch: Partial<Segment>) {
    const next = segs.map((s, idx) => (idx === i ? { ...s, ...patch } : s));
    setSegments(next);
  }

  function insertAfter(i: number) {
    const here = segs[i];
    const newSeg: Segment = { id: i + 1, start: here?.end ?? 0, end: (here?.end ?? 0) + 1, text: "", speaker: here?.speaker ?? null, words: [] };
    const next = [...segs.slice(0, i + 1), newSeg, ...segs.slice(i + 1)].map((s, idx) => ({ ...s, id: idx }));
    setSegments(next);
    setSelected(i + 1);
  }

  function removeSeg(i: number) {
    const next = segs.filter((_, idx) => idx !== i).map((s, idx) => ({ ...s, id: idx }));
    setSegments(next);
    setSelected((i) => Math.max(0, Math.min(i, next.length - 1)));
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border bg-white p-4">
          <h2 className="font-medium mb-2">Load files</h2>
          <div className="flex gap-2">
            <input type="file" accept=".json,.txt" onChange={(e) => handleFileList(e.target.files)} />
            <input type="file" accept="audio/*" onChange={(e) => handleFileList(e.target.files)} />
          </div>
          <AudioPlayer ref={audioRef} className="mt-3" />
        </div>
        <div className="rounded-xl border bg-white p-4">
          <h2 className="font-medium mb-2">Project</h2>
          <label className="text-xs text-slate-500">Project name</label>
          <input className="mt-1 w-full rounded border px-2 py-1" value={projectName} onChange={(e) => setProjectName(e.target.value)} />
          <div className="mt-3 flex gap-2">
            <button className="rounded bg-slate-900 px-3 py-1.5 text-white" onClick={exportJson}>Export JSON</button>
            <button className="rounded border px-3 py-1.5" onClick={undo}>Undo</button>
            <button className="rounded border px-3 py-1.5" onClick={redo}>Redo</button>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-5">
        <div className="md:col-span-2 rounded-xl border bg-white overflow-hidden">
          <SegmentTable
            segments={segs}
            selected={selected}
            onSelect={setSelected}
            onChange={(i, patch) => updateSeg(i, patch)}
            onInsertAfter={insertAfter}
            onRemove={removeSeg}
            onJumpToTime={setAudioTime}
          />
        </div>
        <div className="md:col-span-3 rounded-xl border bg-white p-4">
          <SegmentEditor
            segment={segs[selected]}
            speakers={speakers}
            onChange={(patch) => updateSeg(selected, patch)}
          />
          <div className="mt-3">
            {validation.length ? (
              <ul className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-md p-3 list-disc list-inside">
                {validation.map((v, idx) => <li key={idx}>Segment #{v.i}: {v.msg}</li>)}
              </ul>
            ) : (
              <p className="text-sm text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-md p-3">No validation issues</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
