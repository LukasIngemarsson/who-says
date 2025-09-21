import React, { useEffect, useRef } from "react";
import type { Segment } from "../types/whisperx";
import { parseTime, formatTime } from "../utils/time";

type Props = {
  segment?: Segment;
  speakers: string[];
  onChange: (patch: Partial<Segment>) => void;
  onJumpTo: (t: number) => void;
};

export function SegmentEditor({ segment, speakers, onChange, onJumpTo }: Props) {
  const textRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    textRef.current?.focus();
  }, [segment?.id]);

  if (!segment) return <p className="text-sm text-slate-500">Select a segment to edit.</p>;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <label className="block">
          <div className="text-xs text-slate-500">Start</div>
          <input
            className="mt-1 w-full rounded border px-2 py-1"
            value={formatTime(segment.start)}
            onChange={(e) => onChange({ start: parseTime(e.target.value) })}
          />
        </label>
        <label className="block">
          <div className="text-xs text-slate-500">End</div>
          <input
            className="mt-1 w-full rounded border px-2 py-1"
            value={formatTime(segment.end)}
            onChange={(e) => onChange({ end: parseTime(e.target.value) })}
          />
        </label>
        <label className="block">
          <div className="text-xs text-slate-500">Speaker</div>
          <input
            className="mt-1 w-full rounded border px-2 py-1"
            value={segment.speaker ?? ""}
            onChange={(e) => onChange({ speaker: e.target.value || null })}
            placeholder="—"
          />
        </label>
        <div className="flex items-end gap-2">
          <button
            className="rounded border px-3 py-1.5"
            onClick={() => onJumpTo(Math.max(0, segment.start - 0.05))}
          >
            Jump to start
          </button>
          <button className="rounded border px-3 py-1.5" onClick={() => onJumpTo(segment.end)}>
            Jump to end
          </button>
        </div>
      </div>
      <label className="block">
        <div className="text-xs text-slate-500">Text</div>
        <textarea
          ref={textRef}
          className="mt-1 w-full rounded border px-2 py-1 font-mono"
          rows={8}
          value={segment.text}
          onChange={(e) => onChange({ text: e.target.value })}
        />
      </label>
      {segment.words?.length ? (
        <div>
          <div className="text-xs text-slate-500 mb-1">Words (read-only in MVP)</div>
          <div className="max-h-32 overflow-auto text-xs border rounded p-2 bg-slate-50">
            {segment.words.map((w, i) => (
              <span key={i} className="px-1 py-0.5 inline-block m-0.5 rounded bg-white border">
                {w.word}
                <span className="text-slate-400">@{formatTime(w.start)}</span>
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}