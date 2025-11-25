import React, { useEffect, useRef } from "react";
import type { Segment } from "../types/whisperx";
import { TimeField } from "./TimeField";

type Props = {
  segment?: Segment;
  speakers: string[];
  onChange: (patch: Partial<Segment>) => void;
};

export function SegmentEditor({ segment, speakers, onChange }: Props) {
  const textRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    textRef.current?.focus();
  }, [segment?.id]);

  if (!segment)
    return <p className="text-sm text-slate-500">Select a segment to edit.</p>;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <label className="block">
          <div className="text-xs text-slate-500">Start</div>
          <TimeField
            aria-label="Start time"
            value={segment.start}
            onChange={(v) => onChange({ start: v })}
            min={0}
          />
        </label>

        <label className="block">
          <div className="text-xs text-slate-500">End</div>
          <TimeField
            aria-label="End time"
            value={segment.end}
            onChange={(v) => onChange({ end: v })}
            min={0}
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
    </div>
  );
}
