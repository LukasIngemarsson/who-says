import React from "react";
import type { Segment } from "../types/whisperx";
import clsx from "clsx";
import { TimeField } from "./TimeField";

type Props = {
  segments: Segment[];
  selected: number;
  onSelect: (i: number) => void;
  onChange: (i: number, patch: Partial<Segment>) => void;
  onInsertAfter: (i: number) => void;
  onRemove: (i: number) => void;
  onJumpToTime?: (seconds: number) => void;
};

export function SegmentTable({
  segments,
  selected,
  onSelect,
  onChange,
  onInsertAfter,
  onRemove,
  onJumpToTime,
}: Props) {
  return (
    <div className="overflow-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-slate-50 border-b">
          <tr>
            <th className="text-left p-2 w-10">#</th>
            <th className="text-left p-2">Start</th>
            <th className="text-left p-2">End</th>
            <th className="text-left p-2">Speaker</th>
            <th className="text-right p-2 w-28">Actions</th>
          </tr>
        </thead>
        <tbody>
          {segments.map((s, i) => (
            <tr
              key={s.id}
              className={clsx(
                "border-b cursor-pointer hover:bg-slate-50",
                i === selected && "bg-slate-100"
              )}
              onClick={() => {
                onSelect(i);
                if (typeof onJumpToTime === "function") onJumpToTime(s.start);
              }}
            >
              <td className="p-2 text-slate-500">{i}</td>
              <td className="p-2">
                <TimeField
                  aria-label={`Start ${i}`}
                  value={s.start}
                  onChange={(v) => onChange(i, { start: v })}
                  min={0}
                />
              </td>
              <td className="p-2">
                <TimeField
                  aria-label={`End ${i}`}
                  value={s.end}
                  onChange={(v) => onChange(i, { end: v })}
                  min={0}
                />
              </td>
              <td className="p-2">
                <input
                  className="w-full rounded border px-2 py-1"
                  placeholder="—"
                  value={s.speaker ?? ""}
                  onChange={(e) =>
                    onChange(i, { speaker: e.target.value || null })
                  }
                />
              </td>
              <td className="p-2 text-right">
                <div className="flex justify-end gap-2">
                  <button
                    className="rounded border px-2 py-1"
                    onClick={(e) => {
                      e.stopPropagation();
                      onInsertAfter(i);
                    }}
                  >
                    Insert
                  </button>
                  <button
                    className="rounded border px-2 py-1"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemove(i);
                    }}
                  >
                    Delete
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
