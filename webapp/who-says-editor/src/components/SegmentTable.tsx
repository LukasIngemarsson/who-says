import React, { useState, useRef, useEffect } from "react";
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
  audioRef?: React.RefObject<HTMLAudioElement | null>;
};

export function SegmentTable({
  segments,
  selected,
  onSelect,
  onChange,
  onInsertAfter,
  onRemove,
  audioRef,
}: Props) {
  const [playingSegment, setPlayingSegment] = useState<number | null>(null);
  const [pausedPositions, setPausedPositions] = useState<Map<number, number>>(
    new Map()
  );
  const timeUpdateHandlerRef = useRef<(() => void) | null>(null);
  const currentSegmentIndexRef = useRef<number | null>(null);

  // Cleanup function to remove event listener
  const cleanupListener = () => {
    if (timeUpdateHandlerRef.current && audioRef?.current) {
      audioRef.current.removeEventListener(
        "timeupdate",
        timeUpdateHandlerRef.current
      );
      timeUpdateHandlerRef.current = null;
    }
  };

  // Clean up when component unmounts
  useEffect(() => {
    return () => {
      cleanupListener();
    };
  }, []);

  function playSegment(
    segmentIndex: number,
    start: number,
    end: number,
    e: React.MouseEvent
  ) {
    e.stopPropagation();
    if (!audioRef?.current) return;

    const audio = audioRef.current;

    // Clean up previous listener if any
    cleanupListener();

    // Check if there's a paused position for this segment
    const pausedPosition = pausedPositions.get(segmentIndex);
    const startFrom = pausedPosition !== undefined ? pausedPosition : start;

    // Start playing the segment
    audio.currentTime = startFrom;
    audio.play();
    setPlayingSegment(segmentIndex);
    currentSegmentIndexRef.current = segmentIndex;

    // Stop playback when we reach the end of the segment
    const handleTimeUpdate = () => {
      if (audio.currentTime >= end) {
        audio.pause();
        cleanupListener();
        setPlayingSegment(null);
        currentSegmentIndexRef.current = null;
        // Clear the paused position when segment completes
        setPausedPositions((prev) => {
          const next = new Map(prev);
          next.delete(segmentIndex);
          return next;
        });
      }
    };

    timeUpdateHandlerRef.current = handleTimeUpdate;
    audio.addEventListener("timeupdate", handleTimeUpdate);
  }

  function pauseSegment(e: React.MouseEvent) {
    e.stopPropagation();
    if (!audioRef?.current || currentSegmentIndexRef.current === null) return;

    const audio = audioRef.current;
    const currentSegment = currentSegmentIndexRef.current;

    // Save the current position
    setPausedPositions((prev) => {
      const next = new Map(prev);
      next.set(currentSegment, audio.currentTime);
      return next;
    });

    audio.pause();
    cleanupListener();
    setPlayingSegment(null);
    currentSegmentIndexRef.current = null;
  }

  function resetSegment(segmentIndex: number, e: React.MouseEvent) {
    e.stopPropagation();

    // Clear the paused position for this segment
    setPausedPositions((prev) => {
      const next = new Map(prev);
      next.delete(segmentIndex);
      return next;
    });
  }
  return (
    <div className="overflow-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-slate-50 border-b">
          <tr>
            <th className="text-left p-2 w-12">#</th>
            <th className="text-left p-2 w-36">Controls</th>
            <th className="text-left p-2 w-28">Start</th>
            <th className="text-left p-2 w-28">End</th>
            <th className="text-left p-2 w-32">Speaker</th>
            <th className="text-right p-2 w-40">Actions</th>
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
                if (audioRef?.current) {
                  audioRef.current.currentTime = s.start;
                }
              }}
            >
              <td className="p-2 text-slate-500">{i}</td>
              <td className="p-2">
                <div className="flex gap-1">
                  <button
                    className={clsx(
                      "rounded border px-2 py-1 hover:bg-slate-100",
                      playingSegment === i && "bg-green-100 border-green-300"
                    )}
                    onClick={(e) => playSegment(i, s.start, s.end, e)}
                    title={
                      pausedPositions.has(i)
                        ? "Resume from paused position"
                        : "Play this segment"
                    }
                  >
                    ▶
                  </button>
                  <button
                    className="rounded border px-2 py-1 hover:bg-slate-100"
                    onClick={pauseSegment}
                    disabled={playingSegment !== i}
                    title="Pause playback"
                  >
                    ⏸
                  </button>
                  {pausedPositions.has(i) && (
                    <button
                      className="rounded border px-1 py-1 hover:bg-slate-100 text-xs"
                      onClick={(e) => resetSegment(i, e)}
                      title="Reset to segment start"
                    >
                      ↺
                    </button>
                  )}
                </div>
              </td>
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
