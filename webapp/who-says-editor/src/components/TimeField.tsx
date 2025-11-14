// 🆕 ADDED FILE — TimeField component
// This is a new reusable input for editing times more easily.
// It supports typing times like "62.345" or "1:02.345",
// and lets you nudge values with ↑/↓ keys (±0.1s, Shift=±1s, Alt=±0.01s).

import React, { useEffect, useRef, useState } from "react";
import { parseTime, formatTime } from "../utils/time";

type Props = {
  value: number;
  onChange: (nextSeconds: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
  "aria-label"?: string;
};

// Small helper to clamp value between min and max
function clamp(n: number, min?: number, max?: number) {
  if (typeof min === "number" && n < min) return min;
  if (typeof max === "number" && n > max) return max;
  return n;
}

/**
 * TimeField — simple, user-friendly time input
 *  - Parses "m:ss.mmm" or seconds
 *  - ArrowUp/Down to adjust
 *  - Reverts to last valid value if input is invalid
 */
export function TimeField({
  value,
  onChange,
  min,
  max,
  step = 0.1,
  className,
  ...rest
}: Props) {
  const [text, setText] = useState<string>(formatTime(value));
  const lastGood = useRef<number>(value);
  const focused = useRef<boolean>(false);

  // When value changes externally (e.g. undo/redo), update the text
  useEffect(() => {
    if (!focused.current) setText(formatTime(value));
    lastGood.current = value;
  }, [value]);

  // Validate and commit typed text to seconds
  function commit(nextText: string) {
    const parsed = parseTime(nextText);
    if (Number.isFinite(parsed)) {
      const clamped = clamp(parsed, min, max);
      onChange(clamped);
      setText(formatTime(clamped));
      lastGood.current = clamped;
    } else {
      // revert invalid input
      setText(formatTime(lastGood.current));
    }
  }

  // Handle keyboard shortcuts
  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      const base = parseTime(text);
      const sign = e.key === "ArrowUp" ? 1 : -1;
      const fine = e.altKey ? 0.01 : e.shiftKey ? 1 : step;
      const next = clamp(
        (Number.isFinite(base) ? base : lastGood.current) + sign * fine,
        min,
        max
      );
      onChange(next);
      setText(formatTime(next));
      lastGood.current = next;
    }
    if (e.key === "Enter") {
      commit(text);
      (e.target as HTMLInputElement).blur();
    }
    if (e.key === "Escape") {
      setText(formatTime(lastGood.current));
      (e.target as HTMLInputElement).blur();
    }
  }

  return (
    <input
      {...rest}
      className={className ?? "w-full rounded border px-2 py-1"}
      value={text}
      onFocus={() => (focused.current = true)}
      onBlur={() => {
        focused.current = false;
        commit(text);
      }}
      onChange={(e) => setText(e.target.value)}
      onKeyDown={handleKeyDown}
      inputMode="decimal"
      placeholder="m:ss.mmm or seconds"
      spellCheck={false}
    />
  );
}
