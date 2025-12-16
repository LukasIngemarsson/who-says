import JSON5 from "json5";
import type { WhisperDoc, Segment } from "../types/whisperx";

/**
 * Convert a WhisperX-like TXT dump into a WhisperDoc.
 * Handles:
 *  - np.float64(...) → numbers
 *  - None/True/False → null/true/false
 *  - Single quotes and trailing commas via JSON5
 */
export function fromTxtToWhisperDoc(
  raw: string,
  sourceName = "input.txt"
): WhisperDoc {
  let s = raw.trim();
  s = s.replace(/np\.float64\(([^)]+)\)/g, "$1");
  s = s.replace(/\bNone\b/g, "null");
  s = s.replace(/\bTrue\b/g, "true");
  s = s.replace(/\bFalse\b/g, "false");

  const parsed = JSON5.parse(s);
  const segmentsIn = Array.isArray(parsed) ? parsed : parsed?.segments;
  if (!Array.isArray(segmentsIn)) {
    throw new Error(
      "Expected a list of segments or an object with segments[]"
    );
  }

  const segments: Segment[] = segmentsIn.map((s: any, idx: number) => {
    return {
      id: Number.isFinite(+s?.id) ? +s.id : idx,
      start: Number(s?.start ?? 0),
      end: Number(s?.end ?? 0),
      text: String(s?.text ?? ""),
      speaker: s?.speaker ?? null,
    } as Segment;
  });

  return { segments };
}
