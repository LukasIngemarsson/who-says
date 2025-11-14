export type Word = { start: number; end: number; word: string };
export type Segment = { id: number; start: number; end: number; text: string; speaker: string | null; words: Word[] };
export type WhisperDoc = { segments: Segment[] };