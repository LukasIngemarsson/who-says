import { useCallback, useRef, useState } from "react";

export function useUndoRedo<T>(initial: T) {
  const [present, setPresent] = useState<T>(initial);
  const pastRef = useRef<T[]>([]);
  const futureRef = useRef<T[]>([]);

  const set = useCallback((next: T) => {
    pastRef.current.push(present);
    futureRef.current = [];
    setPresent(next);
  }, [present]);

  const undo = useCallback(() => {
    const past = pastRef.current;
    if (!past.length) return;
    const prev = past.pop()!;
    futureRef.current.push(present);
    setPresent(prev);
  }, [present]);

  const redo = useCallback(() => {
    const future = futureRef.current;
    if (!future.length) return;
    const nxt = future.pop()!;
    pastRef.current.push(present);
    setPresent(nxt);
  }, [present]);

  return { state: present, set, undo, redo } as const;
}