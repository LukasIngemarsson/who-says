import { useEffect, useRef, useState } from "react";

/**
 * Decouples overlap detection from UI display duration.
 *
 * - When `overlapDetected` is true, we display immediately and remember the timestamp.
 * - When `overlapDetected` becomes false, we keep showing for `holdMs`.
 *   Repeated "no overlap" updates do NOT extend the hold.
 */
export function useOverlapDisplay({ overlapDetected, overlapSpeakers, holdMs = 1500 }) {
  const [displayedOverlap, setDisplayedOverlap] = useState(false);
  const [displayedSpeakers, setDisplayedSpeakers] = useState([]);
  const lastOverlapAtRef = useRef(0);
  const timeoutRef = useRef(null);

  useEffect(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    if (overlapDetected) {
      lastOverlapAtRef.current = Date.now();
      setDisplayedOverlap(true);
      setDisplayedSpeakers(overlapSpeakers || []);
      return;
    }

    const last = lastOverlapAtRef.current;
    if (!last) {
      setDisplayedOverlap(false);
      setDisplayedSpeakers([]);
      return;
    }

    const remaining = Math.max(0, (holdMs ?? 0) - (Date.now() - last));
    if (remaining === 0) {
      setDisplayedOverlap(false);
      setDisplayedSpeakers([]);
      return;
    }

    timeoutRef.current = setTimeout(() => {
      setDisplayedOverlap(false);
      setDisplayedSpeakers([]);
      timeoutRef.current = null;
    }, remaining);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [overlapDetected, overlapSpeakers, holdMs]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, []);

  return { displayedOverlap, displayedSpeakers };
}
