import { useEffect, useRef, useState } from "react";

/**
 * Decouples overlap detection from UI display duration.
 *
 * - When `overlapDetected` is true, we display immediately.
 * - When `overlapDetected` becomes false, we keep showing for `holdMs`
 *   before hiding the indicator.
 */
export function useOverlapDisplay({ overlapDetected, overlapSpeakers, holdMs = 2000 }) {
  const [displayedOverlap, setDisplayedOverlap] = useState(false);
  const [displayedSpeakers, setDisplayedSpeakers] = useState([]);
  const hadOverlapRef = useRef(false);
  const timeoutRef = useRef(null);

  useEffect(() => {
    if (overlapDetected) {
      // Overlap detected - clear any pending hold timer and show immediately
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      hadOverlapRef.current = true;
      setDisplayedOverlap(true);
      setDisplayedSpeakers(overlapSpeakers || []);
      return;
    }

    // No overlap detected
    if (!hadOverlapRef.current) {
      // Never had overlap in this cycle, ensure hidden
      setDisplayedOverlap(false);
      setDisplayedSpeakers([]);
      return;
    }

    // Had overlap, now ended - start hold timer if not already running
    if (!timeoutRef.current) {
      timeoutRef.current = setTimeout(() => {
        setDisplayedOverlap(false);
        setDisplayedSpeakers([]);
        hadOverlapRef.current = false; // Reset for next cycle
        timeoutRef.current = null;
      }, holdMs);
    }
    // Don't clear timer on re-renders - we want it to persist until it fires or overlap restarts
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
