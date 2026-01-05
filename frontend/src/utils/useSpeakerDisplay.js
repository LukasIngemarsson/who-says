import { useEffect, useRef, useState } from "react";

/**
 * Decouples speaker detection from UI display duration.
 *
 * - When `hasSpeech` is true, we display `speaker` immediately (no delay).
 * - When `hasSpeech` becomes false, we keep showing the last speaker for `holdMs`
 *   before showing "no speech detected". The hold only applies to the transition
 *   from speech to silence, NOT to speaker switches.
 */
export function useSpeakerDisplay({ speaker, hasSpeech, holdMs = 1500 }) {
  const [displayedSpeaker, setDisplayedSpeaker] = useState(null);
  const lastSpeakerRef = useRef(null);
  const timeoutRef = useRef(null);

  useEffect(() => {
    // Always clear any pending timeout when inputs change
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    if (hasSpeech) {
      // Speech detected - show current speaker IMMEDIATELY (no hold delay)
      // Note: speaker can be null during warmup, preserve that to avoid showing "Unknown"
      if (speaker !== null) {
        lastSpeakerRef.current = speaker;
      }
      setDisplayedSpeaker(speaker);
      return;
    }

    // No speech detected - apply hold before showing "no speech"
    // Only apply hold if we had a previous speaker to hold
    if (lastSpeakerRef.current === null) {
      setDisplayedSpeaker(null);
      return;
    }

    // Keep showing the last speaker, then clear after holdMs
    timeoutRef.current = setTimeout(() => {
      setDisplayedSpeaker(null);
      timeoutRef.current = null;
    }, holdMs);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [speaker, hasSpeech, holdMs]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, []);

  return displayedSpeaker;
}
