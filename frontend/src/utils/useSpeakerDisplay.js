import { useEffect, useRef, useState } from "react";

/**
 * Decouples speaker detection from UI display duration.
 *
 * - When `hasSpeech` is true, we display `speaker` immediately and remember the timestamp.
 * - When `hasSpeech` becomes false, we keep showing the last speaker for `holdMs`.
 *   Repeated "no speech" updates do NOT extend the hold.
 */
export function useSpeakerDisplay({ speaker, hasSpeech, holdMs = 1500 }) {
  const [displayedSpeaker, setDisplayedSpeaker] = useState(null);
  const lastSpeechAtRef = useRef(0);
  const timeoutRef = useRef(null);

  useEffect(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    if (hasSpeech) {
      lastSpeechAtRef.current = Date.now();
      setDisplayedSpeaker(speaker ?? "Unknown");
      return;
    }

    const last = lastSpeechAtRef.current;
    if (!last) {
      setDisplayedSpeaker(null);
      return;
    }

    const remaining = Math.max(0, (holdMs ?? 0) - (Date.now() - last));
    if (remaining === 0) {
      setDisplayedSpeaker(null);
      return;
    }

    timeoutRef.current = setTimeout(() => {
      setDisplayedSpeaker(null);
      timeoutRef.current = null;
    }, remaining);

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
