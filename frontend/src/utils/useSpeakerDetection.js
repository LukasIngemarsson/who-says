import { useState, useRef, useCallback, useEffect } from "react";
import { useSpeakerDisplay } from "./useSpeakerDisplay";
import { useOverlapDisplay } from "./useOverlapDisplay";

/**
 * Manages speaker detection state from ASR responses.
 * Combines raw detection state management with display hold logic.
 *
 * @param {Object} options
 * @param {number} options.speakerHoldMs - How long to show speaker after speech stops (default: 1500)
 * @param {number} options.overlapHoldMs - How long to show overlap indicator after overlap ends (default: 2000)
 * @param {number} options.clearDelayMs - Delay before clearing speaker when no speech (default: 1000)
 */
export function useSpeakerDetection({
  speakerHoldMs = 1500,
  overlapHoldMs = 2000,
  clearDelayMs = 1000,
} = {}) {
  // Raw detection state
  const [currentSpeaker, setCurrentSpeaker] = useState(null);
  const [detectedSpeaker, setDetectedSpeaker] = useState(null);
  const [hasSpeech, setHasSpeech] = useState(false);
  const [overlapDetected, setOverlapDetected] = useState(false);
  const [overlapSpeakers, setOverlapSpeakers] = useState([]);

  // Refs for timeout management
  const speakerClearTimeoutRef = useRef(null);
  const lastSpeechTimeRef = useRef(0);

  // Display state (with hold/TTL)
  const displayedSpeaker = useSpeakerDisplay({
    speaker: detectedSpeaker,
    hasSpeech,
    holdMs: speakerHoldMs,
  });

  const { displayedOverlap, displayedSpeakers } = useOverlapDisplay({
    overlapDetected,
    overlapSpeakers,
    holdMs: overlapHoldMs,
  });

  /**
   * Process an ASR response and update speaker detection state.
   * Handles timeout logic for speaker display persistence.
   *
   * @param {Object} data - Response from /identify_speaker endpoint
   */
  const processResponse = useCallback(
    (data) => {
      if (data.has_speech) {
        // Clear any pending timeout since we have active speech
        if (speakerClearTimeoutRef.current) {
          clearTimeout(speakerClearTimeoutRef.current);
          speakerClearTimeoutRef.current = null;
        }
        lastSpeechTimeRef.current = Date.now();
        // Preserve null during warmup (backend returns null before speaker is identified)
        setCurrentSpeaker(data.speaker);
      } else {
        // Delay clearing speaker name to keep it visible longer
        if (!speakerClearTimeoutRef.current) {
          speakerClearTimeoutRef.current = setTimeout(() => {
            // Double-check that enough time has passed since last speech
            // This handles race conditions with out-of-order responses
            const elapsed = Date.now() - lastSpeechTimeRef.current;
            if (elapsed >= clearDelayMs) {
              setCurrentSpeaker(null);
            }
            speakerClearTimeoutRef.current = null;
          }, clearDelayMs);
        }
      }

      // Handle overlap detection
      if (data.overlap_detected) {
        setOverlapDetected(true);
        setOverlapSpeakers(data.overlap_speakers || []);
      } else {
        setOverlapDetected(false);
        setOverlapSpeakers([]);
      }

      setDetectedSpeaker(data.speaker ?? null);
      setHasSpeech(Boolean(data.has_speech));
    },
    [clearDelayMs]
  );

  /**
   * Reset all speaker detection state.
   */
  const reset = useCallback(() => {
    if (speakerClearTimeoutRef.current) {
      clearTimeout(speakerClearTimeoutRef.current);
      speakerClearTimeoutRef.current = null;
    }
    setCurrentSpeaker(null);
    setDetectedSpeaker(null);
    setHasSpeech(false);
    setOverlapDetected(false);
    setOverlapSpeakers([]);
    lastSpeechTimeRef.current = 0;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (speakerClearTimeoutRef.current) {
        clearTimeout(speakerClearTimeoutRef.current);
      }
    };
  }, []);

  return {
    // Raw state
    currentSpeaker,
    detectedSpeaker,
    hasSpeech,
    overlapDetected,
    overlapSpeakers,

    // Display state (with hold)
    displayedSpeaker,
    displayedOverlap,
    displayedSpeakers,

    // Actions
    processResponse,
    reset,

    // Direct setters for edge cases
    setCurrentSpeaker,
    setDetectedSpeaker,
    setHasSpeech,
    setOverlapDetected,
    setOverlapSpeakers,
  };
}

export default useSpeakerDetection;
