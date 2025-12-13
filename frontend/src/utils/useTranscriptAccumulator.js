import { useState, useCallback, useRef } from "react";

/**
 * Shared hook for accumulating transcript messages from ASR responses.
 * Used by live recording, demo mode, and test sound runner.
 *
 * @param {Object} options
 * @param {number} options.maxWordsPerBubble - Max words before starting a new bubble (default: 18)
 */
export const useTranscriptAccumulator = ({ maxWordsPerBubble = 18 } = {}) => {
  const [messages, setMessages] = useState([]);
  const lastSnippetRef = useRef("");
  const lastKnownSpeakerRef = useRef(null);
  const lastOverlapRef = useRef({ speakers: null, timestamp: 0 });

  /**
   * Backfill speaker names on messages that don't have one yet.
   * Called when we finally identify a speaker.
   */
  const backfillSpeaker = useCallback((speaker) => {
    if (!speaker || speaker === "Unknown") return;

    setMessages((prev) => {
      // Check if any messages need backfilling
      const needsBackfill = prev.some((msg) => !msg.speaker || msg.speaker === "");
      if (!needsBackfill) return prev;

      return prev.map((msg) => {
        if (!msg.speaker || msg.speaker === "") {
          return { ...msg, speaker };
        }
        return msg;
      });
    });
  }, []);

  /**
   * Process an ASR response and append transcript segments to messages.
   * Handles word chunking, speaker grouping, backfill, and merging.
   *
   * @param {Object} data - Response from /identify_speaker endpoint
   * @param {string} data.transcript - Plain text transcript
   * @param {Array} data.transcript_segments - Array of {speaker, text} segments
   * @param {string} data.speaker - Detected speaker name
   * @param {string} data.transcript_speaker - Speaker for transcript (fallback)
   * @param {Object} options - Optional settings
   * @param {string} options.overlapSpeakers - Override speakers during overlap (e.g., "Johan, Lukas")
   */
  const appendFromResponse = useCallback(
    (data, options = {}) => {
      const { overlapSpeakers: externalOverlapSpeakers } = options;

      // Check for separated transcriptions from SOS (Speaker Overlap Separation)
      // These are cleaner transcriptions from separated audio streams
      let separatedSegments = [];
      if (data.overlap_segments && data.overlap_segments.length > 0) {
        for (const ovlp of data.overlap_segments) {
          // Skip short overlap segments (likely early/partial detections)
          // Only process overlaps >= 0.8 seconds for cleaner transcriptions
          const duration = ovlp.duration || 0;
          if (duration < 0.8) {
            console.log("[SOS] Skipping short overlap segment:", duration, "s");
            continue;
          }
          if (ovlp.separated_transcriptions && ovlp.separated_transcriptions.length > 0) {
            const speakers = ovlp.speakers || [];
            for (const sepTrans of ovlp.separated_transcriptions) {
              const text = (sepTrans.transcription || "").trim();
              if (text) {
                // Skip incomplete transcriptions (ending with "...")
                if (text.endsWith("...")) {
                  console.log("[SOS] Skipping incomplete transcription:", text);
                  continue;
                }
                // Use speaker name from overlap speakers list if available
                const speakerName = speakers[sepTrans.speaker_idx] || speakers[0] || "Speaker";
                separatedSegments.push({
                  speaker: speakerName,
                  text: text,
                  isSeparated: true,
                });
              }
            }
          }
        }
      }

      // Build segments from both separated transcriptions AND regular transcript_segments
      // Separated transcriptions are for overlap portions, regular are for non-overlap
      let segments = [];

      // First add separated transcriptions (overlapped speech)
      if (separatedSegments.length > 0) {
        console.log("[SOS] Adding separated transcriptions:", separatedSegments);
        segments.push(...separatedSegments);
      }

      // Then add regular transcript_segments, but filter out overlap-attributed segments
      // (those with comma-separated speakers like "Lukas, Johan")
      const regularSegments =
        Array.isArray(data.transcript_segments) &&
        data.transcript_segments.length > 0
          ? data.transcript_segments
          : typeof data.transcript === "string" && data.transcript.trim()
            ? [{ speaker: data.speaker, text: data.transcript.trim() }]
            : [];

      for (const seg of regularSegments) {
        const speaker = seg.speaker || "";
        // Skip segments attributed to multiple speakers (overlap)
        // These are already handled by separated transcriptions
        if (speaker.includes(",")) {
          console.log("[SOS] Skipping overlap-attributed segment:", speaker, seg.text);
          continue;
        }
        segments.push(seg);
      }

      if (segments.length === 0) return;

      // Check if we have a valid speaker (not null, not empty, not "Unknown")
      const responseSpeaker = data.speaker;
      const isValidSpeaker = responseSpeaker && responseSpeaker !== "Unknown";
      const hasNewSpeaker = isValidSpeaker && responseSpeaker !== lastKnownSpeakerRef.current;

      if (hasNewSpeaker) {
        lastKnownSpeakerRef.current = responseSpeaker;
      }

      // During overlap, show all overlapping speakers joined together (filter out "Unknown")
      const dataOverlapSpeakers = (data.overlap_speakers || []).filter(s => s && s !== "Unknown");
      const isCurrentOverlap = data.overlap_detected && dataOverlapSpeakers.length > 1;
      const overlapSpeakerStr = isCurrentOverlap ? dataOverlapSpeakers.join(", ") : null;

      // Store overlap state when detected, use it for ~3 seconds after
      const now = Date.now();
      const isNewOverlap = isCurrentOverlap && lastOverlapRef.current.speakers !== overlapSpeakerStr;
      if (isCurrentOverlap) {
        lastOverlapRef.current = { speakers: overlapSpeakerStr, timestamp: now };
      }

      // Use external overlap speakers (from UI state), or recent overlap, or current overlap
      const recentOverlap = (now - lastOverlapRef.current.timestamp < 3000) ? lastOverlapRef.current.speakers : null;
      const overlapSpeaker = externalOverlapSpeakers || overlapSpeakerStr || recentOverlap;

      // NOTE: Backend now correctly attributes words to overlap speakers
      // based on timestamps in _assign_words_to_speakers(), so we don't
      // need to override with overlapSpeaker here.

      // Determine the effective speaker - use response speaker or last known (don't override with overlap)
      const effectiveSpeaker = isValidSpeaker ? responseSpeaker : (lastKnownSpeakerRef.current || "");

      // Skip adding transcripts until we have an identified speaker
      if (!effectiveSpeaker) {
        return;
      }

      // Do everything in ONE setMessages call: backfill + merge + append
      setMessages((prev) => {
        let updated = [...prev];

        // Step 1: Backfill any messages with empty speaker
        if (effectiveSpeaker) {
          updated = updated.map((msg) => {
            if (!msg.speaker || msg.speaker === "") {
              return { ...msg, speaker: effectiveSpeaker };
            }
            return msg;
          });
        }

        // Step 2: Process each segment and merge with existing bubbles
        for (const seg of segments) {
          const snippet = (seg.text || "").trim();
          if (!snippet) continue;

          // Use effectiveSpeaker if segment speaker is empty or "Unknown"
          const rawSegSpeaker = seg.speaker || data.transcript_speaker || data.speaker;
          const segSpeaker = (rawSegSpeaker && rawSegSpeaker !== "Unknown") ? rawSegSpeaker : effectiveSpeaker;
          const snippetWords = snippet.split(/\s+/).filter(Boolean);
          if (snippetWords.length === 0) continue;

          const lastMsg = updated[updated.length - 1];

          // Merge with last bubble if same speaker (after backfill)
          // BUT: Never merge separated (overlap) segments with regular segments
          // This keeps overlapped speech visually distinct
          const lastMsgSpeaker = lastMsg?.speaker || "";
          const isSeparatedSegment = seg.isSeparated || false;
          const lastMsgIsSeparated = lastMsg?.isSeparated || false;
          const speakersMatch = lastMsgSpeaker === segSpeaker || (lastMsgSpeaker === effectiveSpeaker && segSpeaker === effectiveSpeaker);
          const canMerge = lastMsg && speakersMatch && (isSeparatedSegment === lastMsgIsSeparated);

          if (canMerge) {
            const lastWords = (lastMsg.text || "").split(/\s+/).filter(Boolean);
            const remaining = maxWordsPerBubble - lastWords.length;

            if (remaining > 0) {
              const toAppend = snippetWords.slice(0, remaining);
              const rest = snippetWords.slice(remaining);

              const newText = lastWords.length
                ? `${lastMsg.text} ${toAppend.join(" ")}`
                : toAppend.join(" ");

              updated[updated.length - 1] = {
                ...lastMsg,
                text: newText,
                speaker: segSpeaker || effectiveSpeaker,
                // Preserve isSeparated flag if segment is separated
                isSeparated: lastMsg.isSeparated || seg.isSeparated || false,
              };

              // Leftover words become new bubbles
              let remainingWords = rest;
              while (remainingWords.length > 0) {
                const chunk = remainingWords.slice(0, maxWordsPerBubble);
                remainingWords = remainingWords.slice(maxWordsPerBubble);
                updated.push({
                  id: `${Date.now()}-${updated.length}-${Math.random()}`,
                  text: chunk.join(" "),
                  speaker: segSpeaker || effectiveSpeaker,
                  isSeparated: seg.isSeparated || false,
                });
              }
              continue;
            }
          }

          // Start new bubbles for this snippet
          let remainingWords = snippetWords;
          while (remainingWords.length > 0) {
            const chunk = remainingWords.slice(0, maxWordsPerBubble);
            remainingWords = remainingWords.slice(maxWordsPerBubble);
            updated.push({
              id: `${Date.now()}-${updated.length}-${Math.random()}`,
              text: chunk.join(" "),
              speaker: segSpeaker || effectiveSpeaker,
              isSeparated: seg.isSeparated || false,
            });
          }
        }

        return updated;
      });
    },
    [maxWordsPerBubble]
  );

  /**
   * Simplified append for test sound runner (doesn't need word chunking logic).
   * Just appends each segment as a new message.
   */
  const appendSimple = useCallback((speaker, text) => {
    if (!text?.trim()) return;
    // Track speaker for backfill
    if (speaker && speaker !== "Unknown") {
      lastKnownSpeakerRef.current = speaker;
    }
    setMessages((prev) => [
      ...prev,
      {
        id: `${Date.now()}_${Math.random()}`,
        speaker: speaker || lastKnownSpeakerRef.current || "",
        text: text.trim(),
      },
    ]);
  }, []);

  /**
   * Clear all messages and reset state.
   */
  const clear = useCallback(() => {
    setMessages([]);
    lastSnippetRef.current = "";
    lastKnownSpeakerRef.current = null;
    lastOverlapRef.current = { speakers: null, timestamp: 0 };
  }, []);

  return {
    messages,
    appendFromResponse,
    appendSimple,
    clear,
    backfillSpeaker,
    setMessages, // Expose for edge cases
  };
};

export default useTranscriptAccumulator;
