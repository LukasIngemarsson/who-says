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

  /**
   * Process an ASR response and append transcript segments to messages.
   * Handles word chunking, speaker grouping, and deduplication.
   *
   * @param {Object} data - Response from /identify_speaker endpoint
   * @param {string} data.transcript - Plain text transcript
   * @param {Array} data.transcript_segments - Array of {speaker, text} segments
   * @param {string} data.speaker - Detected speaker name
   * @param {string} data.transcript_speaker - Speaker for transcript (fallback)
   */
  const appendFromResponse = useCallback(
    (data) => {
      // Extract segments from response - prefer transcript_segments if available
      const segments =
        Array.isArray(data.transcript_segments) &&
        data.transcript_segments.length > 0
          ? data.transcript_segments
          : typeof data.transcript === "string" && data.transcript.trim()
            ? [{ speaker: data.speaker, text: data.transcript.trim() }]
            : [];

      for (const seg of segments) {
        const snippet = (seg.text || "").trim();
        if (!snippet) continue;

        // During warmup, speaker may be null - don't show "Unknown" in that case
        const segSpeaker =
          seg.speaker || data.transcript_speaker || data.speaker || "";

        setMessages((prev) => {
          const updated = [...prev];
          const lastMsg = updated[updated.length - 1];

          const snippetWords = snippet.split(/\s+/).filter(Boolean);
          if (snippetWords.length === 0) {
            return updated;
          }

          // If we have a last bubble for the same speaker, try to
          // append words to it up to maxWordsPerBubble.
          if (lastMsg && lastMsg.speaker === segSpeaker) {
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
              };

              // If there are leftover words beyond the limit,
              // start new bubbles for them.
              let remainingWords = rest;
              while (remainingWords.length > 0) {
                const chunk = remainingWords.slice(0, maxWordsPerBubble);
                remainingWords = remainingWords.slice(maxWordsPerBubble);
                updated.push({
                  id: `${Date.now()}-${updated.length}`,
                  text: chunk.join(" "),
                  speaker: segSpeaker,
                });
              }

              return updated;
            }
          }

          // Otherwise, start one or more new bubbles for this snippet
          let remainingWords = snippetWords;
          while (remainingWords.length > 0) {
            const chunk = remainingWords.slice(0, maxWordsPerBubble);
            remainingWords = remainingWords.slice(maxWordsPerBubble);
            updated.push({
              id: `${Date.now()}-${updated.length}`,
              text: chunk.join(" "),
              speaker: segSpeaker,
            });
          }

          return updated;
        });
      }
    },
    [maxWordsPerBubble]
  );

  /**
   * Simplified append for test sound runner (doesn't need word chunking logic).
   * Just appends each segment as a new message.
   */
  const appendSimple = useCallback((speaker, text) => {
    if (!text?.trim()) return;
    setMessages((prev) => [
      ...prev,
      {
        id: `${Date.now()}_${Math.random()}`,
        speaker: speaker || "",
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
  }, []);

  return {
    messages,
    appendFromResponse,
    appendSimple,
    clear,
    setMessages, // Expose for edge cases
  };
};

export default useTranscriptAccumulator;
