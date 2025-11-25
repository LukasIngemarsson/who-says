import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

import torch


@dataclass
class Segment:
    """
    Simple dataclass for documentation type only.
    The pipeline actually passes plain dicts with at least:
    - start: float
    - end: float
    - text: str
    and possibly other keys (speaker, id, words, ...).
    """
    start: float
    end: float
    text: str


class SpeechBrainPhoneme:
    """
    SpeechBrainPhoneme

    - Wraps SpeechBrain's SoundChoice G2P model
      (speechbrain/soundchoice-g2p) for grapheme-to-phoneme conversion.
    - Adds a 'phonemes' field to each segment.
    - Keeps all other segment fields intact (id, speaker, words, ...).
    - Can be used both inside the pipeline and as a CLI script.
    """

    def __init__(
        self,
        model: str = "speechbrain/soundchoice-g2p",
        savedir: str = "pretrained_models/soundchoice-g2p",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize the SpeechBrain G2P backend.

        Args:
            model: Hugging Face model ID for the G2P model.
            savedir: Local directory where the model will be stored.
            device: "cpu" or "cuda" (if available).
        """
        self.model_id = model
        self.savedir = savedir
        self.device = device

        # Lazy import SpeechBrain and load G2P model
        from speechbrain.inference.text import GraphemeToPhoneme
        from speechbrain.utils.fetching import LocalStrategy

        run_opts = {"device": device} if device is not None else None

        self._g2p = GraphemeToPhoneme.from_hparams(
            source=model,
            savedir=savedir,
            run_opts=run_opts,
            # COPY instead of SYMLINK -> avoids Windows privilege issues
            local_strategy=LocalStrategy.COPY,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _g2p_single(self, text: str) -> str:
        """
        Convert a single text string to a space-separated ARPABET string.
        """
        tokens = self._g2p(text)  # list of tokens
        return " ".join(tokens)

    def _g2p_batch(self, texts: List[str]) -> List[str]:
        """
        Convert a batch of text strings to space-separated ARPABET strings.
        """
        if not texts:
            return []
        tokens_batch = self._g2p(texts)  # list of list-of-tokens
        # SpeechBrain returns list[list[str]]; join each to a string
        return [" ".join(tokens) for tokens in tokens_batch]

    # ------------------------------------------------------------------
    # Public API for pipeline use
    # ------------------------------------------------------------------

    def add_phonemes(
        self,
        segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Add a 'phonemes' field to each segment.

        Args:
            segments: List of dicts with at least:
                      - 'start': float
                      - 'end': float
                      - 'text': str
                      and possibly additional metadata.

        Returns:
            List of dicts, same structure + 'phonemes': str (ARPABET).
        """
        if not segments:
            return []

        # Extract texts (empty string if missing)
        texts = [seg.get("text", "") or "" for seg in segments]

        # Batch G2P
        phonemes_list = self._g2p_batch(texts)

        # Stitch back into segment dicts, preserving everything else
        out: List[Dict[str, Any]] = []
        for seg, ph in zip(segments, phonemes_list):
            seg_out = dict(seg)  # shallow copy to avoid mutating input
            # normalize start/end to float
            seg_out["start"] = float(seg_out.get("start", 0.0))
            seg_out["end"] = float(seg_out.get("end", 0.0))
            seg_out["phonemes"] = ph
            out.append(seg_out)

        return out

    def __call__(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make the class callable, mirroring other pipeline components
        (e.g. SileroVAD, SCD, etc.).

        Usage:
            phoneme = SpeechBrainPhoneme(...)
            segments_out = phoneme(segments_in)
        """
        return self.add_phonemes(segments)


# ----------------------------------------------------------------------
# CLI entry point (replaces your old run_phonemes.py)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add phonemes to segments using SpeechBrain SoundChoice G2P."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input JSON file (either a list of segments, or an object with a 'segments' list).",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output JSON file (segments with 'phonemes').",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device for G2P model ('cpu' or 'cuda'). Default: auto-detect.",
    )
    parser.add_argument(
        "--model",
        default="speechbrain/soundchoice-g2p",
        help="HuggingFace model ID for G2P (default: speechbrain/soundchoice-g2p).",
    )
    parser.add_argument(
        "--savedir",
        default="pretrained_models/soundchoice-g2p",
        help="Directory to cache the G2P model.",
    )

    args = parser.parse_args()

    # Load input JSON
    with open(args.in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept either:
    #  - a top-level list of segments
    #  - or an object with a 'segments' list (like who-says-edit.clean.json)
    if isinstance(data, dict) and "segments" in data and isinstance(data["segments"], list):
        segments: List[Dict[str, Any]] = data["segments"]
    elif isinstance(data, list):
        segments = data
    else:
        raise SystemExit(
            "Input must be either:\n"
            "  - a list of segments, OR\n"
            "  - an object with a 'segments' array."
        )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    phoneme = SpeechBrainPhoneme(
        model=args.model,
        savedir=args.savedir,
        device=device,
    )

    out_segments = phoneme(segments)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out_segments, f, indent=2, ensure_ascii=False)

    print(f"Wrote phonemes to: {args.out_path}")
