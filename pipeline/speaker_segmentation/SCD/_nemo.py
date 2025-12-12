"""
Speaker change point detection using NVIDIA NeMo Sortformer model.
"""
import os
import torch
import numpy as np
from typing import Optional, List


class NemoSCD:
    """
    Speaker change point detection using NVIDIA NeMo Sortformer.

    Uses the diar_sortformer_4spk-v1 model which outputs speaker diarization
    segments. Change points are extracted from segment boundaries.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        min_duration: float = 0.0,
    ):
        """
        Parameters
        ----------
        model_path : str, optional
            Path to local .nemo file OR HuggingFace model ID.
            If None, defaults to "nvidia/diar_sortformer_4spk-v1".
        device : torch.device
            Device to run inference on
        min_duration : float
            Minimum duration between change points in seconds
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_duration = min_duration

        # Import NeMo here to avoid import errors if not installed
        from nemo.collections.asr.models import SortformerEncLabelModel

        # Load model - local file or HuggingFace
        if model_path is None:
            model_path = "nvidia/diar_sortformer_4spk-v1"

        if model_path.endswith(".nemo") and os.path.isfile(model_path):
            # Load from local .nemo file
            self.model = SortformerEncLabelModel.restore_from(model_path, map_location=self.device)
        else:
            # Load from HuggingFace (downloads and caches locally)
            self.model = SortformerEncLabelModel.from_pretrained(model_path, map_location=self.device)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _parse_segment(self, segment_str: str) -> tuple:
        """Parse a segment string 'start end speaker' into (start, end, speaker)."""
        parts = segment_str.strip().split()
        start = float(parts[0])
        end = float(parts[1])
        speaker = parts[2]
        return (start, end, speaker)

    def _call_diarize(self, audio_path: str):
        """
        Call the diarize method with API compatibility handling.
        NeMo API varies across versions.
        """
        import json

        errors = []

        # Try all known API patterns - NeMo 2.x uses audio= parameter
        patterns = [
            ("audio=path", lambda: self.model.diarize(audio=audio_path, batch_size=1)),
            ("audio=[path]", lambda: self.model.diarize(audio=[audio_path], batch_size=1)),
            ("positional path", lambda: self.model.diarize(audio_path, batch_size=1)),
            ("positional [path]", lambda: self.model.diarize([audio_path], batch_size=1)),
            ("paths2audio_files", lambda: self.model.diarize(paths2audio_files=[audio_path], batch_size=1)),
        ]

        for name, attempt in patterns:
            try:
                print(f"[NemoSCD] Trying pattern: {name}")
                result = attempt()
                print(f"[NemoSCD] Success with pattern: {name}")
                return result
            except TypeError as e:
                errors.append(f"{name}: TypeError - {e}")
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__} - {e}")

        # Try manifest-based approach as last resort
        try:
            print("[NemoSCD] Trying manifest file approach")
            manifest_path = audio_path.replace('.wav', '_manifest.json')
            with open(manifest_path, 'w') as f:
                # NeMo manifest format - one JSON object per line
                f.write(json.dumps({"audio_filepath": audio_path, "duration": 1000.0}) + "\n")
            try:
                result = self.model.diarize(manifest_filepath=manifest_path, batch_size=1)
                print("[NemoSCD] Success with manifest approach")
                return result
            finally:
                if os.path.exists(manifest_path):
                    os.unlink(manifest_path)
        except Exception as e:
            errors.append(f"manifest: {type(e).__name__} - {e}")

        # If none work, raise a clear error
        raise RuntimeError(
            f"Could not find compatible diarize() API.\n"
            f"Tried patterns:\n" + "\n".join(f"  - {err}" for err in errors)
        )

    def _extract_change_points_from_segments(
        self,
        segments: List,
        audio_duration: float
    ) -> List[float]:
        """
        Extract speaker change points from diarization segments.

        Parameters
        ----------
        segments : List
            List of segment strings "start end speaker" from Sortformer
        audio_duration : float
            Total duration of audio in seconds

        Returns
        -------
        change_points : List[float]
            Timestamps where speaker changes occur
        """
        if not segments:
            return []

        # Parse segments - they come as strings "start end speaker"
        parsed = [self._parse_segment(s) for s in segments]

        # Sort segments by start time
        parsed = sorted(parsed, key=lambda x: x[0])

        change_points = []
        prev_speaker = None

        for start, end, speaker in parsed:
            if prev_speaker is not None and speaker != prev_speaker:
                change_points.append(start)
            prev_speaker = speaker

        # Filter by min_duration
        if self.min_duration > 0 and len(change_points) > 1:
            filtered = [change_points[0]]
            for cp in change_points[1:]:
                if cp - filtered[-1] >= self.min_duration:
                    filtered.append(cp)
            change_points = filtered

        return change_points

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000
    ) -> List[float]:
        """
        Detect speaker change points.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform, shape (num_samples,) or (1, num_samples)
        sample_rate : int
            Sample rate of the audio

        Returns
        -------
        change_points : List[float]
            List of timestamps (in seconds) where speaker changes occur
        """
        import tempfile
        import soundfile as sf

        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        # Ensure 1D
        if waveform_np.ndim == 2:
            waveform_np = waveform_np.squeeze(0)
        if waveform_np.ndim == 2 and waveform_np.shape[0] > 1:
            waveform_np = waveform_np.mean(axis=0)

        audio_duration = len(waveform_np) / sample_rate

        # Sortformer requires audio file path, so write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, waveform_np, sample_rate)

        try:
            # Run diarization - NeMo API varies by version
            with torch.no_grad():
                predicted_segments = self._call_diarize(temp_path)

            # predicted_segments is a list (one per audio file) of lists of tuples
            # Each tuple is (start, end, speaker_id)
            if predicted_segments and len(predicted_segments) > 0:
                segments = predicted_segments[0]  # First (only) audio file
            else:
                segments = []

            # Extract change points from segments
            change_points = self._extract_change_points_from_segments(
                segments, audio_duration
            )

            return change_points

        finally:
            # Clean up temp file
            os.unlink(temp_path)

    def get_diarization_segments(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000
    ) -> List[dict]:
        """
        Get full diarization segments (not just change points).

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform, shape (num_samples,) or (1, num_samples)
        sample_rate : int
            Sample rate of the audio

        Returns
        -------
        segments : List[dict]
            List of segments with 'start', 'end', 'speaker' keys
        """
        import tempfile
        import soundfile as sf

        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        # Ensure 1D
        if waveform_np.ndim == 2:
            waveform_np = waveform_np.squeeze(0)
        if waveform_np.ndim == 2 and waveform_np.shape[0] > 1:
            waveform_np = waveform_np.mean(axis=0)

        # Sortformer requires audio file path, so write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, waveform_np, sample_rate)

        try:
            # Run diarization - NeMo API varies by version
            with torch.no_grad():
                predicted_segments = self._call_diarize(temp_path)

            # predicted_segments is a list (one per audio file) of lists of segment strings
            segments = []
            if predicted_segments and len(predicted_segments) > 0:
                raw_segments = predicted_segments[0]  # First (only) audio file
                for seg_str in raw_segments:
                    start, end, speaker = self._parse_segment(seg_str)
                    segments.append({
                        'start': start,
                        'end': end,
                        'speaker': speaker
                    })

            return segments

        finally:
            # Clean up temp file
            os.unlink(temp_path)
