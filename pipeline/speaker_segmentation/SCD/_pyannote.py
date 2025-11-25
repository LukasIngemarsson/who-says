"""
Speaker change point detection using pyannote.audio models.
"""
import os
import torch
import numpy as np
from typing import Optional, List
from pyannote.audio import Model


class PyannoteSCD(object):
    """
    Speaker change point detection using pyannote segmentation models.

    Detects boundaries where speaker identity changes by analyzing
    speaker activity patterns in the segmentation model output.
    """

    def __init__(
        self,
        model_name: str = "pyannote/segmentation-3.0",
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration: float = 0.0,
        device: Optional[torch.device] = None
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model ID
        onset : float
            Onset threshold for change point detection
        offset : float
            Offset threshold (hysteresis)
        min_duration : float
            Minimum duration between change points in seconds
        device : torch.device
            Device to run inference on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        use_auth_token = os.getenv("HF_TOKEN")
        if use_auth_token is None:
            raise ValueError("HF_TOKEN not set")

        self.model = Model.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        ).to(self.device)
        self.model.eval()

        self.onset = onset
        self.offset = offset
        self.min_duration = min_duration

    def _extract_change_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker change scores from segmentation model output.

        Computes frame-to-frame dissimilarity to identify speaker changes.
        Uses the difference in speaker distributions between consecutive frames.
        """
        # Check if LogSoftmax output (negative values)
        if scores.max() <= 0:
            scores = torch.exp(scores)

        # Squeeze to ensure we have (num_frames, num_classes) shape
        while scores.ndim > 2:
            scores = scores.squeeze(0)

        # Compute frame-to-frame dissimilarity
        # Use L1 distance between consecutive frame distributions
        diff = torch.abs(scores[1:] - scores[:-1])
        change_scores = diff.sum(dim=-1)

        # Pad beginning to maintain original length
        # Create zero padding with same number of dimensions
        if change_scores.ndim == 0:
            # Edge case: only 1 frame
            change_scores = torch.zeros(1, device=scores.device)
        else:
            zero_pad = torch.zeros(1, device=change_scores.device, dtype=change_scores.dtype)
            change_scores = torch.cat([zero_pad, change_scores], dim=0)

        # Normalize to [0, 1] range
        if change_scores.numel() > 0 and change_scores.max() > 0:
            change_scores = change_scores / change_scores.max()

        return change_scores

    def _detect_peaks(
        self,
        scores: np.ndarray,
        frame_duration: float
    ) -> List[float]:
        """
        Detect speaker change points as peaks in the change score signal.

        Returns
        -------
        change_points : List[float]
            List of change point timestamps in seconds
        """
        change_points = []
        active = False
        peak_idx = 0
        peak_score = 0.0

        for i, score in enumerate(scores):
            if active:
                if score > peak_score:
                    # Update peak
                    peak_idx = i
                    peak_score = score

                if score < self.offset:
                    # End of peak region - record the peak
                    change_points.append(peak_idx * frame_duration)
                    active = False
                    peak_score = 0.0
            else:
                if score >= self.onset:
                    # Start of peak region
                    peak_idx = i
                    peak_score = score
                    active = True

        # Handle final peak
        if active:
            change_points.append(peak_idx * frame_duration)

        # Filter out change points that are too close together
        if len(change_points) > 1 and self.min_duration > 0:
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
        # Convert to mono if stereo
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        # Ensure correct shape
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            waveform = waveform.to(self.device)
            scores = self.model(waveform)
            change_scores = self._extract_change_scores(scores)
            change_scores = change_scores.squeeze().cpu().numpy()

        # Calculate frame duration
        num_frames = len(change_scores)
        audio_duration = waveform.shape[-1] / sample_rate
        frame_duration = audio_duration / num_frames

        # Detect peaks and return change points
        return self._detect_peaks(change_scores, frame_duration)
