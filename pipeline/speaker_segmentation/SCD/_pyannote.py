"""
Speaker change point detection using pyannote.audio models.
"""
import os
import torch
from pyannote.audio.core.task import Specifications, Problem , Resolution # NEW

# Allow pyannote checkpoints to load under PyTorch 2.6+ safe loader
torch.serialization.add_safe_globals([
    torch.torch_version.TorchVersion,
    Specifications,
    Problem,
    Resolution,
])
import numpy as np
from scipy.signal import find_peaks
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
        min_prominence: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model ID
        onset : float
            Onset threshold for change point detection (unused with find_peaks, kept for compatibility)
        offset : float
            Offset threshold (unused with find_peaks, kept for compatibility)
        min_duration : float
            Minimum duration between change points in seconds
        min_prominence : float
            Minimum prominence for peak detection (how much a peak stands out)
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
        self.min_prominence = min_prominence

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

        Uses scipy's find_peaks with prominence-based detection to find
        local maxima that stand out from their surroundings.

        Returns
        -------
        change_points : List[float]
            List of change point timestamps in seconds
        """
        # Calculate minimum distance between peaks in frames
        min_distance = max(1, int(self.min_duration / frame_duration)) if self.min_duration > 0 else 1

        # Find peaks using prominence (how much a peak stands out from surrounding signal)
        peak_indices, properties = find_peaks(
            scores,
            prominence=self.min_prominence,
            distance=min_distance
        )

        # Convert frame indices to timestamps
        change_points = [idx * frame_duration for idx in peak_indices]

        return change_points

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        chunk_duration: float = 60.0
    ) -> List[float]:
        """
        Detect speaker change points.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform, shape (num_samples,) or (1, num_samples)
        sample_rate : int
            Sample rate of the audio
        chunk_duration : float
            Duration of each chunk in seconds (for long audio processing)

        Returns
        -------
        change_points : List[float]
            List of timestamps (in seconds) where speaker changes occur
        """
        # Convert to mono if stereo
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        # Ensure 1D for processing
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)

        audio_duration = waveform.shape[-1] / sample_rate
        chunk_samples = int(chunk_duration * sample_rate)

        # Process long audio in chunks
        if waveform.shape[-1] > chunk_samples:
            return self._process_chunked(waveform, sample_rate, chunk_samples)

        # Short audio - process directly
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            # Ensure contiguous for CUDA/cuDNN operations
            waveform = waveform.contiguous().to(self.device)
            scores = self.model(waveform)
            change_scores = self._extract_change_scores(scores)
            change_scores = change_scores.squeeze().cpu().numpy()

        # Calculate frame duration
        num_frames = len(change_scores)
        frame_duration = audio_duration / num_frames

        # Detect peaks and return change points
        return self._detect_peaks(change_scores, frame_duration)

    def _process_chunked(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_samples: int
    ) -> List[float]:
        """
        Process long audio in chunks to avoid CUDA memory issues.

        Parameters
        ----------
        waveform : torch.Tensor
            1D audio waveform
        sample_rate : int
            Sample rate
        chunk_samples : int
            Number of samples per chunk

        Returns
        -------
        change_points : List[float]
            Combined change points from all chunks
        """
        all_change_scores = []
        total_samples = waveform.shape[-1]
        overlap_samples = int(sample_rate * 2)  # 2 second overlap

        position = 0
        while position < total_samples:
            end_pos = min(position + chunk_samples, total_samples)
            chunk = waveform[position:end_pos]

            # Ensure correct shape
            if chunk.ndim == 1:
                chunk = chunk.unsqueeze(0)

            # Run inference on chunk
            with torch.no_grad():
                chunk = chunk.contiguous().to(self.device)
                scores = self.model(chunk)
                change_scores = self._extract_change_scores(scores)
                change_scores = change_scores.squeeze().cpu().numpy()

            # Calculate frame info for this chunk
            chunk_duration = chunk.shape[-1] / sample_rate
            num_frames = len(change_scores)
            frame_duration = chunk_duration / num_frames
            chunk_offset = position / sample_rate

            # Detect peaks in this chunk
            chunk_changes = self._detect_peaks(change_scores, frame_duration)

            # Offset by chunk position
            for cp in chunk_changes:
                all_change_scores.append(cp + chunk_offset)

            # Move to next chunk (with overlap to avoid missing changes at boundaries)
            position += chunk_samples - overlap_samples
            if position >= total_samples:
                break

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Remove duplicate change points near chunk boundaries
        if len(all_change_scores) > 1:
            all_change_scores = sorted(all_change_scores)
            filtered = [all_change_scores[0]]
            for cp in all_change_scores[1:]:
                if cp - filtered[-1] >= self.min_duration:
                    filtered.append(cp)
            all_change_scores = filtered

        return all_change_scores
