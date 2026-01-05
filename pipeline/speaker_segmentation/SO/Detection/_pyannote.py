"""
Simplified overlapped speech detection using pyannote.audio models.
"""
import os
import torch
import numpy as np
from typing import Optional, List, Tuple
from pyannote.audio import Model
from pyannote.audio.utils.powerset import Powerset


class PyannoteSOD(object):
    """
    Overlapped speech detection using pyannote segmentation models.

    Supports powerset models (e.g., segmentation-3.0) where overlap classes
    are explicitly encoded in the output.
    """

    def __init__(
        self,
        model_name: str = "pyannote/segmentation-3.0",
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration: float = 0.0,
        device: Optional[torch.device] = None,
        *args, **kwargs
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model ID
        onset : float
            Onset threshold for overlap detection
        offset : float
            Offset threshold (hysteresis)
        min_duration : float
            Minimum duration for overlap segments in seconds
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

        # Get model specifications for proper powerset conversion
        specs = self.model.specifications
        if specs.powerset:
            self.powerset = Powerset(
                len(specs.classes),
                specs.powerset_max_classes
            )
            # Pre-compute overlap mask (which classes have 2+ speakers)
            num_classes = self.powerset.num_powerset_classes
            self.overlap_mask = torch.zeros(num_classes)
            for class_idx in range(num_classes):
                one_hot = torch.zeros(1, 1, num_classes)
                one_hot[0, 0, class_idx] = 1.0
                multilabel = self.powerset.to_multilabel(one_hot)
                num_speakers = (multilabel > 0.5).sum().item()
                if num_speakers >= 2:
                    self.overlap_mask[class_idx] = 1.0
            print(f"[PyannoteSOD] Overlap classes: {torch.where(self.overlap_mask > 0)[0].tolist()}")
        else:
            self.powerset = None
            self.overlap_mask = None

    def _extract_overlap_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Extract overlap scores from powerset-encoded model output.

        For segmentation-3.0 with 3 speakers and max 2 per frame:
        - 7 powerset classes total
        - Classes 0-3: no speaker or single speaker
        - Classes 4-6: overlap (2 speakers active)

        The probability of overlap = sum of probabilities of overlap classes.
        """
        # Model outputs log-softmax, convert to probabilities
        if scores.min() < 0:
            probs = torch.exp(scores)  # Convert log-probs to probs
        else:
            probs = scores

        if self.overlap_mask is not None:
            # Use pre-computed overlap mask
            mask = self.overlap_mask.to(probs.device)
            overlap_probs = (probs * mask).sum(dim=-1)
        else:
            # Fallback: assume classes 4+ are overlap (for 7-class powerset)
            overlap_probs = probs[..., 4:].sum(dim=-1)

        return overlap_probs

    def _binarize(
        self,
        scores: np.ndarray,
        frame_duration: float
    ) -> List[Tuple[float, float]]:
        """
        Apply hysteresis thresholding to convert scores to segments.

        Returns
        -------
        segments : List[Tuple[float, float]]
            List of (start, end) tuples in seconds
        """
        segments = []
        active = False
        start_idx = 0

        for i, score in enumerate(scores):
            if active:
                if score < self.offset:
                    # End segment
                    duration = (i - start_idx) * frame_duration
                    if duration >= self.min_duration:
                        segments.append((
                            start_idx * frame_duration,
                            i * frame_duration
                        ))
                    active = False
            else:
                if score >= self.onset:
                    # Start segment
                    start_idx = i
                    active = True

        # Handle final segment
        if active:
            duration = (len(scores) - start_idx) * frame_duration
            if duration >= self.min_duration:
                segments.append((
                    start_idx * frame_duration,
                    len(scores) * frame_duration
                ))

        return segments

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000
    ) -> List[Tuple[float, float]]:
        """
        Detect overlapped speech regions.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform, shape (num_samples,) or (1, num_samples)
        sample_rate : int
            Sample rate of the audio

        Returns
        -------
        segments : List[Tuple[float, float]]
            List of overlap segments as (start, end) in seconds
        """
        # Convert to mono if stereo
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        # Ensure correct shape
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            # Ensure tensor is float32 and contiguous
            waveform = waveform.to(device=self.device, dtype=torch.float32).contiguous()
            # Disable cuDNN for LSTM to avoid CUDNN_STATUS_NOT_SUPPORTED error
            with torch.backends.cudnn.flags(enabled=False):
                scores = self.model(waveform)
            overlap_scores = self._extract_overlap_scores(scores)
            overlap_scores = overlap_scores.squeeze().cpu().numpy()

        # Calculate frame duration
        num_frames = len(overlap_scores)
        audio_duration = waveform.shape[-1] / sample_rate
        frame_duration = audio_duration / num_frames

        # Binarize and return segments
        return self._binarize(overlap_scores, frame_duration)
