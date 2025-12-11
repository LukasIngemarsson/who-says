"""
Speaker separation using pyannote.audio separation models.
"""
import os
import torch
import numpy as np
from typing import Optional, Dict, List
from pyannote.audio import Model

class PyannoteSOS(object):
    """
    Speaker separation using pyannote/separation-ami-1.0 model.

    Separates overlapping speakers in audio waveforms into individual
    speaker waveforms.
    """

    def __init__(
        self,
        model_name: str = "pyannote/separation-ami-1.0",
        device: Optional[torch.device] = None,
        *args, **kwargs
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model ID for separation model
        device : torch.device
            Device to run inference on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        use_auth_token = os.getenv("HF_TOKEN")
        if use_auth_token is None:
            raise ValueError("HF_TOKEN not set")

        self.model = Model.from_pretrained(
            model_name,
            use_auth_token=use_auth_token,
            force_reload=True
        ).to(self.device)
        self.model.eval()

    def __call__(
        self,
        waveform: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Separate overlapping speakers in audio waveform.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform, shape (num_samples,) or (1, num_samples)
        sample_rate : int
            Sample rate of the audio

        Returns
        -------
        separated_waveforms : Dict[int, torch.Tensor]
            Dictionary mapping speaker index to separated waveform.
            Each waveform has shape (num_samples,)
        """
        # Convert to mono if stereo
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        # Ensure correct shape (1, num_samples) for model
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Run separation inference
        with torch.no_grad():
            waveform = waveform.to(self.device).contiguous()

            # The separation model outputs separated sources
            # Shape: (batch, num_sources, num_samples)
            separated = self.model(waveform)

        # Convert to dictionary of individual speaker waveforms
        separated_waveforms = {}

        # Handle different output formats from the model
        if isinstance(separated, torch.Tensor):
            # Shape: (batch, num_sources, num_samples)
            num_sources = separated.shape[1]
            for i in range(num_sources):
                speaker_waveform = separated[0, i, :].cpu()
                separated_waveforms[i] = speaker_waveform
        elif isinstance(separated, (list, tuple)):
            # Some models return a list/tuple of tensors
            for i, source in enumerate(separated):
                if isinstance(source, torch.Tensor):
                    speaker_waveform = source.squeeze().cpu()
                    separated_waveforms[i] = speaker_waveform

        return separated_waveforms

    def separate_regions(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        overlap_segments: List[tuple[float, float]]
    ) -> Dict[tuple[float, float], Dict[int, torch.Tensor]]:
        """
        Separate speakers only in overlapping regions.

        Parameters
        ----------
        waveform : torch.Tensor
            Full audio waveform
        sample_rate : int
            Sample rate of the audio
        overlap_segments : List[tuple[float, float]]
            List of (start, end) time segments in seconds where overlap occurs

        Returns
        -------
        separated_regions : Dict[tuple[float, float], Dict[int, torch.Tensor]]
            Dictionary mapping each overlap segment to separated speaker waveforms
        """
        separated_regions = {}

        for start_time, end_time in overlap_segments:
            # Convert time to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # Extract segment - ensure it's contiguous and a fresh copy
            segment = waveform[..., start_sample:end_sample].clone().contiguous()

            # Separate speakers in this segment
            separated = self(segment)

            separated_regions[(start_time, end_time)] = separated

        return separated_regions
