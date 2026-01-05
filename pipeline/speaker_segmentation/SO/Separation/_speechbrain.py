"""
Speaker separation using SpeechBrain SepFormer model.
"""
import torch
from typing import Optional, Dict, List


class SpeechBrainSOS:
    """
    Speaker separation using SpeechBrain's SepFormer model.

    SepFormer is a state-of-the-art transformer-based separation model
    trained on WSJ0-2mix achieving 22.3 dB SI-SNRi.
    """

    def __init__(
        self,
        model_name: str = "speechbrain/sepformer-wsj02mix",
        device: Optional[torch.device] = None,
        *args, **kwargs
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model ID for SepFormer model
        device : torch.device
            Device to run inference on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from speechbrain.inference.separation import SepformerSeparation

        self.model = SepformerSeparation.from_hparams(
            source=model_name,
            savedir=f"pretrained_models/{model_name.replace('/', '_')}",
            run_opts={"device": str(self.device)}
        )

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 8000
    ) -> Dict[int, torch.Tensor]:
        """
        Separate overlapping speakers in audio waveform.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform, shape (num_samples,) or (1, num_samples)
        sample_rate : int
            Sample rate of the audio (SepFormer expects 8kHz)

        Returns
        -------
        separated_waveforms : Dict[int, torch.Tensor]
            Dictionary mapping speaker index to separated waveform.
            Each waveform has shape (num_samples,)
        """
        import torchaudio

        # Convert to mono if stereo
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        # Ensure 1D
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)

        # SepFormer expects 8kHz audio - resample if needed
        if sample_rate != 8000:
            # Ensure waveform is on CPU and contiguous for resampling
            waveform = waveform.cpu().contiguous()
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=8000
            )
            waveform = resampler(waveform)

        # Run separation
        with torch.no_grad():
            # SepFormer expects batch dimension
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            waveform = waveform.to(self.device)

            # Separate sources - returns tensor of shape (batch, num_sources, samples)
            separated = self.model.separate_batch(waveform)

        # Convert to dictionary of individual speaker waveforms
        separated_waveforms = {}

        if isinstance(separated, torch.Tensor):
            # Shape: (batch, num_sources, num_samples) or (batch, num_samples, num_sources)
            if separated.dim() == 3:
                # Check which dimension is sources vs samples
                if separated.shape[1] < separated.shape[2]:
                    # Shape is (batch, num_sources, num_samples)
                    num_sources = separated.shape[1]
                    for i in range(num_sources):
                        separated_waveforms[i] = separated[0, i, :].cpu()
                else:
                    # Shape is (batch, num_samples, num_sources)
                    num_sources = separated.shape[2]
                    for i in range(num_sources):
                        separated_waveforms[i] = separated[0, :, i].cpu()
            elif separated.dim() == 2:
                # Single source or flattened
                separated_waveforms[0] = separated[0].cpu()

        return separated_waveforms

    def separate_regions(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        overlap_segments: List[tuple]
    ) -> Dict[tuple, Dict[int, torch.Tensor]]:
        """
        Separate speakers only in overlapping regions.

        Parameters
        ----------
        waveform : torch.Tensor
            Full audio waveform
        sample_rate : int
            Sample rate of the audio
        overlap_segments : list[tuple[float, float]]
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
            separated = self(segment, sample_rate)

            separated_regions[(start_time, end_time)] = separated

        return separated_regions
