"""
Speaker overlap detection using NeMo Sortformer model.
"""
import torch
import numpy as np
from typing import Optional, List, Tuple
from loguru import logger


class NemoSOD:
    """
    Overlapped speech detection using NeMo's Sortformer diarization model.

    Uses frame-level speaker activity outputs to detect when multiple
    speakers are active simultaneously.
    """

    def __init__(
        self,
        model_name: str = "nvidia/diar_sortformer_4spk-v1",
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration: float = 0.0,
        max_chunk_duration: float = 30.0,  # Process in 30-second chunks to avoid OOM
        device: Optional[torch.device] = None,
        *args, **kwargs
    ):
        """
        Parameters
        ----------
        model_name : str
            NeMo model name or path
        onset : float
            Onset threshold for speaker activity detection
        offset : float
            Offset threshold (hysteresis)
        min_duration : float
            Minimum duration for overlap segments in seconds
        max_chunk_duration : float
            Maximum chunk duration in seconds (to avoid GPU OOM)
        device : torch.device
            Device to run inference on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.onset = onset
        self.offset = offset
        self.min_duration = min_duration
        self.max_chunk_duration = max_chunk_duration

        from nemo.collections.asr.models import SortformerEncLabelModel

        self.model = SortformerEncLabelModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get model's sample rate requirement
        self.model_sample_rate = getattr(self.model, 'sample_rate', 16000)

    def _detect_overlaps(
        self,
        speaker_probs: np.ndarray,
        frame_duration: float
    ) -> List[Tuple[float, float]]:
        """
        Detect overlap regions from frame-level speaker probabilities.

        Parameters
        ----------
        speaker_probs : np.ndarray
            Shape (num_frames, num_speakers) - probability of each speaker being active
        frame_duration : float
            Duration of each frame in seconds

        Returns
        -------
        segments : List[Tuple[float, float]]
            List of (start, end) tuples in seconds where overlap occurs
        """
        # Count active speakers per frame (above onset threshold)
        active_speakers = (speaker_probs >= self.onset).sum(axis=-1)

        # Overlap = more than 1 speaker active
        overlap_mask = active_speakers > 1

        segments = []
        active = False
        start_idx = 0

        for i, is_overlap in enumerate(overlap_mask):
            if active:
                if not is_overlap:
                    # End segment
                    duration = (i - start_idx) * frame_duration
                    if duration >= self.min_duration:
                        segments.append((
                            start_idx * frame_duration,
                            i * frame_duration
                        ))
                    active = False
            else:
                if is_overlap:
                    # Start segment
                    start_idx = i
                    active = True

        # Handle final segment
        if active:
            duration = (len(overlap_mask) - start_idx) * frame_duration
            if duration >= self.min_duration:
                segments.append((
                    start_idx * frame_duration,
                    len(overlap_mask) * frame_duration
                ))

        return segments

    def _process_chunk(self, waveform: torch.Tensor) -> torch.Tensor:
        """Process a single chunk through the model."""
        input_lengths = torch.tensor([waveform.shape[-1]], device=self.device)

        try:
            outputs = self.model(
                audio_signal=waveform,
                audio_signal_length=input_lengths
            )
        except TypeError:
            try:
                processed, processed_len = self.model.preprocessor(
                    input_signal=waveform,
                    length=input_lengths
                )
                outputs = self.model.encoder(audio_signal=processed, length=processed_len)
            except (AttributeError, TypeError):
                outputs = self.model(waveform, input_lengths)

        if isinstance(outputs, tuple):
            speaker_probs = outputs[0]
        elif isinstance(outputs, dict):
            speaker_probs = outputs.get('logits', outputs.get('preds', list(outputs.values())[0]))
        else:
            speaker_probs = outputs

        if speaker_probs.min() < 0 or speaker_probs.max() > 1:
            speaker_probs = torch.sigmoid(speaker_probs)

        return speaker_probs.squeeze(0)

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

        # Ensure 1D for chunking
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)

        total_samples = waveform.shape[-1]
        audio_duration = total_samples / sample_rate
        chunk_samples = int(self.max_chunk_duration * sample_rate)

        # Process in chunks if audio is too long
        if total_samples > chunk_samples:
            logger.info(f"Processing {audio_duration:.1f}s audio in {self.max_chunk_duration}s chunks")
            all_probs = []

            for start_sample in range(0, total_samples, chunk_samples):
                end_sample = min(start_sample + chunk_samples, total_samples)
                chunk = waveform[start_sample:end_sample].unsqueeze(0).to(self.device).contiguous()

                with torch.no_grad():
                    chunk_probs = self._process_chunk(chunk)
                    all_probs.append(chunk_probs.cpu())

                # Clear GPU memory
                torch.cuda.empty_cache()

            speaker_probs = torch.cat(all_probs, dim=0).numpy()
        else:
            # Process entire audio at once
            waveform = waveform.unsqueeze(0).to(self.device).contiguous()

            with torch.no_grad():
                speaker_probs = self._process_chunk(waveform).cpu().numpy()

        # Calculate frame duration
        num_frames = speaker_probs.shape[0]
        frame_duration = audio_duration / num_frames

        # Detect and return overlap segments
        return self._detect_overlaps(speaker_probs, frame_duration)
