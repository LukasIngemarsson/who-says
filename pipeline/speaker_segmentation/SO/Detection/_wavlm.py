"""
Speaker overlap detection using Microsoft WavLM model.

Uses microsoft/wavlm-base-plus-sd model which is fine-tuned for speaker diarization
and produces frame-level speaker activity predictions.
"""
import torch
import numpy as np
from typing import Optional, List, Tuple
from transformers import Wav2Vec2FeatureExtractor, WavLMForAudioFrameClassification
from loguru import logger


class WavLMSOD:
    """
    Overlapped speech detection using Microsoft's WavLM model fine-tuned for
    speaker diarization (wavlm-base-plus-sd).

    The model outputs frame-level speaker activity logits. Overlap is detected
    when multiple speakers are predicted as active simultaneously.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus-sd",
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration: float = 0.0,
        max_chunk_duration: float = 30.0,
        device: Optional[torch.device] = None,
        *args, **kwargs
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model ID for WavLM speaker diarization model
        onset : float
            Onset threshold for speaker activity detection (sigmoid output)
        offset : float
            Offset threshold for hysteresis (not currently used, kept for API consistency)
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
        self.model_name = model_name

        logger.info(f"Loading WavLM model: {model_name}")

        # Load feature extractor and model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForAudioFrameClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Model's expected sample rate
        self.model_sample_rate = self.feature_extractor.sampling_rate  # 16000

        logger.info(f"WavLM SOD initialized on {self.device}")

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

    def _process_chunk(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Process a single chunk through the model.

        Parameters
        ----------
        waveform : np.ndarray
            Audio waveform as 1D numpy array

        Returns
        -------
        torch.Tensor
            Speaker probabilities of shape (num_frames, num_speakers)
        """
        # Prepare inputs using feature extractor
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.model_sample_rate,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            # outputs.logits shape: (batch, num_frames, num_speakers)
            logits = outputs.logits

        # Convert logits to probabilities using sigmoid (multi-label)
        probs = torch.sigmoid(logits)

        return probs.squeeze(0)  # (num_frames, num_speakers)

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
        # Convert to numpy for feature extractor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        # Convert to mono if stereo
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(axis=0)

        # Ensure 1D
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)

        # Resample if necessary
        if sample_rate != self.model_sample_rate:
            import torchaudio.functional as F
            waveform_tensor = torch.from_numpy(waveform).float()
            waveform_tensor = F.resample(waveform_tensor, sample_rate, self.model_sample_rate)
            waveform = waveform_tensor.numpy()
            sample_rate = self.model_sample_rate

        total_samples = len(waveform)
        audio_duration = total_samples / sample_rate
        chunk_samples = int(self.max_chunk_duration * sample_rate)

        # Process in chunks if audio is too long
        if total_samples > chunk_samples:
            logger.info(f"Processing {audio_duration:.1f}s audio in {self.max_chunk_duration}s chunks")
            all_probs = []

            for start_sample in range(0, total_samples, chunk_samples):
                end_sample = min(start_sample + chunk_samples, total_samples)
                chunk = waveform[start_sample:end_sample]

                chunk_probs = self._process_chunk(chunk)
                all_probs.append(chunk_probs.cpu())

                # Clear GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            speaker_probs = torch.cat(all_probs, dim=0).numpy()
        else:
            # Process entire audio at once
            speaker_probs = self._process_chunk(waveform).cpu().numpy()

        # Calculate frame duration
        num_frames = speaker_probs.shape[0]
        frame_duration = audio_duration / num_frames

        logger.debug(f"WavLM: {num_frames} frames, frame_dur={frame_duration:.4f}s, "
                     f"num_speakers={speaker_probs.shape[-1]}")

        # Detect and return overlap segments
        return self._detect_overlaps(speaker_probs, frame_duration)
