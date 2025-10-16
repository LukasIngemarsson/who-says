from utils.constants import SR
import torch
from typing import Optional


class SileroVAD:

    def __init__(
        self,
        model_repo: str = "snakers4/silero-vad",
        model_name: str = "silero_vad",
        sample_rate: int = SR,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float('inf'),
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30,
        return_seconds: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Parameters:
            model_repo : str
                HuggingFace/GitHub model repository
            model_name : str
                Model name to load from repository
            sample_rate : int
                Target sample rate (8000 or 16000 Hz)
            threshold : float
                Speech threshold (0.0 to 1.0)
            min_speech_duration_ms : int
                Minimum speech segment duration in milliseconds
            max_speech_duration_s : float
                Maximum speech segment duration in seconds
            min_silence_duration_ms : int
                Minimum silence duration between segments in milliseconds
            window_size_samples : int
                Window size for VAD model
            speech_pad_ms : int
                Padding added to speech segments in milliseconds
            return_seconds : bool
                Return timestamps in seconds (True) or samples (False)
            device : torch.device
                Device to run inference on
        """
        self.model_repo = model_repo
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.return_seconds = return_seconds
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.utils = None

    def load(self):
        if self.model is None:
            self.model, self.utils = torch.hub.load(
                repo_or_dir=self.model_repo,
                model=self.model_name,
                trust_repo=True
            )

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000
    ):
        """
        Detect speech segments in audio.

        Parameters:
            waveform : torch.Tensor
                Audio waveform, shape (num_samples,) or (1, num_samples)
            sample_rate : int
                Sample rate of the audio

        Returns:
            segments : List[dict]
                List of speech segments with 'start' and 'end' timestamps.
        """
        if self.model is None:
            self.load()

        # Convert to mono if needed
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        elif waveform.dim() == 2:
            waveform = waveform[0]

        # Get speech timestamps
        get_speech_timestamps = self.utils[0]
        speech_timestamps = get_speech_timestamps(
            waveform,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            max_speech_duration_s=self.max_speech_duration_s,
            min_silence_duration_ms=self.min_silence_duration_ms,
            window_size_samples=self.window_size_samples,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=self.return_seconds
        )

        return [
            {'start': seg['start'], 'end': seg['end']}
            for seg in speech_timestamps
        ]
