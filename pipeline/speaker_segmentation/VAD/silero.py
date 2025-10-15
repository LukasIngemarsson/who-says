from config.constants import DESIRED_FREQUENCY
import torch

class SileroVAD:

    def __init__(self, sample_rate=DESIRED_FREQUENCY):
        """
        Initialize SileroVAD.

        Args:
            sample_rate: Target sample rate (8000 or 16000)
        """
        self.sample_rate = sample_rate
        self.model = None
        self.utils = None

    def load(self):
        if self.model is None:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True
            )

    def __call__(self, waveform):
        """
        Run VAD on waveform tensor.

        Args:
            waveform: Audio tensor

        Returns:
            List of dicts with 'start' and 'end' timestamps in seconds
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
            return_seconds=True
        )

        return [
            {'start': seg['start'], 'end': seg['end']}
            for seg in speech_timestamps
        ]
    
    # def predict(self, audio_path):
    #     """
    #     Detect speech segments in audio file.

    #     Args:
    #         audio_path: Path to audio file

    #     Returns:
    #         List of dicts with 'start' and 'end' timestamps in seconds
    #     """
    #     if self.model is None:
    #         self.load()

    #     wav, sr = torchaudio.load(audio_path)

    #     # Resample if needed
    #     if sr != self.sample_rate:
    #         wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)

    #     # Convert to mono if needed
    #     if wav.shape[0] > 1:
    #         wav = torch.mean(wav, dim=0, keepdim=True)

    #     # Get speech timestamps
    #     get_speech_timestamps = self.utils[0]
    #     speech_timestamps = get_speech_timestamps(
    #         wav[0],
    #         self.model,
    #         sampling_rate=self.sample_rate,
    #         return_seconds=True
    #     )

    #     # Format output
    #     return [
    #         {'start': seg['start'], 'end': seg['end']}
    #         for seg in speech_timestamps
    #     ]
