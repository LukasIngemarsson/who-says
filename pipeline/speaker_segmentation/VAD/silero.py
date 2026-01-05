import argparse
import time
import threading
import torch
from pathlib import Path
from typing import Optional
from loguru import logger
from utils import load_audio_from_file, load_annotation_file, evaluate_segmentation, format_metrics_report
from utils.constants import SR


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
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        with self._load_lock:
            # Double-check after acquiring lock
            if self.model is not None:
                return
            self.model, self.utils = torch.hub.load(
                repo_or_dir=self.model_repo,
                model=self.model_name,
                trust_repo=True
            )
            self.model.to(self.device)

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

        waveform = waveform.to(self.device)

        # Serialize inference - Silero VAD has internal state
        with self._inference_lock:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silero VAD - Voice Activity Detection")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to process")
    parser.add_argument("--annotation", type=Path, help="Path to gold-standard annotation JSON for metrics evaluation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Speech threshold (0.0-1.0, default: 0.5)")
    parser.add_argument("--timing", action="store_true", help="Show timing metrics")

    args = parser.parse_args()

    if not args.audio_file.exists():
        parser.error(f"Audio file not found: {args.audio_file}")

    logger.info(f"Processing: {args.audio_file}")

    if args.timing:
        start_time = time.time()
    waveform, sr = load_audio_from_file(args.audio_file, sr=SR)
    if args.timing:
        audio_loading_time = time.time() - start_time

    if waveform.dim() > 1:
        if waveform.shape[0] > 1 and waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.mean(dim=0)
        elif waveform.shape[1] > 1 and waveform.shape[1] < waveform.shape[0]:
            waveform = waveform.mean(dim=1)
        else:
            waveform = waveform.squeeze()
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)

    logger.info(f"Audio shape: {waveform.shape}, sample rate: {sr}Hz")

    vad = SileroVAD(threshold=args.threshold)

    logger.info("Running Voice Activity Detection...")
    if args.timing:
        start_time = time.time()
    speech_segments = vad(waveform)
    if args.timing:
        vad_time = time.time() - start_time

    logger.info(f"Found {len(speech_segments)} speech segments")

    total_duration = waveform.shape[-1] / sr

    print("\n" + "="*60)
    print("VAD RESULTS")
    print("="*60)
    print(f"Duration: {total_duration:.2f}s")
    print(f"Speech segments: {len(speech_segments)}")

    print("\n" + "-"*60)
    print("SPEECH SEGMENTS")
    print("-"*60)
    for i, seg in enumerate(speech_segments):
        duration = seg['end'] - seg['start']
        print(f"  Segment {i+1}: [{seg['start']:.2f}s - {seg['end']:.2f}s] ({duration:.2f}s)")

    print("\n" + "="*60)

    if args.annotation:
        if not args.annotation.exists():
            logger.warning(f"Annotation file not found: {args.annotation}")
        else:
            try:
                logger.info(f"Loading annotation from {args.annotation}")
                annotation_data = load_annotation_file(args.annotation)

                logger.info("Computing VAD metrics...")
                vad_metrics = evaluate_segmentation(
                    reference_segments=annotation_data['segments'],
                    prediction_segments=speech_segments,
                    total_duration=total_duration
                )

                print(format_metrics_report({'vad': vad_metrics}))
            except Exception as e:
                logger.error(f"Error computing metrics: {e}")

    if args.timing:
        print("\n" + "="*60)
        print("TIMING METRICS")
        print("="*60)
        print(f"Audio Loading:  {audio_loading_time:.2f}s")
        print(f"VAD Processing: {vad_time:.2f}s")
        print("-"*60)
        print(f"Total Time:     {audio_loading_time + vad_time:.2f}s")
        print("="*60)

    logger.info("Done!")
