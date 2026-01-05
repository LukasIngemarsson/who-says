import os
import argparse
import time
import threading
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from loguru import logger
from pyannote.audio import Model, Inference
from utils import load_audio_from_file, load_annotation_file, evaluate_segmentation, format_metrics_report
from utils.constants import SR


class PyannoteVAD:

    def __init__(
        self,
        model: str = "pyannote/segmentation-3.0",
        device: Optional[torch.device] = None,
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0
    ):
        """
        Parameters:
            model : str
                HuggingFace model identifier
            device : torch.device
                Device to run inference on
            onset : float
                Onset threshold for speech detection
            offset : float
                Offset threshold for speech detection
            min_duration_on : float
                Minimum duration of speech region in seconds
            min_duration_off : float
                Minimum duration of silence region in seconds
        """
        self.model_name = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.onset = onset
        self.offset = offset
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.model = None
        self.inference = None
        self._load_lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        with self._load_lock:
            # Double-check after acquiring lock
            if self.model is not None:
                return
            use_auth_token = os.getenv("HF_TOKEN")
            if use_auth_token is None:
                raise ValueError("HF_TOKEN not set")

            self.model = Model.from_pretrained(
                self.model_name,
                use_auth_token=use_auth_token
            )
            self.inference = Inference(self.model, device=self.device)

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
        if self.inference is None:
            self.load()

        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        elif waveform.dim() == 2:
            waveform = waveform[0]

        waveform_2d = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
        audio_dict = {
            'waveform': waveform_2d,
            'sample_rate': sample_rate
        }

        output = self.inference(audio_dict)

        if output.data.ndim > 1:
            speech_probs = np.max(output.data, axis=1)
        else:
            speech_probs = output.data

        speech_probs = np.asarray(speech_probs).flatten()

        segments = []
        is_speech = False
        current_start = None
        frame_step = output.sliding_window.step

        for i in range(len(speech_probs)):
            prob = float(speech_probs[i])
            time = i * frame_step

            if not is_speech:
                if prob > self.onset:
                    is_speech = True
                    current_start = time
            else:
                if prob < self.offset:
                    is_speech = False
                    if current_start is not None:
                        duration = time - current_start
                        if self.min_duration_on == 0 or duration >= self.min_duration_on:
                            segments.append({
                                'start': current_start,
                                'end': time
                            })
                        current_start = None

        if is_speech and current_start is not None:
            end_time = len(speech_probs) * frame_step
            duration = end_time - current_start
            if self.min_duration_on == 0 or duration >= self.min_duration_on:
                segments.append({
                    'start': current_start,
                    'end': end_time
                })

        if self.min_duration_off > 0 and len(segments) > 1:
            filtered_segments = [segments[0]]
            for i in range(1, len(segments)):
                silence_duration = segments[i]['start'] - filtered_segments[-1]['end']
                if silence_duration < self.min_duration_off:
                    filtered_segments[-1]['end'] = segments[i]['end']
                else:
                    filtered_segments.append(segments[i])
            segments = filtered_segments

        return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pyannote VAD - Voice Activity Detection")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to process")
    parser.add_argument("--annotation", type=Path, help="Path to gold-standard annotation JSON for metrics evaluation")
    parser.add_argument("--onset", type=float, default=0.5, help="Onset threshold (0.0-1.0, default: 0.5)")
    parser.add_argument("--offset", type=float, default=0.5, help="Offset threshold (0.0-1.0, default: 0.5)")
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

    vad = PyannoteVAD(onset=args.onset, offset=args.offset)

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
