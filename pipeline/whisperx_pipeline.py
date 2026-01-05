"""
WhisperX end-to-end pipeline for diarization and transcription.
"""
import os
import gc
from pathlib import Path
from typing import Dict, Optional

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from loguru import logger


class WhisperXPipeline:
    """
    End-to-end pipeline using WhisperX for ASR + diarization.

    WhisperX performs:
    1. Transcription (Whisper)
    2. Word-level alignment
    3. Speaker diarization (pyannote-based)
    4. Speaker assignment to words/segments
    """

    def __init__(
        self,
        model_size: str = "large-v2",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        batch_size: int = 16
    ):
        self.model_size = model_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        self.language = language
        self.batch_size = batch_size

        self.asr_model = None
        self.diarize_model = None

    def load(self):
        """Load WhisperX ASR model."""
        if self.asr_model is not None:
            return

        logger.info(f"Loading WhisperX ASR model ({self.model_size}) on {self.device}...")
        self.asr_model = whisperx.load_model(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
        )

    def _get_diarize_model(self) -> DiarizationPipeline:
        """Lazy load diarization model."""
        if self.diarize_model is None:
            hf_token = os.environ.get('HF_TOKEN')
            if not hf_token:
                raise RuntimeError("HF_TOKEN not set for diarization")

            logger.info(f"Loading WhisperX diarization model on {self.device}...")
            self.diarize_model = DiarizationPipeline(
                use_auth_token=hf_token,
                device=self.device,
            )
        return self.diarize_model

    def _cleanup_gpu(self):
        """Free GPU memory."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def process(
        self,
        audio_path: Path,
        num_speakers: Optional[int] = None
    ) -> Dict:
        """
        Process audio file through WhisperX pipeline.

        Args:
            audio_path: Path to audio file
            num_speakers: Optional hint for number of speakers

        Returns:
            Dict with structure:
            {
                'segments': [
                    {
                        'start': float,
                        'end': float,
                        'speaker': str,
                        'text': str
                    }
                ]
            }
        """
        if self.asr_model is None:
            self.load()

        audio_path = str(audio_path)

        # 1. Transcribe
        logger.debug(f"WhisperX: Transcribing {audio_path}")
        result = self.asr_model.transcribe(
            audio_path,
            batch_size=self.batch_size,
        )

        detected_language = result.get("language", self.language or "en")

        # 2. Load audio for alignment and diarization
        audio_data = whisperx.load_audio(audio_path)

        # 3. Alignment (adds per-word timestamps)
        logger.debug("WhisperX: Aligning words...")
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=self.device,
        )

        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_data,
            device=self.device,
        )

        # Free alignment model
        del model_a
        self._cleanup_gpu()

        # 4. Diarization
        logger.debug("WhisperX: Running diarization...")
        diarizer = self._get_diarize_model()

        if num_speakers is not None:
            diarize_segments = diarizer(
                audio_data,
                min_speakers=num_speakers,
                max_speakers=num_speakers,
            )
        else:
            diarize_segments = diarizer(audio_data)

        # 5. Assign speakers to words/segments
        logger.debug("WhisperX: Assigning speakers...")
        result_with_speakers = whisperx.assign_word_speakers(
            diarize_segments,
            result_aligned,
        )

        self._cleanup_gpu()

        # Convert to standard format
        segments = []
        for seg in result_with_speakers.get("segments", []):
            segments.append({
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'speaker': seg.get('speaker', 'UNKNOWN'),
                'text': seg.get('text', '').strip()
            })

        return {'segments': segments}
