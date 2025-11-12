from enum import Enum
from loguru import logger
from torch.cuda import is_available as is_cuda_available
import torch

from pipeline.ASR.whisper import WhisperASR
from pipeline.ASR.faster_whisper import FasterWhisperASR

class TypeASR(Enum):
    WHISPER = "whisper"
    FASTER_WHISPER = "faster_whisper"

class ASR(object):
    def __init__(
        self,
        asr_type: TypeASR = TypeASR.FASTER_WHISPER,
        device: str = "cuda" if is_cuda_available() else "cpu",
        model: str | None = None,
        torch_dtype: torch.dtype | None = None,
        compute_type: str | None = None,
        language: str | None = None,
        return_timestamps: bool = True,
        word_timestamps: bool = True
    ):
        """
        Initialize ASR pipeline with configurable model and parameters.

        Args:
            asr_type: Type of ASR model to use (WHISPER or FASTER_WHISPER)
            device: Device to run model on ("cuda", "cpu")
            model: Model identifier. If None, uses default for the ASR type:
                - WHISPER: "openai/whisper-large-v3-turbo"
                - FASTER_WHISPER: "large-v3-turbo"
            torch_dtype: Data type for Whisper model (float16, float32).
                        If None, uses float16 for cuda, float32 for cpu
            compute_type: Quantization type for Faster-Whisper ("float16", "float32", "int8").
                         If None, uses "float16" for cuda, "float32" for cpu
            language: Language code (e.g., "sv" for Swedish, "en" for English).
                     If None, language will be auto-detected
            return_timestamps: Whether to return timestamps
            word_timestamps: Whether to return word-level timestamps (only for Faster-Whisper)
        """
        # Store configuration for the pipeline
        self.asr_type = asr_type
        self.device = device
        self.language = language
        self.return_timestamps = return_timestamps
        self.word_timestamps = word_timestamps

        # Set default model based on ASR type if not provided
        if model is None:
            if asr_type == TypeASR.WHISPER:
                model = "openai/whisper-large-v3-turbo"
            elif asr_type == TypeASR.FASTER_WHISPER:
                model = "large-v3-turbo"
        self.model = model

        # Set default dtype/compute_type based on device if not provided
        if device == "cuda":
            self.torch_dtype = torch_dtype if torch_dtype is not None else torch.float16
            self.compute_type = compute_type if compute_type is not None else "float16"
        else:
            self.torch_dtype = torch_dtype if torch_dtype is not None else torch.float32
            self.compute_type = compute_type if compute_type is not None else "float32"

        # Initialize pipeline (uses the setter)
        self.asr_pipeline = asr_type

    @property
    def asr_pipeline(self):
        return self._asr_pipeline

    @asr_pipeline.setter
    def asr_pipeline(self, asr_type: TypeASR):
        match asr_type:
            case TypeASR.WHISPER:
                logger.info(f"Initializing ASR with type: {asr_type.value}")
                logger.info(f"Model: {self.model}, Device: {self.device}, Dtype: {self.torch_dtype}")
                self._asr_pipeline = WhisperASR(
                    model=self.model,
                    device=self.device,
                    torch_dtype=self.torch_dtype
                )
            case TypeASR.FASTER_WHISPER:
                logger.info(f"Initializing ASR with type: {asr_type.value}")
                logger.info(f"Model: {self.model}, Device: {self.device}, Compute type: {self.compute_type}")
                self._asr_pipeline = FasterWhisperASR(
                    model=self.model,
                    device=self.device,
                    compute_type=self.compute_type
                )
            case _:
                raise ValueError(f"Invalid ASR Type {asr_type}")

    def transcribe(self, audio: torch.Tensor, **kwargs):
        """
        Transcribe audio using the configured ASR model.

        Args:
            audio: Audio tensor to transcribe
            **kwargs: Additional arguments to pass to the transcribe method.
                     Will override instance defaults if provided.

        Returns:
            Dictionary with transcription results
        """
        # Merge instance defaults with provided kwargs
        transcribe_kwargs = {
            'return_timestamps': kwargs.get('return_timestamps', self.return_timestamps),
            'language': kwargs.get('language', self.language)
        }

        # Add word_timestamps for Faster-Whisper
        if self.asr_type == TypeASR.FASTER_WHISPER:
            transcribe_kwargs['word_timestamps'] = kwargs.get('word_timestamps', self.word_timestamps)

        return self.asr_pipeline.transcribe(audio, **transcribe_kwargs)

    def transcribe_segments(self, audio: torch.Tensor, speech_segments: list[dict[str, float]], **kwargs):
        """
        Transcribe only speech segments from audio detected by VAD.

        Args:
            audio: Full audio tensor
            speech_segments: List of speech segments from VAD
            **kwargs: Additional arguments to pass to transcribe_segments.
                     Will override instance defaults if provided.

        Returns:
            List of dictionaries with transcription results for each segment
        """
        # Merge instance defaults with provided kwargs
        transcribe_kwargs = {
            'return_timestamps': kwargs.get('return_timestamps', self.return_timestamps),
            'language': kwargs.get('language', self.language)
        }

        # Add word_timestamps for Faster-Whisper
        if self.asr_type == TypeASR.FASTER_WHISPER:
            transcribe_kwargs['word_timestamps'] = kwargs.get('word_timestamps', self.word_timestamps)

        return self.asr_pipeline.transcribe_segments(audio, speech_segments, **transcribe_kwargs)

    def __call__(self, audio: torch.Tensor, **kwargs):
        """
        Convenience method to transcribe audio directly.
        """
        return self.transcribe(audio, **kwargs)
