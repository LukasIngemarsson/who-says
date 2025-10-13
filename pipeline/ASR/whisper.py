from utils import load_audio_from_file, match_frequency
from utils.constants import SR, TENSOR_DTYPE

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperASR:
    def __init__(
        self,
        model: str = "openai/whisper-large-v3-turbo",
        device: str = "cuda",
        torch_dtype: torch.dtype = TENSOR_DTYPE
    ) -> None:
        """
        Initialize Whisper ASR model.

        Args:
            model: HuggingFace model ID. Supported:
                - "openai/whisper-large-v3-turbo"
                - "KBLab/kb-whisper-tiny"
            device: Device to run model on ("cuda", "cpu")
            torch_dtype: Data type for model weights (float16, float32)
        """
        self.model_id = model
        self.device = device
        self.torch_dtype = torch_dtype

        # Load model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(model)

        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe(
        self,
        audio: torch.Tensor,
        return_timestamps: bool = True,
        language: str | None = None
    ) -> dict:
        """
        Transcribe audio using Whisper model.

        Args:
            audio: Audio tensor
            return_timestamps: Whether to return word-level timestamps
            language: Language code (e.g., "sv" for Swedish, "en" for English)
                     If None, language will be auto-detected

        Returns:
            Dictionary with transcription results including text and optional timestamps
        """
        # Prepare generate kwargs
        generate_kwargs = {}
        if language is not None:
            generate_kwargs["language"] = language

        # Convert tensor to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio_input = audio.numpy()
        else:
            audio_input = audio

        # Ensure audio is 1D
        if len(audio_input.shape) > 1:
            audio_input = audio_input.squeeze()

        # Run transcription
        result = self.pipe(
            audio_input,
            return_timestamps=return_timestamps,
            generate_kwargs=generate_kwargs
        )

        return result


if __name__ == "__main__":
    # Example usage with whisper-large-v3-turbo
    print("Testing openai/whisper-large-v3-turbo...")
    asr_turbo = WhisperASR(
        model="openai/whisper-large-v3-turbo",
        device="cpu",
        torch_dtype=torch.float32
    )
    file_path = "single_speaker_sample.wav"
    audio, freq = load_audio_from_file(file_path, SR)
    audio = match_frequency(audio, freq, SR)
    result_turbo = asr_turbo.transcribe(audio)
    print(f"Turbo result: {result_turbo}")

    print("\n" + "="*80 + "\n")

    # Example usage with kb-whisper-tiny
    print("Testing KBLab/kb-whisper-tiny...")
    asr_tiny = WhisperASR(
        model="KBLab/kb-whisper-tiny",
        device="cpu",
        torch_dtype=torch.float32
    )
    result_tiny = asr_tiny.transcribe(audio, language="sv")
    print(f"KB-Whisper result: {result_tiny}")
