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

        print(model)
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
            audio = audio.to("cpu")
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

    def transcribe_segments(
        self,
        audio: torch.Tensor,
        speech_segments: list[dict[str, float]],
        sample_rate: int = SR,
        return_timestamps: bool = True,
        language: str | None = None
    ) -> list[dict]:
        """
        Transcribe only speech segments from audio detected by VAD.

        Args:
            audio: Full audio tensor
            speech_segments: List of speech segments from VAD,
                           format: [{'start': 0.7, 'end': 3.5}, ...]
            sample_rate: Sample rate of the audio
            return_timestamps: Whether to return word-level timestamps
            language: Language code (e.g., "sv" for Swedish, "en" for English)
                     If None, language will be auto-detected

        Returns:
            List of dictionaries, each containing:
                - 'text': Transcribed text for the segment
                - 'start': Start time of segment in original audio
                - 'end': End time of segment in original audio
                - 'chunks': Word-level timestamps (if return_timestamps=True)
        """
        # Ensure audio is mono (single channel)
        if audio.dim() > 1:
            if audio.shape[0] > 1:
                # If multiple channels, average them
                audio = audio.mean(dim=0)
            else:
                # If shape is (1, n_samples), squeeze to (n_samples,)
                audio = audio.squeeze(0)

        results = []

        for segment in speech_segments:
            start_time = segment['start']
            end_time = segment['end']

            # Extract audio segment
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            # Skip very short segments (less than 0.1 seconds)
            if (end_sample - start_sample) < int(0.1 * sample_rate):
                continue

            # Ensure segment is 1D for transcription
            if segment_audio.dim() > 1:
                segment_audio = segment_audio.squeeze()

            # Transcribe the segment
            transcription = self.transcribe(
                segment_audio,
                return_timestamps=return_timestamps,
                language=language
            )

            # Add segment timing information
            result = {
                'text': transcription.get('text', ''),
                'start': start_time,
                'end': end_time
            }

            # If word-level timestamps are returned, adjust them to original audio timeline
            if return_timestamps and 'chunks' in transcription:
                adjusted_chunks = []
                for chunk in transcription['chunks']:
                    adjusted_chunk = chunk.copy()
                    if 'timestamp' in chunk and chunk['timestamp']:
                        # Adjust timestamps to align with original audio
                        ts = chunk['timestamp']
                        if isinstance(ts, (list, tuple)) and len(ts) == 2:
                            adjusted_chunk['timestamp'] = (
                                ts[0] + start_time if ts[0] is not None else None,
                                ts[1] + start_time if ts[1] is not None else None
                            )
                    adjusted_chunks.append(adjusted_chunk)
                result['chunks'] = adjusted_chunks

            results.append(result)

        return results


if __name__ == "__main__":
    # Example usage with whisper-large-v3-turbo
    print("Testing openai/whisper-large-v3-turbo...")
    asr_turbo = WhisperASR(
        model="openai/whisper-large-v3-turbo",
        device="cpu",
        torch_dtype=torch.float32
    )
    file_path = "data/single_speaker_sample.wav"
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
