from utils import load_audio_from_file, match_frequency
from utils.constants import SR, TENSOR_DTYPE

import torch
from faster_whisper import WhisperModel


class FasterWhisperASR:
    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float32"
    ) -> None:
        """
        Initialize Faster-Whisper ASR model.

        Args:
            model: Model size. Supported:
                - "tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"
            device: Device to run model on ("cuda", "cpu")
            compute_type: Quantization type ("float16", "float32", "int8")
        """
        self.model_size = model
        self.device = device
        self.compute_type = compute_type

        # Load faster-whisper model
        self.model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type
        )

    def transcribe(
        self,
        audio: torch.Tensor,
        return_timestamps: bool = True,
        language: str | None = None,
        word_timestamps: bool = True,
        **decode_kwargs,
    ) -> dict:
        """
        Transcribe audio using Faster-Whisper model.

        Args:
            audio: Audio tensor
            return_timestamps: Whether to return segment-level timestamps
            language: Language code (e.g., "sv" for Swedish, "en" for English)
                     If None, language will be auto-detected
            word_timestamps: Whether to return word-level timestamps

        Returns:
            Dictionary with transcription results including text and optional timestamps
        """
        # Convert tensor to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.to("cpu")
            audio_input = audio.numpy()
        else:
            audio_input = audio

        # Ensure audio is 1D
        if len(audio_input.shape) > 1:
            audio_input = audio_input.squeeze()

        # Run transcription with optional decoding controls
        segments, info = self.model.transcribe(
            audio_input,
            language=language,
            word_timestamps=word_timestamps,
            **decode_kwargs,
        )

        # Collect segments into result format
        full_text = ""
        chunks = []

        for segment in segments:
            full_text += segment.text

            if return_timestamps:
                if word_timestamps and segment.words:
                    # Add word-level timestamps
                    for word in segment.words:
                        chunks.append({
                            'text': word.word,
                            'timestamp': (word.start, word.end)
                        })
                else:
                    # Add segment-level timestamps
                    chunks.append({
                        'text': segment.text,
                        'timestamp': (segment.start, segment.end)
                    })

        result = {
            'text': full_text.strip()
        }

        if return_timestamps and chunks:
            result['chunks'] = chunks

        return result

    def transcribe_segments(
        self,
        audio: torch.Tensor,
        speech_segments: list[dict[str, float]],
        sample_rate: int = SR,
        return_timestamps: bool = True,
        language: str | None = None,
        word_timestamps: bool = True,
        **decode_kwargs,
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
            word_timestamps: Whether to return word-level timestamps

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

            # Transcribe the segment with optional decoding controls
            transcription = self.transcribe(
                segment_audio,
                return_timestamps=return_timestamps,
                language=language,
                word_timestamps=word_timestamps,
                **decode_kwargs,
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
    # Example usage with large-v3-turbo
    print("Testing Faster-Whisper with large-v3-turbo...")
    asr = FasterWhisperASR(
        model="large-v3-turbo",
        device="cpu",
        compute_type="float32"
    )
    file_path = "data/single_speaker_sample.wav"
    audio, freq = load_audio_from_file(file_path, SR)
    audio = match_frequency(audio, freq, SR)
    result = asr.transcribe(audio)
    print(f"Result: {result}")

    print("\n" + "="*80 + "\n")

    # Example usage with Swedish language
    print("Testing Faster-Whisper with Swedish language...")
    result_sv = asr.transcribe(audio, language="sv")
    print(f"Swedish result: {result_sv}")
