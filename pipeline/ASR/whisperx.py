from utils import load_audio_from_file, match_frequency 

import whisperx
import torch

class WhsiperXASR:
    def __init__(self, model: str = "base", device: str = "cuda", compute_type: str = "float32") -> None:
        self.model = whisperx.load_model(model, device, compute_type=compute_type)

    def transcribe(self, audio: torch.Tensor, batch_size: int) -> list[dict[str, str]]:
        result = self.model.transcribe(audio.numpy(), batch_size=batch_size)
        segments = result["segments"]
        return segments


if __name__ == "__main__":
    asr = WhsiperXASR(device="cpu")
    file_path = "samples/single_speaker_sample.wav"
    audio, freq = load_audio_from_file(file_path)
    print(f"single speaker: audio dim: {audio.shape}, freq: {freq}")
    file_path = "samples/multi_speaker_sample.mp3"
    audio, freq = load_audio_from_file(file_path)
    print(f"multi speaker: audio dim: {audio.shape}, freq: {freq}")
    audio = match_frequency(audio, freq)
    segments = asr.transcribe(audio, batch_size=1)
    print()
    print(segments)

