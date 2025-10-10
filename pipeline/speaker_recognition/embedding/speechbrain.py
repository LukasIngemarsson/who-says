from utils import load_audio_from_file, match_frequency

from speechbrain.inference.speaker import EncoderClassifier
import torch

# TODO:Enforce specific format for audio tensors? 

class SpeechBrainEmbedding:
    def __init__(self, model: str = "speechbrain/spkrec-ecapa-voxceleb") -> None:
        self.model = EncoderClassifier.from_hparams(source=model)


    def embed(self, audio: torch.Tensor, frequency: int) -> torch.Tensor:
        if not self.model:
            raise ValueError("Model is None.")

        audio = match_frequency(audio, frequency)
        embedding = self.model.encode_batch(audio)
        return embedding

    def embed_from_file(self, file_path: str) -> torch.Tensor:
        audio, frequency = load_audio_from_file(file_path)
        return self.embed(audio, frequency)


if __name__ == "__main__":
    embedder = SpeechBrainEmbedding()
    file_path = "single_speaker_sample.wav"
    audio, freq = load_audio_from_file(file_path)
    print(f"audio dim: {audio.shape}, freq: {freq}")
    embd = embedder.embed(audio, freq)
    print(f"embedding dim: {embd.shape}")
