import soundfile as sf
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier, SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering # used directly w/o wrapper class
from speechbrain.inference.separation import SepformerSeparation
import os
from dotenv import load_dotenv
from pyannote.audio import Model, Inference
from torch import Tensor

# TODO: Add types to class/function args
# TODO: Set up general class for different models later (if wanted/needed)

DESIRED_FREQUENCY = 16000

def load_from_file(file_path):
    if file_path.endswith(".wav"):
        audio, frequency = sf.read(file_path)
    elif file_path.endswith(".mp3") or file_path.endswith(".flac"):
        audio, frequency = torchaudio.load(file_path)
    else:
        raise ValueError(f"Unsupported audio format for file: {file_path}")

    return audio, frequency

def match_frequency(audio, frequency):
    if frequency != DESIRED_FREQUENCY:
        audio = torchaudio.functional.resample(audio, orig_freq=frequency, new_freq=DESIRED_FREQUENCY)
    return audio


class SpeechBrainEmbedding:
    def __init__(self, model="speechbrain/spkrec-ecapa-voxceleb"):
        self.model = EncoderClassifier.from_hparams(source=model)


    def embed(self, audio, frequency):
        if not self.model:
            raise ValueError("Model is None.")

        audio = match_frequency(audio, frequency)
        embedding = self.model.encode_batch(audio)
        return embedding

    def embed_from_file(self, file_path):
        audio, frequency = load_from_file(file_path)
        return self.embed(audio, frequency)
    

class PyAnnoteEmbedding:
    def __init__(self, model="pyannote/embedding"):
        load_dotenv("./.env")
        # .env is in root and not ignored
        token = os.getenv("HF_TOKEN_PYANNOTE_EMBEDDING")
        if token is None:
            raise ValueError("Missing HF_TOKEN_PYANNOTE_EMBEDDING in environment variables.")
        
        self.model = Model.from_pretrained(model, use_auth_token=token)
        self.inference = Inference(self.model, window="whole")

    def embed(self, audio, frequency):
        """Embed directly from audio tensor + sample rate"""
        audio = match_frequency(audio, frequency)

        if not isinstance(audio, Tensor):
            import torch
            audio = torch.tensor(audio, dtype=torch.float32)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # (1, num_samples)
        elif audio.ndim > 2:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")
        
        embedding = self.inference({"waveform": audio, "sample_rate": DESIRED_FREQUENCY})
        return embedding

    def embed_from_file(self, file_path):
        """Convenience wrapper for file-based input"""
        return self.inference(file_path)



class SpeechBrainSpeakerRecognition:
    def __init__(self, model="speechbrain/spkrec-ecapa-voxceleb"):
        self.model = SpeakerRecognition.from_hparams(source=model)

    def verify(self, audio1, frequency1, audio2, frequency2):
        if not self.model:
            raise ValueError("Model is None.")

        audio1 = match_frequency(audio1, frequency1)
        audio2 = match_frequency(audio2, frequency2)
        score, prediction = self.model.verify_files(audio1, audio2)
        return score, prediction


class SourceSeparation:
    def __init__(self, model="speechbrain/sepformer-wsj02mix"):
        self.model = SepformerSeparation.from_hparams(source=model)
    
    def separate(self, file_path):
        separation = self.model.separate_file(file_path)
        return separation