from utils import match_frequency

import os
from dotenv import load_dotenv
from pyannote.audio import Model, Inference
from torch import Tensor


# TODO: Add types to class/function args
# TODO: Set up general class for different models later (if wanted)

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
        
        embedding = self.inference({"waveform": audio, "sample_rate": frequency})
        return embedding

    def embed_from_file(self, file_path):
        """Convenience wrapper for file-based input"""
        return self.inference(file_path)

