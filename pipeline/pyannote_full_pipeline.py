from pathlib import Path
from pyannote.audio import Pipeline
from typing import Dict, List
import torch
import os
from loguru import logger


class PyannoteFullPipeline:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipeline = None

    def load(self):
        """Load pyannote.audio 3.1 pipeline."""
        logger.info("Loading pyannote.audio 3.1 pipeline...")

        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            logger.warning("HF_TOKEN not found in environment. Pyannote pipeline may fail to load.")

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        if torch.cuda.is_available():
            self.pipeline.to(torch.device(self.device))

    def process(self, audio_path: Path) -> Dict:
        """
        Process audio file through pyannote pipeline.

        Returns:
            Dict with structure:
            {
                'segments': [
                    {
                        'start': float,
                        'end': float,
                        'speaker': str
                    }
                ]
            }
        """
        if self.pipeline is None:
            self.load()

        diarization = self.pipeline(str(audio_path))

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })

        return {'segments': segments}
