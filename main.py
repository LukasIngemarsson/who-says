from loguru import logger
import argparse
from pathlib import Path

from dotenv import load_dotenv

from pipeline.speaker_segmentation import SO, SCD
from utils import load_audio_from_file
from config import PipelineConfig as config

load_dotenv(".env")

class WhoSays(object):
    def __init__(self):
        self.config = config()
        
        self.sod = SO(
            self.config.SOConfig()
        )
        self.scd = SCD(
            **self.config.SCDConfig().to_dict()
        )
        self.vad = ...
        self.asr = ...
    
    def __call__(
        self,
        audio_file: str,
        num_speakers: int
    ):
        waveform, sr = load_audio_from_file(
            file_path=audio_file,
            sr=self.config.sr
        )
        
        """
        seperated_segments = self.sod(
            waveform,
            sample_rate=sr
        )
        
        segments = self.scd(
            waveform,
            sample_rate=sr 
        )
        """
        
        #print(segments)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhoSays - Speaker diarization pipeline")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to process")
    parser.add_argument("--num-speakers", type=int, default=2, help="Expected number of speakers (default: 2)")

    args = parser.parse_args()

    if not args.audio_file.exists():
        parser.error(f"Audio file not found: {args.audio_file}")

    pipeline = WhoSays()

    logger.info(f"Processing: {args.audio_file}, num speakers: {args.num_speakers}")
    
    output = pipeline(
        args.audio_file,
        args.num_speakers
    )
    
