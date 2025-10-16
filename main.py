from loguru import logger
import argparse
from pathlib import Path

from dotenv import load_dotenv

from pipeline.ASR import WhisperASR
from pipeline.speaker_recognition.clustering.sklearn import SklearnClustering
from pipeline.speaker_segmentation import SO, SCD
from pipeline.speaker_recognition import AgglomerativeClustering
from pipeline.speaker_recognition import SpeechBrainEmbedding
from pipeline.speaker_recognition import SpeechBrainSpeakerRecognition
from pipeline.speaker_segmentation import SileroVAD
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
        
        self.vad = SileroVAD(self.config.sr)
        self.asr = WhisperASR(self.config.ASRConfig.Whisper().to_dict())

        self.embedder = SpeechBrainEmbedding(**self.config.EmbeddingConfig.SpeechBrain.to_dict())
        self.clustering = SklearnClustering(**self.config.ClusteringConfig.KMeans.to_dict()) 
        self.recognition = SpeechBrainSpeakerRecognition(**self.config.RecognitionConfig.SpeechBrain.to_dict())
    
    def __call__(
        self,
        audio_file: str,
        num_speakers: int = 2
    ):
        waveform, sr = load_audio_from_file(
            file_path=audio_file,
            sr=self.config.sr
        )
        
        print(waveform)
        
        speech_segments = self.vad(waveform) # [{'start': 0.7, 'end': 3.5}, {'start': 4.0, 'end': 4.8}]

        print("VAD speech segments: ", speech_segments)

        # TODO: handle so that input params match output of previous component
        # transriptions = self.asr.transcribe(audio, return_timestamps, language)
        
        seperated_segments = self.sod(
            waveform,
            sample_rate=sr
        ) # []
        
        segments = self.scd(
            waveform,
            sample_rate=sr 
        ) # 

        print("Segements after speaker segmentation:", segments)
        
        segment_embeddings = self.embedder.embed_segments(waveform, sr, segments)

        print("Embeddings shape:", segment_embeddings.shape)
        
        # TODO: set up clustering alg. w/ keyword params
        # segment_clusters = self.clustering.cluster()

        # TODO: change so that the audio object is passed instead of file paths
        # recognized_speakers = self.recognition.verify(file1_path, file2_path)
        

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
    
