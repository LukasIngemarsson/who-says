import torch
from dataclasses import dataclass, asdict

from utils.constants import SR
from pipeline.speaker_segmentation.SO.main import TypeSOD, TypeSOS

@dataclass
class BaseConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}

@dataclass
class PipelineConfig:
    sr: int = SR # Sample Rate (Hz)
    
    @dataclass
    class ASRConfig(BaseConfig):
        model: str = "openai/whisper-large-v3-turbo"
    
    @dataclass
    class SCDConfig(BaseConfig):
        model: str = "pyannote/segmentation-3.0"
        onset: float = 0.5
        offset: float = 0.5
        min_duration: float = 0.0

    class SOConfig:
        @dataclass
        class Detection(BaseConfig):
            detection_type: TypeSOD = TypeSOD.PYANNOTE
            model: str = "pyannote/segmentation-3.0"
            onset: float = 0.5
            offset: float = 0.5
            min_duration: float = 0.0

        @dataclass
        class Seperation(BaseConfig):
            seperation_type: TypeSOS = TypeSOS.PYANNOTE
            model_name: str = "pyannote/separation-ami-1.0"
        
    