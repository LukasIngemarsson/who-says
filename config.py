import torch
from dataclasses import dataclass, asdict

from utils.constants import SR, TENSOR_DTYPE
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
        class Whisper:
            model: str = "openai/whisper-large-v3-turbo"
            device: str = "cuda",
            torch_dtype: torch.dtype = TENSOR_DTYPE
    
    @dataclass
    class SCDConfig(BaseConfig):
        model: str = "pyannote/segmentation-3.0"
        onset: float = 0.5
        offset: float = 0.5
        min_duration: float = 0.0

    @dataclass
    class EmbeddingConfig(BaseConfig):
        class PyAnnote:
            model: str = "pyannote/embedding"
            batch_size: int = 1
        class SpeechBrain:
            model: str = "speechbrain/spkrec-ecapa-voxceleb"

    @dataclass
    class RecognitionConfig(BaseConfig):
        class SpeechBrain:
            model: str = "speechbrain/spkrec-ecapa-voxceleb"

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
        
    