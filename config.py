import torch
from dataclasses import dataclass, asdict, field
from utils.constants import SR
from pipeline.speaker_segmentation.SO.main import TypeSOD, TypeSOS
from pipeline.ASR import TypeASR


# -----------------------------
# Base
# -----------------------------
@dataclass
class BaseConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        """Convert to a nested dictionary (recursively)."""
        def _to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [_to_dict(i) for i in obj]
            else:
                return obj
        return _to_dict(self)


# -----------------------------
# Speaker Overlap (SO)
# -----------------------------
@dataclass
class SODetectionPyannoteConfig(BaseConfig):
    detection_type: TypeSOD = TypeSOD.PYANNOTE
    model: str = "pyannote/segmentation-3.0"
    onset: float = 0.5
    offset: float = 0.5
    min_duration: float = 0.0


@dataclass
class SOSeparationPyannoteConfig(BaseConfig):
    separation_type: TypeSOS = TypeSOS.PYANNOTE
    model_name: str = "pyannote/separation-ami-1.0"


@dataclass
class SOConfig:
    detection_pyannote: SODetectionPyannoteConfig = field(default_factory=SODetectionPyannoteConfig)
    separation_pyannote: SOSeparationPyannoteConfig = field(default_factory=SOSeparationPyannoteConfig)


# -----------------------------
# Speaker Change Detection (SCD)
# -----------------------------
@dataclass
class SCDPyannoteConfig(BaseConfig):
    model: str = "pyannote/segmentation-3.0"
    onset: float = 0.5
    offset: float = 0.5
    min_duration: float = 0.0


@dataclass
class SCDConfig:
    pyannote: SCDPyannoteConfig = field(default_factory=SCDPyannoteConfig)


# -----------------------------
# VAD
# -----------------------------

@dataclass
class VADSileroConfig(BaseConfig):
    model_repo: str = "snakers4/silero-vad"
    model_name: str = "silero_vad"
    sample_rate: int = SR
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float('inf')
    min_silence_duration_ms: int = 100
    window_size_samples: int = 512
    speech_pad_ms: int = 30
    return_seconds: bool = True


@dataclass
class VADPyannoteConfig(BaseConfig):
    model: str = "pyannote/segmentation-3.0"
    onset: float = 0.5
    offset: float = 0.5
    min_duration_on: float = 0.0
    min_duration_off: float = 0.0


@dataclass
class VADConfig:
    silero: VADSileroConfig = field(default_factory=VADSileroConfig)
    pyannote: VADPyannoteConfig = field(default_factory=VADPyannoteConfig)

# -----------------------------
# ASR
# -----------------------------
@dataclass
class ASRConfig(BaseConfig):
    asr_type: TypeASR = TypeASR.FASTER_WHISPER
    model: str = "medium" # "large-v3"# "large-v3-turbo" # "KBLab/kb-whisper-large" # openai/whisper-large-v3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "int8" # "float32"
    language: str = "en"
    
# -----------------------------
# Phoneme
# -----------------------------
@dataclass
class PhonemeSpeechbrainConfig(BaseConfig):
    model: str = "speechbrain/soundchoice-g2p"
    savedir: str = "pretrained_models/soundchoice-g2p"


@dataclass
class PhonemeConfig:
    speechbrain: PhonemeSpeechbrainConfig = field(default_factory=PhonemeSpeechbrainConfig)

# -----------------------------
# Speaker Embeddings
# -----------------------------
@dataclass
class EmbeddingPyannoteConfig(BaseConfig):
    model: str = "pyannote/embedding"
    batch_size: int = 1


@dataclass
class EmbeddingSpeechbrainConfig(BaseConfig):
    model: str = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass
class EmbeddingConfig:
    pyannote: EmbeddingPyannoteConfig = field(default_factory=EmbeddingPyannoteConfig)
    speechbrain: EmbeddingSpeechbrainConfig = field(default_factory=EmbeddingSpeechbrainConfig)


# -----------------------------
# Clustering
# -----------------------------
@dataclass
class BaseClusteringConfig(BaseConfig):
    def to_dict(self):
        """
        Convert to a nested dictionary, but exclude the 'device' field
        that scikit-learn doesn't accept.
        """
        # Call the parent's (BaseConfig) to_dict method
        data = super().to_dict()
        
        # Remove 'device' from the resulting dictionary
        data.pop('device', None)
        
        return data
            

@dataclass
class KMeansConfig(BaseClusteringConfig):
    algorithm: str = "kmeans"
    n_clusters: int = 8
    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300


@dataclass
class AgglomerativeConfig(BaseClusteringConfig):
    algorithm: str = "agglomerative"
    n_clusters: int = 2
    affinity: str = "euclidean"
    linkage: str = "ward"


@dataclass
class DBSCANConfig(BaseClusteringConfig):
    algorithm: str = "dbscan"
    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"


@dataclass
class ClusteringConfig:
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    agglomerative: AgglomerativeConfig = field(default_factory=AgglomerativeConfig)
    dbscan: DBSCANConfig = field(default_factory=DBSCANConfig)


# -----------------------------
# Recognition
# -----------------------------
@dataclass
class RecognitionSpeechbrainConfig(BaseConfig):
    model: str = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass
class RecognitionConfig:
    speechbrain: RecognitionSpeechbrainConfig = field(default_factory=RecognitionSpeechbrainConfig)


# -----------------------------
# PipelineConfig (main container)
# -----------------------------
@dataclass
class PipelineConfig(BaseConfig):
    sr: int = SR  # Sample Rate

    so: SOConfig = field(default_factory=SOConfig)
    scd: SCDConfig = field(default_factory=SCDConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    phoneme: PhonemeConfig = field(default_factory=PhonemeConfig)