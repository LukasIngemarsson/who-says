import torch
from dataclasses import dataclass, asdict, field
from utils.constants import SR, TENSOR_DTYPE
from pipeline.speaker_segmentation.SO.main import TypeSOD, TypeSOS


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
class SomeVADLibraryConfig(BaseConfig):
    raise NotImplementedError


@dataclass
class VADConfig:
    raise NotImplementedError

# -----------------------------
# ASR
# -----------------------------
@dataclass
class ASRWhisperConfig(BaseConfig):
    model: str = "openai/whisper-large-v3-turbo"
    device: str = "cuda"
    torch_dtype: torch.dtype = TENSOR_DTYPE


@dataclass
class ASRConfig:
    whisper: ASRWhisperConfig = field(default_factory=ASRWhisperConfig)

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
class KMeansConfig:
    n_clusters: int = 8
    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300


@dataclass
class AgglomerativeConfig:
    n_clusters: int = 2
    affinity: str = "euclidean"
    linkage: str = "ward"


@dataclass
class DBSCANConfig:
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
class PipelineConfig:
    sr: int = SR  # Sample Rate

    so: SOConfig = field(default_factory=SOConfig)
    scd: SCDConfig = field(default_factory=SCDConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
