import torch
from dataclasses import dataclass, asdict, field
from utils.constants import SR, TENSOR_DTYPE
from pipeline.speaker_segmentation.SO.main import TypeSOD, TypeSOS


# -----------------------------
# Base and Sub-Configs
# -----------------------------

@dataclass
class BaseConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}


# --- ASR ---
@dataclass
class ASRWhisperConfig(BaseConfig):
    model: str = "openai/whisper-large-v3-turbo"
    device: str = "cuda"
    torch_dtype: torch.dtype = TENSOR_DTYPE


# --- SCD ---
@dataclass
class SCDConfig(BaseConfig):
    model: str = "pyannote/segmentation-3.0"
    onset: float = 0.5
    offset: float = 0.5
    min_duration: float = 0.0


# --- Embeddings ---
@dataclass
class EmbeddingPyAnnoteConfig(BaseConfig):
    model: str = "pyannote/embedding"
    batch_size: int = 1


@dataclass
class EmbeddingSpeechBrainConfig(BaseConfig):
    model: str = "speechbrain/spkrec-ecapa-voxceleb"


# --- Clustering ---
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


# --- Recognition ---
@dataclass
class RecognitionSpeechBrainConfig(BaseConfig):
    model: str = "speechbrain/spkrec-ecapa-voxceleb"


# --- Speaker Overlap (SO) ---
@dataclass
class SODetectionConfig(BaseConfig):
    detection_type: TypeSOD = TypeSOD.PYANNOTE
    model: str = "pyannote/segmentation-3.0"
    onset: float = 0.5
    offset: float = 0.5
    min_duration: float = 0.0


@dataclass
class SOSeparationConfig(BaseConfig):
    separation_type: TypeSOS = TypeSOS.PYANNOTE
    model_name: str = "pyannote/separation-ami-1.0"


# -----------------------------
# PipelineConfig (container)
# -----------------------------
@dataclass
class PipelineConfig:
    sr: int = SR

    asr: ASRWhisperConfig = field(default_factory=ASRWhisperConfig)
    scd: SCDConfig = field(default_factory=SCDConfig)
    embedding_pyan: EmbeddingPyAnnoteConfig = field(default_factory=EmbeddingPyAnnoteConfig)
    embedding_sb: EmbeddingSpeechBrainConfig = field(default_factory=EmbeddingSpeechBrainConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    recognition_sb: RecognitionSpeechBrainConfig = field(default_factory=RecognitionSpeechBrainConfig)
    so_detection: SODetectionConfig = field(default_factory=SODetectionConfig)
    so_separation: SOSeparationConfig = field(default_factory=SOSeparationConfig)
