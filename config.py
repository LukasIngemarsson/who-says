import torch
from dataclasses import dataclass, asdict, field
from enum import Enum
from utils.constants import SR
from pipeline.speaker_segmentation.SO.main import TypeSOD, TypeSOS
from pipeline.speaker_segmentation.SCD.main import TypeSCD
from pipeline.ASR import TypeASR


class TypeClustering(Enum):
    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"
    DBSCAN = "dbscan"
    COSINE_SIMILARITY = "cosine_similarity"


class TypeEmbedding(Enum):
    PYANNOTE = "pyannote"
    SPEECHBRAIN = "speechbrain"
    WAV2VEC2 = "wav2vec2"


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
class SODetectionNemoConfig(BaseConfig):
    detection_type: TypeSOD = TypeSOD.NEMO
    model_name: str = "nvidia/diar_sortformer_4spk-v1"
    onset: float = 0.5
    offset: float = 0.5
    min_duration: float = 0.0
    max_chunk_duration: float = 30.0  # Process in 30-second chunks to avoid GPU OOM


@dataclass
class SOSeparationPyannoteConfig(BaseConfig):
    separation_type: TypeSOS = TypeSOS.PYANNOTE
    model_name: str = "pyannote/separation-ami-1.0"


@dataclass
class SOSeparationSpeechBrainConfig(BaseConfig):
    separation_type: TypeSOS = TypeSOS.SPEECHBRAIN
    model_name: str = "speechbrain/sepformer-wsj02mix"


@dataclass
class SOConfig:
    detection_type: TypeSOD = TypeSOD.NEMO
    separation_type: TypeSOS = TypeSOS.SPEECHBRAIN
    min_overlap_confidence: float = 0.7  # Minimum confidence to match separated speaker to cluster
    detection_pyannote: SODetectionPyannoteConfig = field(default_factory=SODetectionPyannoteConfig)
    detection_nemo: SODetectionNemoConfig = field(default_factory=SODetectionNemoConfig)
    separation_pyannote: SOSeparationPyannoteConfig = field(default_factory=SOSeparationPyannoteConfig)
    separation_speechbrain: SOSeparationSpeechBrainConfig = field(default_factory=SOSeparationSpeechBrainConfig)

    def get_detection_config(self):
        """Get the config for the selected detection type."""
        if self.detection_type == TypeSOD.PYANNOTE:
            return self.detection_pyannote
        elif self.detection_type == TypeSOD.NEMO:
            return self.detection_nemo
        else:
            raise ValueError(f"Unknown detection type: {self.detection_type}")

    def get_separation_config(self):
        """Get the config for the selected separation type."""
        if self.separation_type == TypeSOS.PYANNOTE:
            return self.separation_pyannote
        elif self.separation_type == TypeSOS.SPEECHBRAIN:
            return self.separation_speechbrain
        else:
            raise ValueError(f"Unknown separation type: {self.separation_type}")


# -----------------------------
# Speaker Change Detection (SCD)
# -----------------------------
@dataclass
class SCDPyannoteConfig(BaseConfig):
    model: str = "pyannote/segmentation-3.0"
    onset: float = 0.5
    offset: float = 0.5
    min_duration: float = 0.0
    min_prominence: float = 0.2


@dataclass
class SCDNemoConfig(BaseConfig):
    min_duration: float = 0.0


@dataclass
class SCDConfig:
    scd_type: TypeSCD = TypeSCD.NEMO
    pyannote: SCDPyannoteConfig = field(default_factory=SCDPyannoteConfig)
    nemo: SCDNemoConfig = field(default_factory=SCDNemoConfig)

    def get_config(self):
        """Get the config for the selected SCD type."""
        if self.scd_type == TypeSCD.PYANNOTE:
            return self.pyannote
        elif self.scd_type == TypeSCD.NEMO:
            return self.nemo
        else:
            raise ValueError(f"Unknown SCD type: {self.scd_type}")


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
    model: str = "small" # "large-v3"# "large-v3-turbo" # "KBLab/kb-whisper-large" # openai/whisper-large-v3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16"  # int8 can cause cuBLAS issues with alignment
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
class EmbeddingWav2Vec2Config(BaseConfig):
    model: str = "facebook/wav2vec2-base"


@dataclass
class EmbeddingConfig:
    embedding_type: TypeEmbedding = TypeEmbedding.SPEECHBRAIN
    pyannote: EmbeddingPyannoteConfig = field(default_factory=EmbeddingPyannoteConfig)
    speechbrain: EmbeddingSpeechbrainConfig = field(default_factory=EmbeddingSpeechbrainConfig)
    wav2vec2: EmbeddingWav2Vec2Config = field(default_factory=EmbeddingWav2Vec2Config)

    def get_config(self):
        """Get the config for the selected embedding type."""
        if self.embedding_type == TypeEmbedding.PYANNOTE:
            return self.pyannote
        elif self.embedding_type == TypeEmbedding.SPEECHBRAIN:
            return self.speechbrain
        elif self.embedding_type == TypeEmbedding.WAV2VEC2:
            return self.wav2vec2
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")


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
    eps: float = 0.3
    min_samples: int = 2
    metric: str = "cosine"


@dataclass
class CosineSimilarityConfig(BaseClusteringConfig):
    algorithm: str = "cosine_similarity"
    threshold: float = 0.7  # Minimum similarity to assign to a known speaker


@dataclass
class ClusteringConfig:
    clustering_type: TypeClustering = TypeClustering.KMEANS
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    agglomerative: AgglomerativeConfig = field(default_factory=AgglomerativeConfig)
    dbscan: DBSCANConfig = field(default_factory=DBSCANConfig)
    cosine_similarity: CosineSimilarityConfig = field(default_factory=CosineSimilarityConfig)

    def get_config(self):
        """Get the config for the selected clustering type."""
        if self.clustering_type == TypeClustering.KMEANS:
            return self.kmeans
        elif self.clustering_type == TypeClustering.AGGLOMERATIVE:
            return self.agglomerative
        elif self.clustering_type == TypeClustering.DBSCAN:
            return self.dbscan
        elif self.clustering_type == TypeClustering.COSINE_SIMILARITY:
            return self.cosine_similarity
        else:
            raise ValueError(f"Unknown clustering type: {self.clustering_type}")


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