from enum import Enum
from pathlib import Path
from typing import Optional, Union, Dict
from loguru import logger
from torch.cuda import is_available as is_cuda_available
from torch import device as torch_device
import torch

from pipeline.speaker_segmentation.SCD._pyannote import PyannoteSCD
from pipeline.speaker_segmentation.SCD._nemo import NemoSCD
from pipeline.speaker_segmentation.SCD._naive import NaiveSCD

class TypeSCD(Enum):
    PYANNOTE = "pyannote"
    NEMO = "nemo"
    NAIVE = "naive"

class SCD(object):
    def __init__(
        self,
        scd_type: TypeSCD = TypeSCD.PYANNOTE,
        device: str = "cuda" if is_cuda_available() else "cpu",
        model: str = "pyannote/segmentation-3.0",
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration: float = 0.0,
        min_prominence: float = 0.1,
        # Naive SCD specific parameters
        reference_dir: Optional[Union[str, Path]] = None,
        reference_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        embedding_model: str = "pyannote",
        window_duration: float = 1.0,
        step_duration: float = 0.5,
        similarity_threshold: float = 0.5
    ):
        # store configuration for the pipeline
        self.device = device
        self.model = model
        self.onset = onset
        self.offset = offset
        self.min_duration = min_duration
        self.min_prominence = min_prominence
        # Naive SCD config
        self.reference_dir = reference_dir
        self.reference_embeddings = reference_embeddings
        self.embedding_model = embedding_model
        self.window_duration = window_duration
        self.step_duration = step_duration
        self.similarity_threshold = similarity_threshold

        # initialize pipeline (uses the setter)
        self.scd_pipeline = scd_type

    @property
    def scd_pipeline(self):
        return self._scd_pipeline

    @scd_pipeline.setter
    def scd_pipeline(self, scd_type: TypeSCD):
        match scd_type:
            case TypeSCD.PYANNOTE:
                logger.info(f"Initializing SCD with type: {scd_type.value}")
                # pass stored config to PyannoteSCD if its constructor accepts them
                self._scd_pipeline = PyannoteSCD(
                    device=torch_device(self.device),
                    model_name=self.model,
                    onset=self.onset,
                    offset=self.offset,
                    min_duration=self.min_duration,
                    min_prominence=self.min_prominence,
                )
            case TypeSCD.NEMO:
                logger.info(f"Initializing SCD with type: {scd_type.value}")
                self._scd_pipeline = NemoSCD(
                    device=torch_device(self.device),
                    min_duration=self.min_duration,
                )
            case TypeSCD.NAIVE:
                logger.info(f"Initializing SCD with type: {scd_type.value}")
                self._scd_pipeline = NaiveSCD(
                    reference_dir=self.reference_dir,
                    reference_embeddings=self.reference_embeddings,
                    embedding_model=self.embedding_model,
                    window_duration=self.window_duration,
                    step_duration=self.step_duration,
                    similarity_threshold=self.similarity_threshold,
                    min_duration=self.min_duration,
                    device=torch_device(self.device),
                )
            case _:
                raise ValueError(f"Invalid SCD Type {scd_type}")

    def __call__(self, *args, **kwds):
        return self.scd_pipeline(*args, **kwds)
