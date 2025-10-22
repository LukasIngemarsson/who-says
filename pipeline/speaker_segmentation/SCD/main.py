from enum import Enum
from loguru import logger
from torch.cuda import is_available as is_cuda_available
from torch import device as torch_device

from pipeline.speaker_segmentation.SCD._pyannote import PyannoteSCD

class TypeSCD(Enum):
    PYANNOTE = "pyannote"

class SCD(object):
    def __init__(
        self,
        scd_type: TypeSCD = TypeSCD.PYANNOTE,
        device: str = "cuda" if is_cuda_available() else "cpu",
        model: str = "pyannote/segmentation-3.0",
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration: float = 0.0
    ):
        # store configuration for the pipeline
        self.device = device
        self.model = model
        self.onset = onset
        self.offset = offset
        self.min_duration = min_duration

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
                )
            case _:
                raise ValueError(f"Invalid SCD Type {scd_type}")

    def __call__(self, *args, **kwds):
        return self.scd_pipeline(*args, **kwds)
