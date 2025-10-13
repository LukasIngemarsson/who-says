from enum import Enum
from loguru import logger

from pipeline.speaker_segmentation.SCD._pyannote import PyannoteSCD

class TypeSCD(Enum):
    PYANNOTE = "pyannote"

class SCD(object):
    def __init__(
        self,
        scd_type: TypeSCD = TypeSCD.PYANNOTE
    ):
        self.scd_pipeline = scd_type

    @property
    def scd_pipeline(self):
        return self._scd_pipeline

    @scd_pipeline.setter
    def scd_pipeline(self, scd_type: TypeSCD):
        match(scd_type):
            case TypeSCD.PYANNOTE:
                logger.info(f"Initializing SCD with type: {scd_type.value}")
                self._scd_pipeline = PyannoteSCD()
            case _:
                raise ValueError(f"Invalid SCD Type {scd_type}")

    def __call__(self, *args, **kwds):
        return self.scd_pipeline(*args, **kwds)
