from enum import Enum
from loguru import logger

from pipeline.speaker_segmentation.SO.Detection import PyannoteSOD
from pipeline.speaker_segmentation.SO.Separation import PyannoteSOS

class TypeSOD(Enum):
    PYANNOTE = "pyannote"
    
class TypeSOS(Enum):
    PYANNOTE = "pyannote"
    
class SO(object):
    def __init__(
        self, config
    ):
        self.sod_pipeline = config.detection_pyannote
        self.sos_pipeline = config.separation_pyannote
        
    @property
    def sod_pipeline(self):
        return self._sod_pipeline

    @sod_pipeline.setter
    def sod_pipeline(self, sod_config):
        match(sod_config.detection_type):
            case TypeSOD.PYANNOTE:
                logger.info(f"Initializing SOD with type: {sod_config.detection_type.value}")
                self._sod_pipeline = PyannoteSOD(
                    **sod_config.to_dict()
                )
            case _:
                raise ValueError(f"Invalid SOD Type {sod_config.detection_type}")            
    
    @property
    def sos_pipeline(self):
        return self._sod_pipeline

    @sos_pipeline.setter
    def sos_pipeline(self, sos_config):
        match(sos_config.separation_type):
            case TypeSOS.PYANNOTE:
                logger.info(f"Initializing SOS with type: {sos_config.separation_type}")
                self._sod_pipeline = PyannoteSOS(
                    **sos_config.to_dict()
                )
            case _:
                raise ValueError(f"Invalid SOD Type {sos_config.separation_type}")            
    
    
    def __call__(self, *args, **kwds):
        overlapped_segments = self.sod_pipeline(*args, **kwds)
        seperated_segments = self.sos_pipeline(
            overlap_segments=overlapped_segments,
            *args, **kwds
        )
        
        return seperated_segments
