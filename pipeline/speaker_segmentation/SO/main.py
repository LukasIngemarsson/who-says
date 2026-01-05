from enum import Enum
from loguru import logger

from pipeline.speaker_segmentation.SO.Detection import PyannoteSOD, NemoSOD
from pipeline.speaker_segmentation.SO.Separation import PyannoteSOS, SpeechBrainSOS

class TypeSOD(Enum):
    PYANNOTE = "pyannote"
    NEMO = "nemo"

class TypeSOS(Enum):
    PYANNOTE = "pyannote"
    SPEECHBRAIN = "speechbrain"

class SO(object):
    def __init__(
        self, config
    ):
        self.sod_pipeline = config.get_detection_config()
        self.sos_pipeline = config.get_separation_config()

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
            case TypeSOD.NEMO:
                logger.info(f"Initializing SOD with type: {sod_config.detection_type.value}")
                self._sod_pipeline = NemoSOD(
                    **sod_config.to_dict()
                )
            case _:
                raise ValueError(f"Invalid SOD Type {sod_config.detection_type}")            
    
    @property
    def sos_pipeline(self):
        return self._sos_pipeline

    @sos_pipeline.setter
    def sos_pipeline(self, sos_config):
        match(sos_config.separation_type):
            case TypeSOS.PYANNOTE:
                logger.info(f"Initializing SOS with type: {sos_config.separation_type}")
                self._sos_pipeline = PyannoteSOS(
                    **sos_config.to_dict()
                )
            case TypeSOS.SPEECHBRAIN:
                logger.info(f"Initializing SOS with type: {sos_config.separation_type}")
                self._sos_pipeline = SpeechBrainSOS(
                    **sos_config.to_dict()
                )
            case _:
                raise ValueError(f"Invalid SOS Type {sos_config.separation_type}")            
    
    
    def __call__(self, waveform, sample_rate: int):
        """
        Detect and separate overlapping speakers.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform
        sample_rate : int
            Sample rate of the audio

        Returns
        -------
        dict with:
            - overlap_segments: List of (start, end) tuples where overlap occurs
            - separated_regions: Dict mapping each overlap segment to separated speaker waveforms
        """
        # Step 1: Detect where overlaps occur
        overlap_segments = self.sod_pipeline(waveform)

        # Step 2: Separate speakers in those regions
        separated_regions = {}
        if overlap_segments:
            separated_regions = self.sos_pipeline.separate_regions(
                waveform=waveform,
                sample_rate=sample_rate,
                overlap_segments=overlap_segments
            )

        return {
            'overlap_segments': overlap_segments,
            'separated_regions': separated_regions
        }
