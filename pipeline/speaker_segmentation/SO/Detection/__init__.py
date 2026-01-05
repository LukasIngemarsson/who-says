from pipeline.speaker_segmentation.SO.Detection._pyannote import PyannoteSOD
from pipeline.speaker_segmentation.SO.Detection._nemo import NemoSOD
from pipeline.speaker_segmentation.SO.Detection._wavlm import WavLMSOD

__all__ = [
    "PyannoteSOD",
    "NemoSOD",
    "WavLMSOD"
]
