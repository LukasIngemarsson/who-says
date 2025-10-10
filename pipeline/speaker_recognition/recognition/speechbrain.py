from utils import match_frequency

from speechbrain.inference.speaker import SpeakerRecognition

# TODO: Add types to class/function args
# TODO: Set up general class for different models later (if wanted)

class SpeechBrainSpeakerRecognition:
    def __init__(self, model="speechbrain/spkrec-ecapa-voxceleb"):
        self.model = SpeakerRecognition.from_hparams(source=model)

    def verify(self, audio1, frequency1, audio2, frequency2):
        if not self.model:
            raise ValueError("Model is None.")

        audio1 = match_frequency(audio1, frequency1)
        audio2 = match_frequency(audio2, frequency2)
        score, prediction = self.model.verify_files(audio1, audio2)
        return score, prediction
