from utils import load_audio_from_file, match_frequency

from speechbrain.inference.speaker import EncoderClassifier

# TODO: Add types to class/function args
# TODO: Set up general class for different models later (if wanted)

class SpeechBrainEmbedding:
    def __init__(self, model="speechbrain/spkrec-ecapa-voxceleb"):
        self.model = EncoderClassifier.from_hparams(source=model)


    def embed(self, audio, frequency):
        if not self.model:
            raise ValueError("Model is None.")

        audio = match_frequency(audio, frequency)
        embedding = self.model.encode_batch(audio)
        return embedding

    def embed_from_file(self, file_path):
        audio, frequency = load_audio_from_file(file_path)
        return self.embed(audio, frequency)
