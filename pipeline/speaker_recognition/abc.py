import soundfile as sf
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier, SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering # used directly w/o wrapper class

# TODO: Add types to class/function args
# TODO: Set up general class for different models later (if wanted/needed)

DESIRED_FREQUENCY = 16000

def load_from_file(file_path):
    if file_path.endswith(".wav"):
        audio, frequency = sf.read(file_path)
    elif file_path.endswith(".mp3") or file_path.endswith(".flac"):
        audio, frequency = torchaudio.load(file_path)
    else:
        raise ValueError(f"Unsupported audio format for file: {file_path}")

    return audio, frequency

def match_frequency(audio, frequency):
    if frequency != DESIRED_FREQUENCY:
        audio = torchaudio.functional.resample(audio, orig_freq=frequency, new_freq=DESIRED_FREQUENCY)
    return audio


class SpeechBrainEmbedding:
    def __init__(self, model="speechbrain/spkrec-ecapa-voxceleb"):
        self.model = EncoderClassifier.from_hparams(source=model)

    def embed(self, audio, frequency):
        audio = match_frequency(audio, frequency)
        embedding = self.model.encode_batch(audio)
        return embedding

    def embed_from_file(self, file_path):
        audio, frequency = load_from_file(file_path)
        return self.embed(audio, frequency)


class SpeechBrainSpeakerRecognition:
    def __init__(self, model="speechbrain/spkrec-ecapa-voxceleb"):
        self.model = SpeakerRecognition.from_hparams(source=model)

    def verify(self, audio1, frequency1, audio2, frequency2):
        audio1 = match_frequency(audio1, frequency1)
        audio2 = match_frequency(audio2, frequency2)
        score, prediction = self.model.verify_files(audio1, audio2)
        return score, prediction



