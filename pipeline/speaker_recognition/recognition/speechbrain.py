from utils import match_frequency, load_audio_from_file

from speechbrain.inference.speaker import SpeakerRecognition

# TODO: Set up general class for different models later (if wanted)

class SpeechBrainSpeakerRecognition:
    def __init__(self, model: str = "speechbrain/spkrec-ecapa-voxceleb") -> None:
        self.model = SpeakerRecognition.from_hparams(source=model)

    def verify(self, audio_path1: str, audio_path2: str) -> tuple:
        if not self.model:
            raise ValueError("Model is None.")

        score, prediction = self.model.verify_files(audio_path1, audio_path2)
        return score, prediction

if __name__ == "__main__":
    path_multi = "multi_speaker_sample.mp3"
    path_sep_1 = "sep_1.wav" #should be true
    path_sep_2 = "sep_2.wav" #should be true
    path_false = "single_speaker_sample.wav" #should be false
    recognizer = SpeechBrainSpeakerRecognition()
    score_1, prediction_1 = recognizer.verify(path_multi, path_sep_1)
    score_2, prediction_2 = recognizer.verify(path_multi, path_sep_2)
    score_false, prediction_false = recognizer.verify(path_multi, path_false)
    print(f"score 1: {score_1}, prediction 1: {prediction_1}") #tensor([0.3468]), tensor([True])
    print(f"score 2: {score_2}, prediction 2: {prediction_2}") #tensor([0.6039]), tensor([True])
    print(f"score false {score_false}, prediction false {prediction_false}") #score false tensor([0.1146]), prediction false tensor([False])
