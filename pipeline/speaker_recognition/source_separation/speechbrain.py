from speechbrain.inference.separation import SepformerSeparation

class SpeechBrainSourceSeparation:
    def __init__(self, model="speechbrain/sepformer-wsj02mix"):
        self.model = SepformerSeparation.from_hparams(source=model)
    
    def separate(self, file_path):
        if not self.model:
            raise ValueError("Model is None.")

        separation = self.model.separate_file(file_path)
        return separation
