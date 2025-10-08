import whisperx

# TODO: Specify types for function/class args

class WhsiperXASR:
    def __init__(self, model="base", device="cuda", compute_type="float16"):
        self.model = whipserx.load_model(model, device, compute_type=compute_type)

    def transcribe(self, audio, batch_size):
        result = self.model.transcribe(audio, batch_size=batch_size)
        return result
