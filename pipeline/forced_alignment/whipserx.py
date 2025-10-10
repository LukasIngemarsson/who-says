import whisperx

# TODO: Specify types to function/class args

class WhsiperXForcedAlignment:
    def __init__(self, language_code="", device="cuda"):
        self.device = device
        self.model, self.metadata = whisperx.load_align_model(language_code=language_code, device=self.device)

    def align(self, segments, audio):
        result = whisperx.align(segments, self.model, self.metadata, audio, self.device, return_char_alignments=False)
        return result
