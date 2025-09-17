import whisperx
import gc
import os
from dotenv import load_dotenv

# huggingface token stored in .env file
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


"""

Both the diarization and segmentation models that are used in the background
in this script (which is taken from the README of the WhsiperX repo) 
had some conditions that you had to manually accept:
    https://huggingface.co/pyannote/speaker-diarization-3.1
    https://huggingface.co/pyannote/segmentation-3.0


"""

device = "cpu"
audio_file = "multi_speaker_sample.mp3"
batch_size = 4 # reduce if low on GPU mem
# compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
compute_type = "int8" # for testing on Mac OSX
output_dir = "output/"

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("base", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, audio_file.split('.')[0] + ".txt"), 'w') as f:
    f.write(str(result["segments"]))

# next: record diarization error rate (DER) and word error rate (WER)

from pyannote.metrics.detection import DetectionErrorRate
# word error rate from lib or define own function?