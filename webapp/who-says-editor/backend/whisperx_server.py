import os
import gc
import tempfile
import traceback
from typing import Optional

import torch
import whisperx

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# -----------------------------
# Config
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
COMPUTE_TYPE = "int8"  # good default for CPU or GPU
LANGUAGE = "sv"        # Swedish as in your original script

# -----------------------------
# FastAPI app + CORS
# -----------------------------

app = FastAPI()

# Allow the Vite dev server to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # adjust if frontend runs elsewhere
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load WhisperX model once
# -----------------------------

print(f"Loading WhisperX model on {DEVICE}...")
base_model = whisperx.load_model(
    "large-v2",
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    language=LANGUAGE,
)


def cleanup_gpu():
  if DEVICE == "cuda":
      gc.collect()
      torch.cuda.empty_cache()
  else:
      gc.collect()


# -----------------------------
# Annotation endpoint
# -----------------------------

@app.post("/annotate")
async def annotate(audio: UploadFile = File(...)):
    """
    Accepts an audio file upload and returns WhisperX's JSON result:
    {
      "segments": [...],
      "text": "...",
      "language": "...",
      ...
    }

    NOTE: This version does *not* do diarization (no speakers),
    only transcription + alignment + word timings.
    """
    # Save uploaded file to a temp path
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        content = await audio.read()
        tmp.write(content)

    try:
        print(f"Transcribing: {audio.filename} -> {temp_path}")
        # 1. Transcribe
        result = base_model.transcribe(temp_path, batch_size=BATCH_SIZE)

        # 2. Alignment (adds per-word timestamps)
        audio_data = whisperx.load_audio(temp_path)
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=DEVICE,
        )

        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_data,
            device=DEVICE,
        )

        del model_a
        cleanup_gpu()

        # No diarization here to avoid pyannote / torchvision issues
        result_final = result_aligned
        return result_final

    except Exception as e:
        # print full stack trace to the terminal so we can see what's wrong
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Annotation failed: {e}") from e

    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass
