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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load WhisperX ASR model once
# -----------------------------

print(f"Loading WhisperX ASR model on {DEVICE}...")
base_model = whisperx.load_model(
    "large-v2",
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    language=LANGUAGE,
)

# -----------------------------
# Load diarization model (lazy)
# -----------------------------

from whisperx.diarize import DiarizationPipeline

diarize_model: Optional[DiarizationPipeline] = None


def get_diarize_model() -> DiarizationPipeline:
    global diarize_model

    if diarize_model is None:
        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is not set in the environment, "
                "but diarization was requested."
            )
        print(f"Loading diarization model on {DEVICE}...")
        diarize_model = DiarizationPipeline(
            use_auth_token=HF_TOKEN,
            device=DEVICE,
        )
    return diarize_model


def cleanup_gpu():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# -----------------------------
# Annotation endpoint
# -----------------------------

@app.post("/annotate")
async def annotate(
    audio: UploadFile = File(...),
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
):
    """
    Accepts an audio file upload and returns WhisperX's JSON result with
    transcription, alignment, AND speaker diarization.

    Result example:
    {
      "segments": [
        {
          "id": 0,
          "start": 0.5,
          "end": 3.2,
          "text": "...",
          "speaker": "SPEAKER_00",
          "words": [
            {"word": "Hej", "start": 0.6, "end": 0.9, "speaker": "SPEAKER_00"},
            ...
          ]
        },
        ...
      ],
      "text": "...",
      "language": "sv"
    }
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
        result = base_model.transcribe(
            temp_path,
            batch_size=BATCH_SIZE,
        )

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

        # Free alignment model from GPU
        del model_a
        cleanup_gpu()

        # 3. Diarization
        diarizer = get_diarize_model()

        # You can pass min/max_speakers from the frontend if you know them
        if min_speakers is not None or max_speakers is not None:
            diarize_segments = diarizer(
                audio_data,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        else:
            diarize_segments = diarizer(audio_data)

        # 4. Assign speakers to words/segments
        result_with_speakers = whisperx.assign_word_speakers(
            diarize_segments,
            result_aligned,
        )

        cleanup_gpu()

        return result_with_speakers

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Annotation failed: {e}"
        ) from e

    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
