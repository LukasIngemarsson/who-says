"""KB-Whisper Swedish transcription (GPU).

Setup:
  python3 -m venv venv && source venv/bin/activate
  pip install faster-whisper
  pip install torch --index-url https://download.pytorch.org/whl/cu121

Run (must set LD_LIBRARY_PATH for cuDNN):
  export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib:$LD_LIBRARY_PATH
  python3 utils/run_kb_whisper.py <input_dir> <output_dir>
"""
import os
import json
import argparse
from faster_whisper import WhisperModel

MODEL_ID = "KBLab/kb-whisper-large"

def transcribe_dir(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    model = WhisperModel(
        MODEL_ID,
        device="cuda",
        compute_type="float16",
        download_root="cache"
    )

    audio_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(('.mp3', '.wav', '.flac')) and not f.startswith('.')
    ])

    print(f"Transcribing {len(audio_files)} files with {MODEL_ID}...")

    for filename in audio_files:
        input_path = os.path.join(input_dir, filename)
        segments_list, info = model.transcribe(
            input_path,
            language="sv",
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        segments = []
        for seg in segments_list:
            segment_data = {"start": seg.start, "end": seg.end, "text": seg.text}
            if seg.words:
                segment_data["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end, "score": w.probability}
                    for w in seg.words
                ]
            segments.append(segment_data)

        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"segments": segments}, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KB-Whisper Swedish transcription")
    parser.add_argument("input_dir", help="Directory with audio files")
    parser.add_argument("output_dir", help="Directory to save JSON transcriptions")
    args = parser.parse_args()
    transcribe_dir(args.input_dir, args.output_dir)
