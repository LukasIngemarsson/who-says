import os
import json
import argparse
import whisperx
import gc

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def transcribe_audio_batch(input_paths, model, diarize_model, batch_size, device):
    results = []
    for path in input_paths:
        audio = whisperx.load_audio(path)
        result = model.transcribe(audio, batch_size=batch_size)
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        results.append(result)
        gc.collect()
    return results

def main(input_dir, output_dir, batch_size):
    os.makedirs(output_dir, exist_ok=True)
    audio_files = []
    filenames = []
    for filename in os.listdir(input_dir):
        if filename.startswith('.'):
            continue
        input_path = os.path.join(input_dir, filename)
        if os.path.isfile(input_path):
            audio_files.append(input_path)
            filenames.append(filename)

    device = "cpu"
    model = whisperx.load_model("base", device=device, compute_type="int8")
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

    total_files = len(audio_files)
    print(f"transcribing {total_files} files in batches of {batch_size}...")

    for i in range(0, total_files, batch_size):
        batch_files = audio_files[i:i+batch_size]
        batch_names = filenames[i:i+batch_size]
        results = transcribe_audio_batch(batch_files, model, diarize_model, batch_size, device)
        for filename, result in zip(batch_names, results):
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"saved: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="batch whisperx transcription + diarization")
    parser.add_argument("input_dir", help="directory with audio files")
    parser.add_argument("output_dir", help="directory to save transcriptions")
    parser.add_argument("--batch_size", type=int, default=8, help="number of files per batch")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.batch_size)
