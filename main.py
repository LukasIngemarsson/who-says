"""
Main pipeline test script

Usage:
    python main.py path/to/audio.wav
"""
import sys

from pipeline.VAD.silero import SileroVAD

if len(sys.argv) < 2:
    print("Usage: python main.py <audio_file>")
    sys.exit(1)

audio_path = sys.argv[1]

vad = SileroVAD()
segments = vad.predict(audio_path)

print(f"\nFound {len(segments)} speech segments:")
for i, seg in enumerate(segments, 1):
    duration = seg['end'] - seg['start']
    print(f"  {i}. {seg['start']:.2f}s - {seg['end']:.2f}s (duration: {duration:.2f}s)")

total_speech = sum(seg['end'] - seg['start'] for seg in segments)
print(f"\nTotal speech time: {total_speech:.2f}s")
