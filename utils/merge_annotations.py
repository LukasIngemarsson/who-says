import os
import sys
import json
import glob
from collections import defaultdict
import re

def collect_files_by_part(input_dir):
    files_by_part = defaultdict(list)
    for file_path in glob.glob(os.path.join(input_dir, "*.json")):
        base = os.path.basename(file_path)
        match = re.search(r'(\d+)', base)
        if match:
            part = match.group(1)
            files_by_part[part].append(file_path)
    return files_by_part

def merge_json_files(file_paths):
    all_segments = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            segments = data.get("segments", [])
            all_segments.extend(segments)

    all_segments.sort(key=lambda seg: seg["start"])
    return {
        "segments": all_segments,
    }

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files_by_part = collect_files_by_part(input_dir)

    for part, files in files_by_part.items():
        merged = merge_json_files(files)
        out_path = os.path.join(output_dir, f"{part}.json")
        with open(out_path, "w", encoding="utf-8") as out_file:
            json.dump(merged, out_file, indent=2)
        print(f"Merged {len(files)} files into {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_annotations.py <input_dir> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
