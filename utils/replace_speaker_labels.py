import os
import json
import argparse

def replace_speaker_labels(input_folder, output_dir, new_label):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for segment in data.get('segments', []):
                segment['speaker'] = new_label
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    # Example usage: python3 replace_speaker_labels.py input_folder --output_folder output_folder --label "MATTIAS"
    parser = argparse.ArgumentParser(description="Replace speaker labels in JSON files.")
    parser.add_argument('input_folder', help='Folder containing input JSON files')
    parser.add_argument('--output-dir', required=True, help='Directory to save modified JSON files')
    parser.add_argument('--label', required=True, help='New speaker label string')
    args = parser.parse_args()
    replace_speaker_labels(args.input_folder, args.output_dir, args.label)

if __name__ == '__main__':
    main()
