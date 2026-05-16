import argparse
import asyncio
import json
import time

import edge_tts
import pandas as pd
from tqdm import tqdm

from s2ut_config import add_num_clusters_arg, build_experiment_config, ensure_parent_dir

TRAIN_CSV = "./data/train.csv"
DEFAULT_UNITS_KEY = "test_audio"


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve the nearest training text for predicted units and synthesize audio.")
    add_num_clusters_arg(parser)
    parser.add_argument("--train-csv", default=TRAIN_CSV)
    parser.add_argument("--units-json", default=None)
    parser.add_argument("--predicted-units", default=None)
    parser.add_argument("--units-key", default=DEFAULT_UNITS_KEY)
    parser.add_argument("--output-audio", default=None)
    return parser.parse_args()


def load_predicted_units(path, units_key):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [int(x) for x in data]
    if isinstance(data, dict):
        if units_key in data and isinstance(data[units_key], list):
            return [int(x) for x in data[units_key]]
        for value in data.values():
            if isinstance(value, list):
                return [int(x) for x in value]
    raise ValueError(f"Could not read predicted units from {path}.")

def levenshtein_distance(s1, s2):
    """Compute edit distance between two unit sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

async def synthesize_text(text, output_file):
    """Synthesize Mandarin speech with Edge-TTS."""
    print(f"\nSynthesizing Mandarin text with Edge-TTS: {text!r}")
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    ensure_parent_dir(output_file)
    await communicate.save(output_file)
    print(f"Saved synthesized audio to: {output_file}")

def _fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def main():
    args = parse_args()
    config = build_experiment_config(args.num_clusters)
    units_json = args.units_json or config.units_json
    predicted_units_path = args.predicted_units or config.predicted_units_path
    output_audio = args.output_audio or config.output_audio_path
    predicted_units = load_predicted_units(predicted_units_path, args.units_key)

    print("Running KNN retrieval over target unit sequences...")
    print(f"Using K = {config.num_clusters}, predicted_units = {predicted_units_path}")
    script_start = time.time()

    train_data = pd.read_csv(args.train_csv)
    with open(units_json, 'r', encoding='utf-8') as f:
        units_dict = json.load(f)

    best_match_id = None
    min_distance = float('inf')

    t_knn = time.time()
    for idx, row in tqdm(train_data.iterrows(), total=len(train_data), desc="KNN search"):
        sample_id = str(row['sample_id'])
        if sample_id not in units_dict:
            continue

        ref_units = units_dict[sample_id]['units']
        dist = levenshtein_distance(predicted_units, ref_units)

        if dist < min_distance:
            min_distance = dist
            best_match_id = sample_id
    print(f"  KNN done in {_fmt(time.time() - t_knn)}")
            
    # Retrieve the Mandarin text attached to the closest unit sequence.
    best_row = train_data[train_data['sample_id'] == best_match_id].iloc[0]
    translated_text = best_row['chinese_text']
    
    print("\n" + "="*40)
    print(f"Predicted sequence length: {len(predicted_units)}")
    print(f"Nearest sample ID: {best_match_id}")
    print(f"Minimum edit distance: {min_distance}")
    print(f"Retrieved Mandarin text: {translated_text}")
    print("="*40)
    
    t_tts = time.time()
    asyncio.run(synthesize_text(translated_text, output_audio))
    print(f"  TTS done in {_fmt(time.time() - t_tts)}")
    print(f"Total time: {_fmt(time.time() - script_start)}")

if __name__ == "__main__":
    main()
