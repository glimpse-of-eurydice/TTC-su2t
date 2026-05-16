import argparse
import os
import json
import random
import pandas as pd

import _repo_path  # noqa: F401

JSON_PATH = "./TCST/text.json"
WAV_BASE_DIR = "./TCST/wav"

TRAIN_CSV = "./data/train.csv"
DEV_CSV = "./data/dev.csv"
TEST_CSV = "./data/test.csv"

from s2ut_config import add_num_clusters_arg, build_experiment_config


def parse_args():
    parser = argparse.ArgumentParser(description="Split the dataset using the extracted unit inventory.")
    add_num_clusters_arg(parser)
    return parser.parse_args()


def split_dataset():
    args = parse_args()
    config = build_experiment_config(args.num_clusters)

    # Load text metadata and extracted target units.
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    with open(config.units_json, "r", encoding="utf-8") as f:
        units_data = json.load(f)
        
    valid_records = []
    
    # Keep only samples that have target units and available Tibetan audio.
    for sample_id, info in raw_data.items():
        if sample_id in units_data:
            rel_audio_path = info.get("audio", "")
            audio_path = os.path.join(WAV_BASE_DIR, rel_audio_path)
            
            if os.path.exists(audio_path):
                valid_records.append({
                    "sample_id": sample_id,
                    "tibetan_audio": audio_path,
                    "chinese_text": info.get("text", {}).get("Chinese", ""),
                    "tibetan_text": info.get("text", {}).get("Tibetan", "")
                })

    print(f"Matched {len(valid_records)} aligned source-target records.")

    # Use a fixed seed so the split is reproducible.
    random.seed(42)
    random.shuffle(valid_records)
    
    total = len(valid_records)
    train_end = int(total * 0.8)
    dev_end = int(total * 0.9)
    
    train_data = valid_records[:train_end]
    dev_data = valid_records[train_end:dev_end]
    test_data = valid_records[dev_end:]
    
    pd.DataFrame(train_data).to_csv(TRAIN_CSV, index=False)
    pd.DataFrame(dev_data).to_csv(DEV_CSV, index=False)
    pd.DataFrame(test_data).to_csv(TEST_CSV, index=False)
    
    print("Dataset split saved as CSV files.")
    print(f"Train: {len(train_data)}")
    print(f"Dev  : {len(dev_data)}")
    print(f"Test : {len(test_data)}")

if __name__ == "__main__":
    split_dataset()
