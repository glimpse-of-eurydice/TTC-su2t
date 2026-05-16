import argparse
import os
import json
import random
import time

import joblib
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from sklearn.cluster import MiniBatchKMeans

from audio_utils import load_audio
from s2ut_config import add_num_clusters_arg, build_experiment_config, ensure_parent_dir

WAV_ZH_DIR = "./data/TCST/wav_zh"
TARGET_LAYER = 6  
SAMPLE_RATE = 16000
DEFAULT_SAMPLE_RATIO = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="Extract HuBERT units with configurable K-means.")
    add_num_clusters_arg(parser)
    parser.add_argument("--sample-ratio", type=float, default=DEFAULT_SAMPLE_RATIO)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_hubert(device):
    print("Loading HuBERT model...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    model.eval()
    return processor, model


def extract_features(audio_path, device, processor, model):
    """Extract HuBERT layer-6 hidden states for one audio file."""
    waveform, sr = load_audio(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values
    input_values = input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[TARGET_LAYER].squeeze(0) 

    return hidden_states.cpu().numpy()


def _fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def train_kmeans(audio_files, sample_ratio, batch_size, config, device, processor, model):
    sample_count = max(1, int(len(audio_files) * sample_ratio))
    print(f"\n[Stage 1/3] Feature extraction for K-means")
    print(f"  K={config.num_clusters}, sampling {sample_count}/{len(audio_files)} files")
    sampled_files = random.sample(audio_files, sample_count)

    t0 = time.time()
    features_list = []
    for f in tqdm(sampled_files, desc="  Extracting features", unit="file"):
        features_list.append(extract_features(f, device, processor, model))
    print(f"  Done in {_fmt(time.time() - t0)}")

    all_features = np.vstack(features_list)
    print(f"\n[Stage 2/3] K-means fitting on {all_features.shape[0]:,} frames")

    t0 = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=config.num_clusters,
        batch_size=batch_size,
        random_state=42,
        n_init="auto",
        verbose=1,
    )
    kmeans.fit(all_features)
    print(f"  Done in {_fmt(time.time() - t0)}")

    ensure_parent_dir(config.kmeans_model_path)
    joblib.dump(kmeans, config.kmeans_model_path)
    print(f"  Saved K-means model -> {config.kmeans_model_path}")
    return kmeans


def remove_duplicates(units):
    """Collapse consecutive duplicate units."""
    if not len(units):
        return []
    reduced = [units[0]]
    for u in units[1:]:
        if u != reduced[-1]:
            reduced.append(u)
    return reduced


def main():
    args = parse_args()
    if not 0 < args.sample_ratio <= 1:
        raise ValueError("--sample-ratio must be in the range (0, 1].")
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = build_experiment_config(args.num_clusters)
    device = get_device()
    print(f"Device : {device}")
    print(f"K      : {config.num_clusters}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    script_start = time.time()

    processor, model = load_hubert(device)
    audio_files = [os.path.join(WAV_ZH_DIR, f) for f in os.listdir(WAV_ZH_DIR) if f.endswith(".wav")]
    print(f"Found {len(audio_files)} valid Chinese audio files.")

    if args.force_retrain or not os.path.exists(config.kmeans_model_path):
        kmeans = train_kmeans(
            audio_files,
            args.sample_ratio,
            args.batch_size,
            config,
            device,
            processor,
            model,
        )
    else:
        kmeans = joblib.load(config.kmeans_model_path)
        print(f"Loaded existing K-means model from {config.kmeans_model_path}")

    print(f"\n[Stage 3/3] Discretizing all {len(audio_files)} audio files")
    t0 = time.time()
    dataset_units = {}
    for f in tqdm(audio_files, desc="  Discretizing", unit="file"):
        sample_id = os.path.basename(f).replace(".wav", "")
        feats = extract_features(f, device, processor, model)
        original_units = kmeans.predict(feats).tolist()
        reduced_units = remove_duplicates(original_units)

        dataset_units[sample_id] = {
            "original_length": len(original_units),
            "reduced_length": len(reduced_units),
            "units": reduced_units
        }
    print(f"  Done in {_fmt(time.time() - t0)}")

    ensure_parent_dir(config.units_json)
    with open(config.units_json, "w", encoding="utf-8") as f:
        json.dump(dataset_units, f, ensure_ascii=False, indent=2)
    print(f"\nAll done! Saved target units -> {config.units_json}")
    print(f"Total time: {_fmt(time.time() - script_start)}")

if __name__ == "__main__":
    main()
