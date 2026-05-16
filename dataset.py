import argparse
import json

import pandas as pd
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from audio_utils import load_audio
from s2ut_config import add_num_clusters_arg, build_experiment_config

TRAIN_CSV = "./data/train.csv"
SAMPLE_RATE = 16000


class S2UTDataset(Dataset):
    def __init__(self, csv_file, units_json, bos_token, eos_token):
        self.data = pd.read_csv(csv_file)
        self.bos_token = bos_token
        self.eos_token = eos_token

        with open(units_json, "r", encoding="utf-8") as f:
            self.units_dict = json.load(f)

        self.valid_data = []
        for _, row in self.data.iterrows():
            sample_id = str(row["sample_id"])
            if sample_id in self.units_dict:
                self.valid_data.append(row)

        print(f"Loaded {len(self.valid_data)} aligned examples.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        row = self.valid_data[idx]
        sample_id = str(row["sample_id"])
        audio_path = row["tibetan_audio"]

        waveform, sr = load_audio(audio_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=SAMPLE_RATE,
            )(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            sample_frequency=SAMPLE_RATE,
        )

        units = self.units_dict[sample_id]["units"]
        units_with_special_tokens = [self.bos_token] + units + [self.eos_token]
        units_tensor = torch.tensor(units_with_special_tokens, dtype=torch.long)

        return fbank, units_tensor


def make_collate_fn(pad_token):
    def collate_fn(batch):
        fbanks = [item[0] for item in batch]
        units = [item[1] for item in batch]

        fbanks_padded = pad_sequence(fbanks, batch_first=True, padding_value=0.0)
        units_padded = pad_sequence(units, batch_first=True, padding_value=pad_token)

        return fbanks_padded, units_padded

    return collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test the S2UT dataset pipeline.")
    add_num_clusters_arg(parser)
    parser.add_argument("--csv-file", default=TRAIN_CSV)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_experiment_config(args.num_clusters)

    dataset = S2UTDataset(
        args.csv_file,
        config.units_json,
        config.bos_token,
        config.eos_token,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(config.pad_token),
    )

    for fbanks, units in dataloader:
        print("\n=== DataLoader smoke test passed ===")
        print(f"K = {config.num_clusters}, vocab_size = {config.vocab_size}")
        print(f"Tibetan speech feature shape (batch, time, mel bins): {fbanks.shape}")
        print(f"Target unit shape (batch, sequence length): {units.shape}")
        print(f"Example unit sequence:\n{units[0]}")
        break


if __name__ == "__main__":
    main()
