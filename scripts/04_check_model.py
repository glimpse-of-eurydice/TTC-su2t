import argparse

import torch

import _repo_path  # noqa: F401
from model import S2UTModel
from s2ut_config import add_num_clusters_arg, build_experiment_config


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test the S2UT model forward pass.")
    add_num_clusters_arg(parser)
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    config = build_experiment_config(args.num_clusters)
    device = get_device()
    print(f"Testing model on {device}...")

    dummy_src = torch.randn(4, 706, 80).to(device)
    dummy_tgt = torch.randint(0, config.vocab_size, (4, 152)).to(device)

    model = S2UTModel(vocab_size=config.vocab_size).to(device)
    output = model(dummy_src, dummy_tgt[:, :-1])

    print("=== Model forward pass succeeded ===")
    print(f"Output shape (batch, target sequence length, vocabulary size): {output.shape}")
    print(f"Expected shape: (4, 151, {config.vocab_size})")


if __name__ == "__main__":
    main()
