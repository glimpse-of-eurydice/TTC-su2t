import argparse
import json
import time
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import sacrebleu
import torch
import torchaudio
from tqdm import tqdm

import _repo_path  # noqa: F401
from audio_utils import load_audio
from checkpoint_utils import load_checkpoint_into_model
from model import S2UTModel
from s2ut_config import add_max_len_arg, add_num_clusters_arg, build_experiment_config

TEST_CSV = "./data/test.csv"
SAMPLE_RATE = 16000
WEIGHTS_TO_TEST = [0.0, 0.2, 0.4, 0.6, 0.8]


def parse_args():
    parser = argparse.ArgumentParser(description="Run LM-weight ablations for a chosen K.")
    add_num_clusters_arg(parser)
    add_max_len_arg(parser)
    parser.add_argument("--units-json", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-csv", default=TEST_CSV)
    return parser.parse_args()


def _fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_ngram_lm(units_dict, bos_token, eos_token, n=3):
    print(f"Building {n}-gram unit language model...")
    lm_probs = defaultdict(Counter)
    for _, info in units_dict.items():
        units = [bos_token] + info["units"] + [eos_token]
        for i in range(len(units) - n + 1):
            history = tuple(units[i : i + n - 1])
            target = units[i + n - 1]
            lm_probs[history][target] += 1
    for history, targets in lm_probs.items():
        total = sum(targets.values())
        for tgt in targets:
            lm_probs[history][tgt] /= total
    print("N-gram LM ready.")
    return lm_probs


def get_lm_prob(lm_probs, current_tokens, vocab_size, n=3):
    lm_logits = torch.ones(vocab_size) * 1e-9
    if len(current_tokens) >= n - 1:
        history = tuple(current_tokens[-(n - 1) :])
        if history in lm_probs:
            for tgt, prob in lm_probs[history].items():
                lm_logits[tgt] = prob
    return lm_logits


def evaluate_with_weight(weight, model, lm_probs, test_data, units_dict, device, config, max_len):
    predictions = []
    references = []

    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc=f"Testing Weight {weight}"):
        sample_id = str(row["sample_id"])
        if sample_id not in units_dict:
            continue

        ref_units = units_dict[sample_id]["units"]
        references.append(" ".join(map(str, ref_units)))

        waveform, sr = load_audio(row["tibetan_audio"])
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, sample_frequency=SAMPLE_RATE)
        src = fbank.unsqueeze(0).to(device)

        tgt_tokens = [config.bos_token]
        with torch.no_grad():
            src_encoded, _ = model.subsampler(src, None)
            src_encoded = model.pos_encoder(src_encoded)

            for _ in range(max_len):
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
                tgt_emb = model.unit_embedding(tgt_tensor) * (model.d_model ** 0.5)
                tgt_emb = model.pos_decoder(tgt_emb)
                tgt_mask = model.generate_square_subsequent_mask(tgt_emb.size(1), device)

                out = model.transformer(src_encoded, tgt_emb, tgt_mask=tgt_mask)
                logits = model.fc_out(out)
                tm_probs = torch.softmax(logits[0, -1, :], dim=0).cpu()
                lm_probs_tensor = get_lm_prob(lm_probs, tgt_tokens, vocab_size=config.vocab_size, n=3)

                fused_probs = (1 - weight) * tm_probs + weight * lm_probs_tensor
                next_token = fused_probs.argmax().item()
                tgt_tokens.append(next_token)
                if next_token == config.eos_token:
                    break

        pred_units = [u for u in tgt_tokens if u not in [config.bos_token, config.eos_token]]
        predictions.append(" ".join(map(str, pred_units)))

    return sacrebleu.corpus_bleu(predictions, [references], tokenize="none").score


def main():
    args = parse_args()
    config = build_experiment_config(args.num_clusters)
    units_json = args.units_json or config.units_json
    model_path = args.model_path or config.checkpoint_path
    device = get_device()
    print(f"Running LM-weight ablation on {device}...")
    print(f"Using K = {config.num_clusters}, checkpoint = {model_path}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    script_start = time.time()

    test_data = pd.read_csv(args.test_csv)
    with open(units_json, "r", encoding="utf-8") as f:
        units_dict = json.load(f)
    t_lm = time.time()
    lm_probs = build_ngram_lm(units_dict, config.bos_token, config.eos_token, n=3)
    print(f"  LM built in {_fmt(time.time() - t_lm)}")

    model = S2UTModel(vocab_size=config.vocab_size).to(device)
    load_checkpoint_into_model(model, model_path, device, expected_num_clusters=config.num_clusters)
    model.eval()

    results = []
    for w in WEIGHTS_TO_TEST:
        print(f"\n{'=' * 40}")
        print(f"Testing LM_WEIGHT = {w}")
        t_w = time.time()
        bleu_score = evaluate_with_weight(
            w,
            model,
            lm_probs,
            test_data,
            units_dict,
            device,
            config,
            args.max_len,
        )
        print(f"LM_WEIGHT = {w} | Unit-BLEU: {bleu_score:.2f} | Elapsed: {_fmt(time.time() - t_w)}")
        results.append((w, bleu_score))

    print("\n" + "=" * 40)
    print("Ablation finished. Result summary:")
    for w, score in results:
        print(f"Weight: {w:.1f} | BLEU: {score:.2f}")

    df = pd.DataFrame(results, columns=["LM_Weight", "Unit_BLEU"])
    df.to_csv(config.ablation_results_path, index=False)
    print(f"Saved results to {config.ablation_results_path}")

    plt.figure(figsize=(8, 5))
    weights = [r[0] for r in results]
    scores = [r[1] for r in results]

    plt.plot(weights, scores, marker="o", linestyle="-", color="b", linewidth=2, markersize=8)
    plt.title("Effect of Unit LM Weight on S2UT Performance", fontsize=14)
    plt.xlabel("LM Weight ($\\lambda$)", fontsize=12)
    plt.ylabel("Unit-BLEU Score", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    for i, txt in enumerate(scores):
        plt.annotate(f"{txt:.2f}", (weights[i], scores[i]), textcoords="offset points", xytext=(0, 10), ha="center")

    plt.savefig(config.ablation_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved ablation plot to {config.ablation_plot_path}")
    print(f"Total time: {_fmt(time.time() - script_start)}")


if __name__ == "__main__":
    main()
