import argparse
import json
import time
from collections import Counter, defaultdict

import pandas as pd
import sacrebleu
import torch
import torchaudio
from tqdm import tqdm

from audio_utils import load_audio
from checkpoint_utils import load_checkpoint_into_model
from model import S2UTModel
from s2ut_config import add_max_len_arg, add_num_clusters_arg, build_experiment_config

TEST_CSV = "./data/test.csv"
SAMPLE_RATE = 16000
LM_WEIGHT = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate S2UT with shallow-fusion LM for a chosen K.")
    add_num_clusters_arg(parser)
    add_max_len_arg(parser)
    parser.add_argument("--units-json", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-csv", default=TEST_CSV)
    parser.add_argument("--lm-weight", type=float, default=LM_WEIGHT)
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


def main():
    args = parse_args()
    config = build_experiment_config(args.num_clusters)
    units_json = args.units_json or config.units_json
    model_path = args.model_path or config.checkpoint_path
    device = get_device()
    print(f"Evaluating shallow-fusion decoding on {device}...")
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

    predictions = []
    references = []

    print(f"Evaluating {len(test_data)} test utterances...")
    t_infer = time.time()
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
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

            for _ in range(args.max_len):
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
                tgt_emb = model.unit_embedding(tgt_tensor) * (model.d_model ** 0.5)
                tgt_emb = model.pos_decoder(tgt_emb)
                tgt_mask = model.generate_square_subsequent_mask(tgt_emb.size(1), device)

                out = model.transformer(src_encoded, tgt_emb, tgt_mask=tgt_mask)
                logits = model.fc_out(out)
                tm_probs = torch.softmax(logits[0, -1, :], dim=0).cpu()
                lm_probs_tensor = get_lm_prob(lm_probs, tgt_tokens, vocab_size=config.vocab_size, n=3)

                fused_probs = (1 - args.lm_weight) * tm_probs + args.lm_weight * lm_probs_tensor
                next_token = fused_probs.argmax().item()
                tgt_tokens.append(next_token)
                if next_token == config.eos_token:
                    break

        pred_units = [u for u in tgt_tokens if u not in [config.bos_token, config.eos_token]]
        predictions.append(" ".join(map(str, pred_units)))

    print(f"  Inference done in {_fmt(time.time() - t_infer)}")
    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="none")
    print("\n" + "=" * 40)
    print("LM-guided evaluation finished.")
    print(f"Unit-BLEU: {bleu.score:.2f}")
    print(f"Total time: {_fmt(time.time() - script_start)}")
    print("=" * 40)


if __name__ == "__main__":
    main()
