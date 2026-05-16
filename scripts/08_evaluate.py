import argparse
import json
import time

import pandas as pd
import sacrebleu
import torch
import torchaudio
from tqdm import tqdm

import _repo_path  # noqa: F401
from audio_utils import load_audio
from checkpoint_utils import load_checkpoint_into_model
from model import S2UTModel
from s2ut_config import (
    add_max_len_arg,
    add_num_clusters_arg,
    build_experiment_config,
    ensure_parent_dir,
)

TEST_CSV = "./data/test.csv"
SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate S2UT with a configurable unit inventory.")
    add_num_clusters_arg(parser)
    add_max_len_arg(parser)
    parser.add_argument("--units-json", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-csv", default=TEST_CSV)
    parser.add_argument(
        "--save-predictions",
        default=None,
        help="Optional JSONL path for per-example predictions and edit-distance stats.",
    )
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


def levenshtein(seq1, seq2):
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1
    if not seq2:
        return len(seq1)
    previous = list(range(len(seq2) + 1))
    for i, item1 in enumerate(seq1, 1):
        current = [i]
        for j, item2 in enumerate(seq2, 1):
            insert_cost = previous[j] + 1
            delete_cost = current[j - 1] + 1
            substitute_cost = previous[j - 1] + (item1 != item2)
            current.append(min(insert_cost, delete_cost, substitute_cost))
        previous = current
    return previous[-1]


def main():
    args = parse_args()
    config = build_experiment_config(args.num_clusters)
    units_json = args.units_json or config.units_json
    model_path = args.model_path or config.checkpoint_path
    device = get_device()
    print(f"Evaluating test split on {device}...")
    print(f"Using K = {config.num_clusters}, checkpoint = {model_path}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    script_start = time.time()

    model = S2UTModel(vocab_size=config.vocab_size).to(device)
    load_checkpoint_into_model(model, model_path, device, expected_num_clusters=config.num_clusters)
    model.eval()

    test_data = pd.read_csv(args.test_csv)
    with open(units_json, "r", encoding="utf-8") as f:
        units_dict = json.load(f)

    predictions = []
    references = []
    prediction_rows = []

    print(f"Evaluating {len(test_data)} test utterances...")
    t_infer = time.time()
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        sample_id = str(row["sample_id"])
        audio_path = row["tibetan_audio"]

        if sample_id not in units_dict:
            continue
        ref_units = units_dict[sample_id]["units"]
        references.append(" ".join(map(str, ref_units)))

        waveform, sr = load_audio(audio_path)
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
                next_token = logits[0, -1, :].argmax().item()
                tgt_tokens.append(next_token)
                if next_token == config.eos_token:
                    break

        pred_units = [u for u in tgt_tokens if u not in [config.bos_token, config.eos_token]]
        predictions.append(" ".join(map(str, pred_units)))
        edit_distance = levenshtein(pred_units, ref_units)
        normalizer = max(len(pred_units), len(ref_units), 1)
        prediction_rows.append(
            {
                "sample_id": sample_id,
                "tibetan_audio": audio_path,
                "tibetan_text": row.get("tibetan_text", ""),
                "chinese_text": row.get("chinese_text", ""),
                "ref_len": len(ref_units),
                "pred_len": len(pred_units),
                "unit_edit_distance": edit_distance,
                "normalized_unit_edit_distance": edit_distance / normalizer,
                "reference_units": ref_units,
                "predicted_units": pred_units,
            }
        )

    print(f"  Inference done in {_fmt(time.time() - t_infer)}")
    if args.save_predictions:
        ensure_parent_dir(args.save_predictions)
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            for item in prediction_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Per-example predictions saved to {args.save_predictions}")

    t_bleu = time.time()
    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="none")
    print(f"  BLEU computed in {_fmt(time.time() - t_bleu)}")
    print("\n" + "=" * 40)
    print("Evaluation finished.")
    print(f"Unit-BLEU: {bleu.score:.2f}")
    print(f"Total time: {_fmt(time.time() - script_start)}")
    print("=" * 40)


if __name__ == "__main__":
    main()
