import argparse
import json
import os
import time

import torch
import torchaudio
from tqdm import tqdm

import _repo_path  # noqa: F401
from audio_utils import load_audio
from checkpoint_utils import load_checkpoint_into_model
from model import S2UTModel
from s2ut_config import add_max_len_arg, add_num_clusters_arg, build_experiment_config, ensure_parent_dir

TEST_AUDIO = "./TCST/wav/Amdo/maqufa/maqufa_002.wav"
SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Run autoregressive S2UT inference for a chosen K.")
    add_num_clusters_arg(parser)
    add_max_len_arg(parser)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-audio", default=TEST_AUDIO)
    parser.add_argument("--output-json", default=None)
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


def main():
    args = parse_args()
    config = build_experiment_config(args.num_clusters)
    model_path = args.model_path or config.checkpoint_path
    output_json = args.output_json or config.predicted_units_path
    device = get_device()
    print(f"Running autoregressive inference on {device}...")
    print(f"Using K = {config.num_clusters}, checkpoint = {model_path}")
    print(f"Audio  : {args.test_audio}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    script_start = time.time()

    model = S2UTModel(vocab_size=config.vocab_size).to(device)
    if os.path.exists(model_path):
        load_checkpoint_into_model(model, model_path, device, expected_num_clusters=config.num_clusters)
        print("Loaded model checkpoint.")
    else:
        print("Checkpoint not found. Train the model first or pass --model-path.")
        return

    model.eval()

    waveform, sr = load_audio(args.test_audio)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, sample_frequency=SAMPLE_RATE)
    src = fbank.unsqueeze(0).to(device)

    print("Decoding target unit sequence...")
    tgt_tokens = [config.bos_token]

    with torch.no_grad():
        t_enc = time.time()
        src_encoded, _ = model.subsampler(src, None)
        src_encoded = model.pos_encoder(src_encoded)
        print(f"  Encoder done in {_fmt(time.time() - t_enc)}")

        t_dec = time.time()
        for _ in tqdm(range(args.max_len), desc="  Decoding", unit="tok"):
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
        print(f"  Decode done in {_fmt(time.time() - t_dec)}")

    predicted_units = [u for u in tgt_tokens if u not in [config.bos_token, config.eos_token]]
    print(f"\nInference finished. Predicted unit sequence length: {len(predicted_units)}")
    print(f"Units: {predicted_units}")

    ensure_parent_dir(output_json)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"test_audio": predicted_units}, f, ensure_ascii=False, indent=2)
    print(f"Saved predicted units to: {output_json}")
    print(f"Total time: {_fmt(time.time() - script_start)}")


if __name__ == "__main__":
    main()
