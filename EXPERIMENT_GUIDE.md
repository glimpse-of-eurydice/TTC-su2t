# Reproduction and Habra GPU Notes

This guide gives the minimum commands needed to reproduce the K sweep on a GPU machine such as Habra.

## Environment

Create a Python environment:

```bash
conda create -n s2ut python=3.10 -y
conda activate s2ut
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Check that PyTorch and the accelerator are visible:

```bash
python 00_check_env.py
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("mps available:", torch.backends.mps.is_available())
PY
```

If HuBERT downloads are slow or home storage is limited, place Hugging Face caches on scratch:

```bash
export HF_HOME=/scratch/$USER/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface/transformers
```

## Expected Data Layout

```text
TCST/text.json
TCST/wav/...
data/TCST/wav_zh/...
data/train.csv
data/dev.csv
data/test.csv
```

Raw audio and generated audio are not committed. Generate Mandarin target speech if needed:

```bash
python 01_synthesize_targets.py \
  --json-path TCST/text.json \
  --out-dir data/TCST/wav_zh
```

## Run One K Setting

```bash
K=100

python 02_extract_units.py \
  --num-clusters "$K" \
  --sample-ratio 0.1 \
  --batch-size 10000 \
  --force-retrain

python 03_split_data.py --num-clusters "$K"
python 05_check_dataset.py --num-clusters "$K" --batch-size 2

python 06_train.py \
  --num-clusters "$K" \
  --batch-size 16 \
  --epochs 40 \
  --learning-rate 5e-4

python 08_evaluate.py \
  --num-clusters "$K" \
  --max-len 600

python 10_ablation_study.py \
  --num-clusters "$K" \
  --max-len 600
```

Lower `--batch-size` to `8` or `4` if training runs out of memory.

## Slurm Array Sketch

Use an array job only after the split CSVs exist, because `03_split_data.py` writes shared `data/train.csv`, `data/dev.csv`, and `data/test.csv`.

```bash
#!/bin/bash
#SBATCH --job-name=s2ut_k_sweep
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/k_sweep_%A_%a.out

set -euo pipefail

source ~/.bashrc
conda activate s2ut

mkdir -p logs
K_VALUES=(100 200 500 1000)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

python 02_extract_units.py --num-clusters "$K" --sample-ratio 0.1 --batch-size 10000 --force-retrain
python 05_check_dataset.py --num-clusters "$K" --batch-size 2
python 06_train.py --num-clusters "$K" --batch-size 16 --epochs 40 --learning-rate 5e-4
python 08_evaluate.py --num-clusters "$K" --max-len 600
python 10_ablation_study.py --num-clusters "$K" --max-len 600
```

Submit and monitor:

```bash
sbatch run_k_sweep.sbatch
squeue -u "$USER"
tail -f logs/k_sweep_<JOBID>_0.out
```

## Common Problems

- CUDA is unavailable: confirm that the job requested a GPU and that the CUDA PyTorch build is installed.
- HuBERT download fails: pre-download on a login node or set a writable Hugging Face cache.
- Checkpoint and K mismatch: avoid manually passing `--model-path` unless you are sure it matches `--num-clusters`.
- Edge-TTS network failure: run synthesis on a machine with internet access, then copy generated audio to the GPU machine.
