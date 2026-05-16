# Tibetan-to-Mandarin Speech-to-Unit Translation Prototype

This repository is a course-scale prototype for Tibetan-to-Mandarin speech translation using discrete target speech units. The goal is specific: explore whether Tibetan source speech can be mapped toward Mandarin speech units without first requiring a complete Tibetan automatic speech recognition system.

The motivating bottleneck is access to Mandarin-dominant public services. For example, an elderly Tibetan speaker in a hospital or government office may need a communication bridge before a robust Tibetan ASR system exists. This prototype is not a deployable translator, but it tests a concrete intervention for that gap.

The method is inspired by Gong, Xu, and Zhao (2025), *Tibetan-Chinese speech-to-speech translation based on discrete units*: https://www.nature.com/articles/s41598-025-85782-w

## What This Repo Contains

The pipeline predicts Mandarin-side discrete speech units from Tibetan speech features:

1. Generate Mandarin target speech from the Chinese text in TCST.
2. Extract HuBERT layer-6 features from the Mandarin target speech.
3. Cluster those features into discrete units with MiniBatch K-means.
4. Train a Transformer speech-to-unit translation model.
5. Evaluate with Unit-BLEU.
6. Test shallow fusion with a 3-gram unit language model.
7. Run a KNN retrieval diagnostic for qualitative error analysis.

This is intentionally smaller than the reference paper. It does not include HuBERT fine-tuning, auxiliary ASR/ST/CTC losses, Fairseq training recipes, or a trained unit vocoder.

## Repository Layout

```text
00_check_env.py              Environment check
01_synthesize_targets.py     Edge-TTS Mandarin target speech generation
02_extract_units.py          HuBERT feature extraction and K-means unit extraction
03_split_data.py             Reproducible TCST train/dev/test split
04_check_model.py            Model forward-pass smoke test
05_check_dataset.py          Dataset smoke test
06_train.py                  Transformer S2UT training
07_inference.py              Single-utterance unit prediction
08_evaluate.py               Greedy Unit-BLEU evaluation
09_lm_guided_inference.py    Evaluation with one LM weight
10_ablation_study.py         LM-weight ablation
11_synthesize_knn.py         KNN retrieval plus Edge-TTS diagnostic synthesis
12_verify_case.py            Qualitative consistency report for one example

dataset.py                   Dataset and collate logic
model.py                     S2UT Transformer model
audio_utils.py               Audio loading helper
checkpoint_utils.py          Checkpoint loading helper
s2ut_config.py               K-specific paths and vocabulary config

RESULTS.md                   Final quantitative and qualitative results
CASE_ANALYSIS.md             Good/bad case analysis procedure
EXPERIMENT_GUIDE.md          Reproduction notes, including Habra GPU usage
DEMO.md                      Five-minute showcase walkthrough
requirements.txt             Python dependencies
```

The numbered files are runnable workflow scripts. The unnumbered Python files are support modules imported by those scripts.

Large artifacts are not meant to be committed to GitHub. Raw Tibetan audio, generated Mandarin audio, checkpoints, K-means `.pkl` files, and generated `.wav` files are ignored. The repository keeps code, metadata, split CSVs, and small result summaries so the experiment can be understood and reproduced.

## Data

The code expects TCST-style metadata and audio paths:

```text
TCST/text.json
TCST/wav/...                 Tibetan source speech, not committed
data/TCST/wav_zh/...         generated Mandarin target speech, not committed
data/train.csv
data/dev.csv
data/test.csv
```

Generate Mandarin target speech with Edge-TTS:

```bash
python 01_synthesize_targets.py \
  --json-path TCST/text.json \
  --out-dir data/TCST/wav_zh
```

Edge-TTS requires network access. If generated audio already exists locally, skip this step.

## Installation

Use a virtual environment or conda environment:

```bash
conda create -n s2ut python=3.10 -y
conda activate s2ut
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Check the environment:

```bash
python 00_check_env.py
python 04_check_model.py --num-clusters 100
```

## Reproducing the Main Experiment

The code supports `K = 100, 200, 500, 1000`.

Run one K setting:

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

Save per-example predictions for qualitative analysis:

```bash
python 08_evaluate.py \
  --num-clusters 100 \
  --max-len 600 \
  --save-predictions results/k100_test_predictions.jsonl
```

GPU-cluster notes are in [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md).

## Final Result

The selected system uses `K=100`, matching the reference paper's reported HuBERT-base layer-6, `k=100` unit choice.

| Setting | LM weight | Unit-BLEU |
|---|---:|---:|
| Greedy S2UT baseline | 0.0 | 12.10 |
| Best LM-guided decoding | 0.6 | 19.50 |

The shallow-fusion result improves over greedy decoding by `+7.40` Unit-BLEU. Larger K values did not help in this small setup, which is a useful negative result. Details are in [RESULTS.md](RESULTS.md).

## Qualitative Diagnostic

Run inference for one Tibetan utterance:

```bash
python 07_inference.py \
  --num-clusters 100 \
  --test-audio ./TCST/wav/Amdo/maqufa/maqufa_002.wav \
  --max-len 600
```

Then run the retrieval check:

```bash
python 12_verify_case.py \
  --num-clusters 100 \
  --test-audio ./TCST/wav/Amdo/maqufa/maqufa_002.wav \
  --knn-pool train
```

This diagnostic is deliberately conservative. It checks whether predicted units retrieve a training example with similar Tibetan and Mandarin text. Current examples show semantic failures, so the prototype should not be presented as a reliable end-to-end translator.

## Limitations

- Unit-BLEU measures overlap between unit sequences, not translation adequacy.
- There is no human evaluation.
- Mandarin target speech is synthesized rather than naturally recorded.
- The synthesis demo uses KNN retrieval plus Edge-TTS, not a trained unit vocoder.
- The model omits the auxiliary objectives and fine-tuned HuBERT setup used in the reference paper.
- The ethical contribution is access-oriented prototyping and transparent evaluation, not deployment readiness.

## License and Data Use

Code in this repository is released under the MIT License. TCST data and generated artifacts should be used only under their original license terms. If redistribution is not permitted, share scripts and metadata pointers rather than raw audio or checkpoints.
