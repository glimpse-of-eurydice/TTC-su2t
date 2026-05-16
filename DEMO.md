# Five-Minute Showcase Walkthrough

## 0:00-0:45 Problem

This repo addresses a concrete low-resource bottleneck: Tibetan-to-Mandarin speech translation. A full Tibetan ASR system is hard to build in this setting, so the project asks whether Tibetan speech can be mapped directly toward Mandarin speech units.

Ethical framing: the motivating use case is language access in Mandarin-dominant public services. The goal is not to claim a deployable medical translator, but to test a transparent technical intervention for a real access gap.

## 0:45-1:45 Method

Walk through `README.md` and the numbered scripts:

1. `synthesize_tcst_edge_tts.py` creates Mandarin target speech from TCST Chinese text.
2. `02_extract_units.py` extracts HuBERT layer-6 features and clusters them into target units.
3. `05_train.py` trains a Transformer S2UT model from Tibetan speech features to Mandarin units.
4. `07_evaluate.py` reports Unit-BLEU.
5. `08b_ablation_study.py` tests LM shallow fusion.
6. `10_verify_case.py` checks qualitative consistency.

## 1:45-2:45 Result

Open `RESULTS.md`.

Main number:

```text
K = 100
LM weight = 0.6
Unit-BLEU = 19.50
```

The greedy baseline is `12.10`, so shallow fusion improves the score by `+7.40`.

Important negative result: larger K values did not help. `K=1000` produced a richer but sparser target inventory that this small model could not predict well.

## 2:45-3:45 Qualitative Failure

Open `CASE_ANALYSIS.md`.

Show the diagnostic failure:

```text
Input sample: maqufa-002
Reference Mandarin: 不断加大高水平对外开放力度，
Retrieved Mandarin: 今天有人买牙膏吗？
```

This is the key evaluation lesson: a model can produce unit sequences and fluent retrieved Mandarin while failing to preserve meaning. Unit-BLEU is useful, but it is not enough.

## 3:45-4:30 Reproducibility

Point to:

- `requirements.txt`
- `EXPERIMENT_GUIDE.md`
- split CSVs in `data/`
- result files such as `ablation_results*.csv`
- ignored large artifacts in `.gitignore`

Explain that checkpoints, raw audio, generated audio, and K-means binaries are not committed, but the scripts document how to reproduce them.

## 4:30-5:00 Takeaway

What I did: built a complete small S2UT prototype covering data construction, representation learning, model training, ablation, and qualitative evaluation.

What I learned: discrete units are promising, but evaluation must include semantic checks. Better unit metrics do not automatically mean useful translation.

What to take away: for marginalized-language technology, honest limitations are part of the contribution. This repo is useful because it is inspectable and reproducible, not because it claims the problem is solved.
