# Five-Minute Showcase Walkthrough and Speaker Notes

Use this as a repo walkthrough, not as slides. Keep the GitHub repository open and move between `README.md`, `docs/RESULTS.md`, `docs/CASE_ANALYSIS.md`, and `.gitignore`.

## 0:00-0:45 Problem

Show: top of `README.md`.

Speaker notes:

> My project addresses a specific low-resource bottleneck: Tibetan-to-Mandarin speech translation. A full Tibetan ASR system is hard to build in this setting, so I tested whether Tibetan speech can be mapped directly toward Mandarin-side speech units.
>
> The motivating scenario is language access in Mandarin-dominant public services, for example hospitals or government offices. I am not claiming this is a deployable medical translator. The contribution is a transparent prototype for an access problem where the usual ASR-first pipeline may be too brittle.

## 0:45-1:45 Method

Show: `README.md` repository layout and the numbered workflow scripts.

Speaker notes:

> The main technical inspiration is Gong, Xu, and Zhao 2025, who study Tibetan-Chinese speech-to-speech translation with discrete units. I did not reproduce their full system. Instead, I built a smaller course-scale version focused on the core idea: target-side discrete units, S2UT training, and honest evaluation.
>
> The workflow is numbered under `scripts/`. `01_synthesize_targets.py` creates Mandarin target speech from TCST Chinese text. `02_extract_units.py` extracts HuBERT layer-6 features and clusters them into target units. `06_train.py` trains a Transformer speech-to-unit model from Tibetan speech features to Mandarin units. `08_evaluate.py` computes Unit-BLEU, `10_ablation_study.py` tests LM shallow fusion, and `12_verify_case.py` checks qualitative consistency.

Quick walkthrough list:

1. `01_synthesize_targets.py`: Mandarin target speech from TCST text.
2. `02_extract_units.py`: HuBERT features and K-means units.
3. `03_split_data.py`: reproducible train/dev/test split.
4. `06_train.py`: Transformer S2UT training.
5. `08_evaluate.py`: Unit-BLEU evaluation.
6. `10_ablation_study.py`: LM shallow-fusion ablation.
7. `12_verify_case.py`: qualitative retrieval diagnostic.

## 1:45-2:45 Result

Show: `docs/RESULTS.md`.

Speaker notes:

> The selected system uses `K=100`, which also matches the discrete-unit choice reported in the reference paper. My best result is `19.50` Unit-BLEU with LM weight `0.6`. The greedy baseline is `12.10`, so shallow fusion gives a `+7.40` absolute improvement.
>
> The negative result is important: larger K values did not help. `K=1000` gives a richer acoustic inventory, but in this small model it becomes too sparse and harder to predict. So the result is not just "bigger is better"; the unit inventory has to match the data and model scale.

Main number:

```text
K = 100
LM weight = 0.6
Unit-BLEU = 19.50
```

## 2:45-3:45 Qualitative Failure

Show: `docs/CASE_ANALYSIS.md`.

Speaker notes:

> Unit-BLEU is only a unit-overlap metric, so I added a qualitative diagnostic. The check takes predicted units, retrieves the closest training unit sequence, and compares the retrieved Tibetan and Mandarin text.
>
> This example is a clear failure. The reference Mandarin is "continuously increasing high-level opening-up," but the retrieved Mandarin is "Did anyone buy toothpaste today?" This is exactly why I do not overclaim the system. Producing units, or even fluent Mandarin through a retrieval/TTS bridge, is not the same as preserving meaning.

Failure case:

```text
Input sample: maqufa-002
Reference Mandarin: 不断加大高水平对外开放力度，
Retrieved Mandarin: 今天有人买牙膏吗？
```

## 3:45-4:30 Reproducibility

Show: `.gitignore`, `requirements.txt`, `docs/EXPERIMENT_GUIDE.md`, and `data/`.

Speaker notes:

> The repo is structured as the submission artifact. It includes the code, split CSVs, metadata, result summaries, and reproduction commands. Large files are intentionally ignored: raw audio, generated audio, checkpoints, K-means binaries, and generated `.wav` files.
>
> This matters because someone can inspect what was done without downloading gigabytes of artifacts, and the heavy artifacts can be regenerated or shared separately if needed.

Point to:

- `requirements.txt`
- `docs/EXPERIMENT_GUIDE.md`
- split CSVs in `data/`
- result files such as `results/ablation_results*.csv`
- ignored large artifacts in `.gitignore`

## 4:30-5:00 Takeaway

Show: final paragraph of `README.md` or `docs/RESULTS.md`.

Speaker notes:

> What I did was build a complete small S2UT prototype covering data construction, representation learning, model training, ablation, and qualitative evaluation.
>
> What I learned is that discrete units are promising, but evaluation has to include semantic checks. Better unit metrics do not automatically mean useful translation.
>
> The main takeaway is ethical as much as technical: for marginalized-language technology, honest limitations are part of the contribution. This repo is useful because it is inspectable and reproducible, not because it claims the problem is solved.
