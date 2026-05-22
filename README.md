# Tibetan-to-Mandarin Speech-to-Unit Translation Prototype

This repository documents and implements a course-scale prototype for Tibetan-to-Mandarin speech translation using discrete speech units. The project asks a narrow question: can Tibetan source speech be mapped toward Mandarin-side acoustic units without first requiring a complete Tibetan automatic speech recognition system?

The short answer is: partly. The system is a working, inspectable prototype. It builds data, extracts target-side units, trains a Transformer speech-to-unit model, evaluates unit predictions, and runs qualitative retrieval diagnostics. It also fails in important ways. Those failures are part of the contribution, because a low-resource language technology project should be explicit about where its outputs are not safe to use.

The main technical inspiration is Gong, Xu, and Zhao (2025), *Tibetan-Chinese speech-to-speech translation based on discrete units*: https://www.nature.com/articles/s41598-025-85782-w

## A Motivating Scene

Imagine an elderly Tibetan speaker from a remote pastoral area traveling to a large hospital in Chengdu. The hospital staff mostly speak Mandarin. The patient may be able to describe pain, medication history, or symptoms in Tibetan, but the clinical interaction depends on a Mandarin-speaking system: reception, triage, payment, diagnosis, and follow-up instructions.

This is not just a benchmark problem. It is an access problem. Speech technology can either reduce or intensify the distance between a marginalized speaker and a public service. A translation system in this setting must therefore be evaluated honestly. A fluent Mandarin sentence is not enough if it does not preserve the meaning of the Tibetan speech.

This repo should be read in that spirit. It is not a deployable medical interpreter. It is a small intervention that tests one idea: avoiding a fragile Tibetan text bottleneck by predicting Mandarin-side speech units.

## Why the Standard ASR-to-MT-to-TTS Pipeline Is Fragile Here

A conventional speech translation system usually works as a cascade:

```text
source speech -> ASR text -> machine translation text -> target speech
```

This architecture is attractive because each component can be trained and evaluated separately. But in a low-resource Tibetan-to-Mandarin setting, every stage can become brittle.

First, the pipeline depends on reliable Tibetan ASR. That is difficult when there is limited transcribed speech, dialectal diversity, and a mismatch between spoken forms and standardized written forms. Errors at the ASR stage then propagate into translation and speech synthesis.

Second, the writing system itself makes the text bridge more complex than a simple character stream. Tibetan is written with syllable structures that combine base letters, prefixes, suffixes, superscribed/subscribed forms, vowel signs, and delimiters. The practical challenge is not "Unicode encoding" in isolation. Unicode can represent Tibetan text, just as it can represent Korean Hangul syllables. The difficulty is that ASR and downstream NLP systems need stable normalization, tokenization, and spoken-to-written alignment. In a better-resourced language, those conventions are often supported by large corpora and mature tools. In this project setting, they are much thinner.

Korean Hangul is a useful contrast only if used carefully. Hangul syllables are also compositional, but modern NLP tooling usually has strong normalization, segmentation, and data support for Korean. Tibetan speech technology has fewer such resources, and the path from acoustic realization to standardized written Tibetan can be less supported in practice. That makes an ASR-first cascade a risky dependency.

Third, even if Tibetan ASR were available, a full speech-to-speech system would still need target-side speech generation. Unit-based speech-to-speech translation systems often train a unit vocoder to convert predicted acoustic units into waveforms. That is a separate data and compute burden. For a six-week course project, a smaller and more honest prototype is preferable to an overclaimed full system.

## Core Idea: Translate Speech Into Units, Not Tibetan Text

This project follows the speech-to-unit translation idea: instead of predicting Mandarin characters directly, the model predicts a target-side "acoustic alphabet."

The pipeline uses HuBERT, a self-supervised speech model, to turn Mandarin target speech into frame-level features. K-means clustering then converts those continuous features into discrete unit IDs. A Mandarin utterance becomes a sequence like:

```text
16 24 55 77 45 99 ...
```

These IDs are not words or characters. They are learned acoustic categories. The translation model then learns:

```text
Tibetan speech features -> Mandarin acoustic unit sequence
```

This reframes the task. The source side remains speech. The target side becomes a symbolic sequence that is easier to train than raw waveform generation, but does not require the model to generate Tibetan text.

## Pipeline

```mermaid
flowchart TD
    A[TCST Tibetan source speech] --> B[80-dim log-Mel filterbank features]
    T[TCST Chinese text metadata] --> C[Edge-TTS Mandarin target speech]
    C --> D[HuBERT base layer-6 features]
    D --> E[MiniBatch K-means target units]
    E --> F[Reduced unit sequences]
    B --> G[Transformer S2UT model]
    F --> G
    F --> H[3-gram unit language model]
    G --> I[Greedy / shallow-fusion decoding]
    H --> I
    I --> J[Predicted Mandarin unit sequence]
    J --> K[Unit-BLEU evaluation]
    J --> L[KNN retrieval diagnostic]
    L --> M[Retrieved Mandarin text]
    M --> N[Edge-TTS diagnostic audio]
```

The implementation is intentionally smaller than the reference paper. It does not include HuBERT fine-tuning, auxiliary ASR/ST/CTC objectives, Fairseq recipes, or a trained unit vocoder. The point is to build a complete prototype that can be inspected and evaluated under realistic course constraints.

## What Is Implemented

The runnable workflow is organized under `scripts/`:

```text
scripts/00_check_env.py              Environment check
scripts/01_synthesize_targets.py     Edge-TTS Mandarin target speech generation
scripts/02_extract_units.py          HuBERT feature extraction and K-means unit extraction
scripts/03_split_data.py             Reproducible TCST train/dev/test split
scripts/04_check_model.py            Model forward-pass smoke test
scripts/05_check_dataset.py          Dataset smoke test
scripts/06_train.py                  Transformer S2UT training
scripts/07_inference.py              Single-utterance unit prediction
scripts/08_evaluate.py               Greedy Unit-BLEU evaluation
scripts/09_lm_guided_inference.py    Evaluation with one LM weight
scripts/10_ablation_study.py         LM-weight ablation
scripts/11_synthesize_knn.py         KNN retrieval plus Edge-TTS diagnostic synthesis
scripts/12_verify_case.py            Qualitative consistency report
```

Support modules stay at the repository root:

```text
dataset.py              Dataset and collate logic
model.py                Transformer S2UT model
audio_utils.py          Audio loading helper
checkpoint_utils.py     Checkpoint loading helper
s2ut_config.py          K-specific paths and vocabulary config
```

Supporting documents and outputs:

```text
docs/RESULTS.md             Short result summary
docs/CASE_ANALYSIS.md       Case-analysis workflow
docs/EXPERIMENT_GUIDE.md    Reproduction and Habra GPU notes
docs/DEMO.md                Five-minute showcase notes
results/                    Small CSV, JSON, and plot outputs
```

Large artifacts are intentionally not committed: raw Tibetan audio, generated Mandarin audio, checkpoints, K-means `.pkl` files, and generated `.wav` files. The repo keeps code, metadata, split CSVs, target units for K=100, and small result summaries.

## Data Construction

The TCST metadata provides Tibetan source speech and Tibetan/Chinese text fields, but this S2UT prototype needs Mandarin target speech in order to extract target acoustic units. I therefore synthesize Mandarin speech from the Chinese text with Edge-TTS:

```bash
python scripts/01_synthesize_targets.py \
  --json-path TCST/text.json \
  --out-dir data/TCST/wav_zh
```

After preprocessing, the split used in this repo contains:

| Split | Utterances |
|---|---:|
| Train | 5,801 |
| Dev | 725 |
| Test | 726 |

The split is generated with a fixed seed:

```bash
python scripts/03_split_data.py --num-clusters 100
```

## Model

The model maps Tibetan source speech to Mandarin target units.

Source side:

- Tibetan audio is loaded as waveform.
- Audio is converted to mono and resampled to 16 kHz.
- The model uses 80-dimensional log-Mel filterbank features.
- A convolutional subsampler reduces the time dimension by a factor of four.

Target side:

- Mandarin target speech is synthesized from Chinese text.
- HuBERT base layer-6 hidden states are extracted from the Mandarin speech.
- MiniBatch K-means maps frame-level HuBERT features to discrete unit IDs.
- Consecutive duplicate unit IDs are collapsed into reduced unit sequences.
- For K=100, the target vocabulary has 100 unit IDs plus BOS, EOS, and PAD tokens.

The sequence model is a compact Transformer encoder-decoder:

| Component | Setting |
|---|---:|
| Model dimension | 256 |
| Encoder layers | 4 |
| Decoder layers | 4 |
| Attention heads | 4 |
| Feed-forward dimension | 1024 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Learning rate | 5e-4 |

## Decoding: Why Add a Tiny Unit Language Model?

Low-resource autoregressive decoding is unstable. The model may choose locally plausible but globally poor unit continuations. To regularize the output, this repo builds a count-based 3-gram language model over target-side unit sequences.

During shallow fusion, the next-unit probability is interpolated:

```text
P_fused = (1 - lambda) * P_translation_model + lambda * P_unit_language_model
```

This language model does not know Tibetan or Mandarin words. It only regularizes short-range unit transitions. That is why the LM weight has to be tuned: too little LM guidance may not help, but too much can drown out source-speech conditioning.

## Results

The selected final system uses `K=100`. This matches the reference paper's reported HuBERT-base layer-6, `k=100` unit choice, but the numbers below are not directly comparable to the paper because this repo uses a smaller setup and reports Unit-BLEU over unit sequences.

### K Sweep

| K | Greedy Unit-BLEU | Best LM Unit-BLEU | Best LM weight |
|---|---:|---:|---:|
| 100 | 12.10 | 19.50 | 0.6 |
| 200 | 1.73 | 11.22 | 0.2 |
| 500 | 8.88 | 8.97 | 0.2 |
| 1000 | 4.27 | 5.24 | 0.6 |

Larger K values were not automatically better. In this small model, larger unit vocabularies became sparser and harder to predict. The best balance came from `K=100`.

### LM Weight Ablation for K=100

| LM weight | Unit-BLEU |
|---:|---:|
| 0.0 | 12.10 |
| 0.2 | 13.92 |
| 0.4 | 13.86 |
| 0.6 | 19.50 |
| 0.8 | 1.18 |

The best setting is:

```text
K = 100
LM weight = 0.6
Unit-BLEU = 19.50
```

The shallow-fusion result improves over greedy decoding by `+7.40` Unit-BLEU.

![K=100 LM weight ablation](results/lm_weight_ablation.png)

## Qualitative Analysis

Unit-BLEU is useful, but it is not semantic evaluation. A unit sequence can overlap with the reference more than another sequence while still failing to preserve meaning. For that reason, this repo includes a retrieval diagnostic.

The diagnostic takes a predicted Mandarin unit sequence, finds the nearest training unit sequence by Levenshtein edit distance, and compares the retrieved Tibetan and Mandarin text. This is deliberately conservative: if the retrieved text is semantically unrelated, the model should not be presented as successful just because it produced units.

### K=100 Failure Cases Despite the Best Unit-BLEU

The most important caution is that the strongest aggregate setting, `K=100`, still fails on individual utterances. These examples were generated with the K=100 checkpoint and checked through the KNN retrieval diagnostic. Three of the held-out test examples below produce the same predicted unit length and retrieve the same unrelated training sentence, which suggests that the model can collapse toward a locally plausible unit pattern instead of preserving the source meaning.

| Query sample | Split | Reference Mandarin | Retrieved Mandarin | Pred. unit length | Unit edit distance | Interpretation |
|---|---|---|---|---:|---:|---|
| `maqufa-002` | train | 不断加大高水平对外开放力度， | 今天有人买牙膏吗？ | 67 | 36 | Fluent retrieved Mandarin, but completely wrong meaning. |
| `maqufb-038` | test | 不计算住院次数，采用公共住院全年统计量线。 | 香港特区政府发言人说， | 148 | 97 | A health/administration sentence is mapped to an unrelated political-news phrase. |
| `bodkb-200` | test | 去看电影吗 | 香港特区政府发言人说， | 148 | 97 | A short everyday question is mapped to the same unrelated phrase. |
| `L_F_0_02_235` | test | 加大对三滇藏区的支持力度。 | 香港特区政府发言人说， | 148 | 97 | A public-policy sentence is again mapped to the same unrelated phrase. |

This is the core evaluation lesson. `K=100` gives the best aggregate Unit-BLEU in this repo, but that does not mean the model is usable for the target user. A patient, receptionist, or doctor would not benefit from a system that produces a fluent Mandarin sentence with the wrong content. The failure is not just a decoding inconvenience; it is an access and safety issue.

### Retrieval Diagnostics Across Unit Inventories

The following examples keep the original `maqufa-002` query fixed and vary only the K-means unit inventory. They are not independent held-out test examples; they are diagnostic checks showing how retrieval behavior changes across unit spaces.

| K | Predicted unit length | Retrieved sample | Retrieved Mandarin | Consistent? |
|---:|---:|---|---|---|
| 100 | 67 | `f58-La68_308` | 今天有人买牙膏吗？ | No |
| 200 | 31 | `cuoxiang-191` | 为什么？ | No |
| 500 | 121 | `L_M_0_13_396` | 周永康在青海考察 | No |
| 1000 | 191 | `f71-La16_44` | 在生产发展和社会财富增长的基础上， | No |

This table is not meant to say that K=100 is uniquely bad or that K=1000 is semantically reliable. It shows a broader limitation: nearest-neighbor retrieval over predicted unit sequences can produce fluent, plausible-looking Mandarin while drifting away from the source meaning.

### How to Generate a Full Good/Bad Case Audit

For a full held-out case audit, run:

```bash
python scripts/08_evaluate.py \
  --num-clusters 100 \
  --max-len 600 \
  --save-predictions results/k100_test_predictions.jsonl
```

That JSONL contains `sample_id`, Tibetan/Mandarin text metadata, predicted units, reference units, unit edit distance, and normalized edit distance. Sorting by `normalized_unit_edit_distance` gives candidate "better" and "worse" examples. This command is slower on CPU because it autoregressively decodes the full test split.

## Reproducing the Main Experiment

Install dependencies:

```bash
conda create -n s2ut python=3.10 -y
conda activate s2ut
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Check the environment and model:

```bash
python scripts/00_check_env.py
python scripts/04_check_model.py --num-clusters 100
python scripts/05_check_dataset.py --num-clusters 100 --batch-size 2
```

Run one K setting:

```bash
K=100

python scripts/02_extract_units.py \
  --num-clusters "$K" \
  --sample-ratio 0.1 \
  --batch-size 10000 \
  --force-retrain

python scripts/03_split_data.py --num-clusters "$K"
python scripts/05_check_dataset.py --num-clusters "$K" --batch-size 2

python scripts/06_train.py \
  --num-clusters "$K" \
  --batch-size 16 \
  --epochs 40 \
  --learning-rate 5e-4

python scripts/08_evaluate.py \
  --num-clusters "$K" \
  --max-len 600

python scripts/10_ablation_study.py \
  --num-clusters "$K" \
  --max-len 600
```

Run the qualitative retrieval diagnostic:

```bash
python scripts/07_inference.py \
  --num-clusters 100 \
  --test-audio ./TCST/wav/Amdo/maqufa/maqufa_002.wav \
  --max-len 600

python scripts/12_verify_case.py \
  --num-clusters 100 \
  --test-audio ./TCST/wav/Amdo/maqufa/maqufa_002.wav \
  --knn-pool train
```

More reproduction notes are in [docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md).

## What This Project Contributes

This project touches three pillars of low-resource NLP and speech work.

Data:

- It constructs Mandarin target speech from TCST Chinese text.
- It creates target-side HuBERT/K-means unit sequences.
- It keeps raw audio and generated artifacts out of GitHub while documenting how to reproduce them.

Learning:

- It trains a compact Transformer S2UT model from Tibetan speech features to Mandarin target units.
- It tests different target unit inventory sizes.
- It adds a small unit-level language model for shallow-fusion decoding.

Evaluation:

- It reports Unit-BLEU rather than claiming semantic translation quality.
- It includes K and LM-weight ablations.
- It uses qualitative retrieval diagnostics to expose semantic failures.

## Limitations

This prototype should not be used for real medical, legal, or public-service interpretation.

Key limitations:

- No human evaluation was run.
- Unit-BLEU is not semantic adequacy.
- The Mandarin target speech is synthesized, not naturally recorded.
- The model does not use fine-tuned HuBERT.
- The model omits auxiliary ASR/ST/CTC objectives used by larger systems.
- The final rendering stage is KNN retrieval plus Edge-TTS, not a trained unit vocoder.
- The retrieval diagnostic shows clear semantic drift.

These limitations are not side notes. They define the safe interpretation of the project.

## Conclusion

The project began from an ethical and technical bottleneck: Tibetan speakers should not have to depend on high-resource Mandarin infrastructure to be understood in public-service settings, yet building reliable Tibetan ASR is itself difficult. A speech-to-unit approach offers a way to test translation without making Tibetan text generation the central dependency.

The results are mixed in a useful way. The K=100 system with LM weight 0.6 reaches `19.50` Unit-BLEU, improving substantially over greedy decoding. At the same time, qualitative retrieval shows that unit overlap and fluent Mandarin rendering do not guarantee meaning preservation.

The takeaway is therefore not "this solves Tibetan-to-Mandarin speech translation." It is: a small discrete-unit prototype can be built and evaluated honestly, and the evaluation shows exactly why low-resource speech translation must be judged by more than fluent output.

## License and Data Use

Code in this repository is released under the MIT License. TCST data and generated artifacts should be used only under their original license terms. If redistribution is not permitted, share scripts and metadata pointers rather than raw audio, generated audio, checkpoints, or K-means binaries.
