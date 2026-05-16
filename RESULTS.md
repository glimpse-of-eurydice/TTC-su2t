# Results and Evaluation

## Research Question

Can a compact speech-to-unit model learn a useful Tibetan-to-Mandarin mapping when Mandarin target speech is represented as HuBERT-derived discrete units?

The experiment tests two design choices:

1. K-means target unit inventory size: `100`, `200`, `500`, and `1000`.
2. Shallow fusion with a simple 3-gram unit language model.

## Main Result

The final selected system uses `K=100`.

| Setting | LM weight | Unit-BLEU |
|---|---:|---:|
| Greedy S2UT baseline | 0.0 | 12.10 |
| LM-guided decoding | 0.2 | 13.92 |
| LM-guided decoding | 0.4 | 13.86 |
| LM-guided decoding | 0.6 | 19.50 |
| LM-guided decoding | 0.8 | 1.18 |

Best result:

```text
K = 100
LM weight = 0.6
Unit-BLEU = 19.50
```

The greedy baseline is `12.10` Unit-BLEU, so shallow fusion gives a `+7.40` absolute improvement.

## K-means Sweep

| K | Greedy Unit-BLEU | Best LM Unit-BLEU | Best LM weight |
|---|---:|---:|---:|
| 100 | 12.10 | 19.50 | 0.6 |
| 200 | 1.73 | 11.22 | 0.2 |
| 500 | 8.88 | 8.97 | 0.2 |
| 1000 | 4.27 | 5.24 | 0.6 |

The best setting is `K=100`. This is consistent with Gong et al. (2025), which selects HuBERT-base layer 6 with `k=100` using PNMI. The comparison is methodological only: this project reports Unit-BLEU over target units, while the paper evaluates a larger speech-to-speech system.

## Interpretation

Larger unit inventories were not automatically better. In this small Transformer setup, larger K values create sparser target sequences and a larger prediction vocabulary. That made exact unit prediction harder and reduced the usefulness of the n-gram language model.

The K sweep is therefore an honest negative result:

- `K=1000` has a richer acoustic inventory but is too sparse for this prototype.
- `K=200` benefits from LM fusion but has a weak greedy score.
- `K=500` is stable but gains little from the LM.
- `K=100` gives the best balance between target granularity and learnability.

## Qualitative Evaluation

The repo includes `10_verify_case.py`, which checks a predicted unit sequence by retrieving the closest training unit sequence and comparing Tibetan and Mandarin text.

One diagnostic failure is:

```text
Input sample: maqufa-002
Reference Mandarin: 不断加大高水平对外开放力度，
Retrieved Mandarin: 今天有人买牙膏吗？
```

This failure is useful: it shows that better Unit-BLEU and fluent synthesized Mandarin are not enough to guarantee semantic preservation. The result should be described as a working representation-learning and unit-prediction prototype, not as a deployable translation system.

## Limitations

- No human evaluation was run.
- Unit-BLEU does not measure semantic adequacy or speech quality.
- Mandarin target speech was synthesized with Edge-TTS.
- The repo does not train a unit vocoder.
- HuBERT is not fine-tuned for this dataset.
- The model omits the auxiliary ASR/ST/CTC objectives used by the reference system.

## Takeaway

The project delivers a complete, inspectable low-resource speech-translation intervention: data construction, target unit extraction, model training, K sweep, LM ablation, and qualitative error checking. The strongest result is `19.50` Unit-BLEU with `K=100` and LM weight `0.6`, but the qualitative diagnostic makes clear that the system is not ready for real-world communication use.
