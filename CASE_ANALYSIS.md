# Qualitative Case Analysis

Unit-BLEU is useful for comparing unit-prediction settings, but it does not prove that the system preserves meaning. This file describes the qualitative check used for the final report and showcase.

## Save Per-Example Predictions

After reproducing the final `K=100` checkpoint, run:

```bash
python 07_evaluate.py \
  --num-clusters 100 \
  --max-len 600 \
  --save-predictions results/k100_test_predictions.jsonl
```

Each JSONL row contains:

```text
sample_id
tibetan_audio
tibetan_text
chinese_text
ref_len
pred_len
unit_edit_distance
normalized_unit_edit_distance
reference_units
predicted_units
```

## Select Good and Bad Candidates

```bash
python - <<'PY'
import json

path = "results/k100_test_predictions.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
rows.sort(key=lambda x: x["normalized_unit_edit_distance"])

print("GOOD CASE CANDIDATES")
for row in rows[:10]:
    print(
        row["sample_id"],
        f'norm_edit={row["normalized_unit_edit_distance"]:.3f}',
        f'pred_len={row["pred_len"]}',
        f'ref_len={row["ref_len"]}',
        row["chinese_text"],
    )

print("\nBAD CASE CANDIDATES")
for row in rows[-10:]:
    print(
        row["sample_id"],
        f'norm_edit={row["normalized_unit_edit_distance"]:.3f}',
        f'pred_len={row["pred_len"]}',
        f'ref_len={row["ref_len"]}',
        row["chinese_text"],
    )
PY
```

## Verify a Selected Case

For a selected test sample:

```bash
python 06_inference.py \
  --num-clusters 100 \
  --test-audio /path/to/selected/audio.wav \
  --max-len 600 \
  --output-json predicted_units_case.json

python 10_verify_case.py \
  --num-clusters 100 \
  --predicted-units predicted_units_case.json \
  --test-audio /path/to/selected/audio.wav \
  --knn-pool train \
  --save-report verify_report_case.json
```

## How to Interpret the Cases

| Case | What to show | Interpretation |
|---|---|---|
| Good case | Low normalized unit edit distance and retrieved text close to the reference | The unit model learned some source-conditioned mapping. |
| Bad case | High edit distance or semantically wrong retrieved text | Unit overlap and fluent TTS do not guarantee meaning preservation. |

If no strong semantic success case is found, state that directly. A failed qualitative check is still useful evidence because it prevents overclaiming.

## Current Diagnostic Failure

The current diagnostic run includes a clear semantic failure:

```text
Input sample: maqufa-002
Reference Mandarin: 不断加大高水平对外开放力度，
Retrieved Mandarin: 今天有人买牙膏吗？
```

This example is useful for the showcase because it separates "producing a fluent Mandarin utterance" from "preserving the meaning of Tibetan speech." It supports the final ethical claim: this work is a transparent prototype for low-resource access, not a system that should be deployed in medical or public-service settings.
