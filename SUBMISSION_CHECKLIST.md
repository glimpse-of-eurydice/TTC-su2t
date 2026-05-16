# Submission Checklist

## Assessment Fit

| Criterion | Status | Evidence |
|---|---|---|
| Problem value and specificity | Strong | The repo targets Tibetan-to-Mandarin speech translation rather than general Tibetan ASR or general speech translation. |
| Methodological approach | Good for a course prototype | HuBERT units, K-means, Transformer S2UT, Unit-BLEU, LM ablation, and qualitative retrieval checks. |
| Engagement with the variety | Adequate | The project treats Tibetan as the source speech modality and avoids assuming a complete Tibetan ASR system. |
| Evaluation rigor | Honest | Held-out split, K sweep, LM-weight ablation, and explicit qualitative failure cases. |
| Scope realism | Strong | Small complete prototype with clear limitations instead of an overclaimed full S2ST system. |

## Commit to GitHub

Keep:

```text
README.md
RESULTS.md
CASE_ANALYSIS.md
EXPERIMENT_GUIDE.md
DEMO.md
LICENSE
requirements.txt
*.py
ablation_results*.csv
predicted_units*.json
verify_report*.json
lm_weight_ablation.png
TCST/text.json
data/train.csv
data/dev.csv
data/test.csv
data/TCST/target_units.json
```

Do not commit:

```text
TCST/wav/
data/TCST/wav_zh/
checkpoints/
*.pth
*.pkl
*.wav
__pycache__/
.DS_Store
.vscode/
```

## Final Checks

```bash
python test_env.py
python model.py --num-clusters 100
python dataset.py --num-clusters 100 --batch-size 2

git status --short
git check-ignore -v checkpoints/best_s2ut_model.pth
git check-ignore -v TCST/wav
git check-ignore -v data/TCST/wav_zh
```

If the checkpoint and audio are present locally:

```bash
python 07_evaluate.py --num-clusters 100 --max-len 600
python 10_verify_case.py --num-clusters 100 --knn-pool train
```

## Showcase Story

1. The problem is a concrete low-resource access gap: Tibetan speech to Mandarin communication.
2. The intervention avoids a fragile ASR-first path by predicting Mandarin-side acoustic units.
3. The implementation covers data, learning, and evaluation.
4. The main result is `19.50` Unit-BLEU with `K=100` and LM weight `0.6`.
5. The negative result is that larger K values were not better.
6. The ethical takeaway is that transparent limitations matter: this is a prototype for access-oriented research, not a deployable medical translator.
