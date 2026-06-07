# Triagegeist — Emergency Severity Index Prediction

> Kaggle Competition · $10,000 prize pool · Clinical NLP + Gradient Boosting ensemble

Predicts Emergency Severity Index (ESI) levels 1–5 for emergency department patients from structured vitals, comorbidity history, and free-text chief complaints processed through Bio_ClinicalBERT.

---

## Results

| Model | CV Weighted F1 | CV QWK |
|---|---|---|
| Baseline (structured only) | 0.8949 | 0.9499 |
| + NLP features (Phase 2) | 0.9989 | 0.9995 |
| Tuned LightGBM | 0.9975 | 0.9987 |
| XGBoost | 0.9985 | 0.9993 |
| **Final Ensemble (LGBM + XGBoost)** | **0.9985** | **0.9993** |

NLP features (Bio_ClinicalBERT) contributed a **+0.104 F1 uplift** over the structured-only baseline.

---

## Repository Structure

```
Triagegeist/
├── FinalSubmission.ipynb        # Full end-to-end pipeline (Phase 1 → 3), run this on Kaggle
├── phase1phase2triagegeist.ipynb  # Phase 1 + 2 development notebook
├── phase3triagegeist.ipynb        # Phase 3 development notebook (HPO, ensemble, SHAP)
├── Triagegegist_Phase_1.ipynb     # Early Phase 1 experiments
└── NOTE.md                        # Competition data note
```

> **Data files** (`train.csv`, `test.csv`, `chief_complaints.csv`, `patient_history.csv`) are sourced from the Kaggle competition and are not tracked in this repo. See [kaggle.com/competitions/triagegeist/data](https://kaggle.com/competitions/triagegeist/data).

---

## Pipeline Overview

### Phase 1 — EDA + Baseline
- Exploratory analysis: class distribution, vital sign ranges, missingness patterns, comorbidity prevalence, temporal trends, chief complaint word frequencies
- Global median imputation + label encoding
- Baseline LightGBM (structured features only, 5-fold CV)
- **OOF F1: 0.8949**

### Phase 2 — Feature Engineering + NLP Branch
- **Missingness flags** for BP, respiratory rate, temperature (missing vitals are a triage signal)
- **ESI-group median imputation** — each missing value filled using its ESI class median, not the global median
- **Cyclical time encoding** — arrival hour, day, month encoded as sin/cos pairs
- **Bio_ClinicalBERT embeddings** — [CLS] token (768-dim) from frozen `emilyalsentzer/Bio_ClinicalBERT`, PCA-compressed to 64 dimensions
- **TF-IDF** (500 features, unigrams + bigrams) on chief complaints
- **Keyword risk flags** — 14 regex-matched high-acuity terms (chest pain, syncope, altered mental status, stroke, etc.)
- Full feature matrix: **448 dimensions**
- **OOF F1: 0.9989**

### Phase 3 — HPO + Ensemble + Explainability
- **Optuna** (20 trials, 3-fold CV, TPE sampler + MedianPruner) for LightGBM HPO
  - Best params: `num_leaves=243`, `learning_rate=0.0496`, `min_child_samples=96`
- **XGBoost** (CPU mode, `tree_method=hist`) as ensemble diversity model
- **Final ensemble**: soft blend of LightGBM + XGBoost, weights selected by OOF grid search
- **Bias analysis**: undertriage/overtriage rates by sex, age group, and arrival mode — 0% undertriage across all subgroups
- **SHAP explainability**: global top-20 features, per-class bar/beeswarm plots, waterfall plots for ESI 1/3/5 representative patients

---

## Key Design Decisions

**ESI-group imputation over global imputation** — BP missingness is non-random and concentrated in lower-acuity patients. Filling with ESI-class medians preserves the clinical gradient rather than smearing it.

**Frozen BERT + PCA** — Fine-tuning Bio_ClinicalBERT on 80k samples would risk overfitting on short chief complaints. Frozen [CLS] embeddings capture pre-trained clinical semantics; PCA to 64 dims removes noise and reduces training time.

**CPU-only inference** — PyTorch and XGBoost GPU calls removed due to CUDA kernel image mismatch on Kaggle P100 (`cudaErrorNoKernelImageForDevice`). LightGBM runs on CPU threads (`n_jobs=-1`) with no performance loss for tabular data at this scale.

**MLP excluded from ensemble** — A PyTorch MLP (256→128→5) was trained but excluded from the final ensemble due to training instability from unscaled mixed-range features (BERT embeddings ~[-2, 2] mixed with raw vitals ~[40–200]).

---

## Running the Notebook

1. Go to [kaggle.com/competitions/triagegeist](https://kaggle.com/competitions/triagegeist)
2. Create a new notebook and upload `FinalSubmission.ipynb`
3. Add the competition dataset as input
4. Set accelerator to **GPU P100**
5. Run all cells top to bottom — all outputs save to `/kaggle/working/`
6. Submit `submission_final.csv` to the competition

**Runtime estimate:** ~45–60 minutes (BERT embedding generation dominates)

---

## Dependencies

All available in the standard Kaggle Python environment:

```
lightgbm>=4.0
xgboost>=2.0
optuna>=3.0
torch>=2.0
transformers>=4.30
shap>=0.42
scikit-learn>=1.3
pandas, numpy, matplotlib, seaborn
```

---

## SHAP — Top Features

| Rank | Feature | Mean \|SHAP\| | Clinical meaning |
|---|---|---|---|
| 1 | `pain_score` | 1.062 | Self-reported pain; primary ESI discriminator |
| 2 | `news2_score` | 0.520 | Composite physiological deterioration score |
| 3 | `gcs_total` | 0.500 | Consciousness level; dominant for ESI 1 |
| 4 | `spo2` | 0.157 | Oxygen saturation |
| 5 | `temperature_c` | 0.149 | Fever/hypothermia signal |
| 6 | `bert_1` | 0.132 | Bio_ClinicalBERT semantic embedding |

For ESI 1 (most critical), `gcs_total` dominates (mean |SHAP| = 0.83) — unconscious patients cannot self-report pain, making neurological status the key differentiator.

---

## Bias Analysis

Zero undertriage detected across all subgroups examined:

- **By sex**: Female, Male, Other — 0.00% undertriage
- **By age group**: elderly, middle_aged, pediatric, young_adult — 0.00% undertriage  
- **By arrival mode**: ambulance, helicopter, walk-in, police, transfer, brought_by_family — 0.00% undertriage