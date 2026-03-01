# EYEMW Project — Claude Code Context

## What This Project Is
Research submission for the [EYEMW Dataset Competition](https://www.eyemindwander.com/competition/) (deadline: April 15, 2026). The paper argues that mind wandering is not monolithic — we use Latent Profile Analysis to discover MW subtypes, map their gaze signatures, and test robustness across environmental conditions.

## Project Structure
```
EYEMW/
├── database/                         # Raw dataset (Excel, not in git)
├── database_description/             # Data dictionary
├── competition/                      # Competition rules
├── deep_research/                    # Literature review
├── outputs/details/                  # Per-study documentation (Study001-026.md)
├── outputs/sources/                  # Source PDFs
├── scripts/
│   ├── phase1_data_preparation.py    # Load, profile, harmonize, eligibility
│   ├── phase2_lpa.py                 # Correlation analysis, GMM-based LPA, bootstrap
│   ├── phase3_gaze_discrimination.py # RF + logistic classifiers, feature importance
│   ├── phase4_slicing_analysis.py    # Environmental robustness slicing
│   ├── phase5_figures.py             # Publication figures and summary tables
│   └── run_all.py                    # Full pipeline runner
├── results/                          # Generated outputs (CSV, JSON, Parquet)
├── figures/                          # Generated figures (PNG)
├── paper/paper.md                    # Manuscript
├── docs/plan.md                      # Full research plan
└── .venv/                            # Python virtual environment
```

## Running the Pipeline
```bash
source .venv/bin/activate
python scripts/run_all.py        # Full pipeline (~34 min)
python scripts/phase1_data_preparation.py   # Individual phases
```

## Key Conventions
- All MW responses harmonized to 0–1 scale (binary kept as 0/1, Likert min-max normalized)
- Gaze features z-scored **within study** to remove study-level scaling
- TUT column is `TUTProbeResponse` (not `TUTResponse`)
- Intentionality data is entirely missing — not used
- Only 8 of 19 gaze features have data; Fixation-Time and AOI-Time columns are empty
- SHAP may fail on Python 3.14; RF Gini importance used as fallback
- scikit-learn 1.8+: don't pass `multi_class` to `LogisticRegression`

## Dataset Quick Reference
- **Source**: `database/Eye Tracking MW Database 1.4.xlsx`
- **Size**: 22,134 rows × 109 columns × 20 studies
- **Eligible studies**: 15 (require TUT + ≥2 other MW dimensions)
- **Subset A** (video): Studies 7,8,10-15,17 — TUT + Disengagement + Valence + Boredom (n=8,167)
- **Subset B** (listening): Studies 3,4 — TUT + FMT + Awareness + Disengagement (n=2,674)
- **MW dimensions**: TUT, Awareness, FMT, Disengagement, Valence, Arousal, Boredom (7 of 8; no Intentionality)
- **Gaze features**: Gazes, Fixations, UniqueGazes, UniqueGazeProportion, OffscreenGazes, OffScreenGazeProportion, AOIGazes, AOIGazeProportion

## Key Results
| Metric | Subset A | Subset B |
|--------|----------|----------|
| Profiles found | 5 | 5 |
| Bootstrap ARI | 0.82 | 0.66 |
| RF AUROC (gaze only) | 0.59 | 0.59 |
| RF AUROC (+ covariates) | 0.71 | 0.75 |
| Slice AUROC range | 0.59–0.71 | 0.57–0.60 |
