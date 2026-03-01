# Mind Wandering is Not Monolithic

Code repository for "Mind Wandering is Not Monolithic: Latent Profiles of Mind Wandering Dimensions Reveal Distinct Gaze Signatures Across Environmental Conditions" — submitted to the [EYEMW Dataset Competition](https://www.eyemindwander.com/competition/) (April 15, 2026).

## Research Question

Do different dimensions of mind wandering form distinct, empirically separable profiles with unique gaze signatures, and do these profiles remain stable across environmental conditions?

## Key Findings

- **MW is not monolithic**: Low-to-moderate correlations between MW dimensions (max |r| = 0.51)
- **5 MW profiles** identified via Latent Profile Analysis in video tasks (n=8,167) and listening/reading tasks (n=2,674)
- **Gaze discrimination**: Webcam-based gaze features achieve AUROC = 0.59 for multi-profile classification
- **Environmental robustness**: Feature importance rankings are stable across lighting (rho=0.71-0.86) and device types (rho=0.88)

## Repository Structure

```
EYEMW/
├── database/                    # EYEMW dataset (not included; obtain from eyemindwander.com)
├── scripts/
│   ├── phase1_data_preparation.py   # Data loading, harmonization, eligibility
│   ├── phase2_lpa.py                # Latent Profile Analysis + bootstrap validation
│   ├── phase3_gaze_discrimination.py # RF classifiers, feature importance, cross-task
│   ├── phase4_slicing_analysis.py   # Environmental robustness slicing
│   ├── phase5_figures.py            # Publication figures and summary tables
│   └── run_all.py                   # Execute full pipeline
├── results/                     # Generated: CSV/JSON/parquet outputs
├── figures/                     # Generated: PNG figures
├── paper/
│   └── paper.md                 # Manuscript
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas openpyxl numpy scipy scikit-learn matplotlib seaborn statsmodels pyarrow
```

## Running

```bash
# Full pipeline
source .venv/bin/activate
python scripts/run_all.py

# Or individual phases
python scripts/phase1_data_preparation.py
python scripts/phase2_lpa.py
python scripts/phase3_gaze_discrimination.py
python scripts/phase4_slicing_analysis.py
python scripts/phase5_figures.py
```

## Methods Summary

1. **Data Harmonization**: Binary probes kept as 0/1; Likert scales min-max normalized to 0-1; gaze features z-scored within study
2. **Latent Profile Analysis**: Gaussian Mixture Models (k=2-6), BIC-selected, bootstrap-validated (1000 resamples)
3. **Gaze Discrimination**: Random Forest (500 trees, balanced weights, 5-fold CV) with multiclass AUROC
4. **Environmental Slicing**: Within-slice AUROC and feature importance rank correlations across lighting, device, and setting conditions
