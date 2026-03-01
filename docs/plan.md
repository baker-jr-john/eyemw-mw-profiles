# EYEMW Competition Plan: "Mind Wandering is Not Monolithic"

## Status: Implemented (All 5 Phases Complete)

Pipeline runs end-to-end via `python scripts/run_all.py` (~34 minutes).

---

## Context

The EYEMW Dataset Competition (deadline: April 15, 2026) invites novel scientific insights from eye-tracking + mind wandering studies. The field overwhelmingly treats mind wandering as a binary state (on-task vs. off-task), but this dataset uniquely captures multiple MW dimensions across multiple task types. Our approach: use Latent Profile Analysis to discover empirically-derived MW subtypes, map their distinct gaze signatures, and validate robustness across environmental conditions/hardware ("slicing analysis").

## Research Question

Do different dimensions of mind wandering form distinct, empirically separable profiles with unique gaze signatures, and do these profiles remain stable across environmental conditions (lighting, device type, experimental setting) and task types?

## Deliverables

1. Short paper (`paper/paper.md`)
2. Public code repository (Python, reproducible)

---

## Phase 1: Data Preparation and Harmonization ✅

**Script**: `scripts/phase1_data_preparation.py`

### What was done
- Loaded `database/Eye Tracking MW Database 1.4.xlsx` (22,134 rows, 109 columns, 20 studies)
- Profiled each study: participant count, probes, MW dimensions, gaze features
- Built 20×8 eligibility matrix (studies × MW dimensions)
- Applied inclusion criterion: TUT + ≥2 additional MW dimensions → 15 eligible studies
- Harmonized MW scales to 0–1 (binary kept as 0/1; Likert min-max normalized; decreasing scales flipped)
- Z-scored 8 gaze features within study
- Tagged environmental metadata (ExpSetting, Lighting, DeviceType, etc.)

### Key decisions
- TUT column is `TUTProbeResponse` (not `TUTResponse` as data dictionary implies)
- Intentionality data completely missing — excluded from analysis
- Awareness lacks ScaleMin/ScaleMax — inferred as 1-7 Likert from observed data
- Some studies use pre-normalized 0-1 proportions for binary probes (e.g., Study 005 TUT: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0) — kept as-is
- Only 8 of 19 gaze features have substantial data; Fixation-Time, AOI-Time, and most Fixation columns are empty

### Outputs
- `results/harmonized_data.parquet` (12,850 rows)
- `results/eligibility_matrix.csv`
- `results/eligibility_info.json`
- `results/study_profiles.csv`
- `results/missingness_report.csv`
- `results/provenance_log.json`

---

## Phase 2: Latent Profile Analysis ✅

**Script**: `scripts/phase2_lpa.py`

### What was done
- Computed correlation matrices for MW dimensions (overall and per-subset)
- Fitted Gaussian Mixture Models for k=2 through 6 profiles
- Selected optimal k using BIC elbow method (capped at 5 for interpretability)
- Characterized profiles with mean/SD per dimension, auto-named based on MW signatures
- Created radar/spider plots
- Bootstrap validation (1,000 resamples): Adjusted Rand Index + average posterior probability

### Key findings
- **MW dimensions are NOT monolithic**: max |r| = 0.51 (Valence-Boredom); most r < 0.3
- **Subset A (video, 9 studies, n=8,167)**: 5 profiles identified
  - On-Task (49.4%), MW (22.7%), On-Task + Positive (10.1%), MW + Disengaged (10.0%), On-Task + Disengaged (7.8%)
  - Bootstrap ARI = 0.82, avg posterior = 1.00
- **Subset B (listening, 2 studies, n=2,674)**: 5 profiles identified
  - On-Task + Aware (High FMT) (38.1%), MW (30.4%), On-Task + Aware (14.9%), On-Task (11.4%), MW + Disengaged + Aware + Free-Flowing (5.2%)
  - Bootstrap ARI = 0.66, avg posterior = 0.99

### Outputs
- `results/labeled_data_SubsetA.parquet`, `results/labeled_data_SubsetB.parquet`
- `results/profile_stats_SubsetA.csv`, `results/profile_stats_SubsetB.csv`
- `results/bootstrap_results_SubsetA.json`, `results/bootstrap_results_SubsetB.json`
- `results/lpa_model_selection_SubsetA.csv`, `results/lpa_model_selection_SubsetB.csv`
- `figures/fig2_correlation_subset_A.png`, `figures/fig2_correlation_subset_B.png`
- `figures/fig3_radar_profiles_SubsetA.png`, `figures/fig3_radar_profiles_SubsetB.png`
- `figures/model_selection_SubsetA.png`, `figures/model_selection_SubsetB.png`

---

## Phase 3: Gaze Signature Discrimination ✅

**Script**: `scripts/phase3_gaze_discrimination.py`

### What was done
- Multinomial logistic regression (5-fold CV)
- Random Forest classifier (500 trees, balanced weights, 5-fold CV)
- Feature importance via RF Gini importance (SHAP failed on Python 3.14)
- Cross-task comparison (only 1 task type per subset, so limited)
- Binary baseline comparison (on-task vs. any-MW)
- Covariate analysis (age, gender, screen dimensions)

### Key findings
- **RF AUROC = 0.59** for both subsets (above chance 0.50, modest)
- **Binary baseline AUROC = 0.70 (A) / 0.63 (B)** — binary detection easier than multi-profile
- **Covariates boost AUROC to 0.71 (A) / 0.75 (B)** — substantial improvement
- Top gaze features: Gazes, UniqueGazeProportion, UniqueGazes, AOIGazeProportion
- Fixations contributed zero importance (data from only 1 study per subset)

### Outputs
- `results/feature_importance_SubsetA.csv`, `results/feature_importance_SubsetB.csv`
- `results/logistic_coefs_SubsetA.csv`, `results/logistic_coefs_SubsetB.csv`
- `results/binary_comparison_SubsetA.json`, `results/binary_comparison_SubsetB.json`
- `results/phase3_summary.json`
- `figures/fig4_shap_SubsetA.png`, `figures/fig4_shap_SubsetB.png`
- `figures/confusion_matrix_SubsetA.png`, `figures/confusion_matrix_SubsetB.png`

---

## Phase 4: Environmental Robustness Slicing Analysis ✅

**Script**: `scripts/phase4_slicing_analysis.py`

### What was done
- Chi-squared tests for profile distribution differences across environmental slices
- Within-slice LPA to assess structural consistency (ARI vs. full-sample labels)
- Within-slice gaze discrimination AUROCs
- Feature importance rank correlation (Spearman rho) across slices
- Moderation analysis identifying robust vs. environment-sensitive features

### Key findings
- **Profile distributions differ** across lighting (p=.002), device (p<.001), setting (p<.001) — but same profile *structures* emerge within slices (ARI=0.80–0.91)
- **Gaze discrimination stable**: AUROC range 0.59–0.71 across slices
- **Feature importance robust** across lighting (rho=0.71–0.86) and devices (rho=0.88)
- **Feature importance unstable** for home vs. public (rho=0.14)
- Surprising: dim-lit and no-lighting conditions yield *higher* AUROCs

### Outputs
- `results/robustness_table_SubsetA.csv`, `results/robustness_table_SubsetB.csv`
- `results/profile_stability_SubsetA.csv`, `results/profile_stability_SubsetB.csv`
- `results/moderation_analysis_SubsetA.csv`, `results/moderation_analysis_SubsetB.csv`
- `figures/fig5_robustness_heatmap_SubsetA.png`, `figures/fig5_robustness_heatmap_SubsetB.png`

---

## Phase 5: Synthesis and Paper ✅

**Script**: `scripts/phase5_figures.py`

### What was done
- Generated Figure 1 (eligibility matrix)
- Verified all other figures exist
- Created summary results table (`results/paper_summary.json`)
- Wrote full manuscript (`paper/paper.md`)

### Paper structure
1. Abstract
2. Introduction (~1 page): monolithic MW problem
3. Related Work (~0.5 page): MW detection, MW theory, environmental gap
4. Data and Methods (~1.5 pages): dataset, harmonization, LPA, discrimination, slicing
5. Results (~2 pages): correlations, profiles, gaze discrimination, robustness
6. Discussion (~1 page): implications, limitations, future work
7. References

### All figures (13 total)
| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `fig1_eligibility_matrix.png` | 20×8 study eligibility matrix |
| Fig 2a | `fig2_correlation_subset_A.png` | MW correlation heatmap (Subset A) |
| Fig 2b | `fig2_correlation_subset_B.png` | MW correlation heatmap (Subset B) |
| Fig 3a | `fig3_radar_profiles_SubsetA.png` | Radar plot of 5 MW profiles (video) |
| Fig 3b | `fig3_radar_profiles_SubsetB.png` | Radar plot of 5 MW profiles (listening) |
| Fig 4a | `fig4_shap_SubsetA.png` | Gaze feature importance (Subset A) |
| Fig 4b | `fig4_shap_SubsetB.png` | Gaze feature importance (Subset B) |
| Fig 5a | `fig5_robustness_heatmap_SubsetA.png` | AUROC by environmental slice (A) |
| Fig 5b | `fig5_robustness_heatmap_SubsetB.png` | AUROC by environmental slice (B) |
| — | `model_selection_SubsetA.png` | BIC/AIC/entropy curves (A) |
| — | `model_selection_SubsetB.png` | BIC/AIC/entropy curves (B) |
| — | `confusion_matrix_SubsetA.png` | RF confusion matrix (A) |
| — | `confusion_matrix_SubsetB.png` | RF confusion matrix (B) |

---

## Contingency Assessment

| Risk | Status | Outcome |
|------|--------|---------|
| LPA yields only 2 profiles | Not triggered | 5 profiles found in both subsets |
| Gaze can't discriminate profiles | Partially triggered | AUROC=0.59 is modest; binary does better (0.63–0.70) |
| Missing data too severe | Not triggered | 8,167 complete cases in Subset A |
| Timeline slips | Not triggered | All 5 phases complete |

The modest gaze discrimination is framed honestly in the paper as "the precision ceiling" — what webcam gaze can tell us about MW *structure*. The covariates finding (AUROC to 0.71–0.75) and the robustness analysis provide additional value.
