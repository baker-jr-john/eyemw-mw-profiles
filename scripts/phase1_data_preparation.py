"""
Phase 1: Data Preparation and Harmonization
============================================
Loads the EYEMW dataset, profiles studies, builds eligibility matrix,
harmonizes MW scales and gaze features, tags environmental metadata.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "database", "Eye Tracking MW Database 1.4.xlsx")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── MW dimension definitions ──
MW_DIMENSIONS = ['TUT', 'Intentionality', 'Awareness', 'FMT',
                 'Disengagement', 'Valence', 'Arousal', 'Boredom']

# TUT uses "TUTProbeResponse" while others use "{dim}Response"
MW_RESPONSE_COLS = {dim: f"{dim}Response" for dim in MW_DIMENSIONS}
MW_RESPONSE_COLS['TUT'] = 'TUTProbeResponse'  # actual column name in dataset

MW_SCALE_DIR_COLS = {dim: f"{dim}ScaleDirection" for dim in MW_DIMENSIONS}
MW_SCALE_MIN_COLS = {dim: f"{dim}ScaleMin" for dim in MW_DIMENSIONS}
MW_SCALE_MAX_COLS = {dim: f"{dim}ScaleMax" for dim in MW_DIMENSIONS}
MW_PROBE_TYPE_COLS = {dim: f"{dim}ProbeType" for dim in MW_DIMENSIONS}

GAZE_FEATURES = [
    'Gazes', 'Fixations', 'FixationTime', 'UniqueGazes', 'UniqueGazeProportion',
    'UniqueFixations', 'UniqueFixProportion', 'OffscreenGazes', 'OffScreenGazeProportion',
    'OffScreenGazeTime', 'OffScreenFixations', 'OffScreenFixProportion', 'OffScreenFixTime',
    'AOIGazes', 'AOIGazeProportion', 'AOIGazeTime', 'AOIFixations',
    'AOIFixationProportion', 'AOIFixationTime'
]

ENV_METADATA_COLS = ['ExpSetting', 'Lighting', 'DeviceType', 'IOS',
                     'EyeTrackerType', 'RefreshRate', 'SamplingRate', 'TaskType']

DEMOGRAPHIC_COLS = ['Age', 'Gender']


def load_data():
    """Load the Excel dataset."""
    print("Loading dataset...")
    df = pd.read_excel(DATA_PATH)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Studies: {df['StudyNum'].nunique()}")
    return df


def profile_dataset(df):
    """Profile overall dataset: missingness, dtypes, basic stats."""
    print("\n=== Dataset Profile ===")

    # Missingness per column
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    miss_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
    miss_df = miss_df[miss_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    print(f"\n  Columns with missing data: {len(miss_df)} / {len(df.columns)}")
    print(f"  Top 10 most missing:")
    for col, row in miss_df.head(10).iterrows():
        print(f"    {col}: {row['missing_pct']:.1f}% ({int(row['missing_count']):,} rows)")

    miss_df.to_csv(os.path.join(RESULTS_DIR, "missingness_report.csv"))
    return miss_df


def profile_studies(df):
    """Profile each study: participant count, probes, MW dimensions, gaze features."""
    print("\n=== Study Profiles ===")

    study_profiles = []
    for study_num in sorted(df['StudyNum'].unique()):
        sdf = df[df['StudyNum'] == study_num]

        profile = {
            'StudyNum': study_num,
            'n_rows': len(sdf),
            'n_participants': sdf['ParticipantNum'].nunique(),
            'probes_per_participant': round(len(sdf) / max(sdf['ParticipantNum'].nunique(), 1), 1),
            'task_type': sdf['TaskType'].mode().iloc[0] if not sdf['TaskType'].isna().all() else np.nan,
        }

        # MW dimension availability
        for dim in MW_DIMENSIONS:
            resp_col = MW_RESPONSE_COLS[dim]
            if resp_col in sdf.columns:
                non_null = sdf[resp_col].notna().sum()
                profile[f'{dim}_available'] = non_null > 0
                profile[f'{dim}_n'] = non_null
                profile[f'{dim}_pct'] = round(non_null / len(sdf) * 100, 1)
            else:
                profile[f'{dim}_available'] = False
                profile[f'{dim}_n'] = 0
                profile[f'{dim}_pct'] = 0.0

        # Gaze feature availability
        gaze_available = []
        for feat in GAZE_FEATURES:
            if feat in sdf.columns and sdf[feat].notna().sum() > 0:
                gaze_available.append(feat)
        profile['n_gaze_features'] = len(gaze_available)
        profile['gaze_features'] = ', '.join(gaze_available)

        # Environmental metadata
        for col in ENV_METADATA_COLS:
            if col in sdf.columns and not sdf[col].isna().all():
                profile[col] = sdf[col].mode().iloc[0] if not sdf[col].mode().empty else np.nan
            else:
                profile[col] = np.nan

        study_profiles.append(profile)

    profiles_df = pd.DataFrame(study_profiles)
    profiles_df.to_csv(os.path.join(RESULTS_DIR, "study_profiles.csv"), index=False)

    # Print summary
    task_map = {1: 'reading', 2: 'listening', 3: 'math', 4: 'video', 5: 'other'}
    for _, p in profiles_df.iterrows():
        tt = task_map.get(p['task_type'], '?')
        dims = [d for d in MW_DIMENSIONS if p.get(f'{d}_available', False)]
        print(f"  Study {int(p['StudyNum']):03d}: {p['n_participants']:3.0f} participants, "
              f"{p['n_rows']:5d} rows, task={tt:8s}, "
              f"MW dims=[{', '.join(dims)}], gaze_feats={p['n_gaze_features']}")

    return profiles_df


def build_eligibility_matrix(profiles_df):
    """Build 26x8 study eligibility matrix and determine eligible studies."""
    print("\n=== Study Eligibility Matrix ===")

    # Build matrix
    matrix = pd.DataFrame(index=[f"Study {int(s):03d}" for s in profiles_df['StudyNum']])
    for dim in MW_DIMENSIONS:
        matrix[dim] = profiles_df[f'{dim}_available'].values

    matrix.to_csv(os.path.join(RESULTS_DIR, "eligibility_matrix.csv"))
    print(matrix.to_string())

    # Apply inclusion criterion: TUT + at least 2 additional dimensions
    eligible = []
    for idx, row in matrix.iterrows():
        study_num = int(idx.split()[-1])
        if row['TUT']:
            other_dims = sum(row[d] for d in MW_DIMENSIONS if d != 'TUT')
            if other_dims >= 2:
                eligible.append(study_num)

    print(f"\n  Eligible studies (TUT + ≥2 other dims): {eligible}")
    print(f"  Total eligible: {len(eligible)}")

    # Categorize by subset
    subset_a_dims = ['TUT', 'Disengagement', 'Valence', 'Boredom']
    subset_b_dims = ['TUT', 'FMT', 'Awareness', 'Disengagement']

    subset_a = []
    subset_b = []
    for idx, row in matrix.iterrows():
        study_num = int(idx.split()[-1])
        if all(row[d] for d in subset_a_dims):
            subset_a.append(study_num)
        if all(row[d] for d in subset_b_dims):
            subset_b.append(study_num)

    print(f"\n  Subset A (TUT+Disengagement+Valence+Boredom): {subset_a}")
    print(f"  Subset B (TUT+FMT+Awareness+Disengagement): {subset_b}")

    eligibility_info = {
        'eligible_studies': eligible,
        'subset_a': subset_a,
        'subset_a_dims': subset_a_dims,
        'subset_b': subset_b,
        'subset_b_dims': subset_b_dims,
    }

    with open(os.path.join(RESULTS_DIR, "eligibility_info.json"), 'w') as f:
        json.dump(eligibility_info, f, indent=2)

    return eligibility_info


def harmonize_mw_scales(df, eligible_studies):
    """Normalize MW responses to 0-1 scale, respecting ScaleDirection."""
    print("\n=== Harmonizing MW Scales ===")

    df_elig = df[df['StudyNum'].isin(eligible_studies)].copy()

    harmonized_cols = {}
    for dim in MW_DIMENSIONS:
        resp_col = MW_RESPONSE_COLS[dim]
        dir_col = MW_SCALE_DIR_COLS[dim]
        min_col = MW_SCALE_MIN_COLS[dim]
        max_col = MW_SCALE_MAX_COLS[dim]
        probe_type_col = MW_PROBE_TYPE_COLS[dim]
        norm_col = f"{dim}_norm"

        if resp_col not in df_elig.columns:
            print(f"  {dim}: column {resp_col} not found, skipping")
            continue

        has_data = df_elig[resp_col].notna()
        if has_data.sum() == 0:
            print(f"  {dim}: no data, skipping")
            continue

        # Initialize normalized column
        df_elig[norm_col] = np.nan

        # Process per-study to handle different encodings
        for study_num in df_elig['StudyNum'].unique():
            study_mask = (df_elig['StudyNum'] == study_num) & has_data
            if study_mask.sum() == 0:
                continue

            resp = df_elig.loc[study_mask, resp_col].astype(float)

            # Determine if this is binary/pre-normalized or needs scaling
            is_binary = False
            if probe_type_col in df_elig.columns:
                pt_vals = df_elig.loc[study_mask, probe_type_col].dropna().unique()
                is_binary = len(pt_vals) > 0 and pt_vals[0] == 1

            if is_binary:
                # ProbeType=1: responses are already 0/1 or pre-normalized to 0-1
                df_elig.loc[study_mask, norm_col] = resp.clip(0, 1)
            else:
                # ProbeType=2 (scale): min-max normalize
                # Try to use provided ScaleMin/ScaleMax
                has_scale_info = (min_col in df_elig.columns and max_col in df_elig.columns and
                                  df_elig.loc[study_mask, min_col].notna().any() and
                                  df_elig.loc[study_mask, max_col].notna().any())

                if has_scale_info:
                    smin = df_elig.loc[study_mask, min_col].astype(float)
                    smax = df_elig.loc[study_mask, max_col].astype(float)
                else:
                    # Infer from observed range (e.g., Awareness has no ScaleMin/Max)
                    observed_min = resp.min()
                    observed_max = resp.max()
                    # Use 1-7 default for Likert if observed range suggests it
                    if observed_min >= 1 and observed_max <= 7:
                        smin = pd.Series(1.0, index=resp.index)
                        smax = pd.Series(7.0, index=resp.index)
                    elif observed_min >= 1 and observed_max <= 9:
                        smin = pd.Series(1.0, index=resp.index)
                        smax = pd.Series(9.0, index=resp.index)
                    else:
                        smin = pd.Series(observed_min, index=resp.index)
                        smax = pd.Series(observed_max, index=resp.index)
                    print(f"    {dim} Study {study_num:03d}: inferred scale [{smin.iloc[0]}-{smax.iloc[0]}]")

                denom = smax - smin
                denom = denom.replace(0, np.nan)
                normalized = (resp - smin) / denom

                # Flip decreasing scales (direction == 2)
                if dir_col in df_elig.columns:
                    dir_vals = df_elig.loc[study_mask, dir_col]
                    decreasing = dir_vals == 2
                    normalized[decreasing] = 1.0 - normalized[decreasing]

                df_elig.loc[study_mask, norm_col] = normalized.clip(0, 1)

        n_harmonized = df_elig[norm_col].notna().sum()
        if n_harmonized > 0:
            print(f"  {dim}: {n_harmonized:,} values harmonized "
                  f"(range: {df_elig[norm_col].min():.2f} - {df_elig[norm_col].max():.2f})")
            harmonized_cols[dim] = norm_col
        else:
            print(f"  {dim}: no values harmonized")

    return df_elig, harmonized_cols


def harmonize_gaze_features(df_elig):
    """Z-score gaze features within study to remove study-level scaling."""
    print("\n=== Harmonizing Gaze Features ===")

    gaze_z_cols = {}
    for feat in GAZE_FEATURES:
        if feat not in df_elig.columns:
            continue
        if df_elig[feat].notna().sum() == 0:
            continue

        z_col = f"{feat}_z"
        # Z-score within study
        df_elig[z_col] = df_elig.groupby('StudyNum')[feat].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

        n_valid = df_elig[z_col].notna().sum()
        if n_valid > 0:
            gaze_z_cols[feat] = z_col
            print(f"  {feat}: z-scored ({n_valid:,} values)")

    print(f"\n  Total gaze features harmonized: {len(gaze_z_cols)}")
    return df_elig, gaze_z_cols


def tag_environmental_metadata(df_elig):
    """Ensure environmental metadata is properly tagged and categorized."""
    print("\n=== Environmental Metadata ===")

    label_maps = {
        'TaskType': {1: 'reading', 2: 'listening', 3: 'math', 4: 'video', 5: 'other'},
        'ExpSetting': {1: 'lab', 2: 'home', 3: 'classroom', 4: 'public'},
        'Lighting': {1: 'well_lit', 2: 'dim_lit', 3: 'no_lighting'},
        'DeviceType': {1: 'computer', 2: 'laptop', 3: 'phone', 4: 'VR'},
        'EyeTrackerType': {1: 'commercial', 2: 'webcam', 3: 'VR'},
    }

    for col, mapping in label_maps.items():
        label_col = f"{col}_label"
        if col in df_elig.columns:
            df_elig[label_col] = df_elig[col].map(mapping)
            vc = df_elig[label_col].value_counts()
            print(f"  {col}:")
            for val, count in vc.items():
                print(f"    {val}: {count:,}")

    return df_elig


def save_outputs(df_elig, harmonized_cols, gaze_z_cols, eligibility_info):
    """Save the harmonized dataset and provenance log."""
    print("\n=== Saving Outputs ===")

    # Save harmonized dataset
    out_path = os.path.join(RESULTS_DIR, "harmonized_data.parquet")
    df_elig.to_parquet(out_path, index=False)
    print(f"  Saved harmonized data: {out_path} ({len(df_elig):,} rows)")

    # Also save as CSV for inspection
    csv_path = os.path.join(RESULTS_DIR, "harmonized_data.csv")
    df_elig.to_csv(csv_path, index=False)

    # Save provenance log
    provenance = {
        'source_file': DATA_PATH,
        'total_rows_original': None,  # filled in main
        'eligible_studies': eligibility_info['eligible_studies'],
        'rows_after_filtering': len(df_elig),
        'mw_harmonized_columns': harmonized_cols,
        'gaze_z_columns': gaze_z_cols,
        'subset_a': eligibility_info['subset_a'],
        'subset_b': eligibility_info['subset_b'],
        'harmonization_notes': [
            'Binary probes (ProbeType=1): kept as 0/1',
            'Likert scales (ProbeType=2): normalized to 0-1 via (resp-min)/(max-min)',
            'Decreasing scales (ScaleDirection=2): flipped via 1 - normalized',
            'Gaze features: z-scored within each study',
        ]
    }

    with open(os.path.join(RESULTS_DIR, "provenance_log.json"), 'w') as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"  Saved provenance log")

    return provenance


def main():
    print("=" * 70)
    print("PHASE 1: Data Preparation and Harmonization")
    print("=" * 70)

    # 1.1 Load and profile
    df = load_data()
    miss_df = profile_dataset(df)
    profiles_df = profile_studies(df)

    # 1.2 Build eligibility matrix
    eligibility_info = build_eligibility_matrix(profiles_df)

    # 1.3 Harmonize MW scales
    df_elig, harmonized_cols = harmonize_mw_scales(df, eligibility_info['eligible_studies'])

    # 1.4 Harmonize gaze features
    df_elig, gaze_z_cols = harmonize_gaze_features(df_elig)

    # 1.5 Tag environmental metadata
    df_elig = tag_environmental_metadata(df_elig)

    # Save
    provenance = save_outputs(df_elig, harmonized_cols, gaze_z_cols, eligibility_info)
    provenance['total_rows_original'] = len(df)

    # Re-save with original count
    with open(os.path.join(RESULTS_DIR, "provenance_log.json"), 'w') as f:
        json.dump(provenance, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("Phase 1 Complete!")
    print(f"  Original dataset: {len(df):,} rows across {df['StudyNum'].nunique()} studies")
    print(f"  Eligible studies: {len(eligibility_info['eligible_studies'])}")
    print(f"  Harmonized dataset: {len(df_elig):,} rows")
    print(f"  MW dimensions harmonized: {len(harmonized_cols)}")
    print(f"  Gaze features z-scored: {len(gaze_z_cols)}")
    print("=" * 70)

    return df_elig, eligibility_info, harmonized_cols, gaze_z_cols


if __name__ == "__main__":
    main()
