"""
Phase 4: Environmental Robustness Slicing Analysis
====================================================
Tests whether MW profiles and gaze discrimination hold
across environmental conditions, hardware, and task types.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from scipy import stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

GAZE_FEATURES_Z = [
    'Gazes_z', 'Fixations_z', 'FixationTime_z', 'UniqueGazes_z',
    'UniqueGazeProportion_z', 'UniqueFixations_z', 'UniqueFixProportion_z',
    'OffscreenGazes_z', 'OffScreenGazeProportion_z', 'OffScreenGazeTime_z',
    'OffScreenFixations_z', 'OffScreenFixProportion_z', 'OffScreenFixTime_z',
    'AOIGazes_z', 'AOIGazeProportion_z', 'AOIGazeTime_z',
    'AOIFixations_z', 'AOIFixationProportion_z', 'AOIFixationTime_z'
]

# Slicing dimensions with their labels
SLICING_DIMS = {
    'Lighting': {1: 'well_lit', 2: 'dim_lit', 3: 'no_lighting'},
    'DeviceType': {1: 'computer', 2: 'laptop', 3: 'phone', 4: 'VR'},
    'ExpSetting': {1: 'lab', 2: 'home', 3: 'classroom', 4: 'public'},
    'EyeTrackerType': {1: 'commercial', 2: 'webcam', 3: 'VR'},
    'TaskType': {1: 'reading', 2: 'listening', 3: 'math', 4: 'video', 5: 'other'},
}


def load_labeled_data(subset_name):
    """Load LPA-labeled data."""
    path = os.path.join(RESULTS_DIR, f"labeled_data_{subset_name}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def profile_stability_across_slices(df, subset_name):
    """Test if MW profiles emerge similarly within each environmental slice."""
    print(f"\n=== 4.2 Profile Stability Across Slices ({subset_name}) ===")

    df_valid = df[df['profile'].notna()].copy()
    df_valid['profile'] = df_valid['profile'].astype(int)

    stability_results = []

    for dim, label_map in SLICING_DIMS.items():
        if dim not in df_valid.columns:
            continue

        print(f"\n  Slicing by {dim}:")

        # Test whether profile distributions differ across slices (chi-squared)
        contingency_data = {}
        for code, label in label_map.items():
            slice_df = df_valid[df_valid[dim] == code]
            if len(slice_df) >= 20:
                contingency_data[label] = slice_df['profile'].value_counts().sort_index()

        if len(contingency_data) < 2:
            print(f"    Too few slices with sufficient data, skipping.")
            continue

        # Build contingency table
        contingency = pd.DataFrame(contingency_data).fillna(0).astype(int)
        print(f"    Contingency table:")
        print(f"    {contingency.to_string()}")

        # Chi-squared test
        chi2, p, dof, expected = sp_stats.chi2_contingency(contingency.values)
        print(f"    Chi-squared: {chi2:.2f}, df={dof}, p={p:.4f}")

        stability_results.append({
            'dimension': dim,
            'n_slices': len(contingency_data),
            'chi2': chi2,
            'p_value': p,
            'dof': dof,
            'significant': p < 0.05,
            'interpretation': 'Profile distribution differs across slices' if p < 0.05
                            else 'Profile distribution stable across slices'
        })

        # Re-run LPA within each slice and compare (for slices with enough data)
        slice_aris = []
        for code, label in label_map.items():
            slice_df = df_valid[df_valid[dim] == code]
            if len(slice_df) < 50:
                continue

            # Get MW dimensions for this subset
            mw_cols = [c for c in df_valid.columns if c.endswith('_norm') and
                       df_valid[c].notna().sum() > len(df_valid) * 0.3]
            slice_mw = slice_df[mw_cols].dropna()

            if len(slice_mw) < 30:
                continue

            n_profiles = df_valid['profile'].nunique()
            scaler = StandardScaler()
            X_slice = scaler.fit_transform(slice_mw.values)

            try:
                gmm = GaussianMixture(n_components=min(n_profiles, len(slice_mw) // 10),
                                       covariance_type='full', n_init=10,
                                       max_iter=300, random_state=42, reg_covar=1e-3)
                gmm.fit(X_slice)
                slice_labels = gmm.predict(X_slice)

                # Compare against original labels
                orig_labels = slice_df.loc[slice_mw.index, 'profile'].values
                ari = adjusted_rand_score(orig_labels, slice_labels)
                slice_aris.append({'label': label, 'ari': ari, 'n': len(slice_mw)})
                print(f"    Within-slice LPA ({label}, n={len(slice_mw)}): ARI={ari:.3f}")
            except Exception as e:
                print(f"    Within-slice LPA ({label}): failed ({e})")

    stability_df = pd.DataFrame(stability_results)
    stability_df.to_csv(os.path.join(RESULTS_DIR, f"profile_stability_{subset_name}.csv"),
                        index=False)
    return stability_df


def gaze_discrimination_by_slice(df, subset_name):
    """Compute AUROC for gaze-based profile discrimination within each slice."""
    print(f"\n=== 4.3 Gaze Discrimination by Slice ({subset_name}) ===")

    df_valid = df[df['profile'].notna()].copy()
    df_valid['profile'] = df_valid['profile'].astype(int)

    available_gaze = [c for c in GAZE_FEATURES_Z if c in df_valid.columns]

    robustness_results = []

    for dim, label_map in SLICING_DIMS.items():
        if dim not in df_valid.columns:
            continue

        print(f"\n  Slicing by {dim}:")

        for code, label in label_map.items():
            slice_df = df_valid[df_valid[dim] == code]
            if len(slice_df) < 30 or slice_df['profile'].nunique() < 2:
                continue

            X_slice = slice_df[available_gaze].fillna(0)
            y_slice = slice_df['profile']

            # Need at least 2 samples per class for CV
            min_class_size = y_slice.value_counts().min()
            if min_class_size < 5:
                continue

            n_folds = min(5, min_class_size)
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                         class_weight='balanced', random_state=42, n_jobs=-1)
            try:
                y_pred_proba = cross_val_predict(rf, X_slice, y_slice, cv=cv,
                                                  method='predict_proba')
                y_pred = cross_val_predict(rf, X_slice, y_slice, cv=cv)

                acc = accuracy_score(y_slice, y_pred)
                if y_slice.nunique() == 2:
                    auroc = roc_auc_score(y_slice, y_pred_proba[:, 1])
                else:
                    auroc = roc_auc_score(y_slice, y_pred_proba,
                                          multi_class='ovr', average='weighted')

                robustness_results.append({
                    'dimension': dim,
                    'slice': label,
                    'n': len(X_slice),
                    'n_profiles': y_slice.nunique(),
                    'auroc': auroc,
                    'accuracy': acc
                })
                print(f"    {label}: n={len(X_slice)}, AUROC={auroc:.3f}, acc={acc:.3f}")
            except Exception as e:
                print(f"    {label}: failed ({e})")

    robustness_df = pd.DataFrame(robustness_results)
    robustness_df.to_csv(os.path.join(RESULTS_DIR, f"robustness_table_{subset_name}.csv"),
                         index=False)
    return robustness_df


def create_robustness_heatmap(robustness_df, subset_name):
    """Create heatmap of AUROC by environmental slice."""
    print(f"\n  Creating robustness heatmap...")

    if len(robustness_df) == 0:
        print("    No data for heatmap.")
        return

    # Pivot for heatmap
    pivot = robustness_df.pivot_table(index='dimension', columns='slice',
                                       values='auroc', aggfunc='first')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                vmin=0.3, vmax=1.0, ax=ax, linewidths=0.5)
    ax.set_title(f'Gaze Discrimination AUROC by Environmental Slice - {subset_name}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Environmental Slice')
    ax.set_ylabel('Slicing Dimension')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"fig5_robustness_heatmap_{subset_name}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: fig5_robustness_heatmap_{subset_name}.png")


def moderation_analysis(df, subset_name):
    """Test interaction effects: gaze feature x environmental variable."""
    print(f"\n=== 4.4 Moderation Analysis ({subset_name}) ===")

    df_valid = df[df['profile'].notna()].copy()
    df_valid['profile'] = df_valid['profile'].astype(int)

    available_gaze = [c for c in GAZE_FEATURES_Z if c in df_valid.columns]
    moderation_results = []

    for dim, label_map in SLICING_DIMS.items():
        if dim not in df_valid.columns or df_valid[dim].isna().all():
            continue

        # Compare feature importances across slices
        slice_importances = {}
        for code, label in label_map.items():
            slice_df = df_valid[df_valid[dim] == code]
            if len(slice_df) < 50 or slice_df['profile'].nunique() < 2:
                continue

            X = slice_df[available_gaze].fillna(0)
            y = slice_df['profile']
            min_class = y.value_counts().min()
            if min_class < 5:
                continue

            rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                         class_weight='balanced', random_state=42, n_jobs=-1)
            rf.fit(X, y)
            slice_importances[label] = rf.feature_importances_

        if len(slice_importances) < 2:
            continue

        # Compare importance rankings across slices
        labels = list(slice_importances.keys())
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                rho, p = sp_stats.spearmanr(slice_importances[labels[i]],
                                             slice_importances[labels[j]])
                moderation_results.append({
                    'dimension': dim,
                    'slice_1': labels[i],
                    'slice_2': labels[j],
                    'spearman_rho': rho,
                    'p_value': p,
                    'interpretation': 'Robust' if rho > 0.5 else 'Environment-sensitive'
                })
                print(f"  {dim}: {labels[i]} vs {labels[j]}: rho={rho:.3f}, p={p:.4f}")

        # Identify which features change most across slices
        if len(slice_importances) >= 2:
            imp_df = pd.DataFrame(slice_importances,
                                   index=[c.replace('_z', '') for c in available_gaze])
            imp_df['cv'] = imp_df.std(axis=1) / (imp_df.mean(axis=1) + 1e-10)
            robust_feats = imp_df.nsmallest(5, 'cv').index.tolist()
            sensitive_feats = imp_df.nlargest(5, 'cv').index.tolist()
            print(f"\n  {dim}: Most robust features: {robust_feats}")
            print(f"  {dim}: Most sensitive features: {sensitive_feats}")

    mod_df = pd.DataFrame(moderation_results)
    mod_df.to_csv(os.path.join(RESULTS_DIR, f"moderation_analysis_{subset_name}.csv"),
                  index=False)
    return mod_df


def run_phase4_subset(subset_name):
    """Run full Phase 4 for a subset."""
    print(f"\n{'='*60}")
    print(f"Phase 4: Slicing Analysis - {subset_name}")
    print(f"{'='*60}")

    df = load_labeled_data(subset_name)
    if df is None:
        print(f"  No labeled data for {subset_name}")
        return None

    results = {}

    # 4.2 Profile stability
    results['stability'] = profile_stability_across_slices(df, subset_name)

    # 4.3 Gaze discrimination by slice
    robustness_df = gaze_discrimination_by_slice(df, subset_name)
    results['robustness'] = robustness_df
    create_robustness_heatmap(robustness_df, subset_name)

    # 4.4 Moderation analysis
    results['moderation'] = moderation_analysis(df, subset_name)

    return results


def main():
    print("=" * 70)
    print("PHASE 4: Environmental Robustness Slicing Analysis")
    print("=" * 70)

    all_results = {}
    for subset_name in ['SubsetA', 'SubsetB']:
        path = os.path.join(RESULTS_DIR, f"labeled_data_{subset_name}.parquet")
        if os.path.exists(path):
            all_results[subset_name] = run_phase4_subset(subset_name)

    # Summary
    print("\n" + "=" * 70)
    print("Phase 4 Complete!")
    for subset, res in all_results.items():
        if res and res.get('robustness') is not None:
            rob = res['robustness']
            if len(rob) > 0:
                print(f"  {subset}: {len(rob)} slice-level AUROCs computed")
                print(f"    Mean AUROC: {rob['auroc'].mean():.3f}")
                print(f"    Range: [{rob['auroc'].min():.3f}, {rob['auroc'].max():.3f}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
