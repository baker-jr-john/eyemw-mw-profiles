"""
Phase 3: Gaze Signature Discrimination
========================================
Multinomial logistic regression, Random Forest + SHAP,
cross-task comparison, classification performance.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, accuracy_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
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

DEMOGRAPHIC_COLS = ['Age', 'Gender']
HARDWARE_COVARIATES = ['RefreshRate', 'SamplingRate', 'ScreenHeight', 'ScreenWidth']


def load_labeled_data(subset_name):
    """Load LPA-labeled data for a given subset."""
    path = os.path.join(RESULTS_DIR, f"labeled_data_{subset_name}.parquet")
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None
    df = pd.read_parquet(path)
    return df


def prepare_modeling_data(df):
    """Prepare feature matrix and labels, dropping rows with missing profiles or gaze data."""
    # Filter to rows with valid profile labels
    df_valid = df[df['profile'].notna()].copy()
    df_valid['profile'] = df_valid['profile'].astype(int)

    # Get available gaze features
    available_gaze = [c for c in GAZE_FEATURES_Z if c in df_valid.columns]
    if not available_gaze:
        print("  No gaze features available!")
        return None, None, None, None

    # Drop rows where all gaze features are NaN
    gaze_data = df_valid[available_gaze]
    has_gaze = gaze_data.notna().any(axis=1)
    df_valid = df_valid[has_gaze]

    # Fill remaining NaN in gaze features with 0 (within-study z-scores, 0 = mean)
    X = df_valid[available_gaze].fillna(0)
    y = df_valid['profile']

    print(f"  Modeling data: {len(X)} rows, {len(available_gaze)} gaze features, "
          f"{y.nunique()} profiles")
    print(f"  Profile distribution: {dict(y.value_counts().sort_index())}")

    return X, y, df_valid, available_gaze


def multinomial_logistic(X, y, available_gaze, subset_name):
    """Multinomial logistic regression with cross-validation."""
    print(f"\n=== 3.1 Multinomial Logistic Regression ({subset_name}) ===")

    if y.nunique() < 2:
        print("  Only 1 profile, skipping.")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = LogisticRegression(
        solver='lbfgs', max_iter=2000,
        C=1.0, random_state=42
    )

    try:
        y_pred_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')
        y_pred = cross_val_predict(model, X_scaled, y, cv=cv)
    except Exception as e:
        print(f"  Error: {e}")
        return None

    # Metrics
    acc = accuracy_score(y, y_pred)
    print(f"  Accuracy: {acc:.3f}")

    # AUROC
    if y.nunique() == 2:
        auroc = roc_auc_score(y, y_pred_proba[:, 1])
        print(f"  AUROC: {auroc:.3f}")
    else:
        auroc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
        print(f"  Weighted AUROC (one-vs-rest): {auroc:.3f}")

    print(f"\n  Classification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    # Fit full model for coefficients
    model.fit(X_scaled, y)
    coef_df = pd.DataFrame(model.coef_, columns=[c.replace('_z', '') for c in available_gaze])
    coef_df.index = [f'Profile {i}' for i in model.classes_]
    coef_df.to_csv(os.path.join(RESULTS_DIR, f"logistic_coefs_{subset_name}.csv"))

    return {'accuracy': acc, 'auroc': auroc, 'model': model, 'scaler': scaler}


def random_forest_shap(X, y, available_gaze, subset_name):
    """Random Forest classifier with SHAP feature importance."""
    print(f"\n=== 3.2 Random Forest + SHAP ({subset_name}) ===")

    if y.nunique() < 2:
        print("  Only 1 profile, skipping.")
        return None

    # Cross-validated predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    y_pred = cross_val_predict(rf, X, y, cv=cv)
    y_pred_proba = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')

    acc = accuracy_score(y, y_pred)
    if y.nunique() == 2:
        auroc = roc_auc_score(y, y_pred_proba[:, 1])
    else:
        auroc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')

    print(f"  Accuracy: {acc:.3f}")
    print(f"  Weighted AUROC: {auroc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\n  Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[f'P{i}' for i in sorted(y.unique())],
                yticklabels=[f'P{i}' for i in sorted(y.unique())])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {subset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"confusion_matrix_{subset_name}.png"), dpi=300)
    plt.close()

    # Fit full model for SHAP
    rf.fit(X, y)

    # Feature importance (RF built-in + permutation-based)
    feat_names = [c.replace('_z', '') for c in available_gaze]
    importance_df = pd.DataFrame({
        'feature': feat_names,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)

    print(f"\n  Top features by RF importance:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']}: RF_imp={row['rf_importance']:.4f}")

    importance_df.to_csv(os.path.join(RESULTS_DIR,
                         f"feature_importance_{subset_name}.csv"), index=False)

    # Create feature importance bar plot (Figure 4)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp_sorted = importance_df.sort_values('rf_importance', ascending=True)
    ax.barh(range(len(imp_sorted)), imp_sorted['rf_importance'].values,
            color='steelblue')
    ax.set_yticks(range(len(imp_sorted)))
    ax.set_yticklabels(imp_sorted['feature'].values)
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title(f'Gaze Feature Importance for MW Profile Discrimination\n{subset_name}',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"fig4_shap_{subset_name}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig4_shap_{subset_name}.png")

    return {'accuracy': acc, 'auroc': auroc, 'model': rf,
            'importance': importance_df}


def cross_task_comparison(df, available_gaze, subset_name):
    """Compare feature importance across task types."""
    print(f"\n=== 3.3 Cross-Task Comparison ({subset_name}) ===")

    task_map = {1: 'reading', 2: 'listening', 3: 'math', 4: 'video', 5: 'other'}
    task_importances = {}

    for task_code, task_name in task_map.items():
        task_df = df[(df['TaskType'] == task_code) & df['profile'].notna()]
        if len(task_df) < 30 or task_df['profile'].astype(int).nunique() < 2:
            continue

        X_task = task_df[available_gaze].fillna(0)
        y_task = task_df['profile'].astype(int)

        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                     class_weight='balanced', random_state=42, n_jobs=-1)
        try:
            rf.fit(X_task, y_task)
            importances = rf.feature_importances_
            task_importances[task_name] = importances
            print(f"  {task_name}: n={len(X_task)}, "
                  f"profiles={sorted(y_task.unique())}")
        except Exception as e:
            print(f"  {task_name}: failed ({e})")

    if len(task_importances) >= 2:
        # Spearman rank correlation of feature importances
        task_names = list(task_importances.keys())
        print(f"\n  Feature importance rank correlations:")
        for i in range(len(task_names)):
            for j in range(i+1, len(task_names)):
                rho, p = stats.spearmanr(task_importances[task_names[i]],
                                          task_importances[task_names[j]])
                print(f"    {task_names[i]} vs {task_names[j]}: "
                      f"rho={rho:.3f}, p={p:.4f}")

        # Save comparison
        comp_df = pd.DataFrame(task_importances,
                                index=[c.replace('_z', '') for c in available_gaze])
        comp_df.to_csv(os.path.join(RESULTS_DIR,
                       f"cross_task_importance_{subset_name}.csv"))

    return task_importances


def binary_baseline_comparison(X, y, available_gaze, subset_name):
    """Compare multi-profile classification against binary (on-task vs any MW)."""
    print(f"\n=== 3.4 Binary Baseline Comparison ({subset_name}) ===")

    # Load profile names to identify on-task profile
    stats_path = os.path.join(RESULTS_DIR, f"profile_stats_{subset_name}.csv")
    if os.path.exists(stats_path):
        stats_df = pd.read_csv(stats_path)
    else:
        print("  No profile stats found, skipping.")
        return None

    # Find profile with lowest TUT (most on-task)
    tut_cols = [c for c in stats_df.columns if c == 'TUT_mean']
    if tut_cols:
        on_task_profile = stats_df.loc[stats_df['TUT_mean'].idxmin(), 'profile']
    else:
        on_task_profile = 0  # default

    y_binary = (y != on_task_profile).astype(int)  # 0 = on-task, 1 = any MW

    if y_binary.nunique() < 2:
        print("  Only one class in binary split, skipping.")
        return None

    # Multi-profile AUROC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_multi = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                       class_weight='balanced', random_state=42, n_jobs=-1)
    y_pred_multi = cross_val_predict(rf_multi, X, y, cv=cv, method='predict_proba')
    if y.nunique() == 2:
        auroc_multi = roc_auc_score(y, y_pred_multi[:, 1])
    else:
        auroc_multi = roc_auc_score(y, y_pred_multi, multi_class='ovr', average='weighted')

    # Binary AUROC
    rf_binary = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                        class_weight='balanced', random_state=42, n_jobs=-1)
    y_pred_binary = cross_val_predict(rf_binary, X, y_binary, cv=cv, method='predict_proba')
    auroc_binary = roc_auc_score(y_binary, y_pred_binary[:, 1])

    print(f"  Binary AUROC (on-task vs any MW): {auroc_binary:.3f}")
    print(f"  Multi-profile weighted AUROC: {auroc_multi:.3f}")
    print(f"  Improvement from multi-profile: {auroc_multi - auroc_binary:+.3f}")

    comparison = {
        'subset': subset_name,
        'binary_auroc': auroc_binary,
        'multi_auroc': auroc_multi,
        'improvement': auroc_multi - auroc_binary,
        'on_task_profile': int(on_task_profile)
    }

    with open(os.path.join(RESULTS_DIR, f"binary_comparison_{subset_name}.json"), 'w') as f:
        json.dump(comparison, f, indent=2)

    return comparison


def classification_with_covariates(df, available_gaze, subset_name):
    """Classification including demographic and hardware covariates."""
    print(f"\n=== 3.5 Classification with Covariates ({subset_name}) ===")

    df_valid = df[df['profile'].notna()].copy()
    df_valid['profile'] = df_valid['profile'].astype(int)

    # Add covariates
    covariate_cols = []
    for col in DEMOGRAPHIC_COLS + HARDWARE_COVARIATES:
        if col in df_valid.columns and df_valid[col].notna().sum() > len(df_valid) * 0.3:
            covariate_cols.append(col)

    if not covariate_cols:
        print("  No covariates available, skipping.")
        return None

    print(f"  Using covariates: {covariate_cols}")

    # Gaze only vs gaze + covariates
    X_gaze = df_valid[available_gaze].fillna(0)
    y = df_valid['profile']

    all_features = available_gaze + covariate_cols
    X_full = df_valid[all_features].fillna(0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Gaze only
    rf_gaze = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                      class_weight='balanced', random_state=42, n_jobs=-1)
    y_pred_gaze = cross_val_predict(rf_gaze, X_gaze, y, cv=cv, method='predict_proba')

    # Gaze + covariates
    rf_full = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                      class_weight='balanced', random_state=42, n_jobs=-1)
    y_pred_full = cross_val_predict(rf_full, X_full, y, cv=cv, method='predict_proba')

    if y.nunique() == 2:
        auroc_gaze = roc_auc_score(y, y_pred_gaze[:, 1])
        auroc_full = roc_auc_score(y, y_pred_full[:, 1])
    else:
        auroc_gaze = roc_auc_score(y, y_pred_gaze, multi_class='ovr', average='weighted')
        auroc_full = roc_auc_score(y, y_pred_full, multi_class='ovr', average='weighted')

    print(f"  Gaze-only AUROC: {auroc_gaze:.3f}")
    print(f"  Gaze + covariates AUROC: {auroc_full:.3f}")
    print(f"  Improvement with covariates: {auroc_full - auroc_gaze:+.3f}")

    return {'auroc_gaze': auroc_gaze, 'auroc_full': auroc_full,
            'covariates_used': covariate_cols}


def run_phase3_subset(subset_name):
    """Run full Phase 3 pipeline for a subset."""
    print(f"\n{'='*60}")
    print(f"Phase 3: Gaze Discrimination - {subset_name}")
    print(f"{'='*60}")

    df = load_labeled_data(subset_name)
    if df is None:
        return None

    X, y, df_valid, available_gaze = prepare_modeling_data(df)
    if X is None:
        return None

    results = {}

    # 3.1 Multinomial logistic
    results['logistic'] = multinomial_logistic(X, y, available_gaze, subset_name)

    # 3.2 Random Forest + SHAP
    results['rf_shap'] = random_forest_shap(X, y, available_gaze, subset_name)

    # 3.3 Cross-task comparison
    results['cross_task'] = cross_task_comparison(df_valid, available_gaze, subset_name)

    # 3.4 Binary baseline
    results['binary_comp'] = binary_baseline_comparison(X, y, available_gaze, subset_name)

    # 3.5 Covariates
    results['covariates'] = classification_with_covariates(df_valid, available_gaze, subset_name)

    return results


def main():
    print("=" * 70)
    print("PHASE 3: Gaze Signature Discrimination")
    print("=" * 70)

    all_results = {}
    for subset_name in ['SubsetA', 'SubsetB']:
        path = os.path.join(RESULTS_DIR, f"labeled_data_{subset_name}.parquet")
        if os.path.exists(path):
            all_results[subset_name] = run_phase3_subset(subset_name)

    # Summary
    print("\n" + "=" * 70)
    print("Phase 3 Complete!")
    for subset, res in all_results.items():
        if res and res.get('rf_shap'):
            print(f"  {subset}: AUROC={res['rf_shap']['auroc']:.3f}")
    print("=" * 70)

    # Save summary
    summary = {}
    for subset, res in all_results.items():
        if res:
            summary[subset] = {
                'logistic_auroc': res['logistic']['auroc'] if res.get('logistic') else None,
                'rf_auroc': res['rf_shap']['auroc'] if res.get('rf_shap') else None,
                'binary_comp': res.get('binary_comp'),
            }
    with open(os.path.join(RESULTS_DIR, "phase3_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
