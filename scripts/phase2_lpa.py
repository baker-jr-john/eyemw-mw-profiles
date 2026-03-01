"""
Phase 2: Latent Profile Analysis
=================================
Exploratory correlation analysis, Gaussian Mixture Model-based LPA,
profile characterization, and bootstrap validation.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MW_DIMENSIONS = ['TUT', 'Intentionality', 'Awareness', 'FMT',
                 'Disengagement', 'Valence', 'Arousal', 'Boredom']


def load_harmonized_data():
    """Load harmonized dataset and eligibility info."""
    df = pd.read_parquet(os.path.join(RESULTS_DIR, "harmonized_data.parquet"))
    with open(os.path.join(RESULTS_DIR, "eligibility_info.json")) as f:
        elig = json.load(f)
    return df, elig


def get_mw_matrix(df, dimensions):
    """Extract MW response matrix for given dimensions, dropping rows with any NaN."""
    norm_cols = [f"{d}_norm" for d in dimensions]
    available = [c for c in norm_cols if c in df.columns]
    if len(available) < len(norm_cols):
        missing = set(norm_cols) - set(available)
        print(f"  Warning: columns not found: {missing}")
    matrix = df[available].dropna()
    dim_names = [c.replace('_norm', '') for c in available]
    return matrix, dim_names, matrix.index


def correlation_analysis(df, elig):
    """Compute and visualize MW dimension correlations."""
    print("\n=== 2.1 Correlation Analysis ===")

    # Overall correlation
    all_norm = [f"{d}_norm" for d in MW_DIMENSIONS if f"{d}_norm" in df.columns]
    corr_data = df[all_norm].dropna()

    if len(corr_data) < 10:
        print("  Insufficient data for overall correlation. Trying subset-specific.")
    else:
        corr_matrix = corr_data.corr()
        dim_labels = [c.replace('_norm', '') for c in all_norm]
        corr_matrix.index = dim_labels
        corr_matrix.columns = dim_labels

        print(f"  Overall correlation matrix ({len(corr_data)} complete rows):")
        print(corr_matrix.round(3).to_string())

        # Check monolithic assumption
        off_diag = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        print(f"\n  Off-diagonal correlations: mean={np.nanmean(off_diag):.3f}, "
              f"max={np.nanmax(off_diag):.3f}, min={np.nanmin(off_diag):.3f}")
        if np.nanmean(np.abs(off_diag)) > 0.8:
            print("  WARNING: High correlations suggest monolithic structure!")
        else:
            print("  Correlations suggest MW dimensions are NOT monolithic - proceed with LPA.")

        # Plot overall heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, mask=mask, ax=ax,
                    square=True, linewidths=0.5)
        ax.set_title('MW Dimension Correlations (Overall)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "fig2_mw_correlation_heatmap.png"), dpi=300)
        plt.close()
        print("  Saved: fig2_mw_correlation_heatmap.png")

    # Stratified by task type
    task_map = {1: 'reading', 2: 'listening', 3: 'math', 4: 'video', 5: 'other'}
    for task_code, task_name in task_map.items():
        task_df = df[df['TaskType'] == task_code]
        task_corr_data = task_df[all_norm].dropna()
        if len(task_corr_data) >= 20:
            corr = task_corr_data.corr()
            dim_labels = [c.replace('_norm', '') for c in all_norm]
            corr.index = dim_labels
            corr.columns = dim_labels
            print(f"\n  {task_name} task correlation ({len(task_corr_data)} rows):")
            print(corr.round(3).to_string())

    # Per-subset correlations
    for subset_name, subset_key in [('A', 'subset_a'), ('B', 'subset_b')]:
        subset_studies = elig[subset_key]
        subset_dims = elig[f'{subset_key}_dims']
        subset_df = df[df['StudyNum'].isin(subset_studies)]
        norm_cols = [f"{d}_norm" for d in subset_dims if f"{d}_norm" in subset_df.columns]
        sub_data = subset_df[norm_cols].dropna()
        if len(sub_data) >= 20:
            corr = sub_data.corr()
            dim_labels = [c.replace('_norm', '') for c in norm_cols]
            corr.index = dim_labels
            corr.columns = dim_labels
            print(f"\n  Subset {subset_name} correlation ({len(sub_data)} rows, "
                  f"studies {subset_studies}):")
            print(corr.round(3).to_string())

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                        center=0, vmin=-1, vmax=1, ax=ax,
                        square=True, linewidths=0.5)
            ax.set_title(f'Subset {subset_name}: MW Dimension Correlations')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR,
                        f"fig2_correlation_subset_{subset_name}.png"), dpi=300)
            plt.close()


def run_lpa(data, dim_names, subset_name, k_range=range(2, 7)):
    """Run Gaussian Mixture Model (LPA) for various k values."""
    print(f"\n  Running LPA on {subset_name} ({len(data)} rows, dims={dim_names})...")

    # Standardize for fitting
    scaler = StandardScaler()
    X = scaler.fit_transform(data.values)

    results = []
    models = {}
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            n_init=20,
            max_iter=500,
            random_state=42,
            reg_covar=1e-4
        )
        gmm.fit(X)

        bic = gmm.bic(X)
        aic = gmm.aic(X)

        # Compute entropy-based classification accuracy
        probs = gmm.predict_proba(X)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        max_entropy = np.log(k)
        rel_entropy = 1 - np.mean(entropy) / max_entropy if max_entropy > 0 else 1.0

        # Average posterior probability (max per row)
        avg_posterior = np.mean(np.max(probs, axis=1))

        results.append({
            'k': k, 'BIC': bic, 'AIC': aic,
            'entropy': rel_entropy, 'avg_posterior': avg_posterior,
            'log_likelihood': gmm.score(X) * len(X),
            'converged': gmm.converged_
        })
        models[k] = (gmm, scaler)

        print(f"    k={k}: BIC={bic:.1f}, AIC={aic:.1f}, "
              f"entropy={rel_entropy:.3f}, avg_post={avg_posterior:.3f}")

    results_df = pd.DataFrame(results)

    # Select optimal k using BIC elbow method
    # Find where the rate of BIC improvement drops substantially
    bic_vals = results_df['BIC'].values
    bic_diffs = np.diff(bic_vals)  # negative = improvement
    if len(bic_diffs) > 1:
        # Find elbow: where improvement drops to < 20% of first improvement
        first_improvement = abs(bic_diffs[0])
        for i, diff in enumerate(bic_diffs[1:], 1):
            if abs(diff) < 0.20 * first_improvement:
                best_k = results_df.iloc[i]['k']
                break
        else:
            best_k = results_df.loc[results_df['BIC'].idxmin(), 'k']
    else:
        best_k = results_df.loc[results_df['BIC'].idxmin(), 'k']

    # Cap at 5 for interpretability
    best_k = min(best_k, 5)
    print(f"\n  Optimal k (BIC elbow, max 5): {best_k}")

    # Check posterior probability criterion
    best_post = results_df.loc[results_df['k'] == best_k, 'avg_posterior'].values[0]
    if best_post < 0.70:
        print(f"  WARNING: Avg posterior ({best_post:.3f}) below 0.70 threshold")

    return results_df, models, int(best_k), scaler


def characterize_profiles(df_subset, data, dim_names, gmm, scaler, best_k, subset_name):
    """Compute profile means and create radar plots."""
    print(f"\n=== Characterizing {best_k} Profiles ({subset_name}) ===")

    X = scaler.transform(data.values)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    # Add labels back using original index
    data_with_labels = data.copy()
    data_with_labels['profile'] = labels
    data_with_labels['max_posterior'] = np.max(probs, axis=1)

    # Compute profile statistics (on original 0-1 normalized scale)
    profile_stats = []
    for p in range(best_k):
        mask = data_with_labels['profile'] == p
        pdata = data_with_labels[mask]
        stats_dict = {'profile': p, 'n': mask.sum(),
                      'pct': round(mask.sum() / len(data_with_labels) * 100, 1),
                      'avg_posterior': pdata['max_posterior'].mean()}
        for col in data.columns:
            dim = col.replace('_norm', '')
            stats_dict[f'{dim}_mean'] = pdata[col].mean()
            stats_dict[f'{dim}_sd'] = pdata[col].std()
        profile_stats.append(stats_dict)

    stats_df = pd.DataFrame(profile_stats)
    print(stats_df.to_string(index=False))
    stats_df.to_csv(os.path.join(RESULTS_DIR, f"profile_stats_{subset_name}.csv"), index=False)

    # Name profiles based on patterns
    profile_names = name_profiles(stats_df, dim_names)
    print(f"\n  Profile names: {profile_names}")

    # Radar/spider plot
    create_radar_plot(stats_df, dim_names, profile_names, subset_name)

    return data_with_labels, stats_df, profile_names


def name_profiles(stats_df, dim_names):
    """Auto-name profiles based on their MW signatures."""
    names = {}
    # Track used names to avoid duplicates
    used_names = set()

    for _, row in stats_df.iterrows():
        p = int(row['profile'])
        means = {d: row[f'{d}_mean'] for d in dim_names}

        tut = means.get('TUT', 0)
        boredom = means.get('Boredom', 0)
        valence = means.get('Valence', 0)
        diseng = means.get('Disengagement', 0)
        awareness = means.get('Awareness', 0)
        fmt = means.get('FMT', 0)

        # Build descriptive name from MW signature
        descriptors = []
        if tut > 0.7:
            descriptors.append("MW")
        elif tut < 0.3:
            descriptors.append("On-Task")

        if diseng > 0.7:
            descriptors.append("Disengaged")
        if boredom > 0.7:
            descriptors.append("Bored")
        if 'Valence' in dim_names:
            if valence > 0.7:
                descriptors.append("Positive")
            elif valence < 0.3:
                descriptors.append("Negative")
        if awareness > 0.7:
            descriptors.append("Aware")
        if fmt > 0.7:
            descriptors.append("Free-Flowing")

        if not descriptors:
            # Use relative patterns
            if tut > 0.5 and diseng > 0.5:
                descriptors = ["MW", "Disengaged"]
            elif tut > 0.5:
                descriptors = ["Mild MW"]
            elif tut < 0.5 and diseng < 0.5:
                descriptors = ["On-Task"]
            else:
                descriptors = ["Mixed"]

        name = " + ".join(descriptors)

        # Ensure uniqueness
        base_name = name
        counter = 2
        while name in used_names:
            # Add distinguishing detail based on secondary features
            if 'Boredom' in dim_names and boredom > 0.5:
                name = f"{base_name} (High Boredom)"
            elif 'Boredom' in dim_names and boredom < 0.3:
                name = f"{base_name} (Low Boredom)"
            elif 'FMT' in dim_names and fmt > 0.5:
                name = f"{base_name} (High FMT)"
            elif 'Awareness' in dim_names and awareness > 0.5:
                name = f"{base_name} (Aware)"
            else:
                name = f"{base_name} ({counter})"
            counter += 1

        used_names.add(name)
        names[p] = name

    return names


def create_radar_plot(stats_df, dim_names, profile_names, subset_name):
    """Create radar/spider plot of profile means."""
    n_dims = len(dim_names)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(stats_df)))

    for i, (_, row) in enumerate(stats_df.iterrows()):
        p = int(row['profile'])
        values = [row[f'{d}_mean'] for d in dim_names]
        values += values[:1]
        name = profile_names.get(p, f"Profile {p}")
        label = f"{name} (n={int(row['n'])}, {row['pct']}%)"
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=label)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f'MW Profiles - {subset_name}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"fig3_radar_profiles_{subset_name}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_radar_profiles_{subset_name}.png")


def plot_model_selection(results_df, subset_name):
    """Plot BIC/AIC/entropy across k values."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(results_df['k'], results_df['BIC'], 'bo-', linewidth=2)
    axes[0].set_xlabel('Number of Profiles (k)')
    axes[0].set_ylabel('BIC')
    axes[0].set_title('BIC (lower is better)')

    axes[1].plot(results_df['k'], results_df['AIC'], 'ro-', linewidth=2)
    axes[1].set_xlabel('Number of Profiles (k)')
    axes[1].set_ylabel('AIC')
    axes[1].set_title('AIC (lower is better)')

    axes[2].plot(results_df['k'], results_df['entropy'], 'go-', linewidth=2)
    axes[2].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='0.8 threshold')
    axes[2].set_xlabel('Number of Profiles (k)')
    axes[2].set_ylabel('Relative Entropy')
    axes[2].set_title('Classification Entropy')
    axes[2].legend()

    plt.suptitle(f'Model Selection - {subset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"model_selection_{subset_name}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: model_selection_{subset_name}.png")


def bootstrap_validation(data, scaler, best_k, n_boot=1000, subset_name=""):
    """Bootstrap LPA to assess assignment stability."""
    print(f"\n=== Bootstrap Validation ({subset_name}, n_boot={n_boot}) ===")

    X = scaler.transform(data.values)
    n = len(X)

    # Fit reference model
    ref_gmm = GaussianMixture(n_components=best_k, covariance_type='full',
                               n_init=20, max_iter=500, random_state=42, reg_covar=1e-4)
    ref_gmm.fit(X)
    ref_labels = ref_gmm.predict(X)

    stability_scores = []
    boot_posteriors = []

    for b in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]

        boot_gmm = GaussianMixture(n_components=best_k, covariance_type='full',
                                    n_init=5, max_iter=300, random_state=b, reg_covar=1e-4)
        try:
            boot_gmm.fit(X_boot)
            boot_probs = boot_gmm.predict_proba(X)
            boot_posteriors.append(np.mean(np.max(boot_probs, axis=1)))

            # Stability: how often does the bootstrap agree with reference?
            boot_labels = boot_gmm.predict(X)
            # Use adjusted Rand index as stability metric
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(ref_labels, boot_labels)
            stability_scores.append(ari)
        except Exception:
            continue

        if (b + 1) % 200 == 0:
            print(f"    Bootstrap {b+1}/{n_boot} done...")

    mean_ari = np.mean(stability_scores)
    mean_post = np.mean(boot_posteriors)
    print(f"\n  Bootstrap results (n={len(stability_scores)} successful):")
    print(f"    Mean Adjusted Rand Index: {mean_ari:.3f} (95% CI: "
          f"[{np.percentile(stability_scores, 2.5):.3f}, "
          f"{np.percentile(stability_scores, 97.5):.3f}])")
    print(f"    Mean Avg Posterior: {mean_post:.3f} (95% CI: "
          f"[{np.percentile(boot_posteriors, 2.5):.3f}, "
          f"{np.percentile(boot_posteriors, 97.5):.3f}])")

    boot_results = {
        'subset': subset_name,
        'n_boot': int(len(stability_scores)),
        'best_k': int(best_k),
        'mean_ari': float(mean_ari),
        'ari_ci_lower': float(np.percentile(stability_scores, 2.5)),
        'ari_ci_upper': float(np.percentile(stability_scores, 97.5)),
        'mean_posterior': float(mean_post),
        'posterior_ci_lower': float(np.percentile(boot_posteriors, 2.5)),
        'posterior_ci_upper': float(np.percentile(boot_posteriors, 97.5)),
        'target_posterior': 0.70,
        'meets_posterior_target': bool(mean_post >= 0.70)
    }

    with open(os.path.join(RESULTS_DIR, f"bootstrap_results_{subset_name}.json"), 'w') as f:
        json.dump(boot_results, f, indent=2)

    return boot_results


def run_lpa_subset(df, elig, subset_key, subset_name):
    """Run full LPA pipeline for a given subset."""
    print(f"\n{'='*60}")
    print(f"LPA for {subset_name}")
    print(f"{'='*60}")

    subset_studies = elig[subset_key]
    subset_dims = elig[f'{subset_key}_dims']
    print(f"  Studies: {subset_studies}")
    print(f"  Dimensions: {subset_dims}")

    subset_df = df[df['StudyNum'].isin(subset_studies)]
    print(f"  Total rows in subset: {len(subset_df)}")

    data, dim_names, valid_idx = get_mw_matrix(subset_df, subset_dims)
    print(f"  Complete cases: {len(data)}")

    if len(data) < 50:
        print(f"  WARNING: Too few complete cases ({len(data)}). Skipping LPA.")
        return None, None, None, None, None

    # Run LPA
    results_df, models, best_k, scaler = run_lpa(data, dim_names, subset_name)
    plot_model_selection(results_df, subset_name)

    # Characterize with best k
    best_gmm = models[best_k][0]
    data_labeled, stats_df, profile_names = characterize_profiles(
        subset_df, data, dim_names, best_gmm, scaler, best_k, subset_name)

    # Bootstrap validation
    boot_results = bootstrap_validation(data, scaler, best_k, n_boot=1000,
                                         subset_name=subset_name)

    # Save LPA results
    results_df.to_csv(os.path.join(RESULTS_DIR, f"lpa_model_selection_{subset_name}.csv"),
                      index=False)

    # Save labeled data for Phase 3
    # Merge profile labels back to subset_df
    labeled_df = subset_df.copy()
    labeled_df['profile'] = np.nan
    labeled_df['profile_name'] = ''
    labeled_df['max_posterior'] = np.nan
    labeled_df.loc[data_labeled.index, 'profile'] = data_labeled['profile'].values
    labeled_df.loc[data_labeled.index, 'max_posterior'] = data_labeled['max_posterior'].values
    for idx in data_labeled.index:
        p = int(data_labeled.loc[idx, 'profile'])
        labeled_df.loc[idx, 'profile_name'] = profile_names.get(p, f'Profile {p}')

    labeled_df.to_parquet(os.path.join(RESULTS_DIR, f"labeled_data_{subset_name}.parquet"),
                          index=False)
    print(f"  Saved labeled data: labeled_data_{subset_name}.parquet")

    return results_df, best_k, stats_df, profile_names, boot_results


def main():
    print("=" * 70)
    print("PHASE 2: Latent Profile Analysis")
    print("=" * 70)

    df, elig = load_harmonized_data()

    # 2.1 Correlation analysis
    correlation_analysis(df, elig)

    # 2.2-2.4 LPA for each subset
    results = {}
    for subset_key, subset_name in [('subset_a', 'SubsetA'), ('subset_b', 'SubsetB')]:
        if elig[subset_key]:
            out = run_lpa_subset(df, elig, subset_key, subset_name)
            results[subset_name] = out
        else:
            print(f"\n  No studies for {subset_name}, skipping.")

    # Summary
    print("\n" + "=" * 70)
    print("Phase 2 Complete!")
    for name, (res_df, best_k, stats_df, pnames, boot) in results.items():
        if best_k is not None:
            print(f"  {name}: {best_k} profiles identified")
            for p, pname in pnames.items():
                n = stats_df[stats_df['profile'] == p]['n'].values[0]
                print(f"    Profile {p}: {pname} (n={int(n)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
