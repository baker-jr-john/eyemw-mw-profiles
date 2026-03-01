"""
Phase 5: Final Figures and Paper Artifacts
===========================================
Generates all publication-quality figures and summary tables.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
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


def figure1_eligibility_matrix():
    """Figure 1: Study eligibility matrix (26 studies x 8 MW dimensions)."""
    print("Creating Figure 1: Study Eligibility Matrix...")

    matrix = pd.read_csv(os.path.join(RESULTS_DIR, "eligibility_matrix.csv"), index_col=0)

    # Add task type info from study profiles
    profiles = pd.read_csv(os.path.join(RESULTS_DIR, "study_profiles.csv"))
    task_map = {1: 'Read', 2: 'Listen', 3: 'Math', 4: 'Video', 5: 'Other'}

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create numeric matrix for heatmap
    numeric = matrix.astype(int)

    # Color eligible studies
    with open(os.path.join(RESULTS_DIR, "eligibility_info.json")) as f:
        elig = json.load(f)

    cmap = plt.cm.colors.ListedColormap(['#f0f0f0', '#2196F3'])
    sns.heatmap(numeric, cmap=cmap, linewidths=0.5, linecolor='gray',
                cbar=False, ax=ax, annot=False)

    # Add check marks for available dimensions
    for i, (idx, row) in enumerate(numeric.iterrows()):
        for j, col in enumerate(numeric.columns):
            if row[col] == 1:
                ax.text(j + 0.5, i + 0.5, '✓', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')

    # Add task type labels on right
    for i, (_, prof) in enumerate(profiles.iterrows()):
        task_label = task_map.get(prof['task_type'], '?')
        ax.text(len(numeric.columns) + 0.3, i + 0.5, task_label,
               ha='left', va='center', fontsize=8)

    # Highlight eligible studies
    eligible_studies = elig['eligible_studies']
    for i, idx in enumerate(numeric.index):
        study_num = int(idx.split()[-1])
        if study_num in eligible_studies:
            ax.add_patch(plt.Rectangle((0, i), len(numeric.columns), 1,
                        fill=False, edgecolor='red', linewidth=2))

    ax.set_title('Figure 1: Study Eligibility Matrix\n(Red border = eligible; ✓ = dimension measured)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Mind Wandering Dimension', fontsize=11)
    ax.set_ylabel('Study', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_eligibility_matrix.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig1_eligibility_matrix.png")


def figure2_correlation_heatmap():
    """Figure 2 is created in Phase 2 - just verify it exists."""
    path = os.path.join(FIGURES_DIR, "fig2_mw_correlation_heatmap.png")
    if os.path.exists(path):
        print(f"Figure 2 exists: {path}")
    else:
        print("Figure 2 not found - was created in Phase 2")


def figure3_radar_plots():
    """Figure 3 is created in Phase 2 - just verify."""
    for subset in ['SubsetA', 'SubsetB']:
        path = os.path.join(FIGURES_DIR, f"fig3_radar_profiles_{subset}.png")
        if os.path.exists(path):
            print(f"Figure 3 ({subset}) exists: {path}")
        else:
            print(f"Figure 3 ({subset}) not found - was created in Phase 2")


def figure4_shap():
    """Figure 4 is created in Phase 3 - just verify."""
    for subset in ['SubsetA', 'SubsetB']:
        path = os.path.join(FIGURES_DIR, f"fig4_shap_{subset}.png")
        if os.path.exists(path):
            print(f"Figure 4 ({subset}) exists: {path}")
        else:
            print(f"Figure 4 ({subset}) not found - was created in Phase 3")


def figure5_robustness():
    """Figure 5 is created in Phase 4 - verify and create combined if needed."""
    for subset in ['SubsetA', 'SubsetB']:
        path = os.path.join(FIGURES_DIR, f"fig5_robustness_heatmap_{subset}.png")
        if os.path.exists(path):
            print(f"Figure 5 ({subset}) exists: {path}")
        else:
            print(f"Figure 5 ({subset}) not found - was created in Phase 4")


def create_summary_table():
    """Create a summary results table for the paper."""
    print("\nCreating summary results table...")

    summary = {}

    # Phase 2: LPA results
    for subset in ['SubsetA', 'SubsetB']:
        stats_path = os.path.join(RESULTS_DIR, f"profile_stats_{subset}.csv")
        if os.path.exists(stats_path):
            stats = pd.read_csv(stats_path)
            summary[f'{subset}_n_profiles'] = len(stats)
            summary[f'{subset}_total_n'] = stats['n'].sum()
            summary[f'{subset}_avg_posterior'] = stats['avg_posterior'].mean()

        boot_path = os.path.join(RESULTS_DIR, f"bootstrap_results_{subset}.json")
        if os.path.exists(boot_path):
            with open(boot_path) as f:
                boot = json.load(f)
            summary[f'{subset}_bootstrap_ari'] = boot['mean_ari']
            summary[f'{subset}_bootstrap_post'] = boot['mean_posterior']

    # Phase 3: Classification results
    phase3_path = os.path.join(RESULTS_DIR, "phase3_summary.json")
    if os.path.exists(phase3_path):
        with open(phase3_path) as f:
            p3 = json.load(f)
        for subset, vals in p3.items():
            if vals:
                for k, v in vals.items():
                    summary[f'{subset}_{k}'] = v

    # Phase 4: Robustness
    for subset in ['SubsetA', 'SubsetB']:
        rob_path = os.path.join(RESULTS_DIR, f"robustness_table_{subset}.csv")
        if os.path.exists(rob_path):
            rob = pd.read_csv(rob_path)
            if len(rob) > 0:
                summary[f'{subset}_mean_slice_auroc'] = rob['auroc'].mean()
                summary[f'{subset}_min_slice_auroc'] = rob['auroc'].min()
                summary[f'{subset}_max_slice_auroc'] = rob['auroc'].max()

    with open(os.path.join(RESULTS_DIR, "paper_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("  Summary saved to paper_summary.json")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.3f}")
        else:
            print(f"    {k}: {v}")

    return summary


def main():
    print("=" * 70)
    print("PHASE 5: Final Figures and Summary")
    print("=" * 70)

    figure1_eligibility_matrix()
    figure2_correlation_heatmap()
    figure3_radar_plots()
    figure4_shap()
    figure5_robustness()
    create_summary_table()

    # List all figures
    print("\n=== Generated Figures ===")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(FIGURES_DIR, f))
            print(f"  {f} ({size/1024:.0f} KB)")

    print("\n" + "=" * 70)
    print("Phase 5 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
