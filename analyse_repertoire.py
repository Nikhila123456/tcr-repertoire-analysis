"""
analyse_repertoire.py — Run complete immune repertoire analysis pipeline

Run: python analyse_repertoire.py
Outputs in results/:
  diversity_metrics.csv
  clonotype_overlap.csv
  vgene_usage.csv
  repertoire_summary.png  (4-panel figure)
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, ".")
from src.repertoire import (
    count_clonotypes, classify_expansion, compute_diversity,
    vgene_usage, clonotype_overlap,
    SAMPLE_COL, CDR3_COL, V_GENE_COL, CLONE_ID_COL, COUNT_COL
)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} cells from {path}")
    print(f"  Samples: {sorted(df[SAMPLE_COL].unique())}")
    return df


def run_pipeline(df: pd.DataFrame, results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)

    # ── 1. Clonotype counts and expansion ────────────────────────────────────
    print("\nCounting clonotypes...")
    ct = count_clonotypes(df, by=CDR3_COL)
    ct = classify_expansion(ct)
    ct.to_csv(f"{results_dir}/clonotype_counts.csv", index=False)

    # ── 2. Diversity metrics ──────────────────────────────────────────────────
    print("Computing diversity metrics...")
    div = compute_diversity(ct, group_col=SAMPLE_COL)
    div.to_csv(f"{results_dir}/diversity_metrics.csv", index=False)
    print(div.to_string(index=False))

    # ── 3. V-gene usage ───────────────────────────────────────────────────────
    print("\nComputing V-gene usage...")
    vg = vgene_usage(df, group_col=SAMPLE_COL)
    vg.to_csv(f"{results_dir}/vgene_usage.csv", index=False)

    # ── 4. Clonotype overlap ──────────────────────────────────────────────────
    print("Computing clonotype overlap...")
    overlap = clonotype_overlap(df, cdr3_col=CDR3_COL, sample_col=SAMPLE_COL)
    overlap.to_csv(f"{results_dir}/clonotype_overlap.csv")

    # ── 5. Visualisation ──────────────────────────────────────────────────────
    print("\nGenerating figures...")

    # Colour palette by cohort
    cohort_colors = {
        "healthy":    "#2196F3",
        "autoimmune": "#F44336",
        "cart":       "#4CAF50",
        "til":        "#FF9800",
    }
    div["cohort"] = div["sample_id"].apply(
        lambda s: next((k for k in cohort_colors if s.startswith(k)), "other")
    )
    div["color"] = div["cohort"].map(cohort_colors)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Immune Repertoire Analysis — Cohort Comparison", fontsize=14, fontweight="bold")

    # Panel A: Shannon diversity
    ax = axes[0, 0]
    bars = ax.bar(div["sample_id"], div["shannon_entropy"], color=div["color"], edgecolor="white")
    ax.set_title("A. Shannon Diversity (H)", fontsize=12)
    ax.set_ylabel("Shannon Entropy")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Gini coefficient
    ax = axes[0, 1]
    ax.bar(div["sample_id"], div["gini_coefficient"], color=div["color"], edgecolor="white")
    ax.set_title("B. Gini Coefficient (Clonal Inequality)", fontsize=12)
    ax.set_ylabel("Gini Coefficient")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Top-10 clonal frequency
    ax = axes[1, 0]
    ax.bar(div["sample_id"], div["top10_freq"], color=div["color"], edgecolor="white")
    ax.set_title("C. Top-10 Clone Cumulative Frequency", fontsize=12)
    ax.set_ylabel("Fraction of Cells")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel D: Clonotype overlap heatmap
    ax = axes[1, 1]
    sns.heatmap(
        overlap.astype(float), ax=ax, cmap="Blues",
        vmin=0, vmax=0.1, annot=False,
        xticklabels=True, yticklabels=True,
        cbar_kws={"label": "Jaccard index"},
        linewidths=0.3, linecolor="white",
    )
    ax.set_title("D. Clonotype Overlap (Jaccard)", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=k) for k, c in cohort_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_fig = f"{results_dir}/repertoire_summary.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {out_fig}")

    return div, overlap


if __name__ == "__main__":
    import subprocess, sys as _sys
    data_path = "data/synthetic_repertoires.csv"
    if not os.path.exists(data_path):
        print("Generating synthetic data first...")
        subprocess.run([_sys.executable, "data/generate_synthetic.py"], check=True)

    df  = load_data(data_path)
    div, overlap = run_pipeline(df)

    print("\n=== BIOLOGICAL INTERPRETATION ===")
    print("Healthy donors:     high Shannon entropy, low Gini → diverse, balanced repertoire")
    print("Autoimmune samples: low Shannon entropy, high Gini → oligoclonal, skewed V-gene")
    print("Post-CAR-T:         dominant clone (CAR-T construct) visible in top-1 frequency")
    print("TIL samples:        intermediate diversity with expanded tumour-reactive clones")
    print("\nResults saved to results/")
