"""
generate_synthetic.py — Synthetic immune repertoire data for testing

Generates realistic TCR/BCR repertoire data with:
  - Healthy donors: high diversity, moderate clonal expansion
  - Autoimmune disease patients: oligoclonal, skewed V-gene usage
  - Cell therapy (post-CAR-T): dominant CAR-T clones + residual polyclonal
  - Cancer patients: expanded tumour-infiltrating T-cell clones

Run: python data/generate_synthetic.py
Output: data/synthetic_repertoires.csv
"""

import random
import string
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# Common human TRBV genes (from IMGT database)
TRBV_GENES = [
    "TRBV2", "TRBV3-1", "TRBV4-1", "TRBV4-2", "TRBV5-1", "TRBV5-4",
    "TRBV6-1", "TRBV6-5", "TRBV7-2", "TRBV9", "TRBV10-3", "TRBV11-2",
    "TRBV12-3", "TRBV12-4", "TRBV13", "TRBV14", "TRBV18", "TRBV20-1",
    "TRBV25-1", "TRBV27", "TRBV28", "TRBV29-1", "TRBV30"
]
TRBJGENES = ["TRBJ1-1", "TRBJ1-2", "TRBJ1-3", "TRBJ1-4", "TRBJ1-5",
             "TRBJ2-1", "TRBJ2-2", "TRBJ2-3", "TRBJ2-4", "TRBJ2-5"]
AA = "ACDEFGHIKLMNPQRSTVWY"


def random_cdr3(length: int = None) -> str:
    if length is None:
        length = random.randint(10, 18)
    return "C" + "".join(random.choices(AA, k=length - 2)) + "F"


def make_clonotype_pool(
    n_clones: int,
    skew_factor: float = 1.0,
    dominant_cdr3: str = None,
    dominant_fraction: float = 0.0,
    skewed_vgene: str = None,
    skewed_vgene_frac: float = 0.0,
) -> pd.DataFrame:
    """
    Generate a pool of clonotypes with configurable diversity properties.

    Args:
        n_clones:           number of distinct clonotypes
        skew_factor:        power-law exponent for frequency distribution
                            (1 = moderate expansion; higher = more oligoclonal)
        dominant_cdr3:      CDR3 sequence for the dominant clone (e.g. CAR-T)
        dominant_fraction:  fraction of cells belonging to dominant clone
        skewed_vgene:       V-gene to over-represent (autoimmune bias)
        skewed_vgene_frac:  fraction of clones assigned the skewed V-gene
    """
    cdr3s = [random_cdr3() for _ in range(n_clones)]
    if dominant_cdr3:
        cdr3s[0] = dominant_cdr3

    v_genes = []
    for i in range(n_clones):
        if skewed_vgene and random.random() < skewed_vgene_frac:
            v_genes.append(skewed_vgene)
        else:
            v_genes.append(random.choice(TRBV_GENES))

    j_genes = [random.choice(TRBJGENES) for _ in range(n_clones)]

    # Power-law frequencies
    raw_freqs = np.random.zipf(skew_factor, n_clones).astype(float)
    raw_freqs /= raw_freqs.sum()

    if dominant_cdr3 and dominant_fraction > 0:
        raw_freqs[0] = dominant_fraction
        raw_freqs[1:] *= (1 - dominant_fraction) / raw_freqs[1:].sum()

    counts = np.maximum(1, (raw_freqs * 1000).astype(int))

    return pd.DataFrame({
        "junction_aa":      cdr3s,
        "v_call":           v_genes,
        "j_call":           j_genes,
        "consensus_count":  counts,
        "productive":       "True",
        "chain":            "TRB",
    })


def expand_to_cells(clonotype_df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    """Expand clonotype table to single-cell rows."""
    rows = []
    clone_counter = 1
    for _, row in clonotype_df.iterrows():
        clone_id = f"clonotype{clone_counter}"
        for cell_i in range(int(row["consensus_count"])):
            barcode = "".join(random.choices(string.ascii_uppercase + string.digits, k=16)) + "-1"
            rows.append({
                "barcode":          barcode,
                "clone_id":         clone_id,
                "chain":            row["chain"],
                "v_call":           row["v_call"],
                "j_call":           row["j_call"],
                "junction_aa":      row["junction_aa"],
                "consensus_count":  row["consensus_count"],
                "productive":       row["productive"],
                "sample_id":        sample_id,
            })
        clone_counter += 1
    return pd.DataFrame(rows)


def generate_dataset():
    random.seed(42)
    np.random.seed(42)

    cohorts = [
        # Healthy donors — high diversity
        dict(sample_id="healthy_1", n_clones=500, skew=1.5),
        dict(sample_id="healthy_2", n_clones=480, skew=1.6),
        dict(sample_id="healthy_3", n_clones=510, skew=1.4),
        # Autoimmune (rheumatoid arthritis-like) — oligoclonal, TRBV14 bias
        dict(sample_id="autoimmune_1", n_clones=120, skew=3.0,
             skewed_vgene="TRBV14", skewed_vgene_frac=0.45),
        dict(sample_id="autoimmune_2", n_clones=100, skew=3.5,
             skewed_vgene="TRBV14", skewed_vgene_frac=0.50),
        # Post-CAR-T cell therapy — dominant CAR-T clone
        dict(sample_id="cart_1", n_clones=80, skew=2.0,
             dominant_cdr3="CASSLGQAYEQYF", dominant_fraction=0.45),
        dict(sample_id="cart_2", n_clones=75, skew=2.5,
             dominant_cdr3="CASSLGQAYEQYF", dominant_fraction=0.55),
        # Tumour-infiltrating lymphocytes (TIL) — expanded tumour-reactive clones
        dict(sample_id="til_1", n_clones=150, skew=2.8),
        dict(sample_id="til_2", n_clones=130, skew=3.0),
    ]

    all_cells = []
    for c in cohorts:
        pool = make_clonotype_pool(
            n_clones=c["n_clones"],
            skew_factor=c.get("skew", 1.5),
            dominant_cdr3=c.get("dominant_cdr3"),
            dominant_fraction=c.get("dominant_fraction", 0.0),
            skewed_vgene=c.get("skewed_vgene"),
            skewed_vgene_frac=c.get("skewed_vgene_frac", 0.0),
        )
        cells = expand_to_cells(pool, c["sample_id"])
        all_cells.append(cells)
        print(f"  {c['sample_id']}: {len(cells)} cells, {c['n_clones']} clonotypes")

    combined = pd.concat(all_cells, ignore_index=True)
    out = os.path.join(os.path.dirname(__file__), "synthetic_repertoires.csv")
    combined.to_csv(out, index=False)
    print(f"\nSaved {len(combined)} cells → {out}")
    return combined


if __name__ == "__main__":
    print("Generating synthetic immune repertoire data...")
    print("Cohorts: healthy (n=3), autoimmune (n=2), post-CAR-T (n=2), TIL (n=2)\n")
    generate_dataset()
    print("\nNext: python analyse_repertoire.py")
