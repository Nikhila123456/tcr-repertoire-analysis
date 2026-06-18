"""
repertoire.py — Core analysis module for TCR/BCR immune repertoire data

Analyses clonotype diversity, expansion, and sharing from 10x VDJ or
AIRR-format CSV files. Designed for integration with scRNA-seq pipelines
(Seurat / Scanpy) for paired transcriptomic + repertoire analysis.

Biological context:
    T-cell receptor (TCR) and B-cell receptor (BCR) repertoire analysis
    measures the diversity and clonal structure of the adaptive immune response.
    In immune-mediated diseases and cell therapy:
    - Clonal expansion: dominant clones indicate antigen-driven response
    - Diversity: low diversity = oligoclonal response (autoimmune, CAR-T exhaustion)
    - Clonotype sharing: tracks convergent immune responses across patients
    - CDR3 properties: motifs that define antigen specificity
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings


# ── AIRR-standard column names ─────────────────────────────────────────────────
CLONE_ID_COL    = "clone_id"
BARCODE_COL     = "barcode"
CHAIN_COL       = "chain"
V_GENE_COL      = "v_call"
J_GENE_COL      = "j_call"
CDR3_COL        = "junction_aa"
COUNT_COL       = "consensus_count"
PRODUCTIVE_COL  = "productive"
SAMPLE_COL      = "sample_id"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_vdj(path: str, sample_id: str = None) -> pd.DataFrame:
    """
    Load 10x Genomics VDJ CSV (filtered_contig_annotations.csv) or
    AIRR-format TSV into a standardised DataFrame.

    Args:
        path:      path to CSV/TSV file
        sample_id: optional sample label to add as a column

    Returns:
        DataFrame with standardised column names
    """
    path = Path(path)
    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)

    # 10x column → AIRR-standard mapping
    rename = {
        "barcode":          BARCODE_COL,
        "raw_clonotype_id": CLONE_ID_COL,
        "chain":            CHAIN_COL,
        "v_gene":           V_GENE_COL,
        "j_gene":           J_GENE_COL,
        "cdr3":             CDR3_COL,
        "reads":            COUNT_COL,
        "productive":       PRODUCTIVE_COL,
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if sample_id:
        df[SAMPLE_COL] = sample_id

    # Keep only productive, full-length chains
    if PRODUCTIVE_COL in df.columns:
        df = df[df[PRODUCTIVE_COL].astype(str).str.upper() == "TRUE"].copy()

    return df


def combine_samples(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple sample DataFrames, ensuring unique clonotype IDs."""
    combined = []
    for i, df in enumerate(dfs):
        sample = df.get(SAMPLE_COL, pd.Series([f"sample_{i}"] * len(df)))
        if SAMPLE_COL not in df.columns:
            df = df.copy()
            df[SAMPLE_COL] = f"sample_{i}"
        # Make clone_id unique across samples
        if CLONE_ID_COL in df.columns:
            df = df.copy()
            df[CLONE_ID_COL] = df[SAMPLE_COL].astype(str) + "_" + df[CLONE_ID_COL].astype(str)
        combined.append(df)
    return pd.concat(combined, ignore_index=True)


# ── Clonotype counting & frequency ─────────────────────────────────────────────

def count_clonotypes(
    df: pd.DataFrame,
    by: str = CLONE_ID_COL,
    sample_col: str = SAMPLE_COL,
) -> pd.DataFrame:
    """
    Count cells per clonotype per sample, compute frequency.

    Returns DataFrame with columns:
        sample_id, clone_id, count, frequency
    """
    if sample_col in df.columns:
        groups = df.groupby([sample_col, by]).size().reset_index(name="count")
        totals = groups.groupby(sample_col)["count"].transform("sum")
    else:
        groups = df.groupby(by).size().reset_index(name="count")
        totals = groups["count"].sum()

    groups["frequency"] = groups["count"] / totals
    return groups.sort_values("frequency", ascending=False).reset_index(drop=True)


def classify_expansion(df: pd.DataFrame, freq_col: str = "frequency") -> pd.DataFrame:
    """
    Classify clonotypes by expansion level (clinical convention):
        rare      : freq < 0.001
        small     : 0.001 ≤ freq < 0.01
        medium    : 0.01 ≤ freq < 0.1
        large     : 0.1 ≤ freq < 0.3
        hyperexpanded: freq ≥ 0.3
    """
    bins   = [0, 0.001, 0.01, 0.1, 0.3, 1.01]
    labels = ["rare", "small", "medium", "large", "hyperexpanded"]
    df = df.copy()
    df["expansion_class"] = pd.cut(
        df[freq_col], bins=bins, labels=labels, right=False
    )
    return df


# ── Diversity metrics ──────────────────────────────────────────────────────────

def shannon_entropy(frequencies: np.ndarray) -> float:
    """
    Shannon entropy H = -Σ p_i * log2(p_i)
    High H = high diversity (healthy repertoire)
    Low H  = oligoclonal (autoimmune, tumour-infiltrating, exhausted CAR-T)
    """
    p = np.array(frequencies, dtype=float)
    p = p[p > 0]
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def simpson_index(frequencies: np.ndarray) -> float:
    """
    Simpson diversity index D = 1 - Σ p_i²
    Ranges 0 (no diversity) to 1 (maximum diversity)
    """
    p = np.array(frequencies, dtype=float)
    p = p / p.sum()
    return float(1 - np.sum(p ** 2))


def clonal_gini(frequencies: np.ndarray) -> float:
    """
    Gini coefficient for clonal inequality.
    0 = perfectly equal (diverse); 1 = one clone dominates (oligoclonal)
    Widely used in immune repertoire analysis (Rosenblatt et al.)
    """
    p = np.sort(np.array(frequencies, dtype=float))
    n = len(p)
    if n == 0 or p.sum() == 0:
        return 0.0
    p = p / p.sum()
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * p) - (n + 1)) / n)


def compute_diversity(
    clonotype_df: pd.DataFrame,
    freq_col: str = "frequency",
    group_col: str = SAMPLE_COL,
) -> pd.DataFrame:
    """
    Compute diversity metrics per sample.

    Returns DataFrame with:
        sample_id, n_clonotypes, shannon_entropy, simpson_index,
        gini_coefficient, top1_freq, top10_freq
    """
    rows = []
    groups = clonotype_df.groupby(group_col) if group_col in clonotype_df.columns \
             else [("all", clonotype_df)]

    for sample, grp in groups:
        freqs = grp[freq_col].values
        sorted_f = np.sort(freqs)[::-1]
        rows.append({
            "sample_id":       sample,
            "n_clonotypes":    len(grp),
            "shannon_entropy": shannon_entropy(freqs),
            "simpson_index":   simpson_index(freqs),
            "gini_coefficient": clonal_gini(freqs),
            "top1_freq":       float(sorted_f[0]) if len(sorted_f) > 0 else 0.0,
            "top10_freq":      float(sorted_f[:10].sum()) if len(sorted_f) > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ── V-gene usage ───────────────────────────────────────────────────────────────

def vgene_usage(
    df: pd.DataFrame,
    v_col: str = V_GENE_COL,
    group_col: str = SAMPLE_COL,
    normalise: bool = True,
) -> pd.DataFrame:
    """
    V-gene usage frequencies per sample.
    Skewed V-gene usage can indicate antigen-driven clonal selection.
    """
    if group_col in df.columns:
        counts = df.groupby([group_col, v_col]).size().reset_index(name="count")
        if normalise:
            totals = counts.groupby(group_col)["count"].transform("sum")
            counts["frequency"] = counts["count"] / totals
    else:
        counts = df[v_col].value_counts().reset_index()
        counts.columns = [v_col, "count"]
        if normalise:
            counts["frequency"] = counts["count"] / counts["count"].sum()
    return counts


# ── Clonotype sharing ──────────────────────────────────────────────────────────

def clonotype_overlap(
    clonotype_df: pd.DataFrame,
    cdr3_col: str = CDR3_COL,
    sample_col: str = SAMPLE_COL,
) -> pd.DataFrame:
    """
    Compute pairwise Jaccard overlap of CDR3 sequences between samples.
    Shared clonotypes can indicate:
        - Convergent immune responses (public clonotypes)
        - Contaminant/technical artefact (if unexpected)
        - Shared antigen exposure in a disease cohort

    Returns:
        DataFrame of shape (n_samples, n_samples) with Jaccard scores
    """
    if sample_col not in clonotype_df.columns:
        raise ValueError(f"Column '{sample_col}' not found.")

    samples = clonotype_df[sample_col].unique()
    sets = {
        s: set(clonotype_df.loc[clonotype_df[sample_col] == s, cdr3_col].dropna())
        for s in samples
    }

    matrix = {}
    for s1 in samples:
        matrix[s1] = {}
        for s2 in samples:
            intersection = len(sets[s1] & sets[s2])
            union = len(sets[s1] | sets[s2])
            matrix[s1][s2] = intersection / union if union > 0 else 0.0

    return pd.DataFrame(matrix, index=samples, columns=samples)
