# Stats/correlation_analysis.py
# ------------------------------------------------------------
# Correlation analysis for LipidQuest pipeline (MetaboScape exports)
# - Loads via Stats.utils.load_dataset(file_path, group_file) when available
# - ANOVA per feature across groups + BH-FDR, select top-N features
# - Compound–compound and sample–sample correlation heatmaps (PNG+SVG) + CSVs
# - Optional per-group compound correlation heatmaps
# - Saves under save_dir/Correlation/
# ------------------------------------------------------------

from __future__ import annotations
import os
import re
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

MIN_SAMPLES_PER_GROUP = 3

# House style
warnings.filterwarnings("ignore", category=FutureWarning)
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["mathtext.default"] = "regular"

# Optional imports from your pipeline
try:
    from Stats.utils import load_dataset, prepare_output_dir
except Exception:
    load_dataset = None
    prepare_output_dir = None


# ---------------------------
# Helpers
# ---------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _clean_feature_label(s: str) -> str:
    """
    Make labels compact and comparable across MetaboScape/LipidQuest exports.
    Examples:
      'PC 34:1|C00123|[M+H]+' -> 'PC 34:1'
      'TG 52:3 [M+NH4]+' -> 'TG 52:3'
    """
    if not isinstance(s, str):
        return str(s)
    out = s
    # keep last token after '|' if present
    if '|' in out:
        out = out.split('|')[-1]
    # drop adduct in square brackets
    out = re.sub(r"\s*\[[^\]]+\]\s*$", "", out)
    # normalize spaces
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _select_numeric_features(df: pd.DataFrame, exclude_cols: Iterable[str]) -> List[str]:
    exclude = set(exclude_cols)
    numeric_cols = [c for c in df.columns if c not in exclude and _is_numeric_series(df[c])]
    return numeric_cols


def _anova_top_features(X: pd.DataFrame, groups: pd.Series, top_n: int) -> List[str]:
    """ANOVA per feature across groups + BH-FDR; return top_n by adjusted p."""
    # Align indices
    X = X.loc[groups.index]
    group_names = pd.Index(sorted(groups.dropna().unique()))

    pvals = []
    cols = []
    for col in X.columns:
        # collect values per group and ensure at least 2 observations per group
        arrays = []
        ok = True
        for g in group_names:
            v = X.loc[groups == g, col].dropna()
            if v.size < 2:
                ok = False
                break
            arrays.append(v.values)
        if not ok:
            pvals.append(1.0)
            cols.append(col)
            continue
        try:
            p = float(f_oneway(*arrays).pvalue)
        except Exception:
            p = 1.0
        pvals.append(p)
        cols.append(col)

    pvals = np.asarray(pvals, dtype=float)
    # BH-FDR
    try:
        _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
    except Exception:
        p_adj = pvals.copy()

    dfp = pd.DataFrame({"feature": cols, "p": pvals, "p_adj": p_adj})
    dfp = dfp.sort_values("p_adj", kind="mergesort").reset_index(drop=True)
    return dfp.head(min(top_n, len(dfp))).feature.tolist()


def _impute_groupwise_min(X: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """
    Group-wise missing value substitution:
      - if a feature in a group has >=50% observed, fill its NaNs with group min
      - else fill with (global min / 5) of that feature across all samples
    """
    X = X.copy()
    groups = groups.copy()
    for col in X.columns:
        for g in groups.dropna().unique():
            mask_g = (groups == g)
            vals = X.loc[mask_g, col]
            if vals.notna().sum() == 0:
                continue
            frac_obs = vals.notna().mean()
            if frac_obs >= 0.5:
                fill_val = vals.min(skipna=True)
                X.loc[mask_g & X[col].isna(), col] = fill_val
            else:
                global_min = X[col].min(skipna=True)
                if pd.isna(global_min):
                    continue
                X.loc[mask_g & X[col].isna(), col] = global_min / 5.0
    # Any remaining NaNs -> column min or 0
    for col in X.columns:
        if X[col].isna().any():
            col_min = X[col].min(skipna=True)
            if pd.isna(col_min):
                X[col] = X[col].fillna(0.0)
            else:
                X[col] = X[col].fillna(col_min)
    return X


def _standardize(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    # Drop constant columns to avoid zero-variance errors
    variances = X.var(axis=0, skipna=True)
    keep = variances[variances > 0].index.tolist()
    if not keep:
        return pd.DataFrame(index=X.index)
    scaler = StandardScaler()
    Z = scaler.fit_transform(X[keep].values)
    return pd.DataFrame(Z, index=X.index, columns=keep)


def _cluster_corr(C: pd.DataFrame) -> pd.DataFrame:
    """Average-linkage clustering of a correlation matrix; returns re-ordered matrix."""
    # numeric safety
    C = C.copy()
    C = C.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # small jitter to avoid singularities
    C = C + np.eye(C.shape[0]) * 1e-12
    try:
        Z = linkage(C, method="average")
        order = leaves_list(Z)
        return C.iloc[order, order]
    except Exception:
        return C


def _plot_lower_triangle_heatmap(C: pd.DataFrame,
                                 title: str,
                                 out_png: Path,
                                 out_svg: Path,
                                 ylabels_clean: bool = True,
                                 tick_fs: int = 8) -> None:
    """Lower-triangle clustered heatmap with labels on left and on-diagonal rotated labels on top."""
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(C, dtype=bool), k=1)
    ax = sns.heatmap(
        C,
        cmap="coolwarm",
        center=0,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient", "orientation": "horizontal", "shrink": 0.6, "pad": 0.1},
        xticklabels=False,
        yticklabels=False
    )

    labels = list(C.columns)
    if ylabels_clean:
        labels = [_clean_feature_label(x) for x in labels]

    ticks = np.arange(len(labels)) + 0.5
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=tick_fs, rotation=0, va="center")

    # place rotated labels near diagonal
    for i, lab in enumerate(labels):
        ax.text(i + 0.5, i - 0.7, lab, rotation=90, fontsize=tick_fs,
                ha="center", va="bottom", clip_on=False)
        ax.plot(i + 0.5, i - 0.125, marker="|", color="black", markersize=4,
                markeredgewidth=1, clip_on=False)

    ax.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.savefig(out_svg, dpi=100, bbox_inches="tight")
    plt.close()

def _make_feature_labels(names: Iterable[str], feature_meta: Optional[pd.DataFrame]) -> pd.Index:
    names = pd.Index(names).astype(str)

    if feature_meta is None or feature_meta.empty or "UniqueID" not in feature_meta.columns:
        return names

    meta = feature_meta.copy()
    meta["UniqueID"] = meta["UniqueID"].astype(str)

    # prefer Annotation; fallback to Name if present
    label_col = "Annotation" if "Annotation" in meta.columns else ("Name" if "Name" in meta.columns else None)
    if label_col is None:
        return names

    map_ = meta.set_index("UniqueID")[label_col].astype(str)
    mapped = names.to_series().map(map_)
    mapped = mapped.fillna(names.to_series())              # <-- crucial: fill with a *Series*, not an Index
    mapped = mapped.map(_clean_feature_label)

    # de-duplicate labels (append UID where repeated)
    dup = mapped.duplicated(keep=False)
    if dup.any():
        mapped.loc[dup] = mapped.loc[dup] + " [" + names[dup] + "]"

    return pd.Index(mapped.values)


# ---------------------------
# Public API
# ---------------------------
def run_correlation_analysis(
    file_path: Optional[str],
    group_file: Optional[str],
    save_dir: str,
    group_col: str = "Group",
    sample_id_col: Optional[str] = None,
    top_list: Iterable[int] = (15, 25, 50, 100),
    do_groupwise: bool = True,
    group_order: Optional[list[str]] = None,
) -> None:

    """
    Main entry point.
    If Stats.utils.load_dataset is available, uses it; otherwise expects file_path to be a CSV
    where the first columns include sample metadata and numeric feature columns follow.

    Parameters
    ----------
    file_path : str or None
        Raw data table path (MetaboScape wide table). Ignored if load_dataset handles reading.
    group_file : str or None
        Path to group annotations for load_dataset; can be None if embedded.
    save_dir : str
        Output base directory for Correlation results.
    group_col : str
        Column in sample metadata that defines groups.
    sample_id_col : str or None
        Column in sample metadata with unique sample IDs. If None, inferred.
    top_list : iterable of int
        List of top-N sizes to compute (ANOVA pre-selection).
    do_groupwise : bool
        If True, produce per-group compound correlation heatmaps.
    """
    base = Path(save_dir)
    outdir = _ensure_dir(base)

    print('[Correlations] Running correlation analysis...', flush = True)

    # ---------------- Load data ----------------
    if load_dataset is not None:
        res = load_dataset(file_path, group_file)

        # Handle both signatures:
        #  (X, y, feature_meta)  OR  (data_wide, feature_meta, sample_meta)
        if isinstance(res, tuple) and len(res) == 3:
            A, B, C = res

            # Case 1: (X, y, feature_meta)
            if isinstance(B, (pd.Series, pd.DataFrame)) and getattr(A, "shape", (None,))[0] == len(B):
                data_wide   = A
                y_series    = B.iloc[:, 0] if isinstance(B, pd.DataFrame) else B
                feature_meta = C if isinstance(C, pd.DataFrame) else None

                # Build a minimal sample_meta compatible with the rest of the code
                sample_meta = pd.DataFrame(index=y_series.index).copy()
                sample_meta["Group"] = y_series.astype(str).values

                # Try to provide a readable sample id column if caller wants one
                if sample_id_col is None:
                    sample_id_col = "Sample"
                sample_meta[sample_id_col] = sample_meta.index

            # Case 2: (data_wide, feature_meta, sample_meta)
            else:
                data_wide, feature_meta, sample_meta = A, B, C

        else:
            raise ValueError("Unsupported return signature from load_dataset().")

        # Resolve group column name if 'Group' is not present
        if group_col not in sample_meta.columns:
            alt = [c for c in sample_meta.columns if c.lower() in {"group", "class", "condition"}]
            if alt:
                group_col = alt[0]
            else:
                raise ValueError(f"Group column '{group_col}' not found in sample metadata.")

        groups = sample_meta[group_col].astype("category")

        # Identify numeric feature columns and align indices
        feature_cols = _select_numeric_features(data_wide, exclude_cols=[])
        X_full = data_wide[feature_cols].copy()
        X_full.index = sample_meta.index

        # Sample IDs for labeling
        if sample_id_col and sample_id_col in sample_meta.columns:
            sample_ids = sample_meta[sample_id_col]
        else:
            sample_ids = pd.Series(sample_meta.index, index=sample_meta.index, name="Sample")

    else:
        if not file_path:
            raise ValueError("file_path is required when Stats.utils.load_dataset is not available.")
        df = pd.read_csv(file_path)
        # Heuristic: first two columns are [Sample, Group] in your older CSVs; new pipeline can be different.
        # Try to infer:
        candidate_group_cols = [c for c in df.columns if c.lower() in {"group", "class", "condition"}]
        if group_col in df.columns:
            gcol = group_col
        elif candidate_group_cols:
            gcol = candidate_group_cols[0]
        else:
            raise ValueError("Could not find a group column. Pass group_col=... explicitly or provide a compatible table.")
        # Sample id
        if sample_id_col is None:
            for cand in ("Sample", "SampleID", "Name"):
                if cand in df.columns:
                    sample_id_col = cand
                    break
            if sample_id_col is None:
                sample_id_col = df.columns[0]

        groups = df[gcol].astype("category")
        meta_cols = {sample_id_col, gcol}
        feature_cols = _select_numeric_features(df, exclude_cols=meta_cols)
        X_full = df[feature_cols].copy()
        X_full.index = df.index
        sample_ids = df[sample_id_col]

    # Align groups to X_full
    groups = groups.loc[X_full.index]
    
    # ---- normalize requested group order ----
    all_groups = list(pd.Index(groups.dropna().unique()))
    if group_order:
        wanted = [g for g in group_order if g in all_groups]
        leftovers = [g for g in all_groups if g not in wanted]
        ordered_groups = wanted + leftovers
    else:
        ordered_groups = all_groups

    # persist order used
    pd.Series(ordered_groups, name="GroupOrder").to_csv(outdir / "group_order_used.csv", index=False)

    # Save basic bookkeeping
    pd.DataFrame({"SampleID": sample_ids, "Group": groups}).to_csv(outdir / "samples_groups.csv", index=False)

    # Choose a single N for groupwise panels (smallest in the list)
    _groupwise_top_n = min(top_list) if top_list else 25
    _groupwise_feats = _anova_top_features(X_full, groups, top_n=_groupwise_top_n)

    # ---------------- Top-N loops ----------------
    for top_n in top_list:
        print(f"[Correlation] Top {top_n} — ANOVA+FDR pre-selection")
        # Guard for small panels
        if X_full.shape[1] == 0:
            print("No numeric features found. Skipping.")
            continue

        top_feats = _anova_top_features(X_full, groups, top_n=top_n)
        if len(top_feats) < 2:
            print(f"Found <2 usable features for top {top_n}. Skipping.")
            continue

        # Impute
        Xi = _impute_groupwise_min(X_full[top_feats], groups)

        # Standardize
        Z = _standardize(Xi)
        if Z.empty or Z.shape[1] < 2:
            print(f"Standardization dropped to <2 features for top {top_n}. Skipping.")
            continue

        # ---------------- Compound–compound ----------------
        Cc = pd.DataFrame(np.corrcoef(Z.values.T), index=Z.columns, columns=Z.columns)
        # relabel with annotations before clustering
        Cc = _cluster_corr(Cc)
        Cc.index = Cc.columns = _make_feature_labels(Cc.columns, feature_meta)
        Cc.to_csv(outdir / f"correlation_compounds_top{top_n}.csv")

        _plot_lower_triangle_heatmap(
            Cc,
            title=f"Compound Correlation — top {top_n}",
            out_png=outdir / f"correlation_compounds_top{top_n}.png",
            out_svg=outdir / f"correlation_compounds_top{top_n}.svg",
            ylabels_clean=True,
            tick_fs=8 if top_n >= 50 else 10 if top_n >= 25 else 12
        )

        # ---------------- Sample–sample ----------------
        # If a group order is provided, enforce group blocks and cluster within each block.
        Z_for_samples = Z
        if group_order:
            block_indices = []
            for g in ordered_groups:
                idx_g = Z.index[groups.loc[Z.index] == g]
                if len(idx_g) == 0:
                    continue
                # local clustering within group block
                try:
                    Cs_g = np.corrcoef(Z.loc[idx_g].values)
                    Zg = linkage(Cs_g, method="average")
                    ord_idx = idx_g[leaves_list(Zg)]
                except Exception:
                    ord_idx = idx_g
                block_indices.extend(list(ord_idx))
            if len(block_indices) >= 2:
                Z_for_samples = Z.loc[block_indices]

        Cs = pd.DataFrame(np.corrcoef(Z_for_samples.values),
                  index=Z_for_samples.index, columns=Z_for_samples.index)

        # Label with readable sample IDs first
        Cs.index = [str(sample_ids.loc[i]) for i in Cs.index]
        Cs.columns = [str(sample_ids.loc[i]) for i in Cs.columns]

        # ---- Select the top-N most correlated samples (by mean absolute r to others) ----
        if Cs.shape[0] > top_n:
            Cs_no_diag = Cs.copy()
            np.fill_diagonal(Cs_no_diag.values, np.nan)  # ignore self-corr in the score
            mean_abs = Cs_no_diag.abs().mean(axis=1, skipna=True)
            keep = (mean_abs.sort_values(ascending=False)
                        .head(top_n)
                        .index.tolist())
            Cs = Cs.loc[keep, keep]
            print(f"[Correlation] Sample panel reduced to top-{top_n} by mean|r|. "
                f"Retained: {len(keep)}/{len(mean_abs)} samples.", flush=True)
        else:
            print(f"[Correlation] {Cs.shape[0]} samples ≤ top-{top_n}; keeping all.", flush=True)

        # Save & plot the (possibly reduced) matrix
        Cs.to_csv(outdir / f"sample_correlation_matrix_top{top_n}.csv")
        pd.Series(Cs.index, name="SampleOrder").to_csv(outdir / f"sample_order_top{top_n}.csv", index=False)

        _plot_lower_triangle_heatmap(
            Cs,
            title=f"Sample Correlation — top {top_n} samples",
            out_png=outdir / f"sample_correlation_top{top_n}.png",
            out_svg=outdir / f"sample_correlation_top{top_n}.svg",
            ylabels_clean=False,
            tick_fs=8 if Cs.shape[0] >= 50 else 10 if Cs.shape[0] >= 25 else 12
        )


        # ---------------- Groupwise heatmaps (optional) ----------------
        if do_groupwise:
            gdir = _ensure_dir(outdir / f"Groupwise_top{top_n}")
            for idx_g, g in enumerate(ordered_groups, start=1):
                mask_g = (groups == g)
                Xg = X_full.loc[mask_g, _groupwise_feats].copy()   # <-- restrict to top-N features
                n_g = Xg.shape[0]
                if n_g < MIN_SAMPLES_PER_GROUP:
                    print(f"[Groupwise] {g} (top{top_n}): <{MIN_SAMPLES_PER_GROUP} samples (n={n_g}). Skipping.")
                    continue

                # simple within-group imputation (no leakage)
                Xi = Xg.copy()
                for c in Xi.columns:
                    col = Xi[c]
                    Xi[c] = col.fillna(col.min(skipna=True)) if col.notna().any() else 0.0

                Zg = _standardize(Xi)
                if Zg.shape[1] < 2:
                    print(f"[Groupwise] {g} (top{top_n}): <2 features after standardize. Skipping.")
                    continue

                Cg = pd.DataFrame(np.corrcoef(Zg.values.T), index=Zg.columns, columns=Zg.columns)
                Cg = _cluster_corr(Cg)
                Cg.index = Cg.columns = _make_feature_labels(Cg.columns, feature_meta)

                prefix = f"{idx_g:02d}_{g}_top{top_n}"
                _plot_lower_triangle_heatmap(
                    Cg,
                    title=f"Compound Correlation — {g} (top {top_n})",
                    out_png=gdir / f"{prefix}_compound_correlation.png",
                    out_svg=gdir / f"{prefix}_compound_correlation.svg",
                    ylabels_clean=True,
                    tick_fs=7 if Cg.shape[0] >= 60 else 8 if Cg.shape[0] >= 40 else 10
                )

# ---------------------------
# CLI (optional)
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Correlation analysis for LipidQuest pipeline.")
    parser.add_argument("--file", dest="file_path", default=None, help="Input table (CSV). Ignored if Stats.utils.load_dataset is available.")
    parser.add_argument("--groups", dest="group_file", default=None, help="Group file for load_dataset (optional).")
    parser.add_argument("--out", dest="save_dir", required=True, help="Output directory.")
    parser.add_argument("--group-col", dest="group_col", default="Group", help="Group column name.")
    parser.add_argument("--sample-col", dest="sample_id_col", default=None, help="Sample ID column name (optional).")
    parser.add_argument("--top", dest="top_list", default="15,25,50", help="Comma-separated top-N list.")
    parser.add_argument("--no-groupwise", action="store_true", help="Disable per-group heatmaps.")
    args = parser.parse_args()

    top_list = [int(x) for x in str(args.top_list).split(",") if str(x).strip().isdigit()]
    run_correlation_analysis(
        file_path=args.file_path,
        group_file=args.group_file,
        save_dir=args.save_dir,
        group_col=args.group_col,
        sample_id_col=args.sample_id_col,
        top_list=top_list,
        do_groupwise=(not args.no_groupwise)
    )
