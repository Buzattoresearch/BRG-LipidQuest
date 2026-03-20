# Stats/correlation_analysis.py
# ------------------------------------------------------------
# Correlation analysis for LipidQuest pipeline (MetaboScape exports)
# - Loads via Stats.utils.load_dataset(file_path, group_file) when available
# - Kruskal-Wallis per feature across groups + BH-FDR, select top-N features
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
from scipy.stats import kruskal, spearmanr, pearsonr, norm
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


def _kruskal_top_features(X: pd.DataFrame, groups: pd.Series, top_n: int) -> List[str]:
    """Kruskal-Wallis per feature across groups + BH-FDR; return top_n by adjusted p."""
    # Align indices
    X = X.loc[groups.index]
    group_names = pd.Index(sorted(groups.dropna().unique()))

    pvals = []
    cols = []
    for col in X.columns:
        arrays = []
        for g in group_names:
            v = X.loc[groups == g, col].dropna()
            if v.size > 0:
                arrays.append(v.values)
        if len(arrays) < 2:
            pvals.append(1.0)
            cols.append(col)
            continue
        try:
            p = float(kruskal(*arrays).pvalue)
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
    
def _fisher_z_compare(r1: float, n1: int, r2: float, n2: int) -> Tuple[float, float]:
    """
    Compare two independent correlations using Fisher z transform.
    Returns:
        z_stat, p_value
    """
    if n1 < 4 or n2 < 4:
        return np.nan, np.nan

    if not np.isfinite(r1) or not np.isfinite(r2):
        return np.nan, np.nan

    # avoid infinities at |r| = 1
    r1 = float(np.clip(r1, -0.999999, 0.999999))
    r2 = float(np.clip(r2, -0.999999, 0.999999))

    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    se = np.sqrt((1.0 / (n1 - 3)) + (1.0 / (n2 - 3)))
    if se <= 0 or not np.isfinite(se):
        return np.nan, np.nan

    z_stat = (z1 - z2) / se
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    return float(z_stat), float(p_value)

def _pair_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> tuple[float, int]:
    """
    Compute within-group correlation for one lipid pair.
    Returns:
        correlation, n_used
    """
    df = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    if n < 3:
        return np.nan, n

    xvals = df.iloc[:, 0].astype(float).values
    yvals = df.iloc[:, 1].astype(float).values

    try:
        if method == "pearson":
            r, _ = pearsonr(xvals, yvals)
        else:
            r, _ = spearmanr(xvals, yvals)
    except Exception:
        return np.nan, n

    if not np.isfinite(r):
        return np.nan, n
    return float(r), n

def _differential_correlation(
    X: pd.DataFrame,
    groups: pd.Series,
    group1: str,
    group2: str,
    feature_meta: Optional[pd.DataFrame] = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compare pairwise lipid correlations between two groups.
    """
    X = X.copy()
    groups = groups.astype(str)

    idx1 = groups[groups == str(group1)].index
    idx2 = groups[groups == str(group2)].index

    X1 = X.loc[idx1]
    X2 = X.loc[idx2]
    if X1.shape[0] < 6 or X2.shape[0] < 6:
        print(f"[DiffCorr] Warning: small n for {group1} vs {group2}. Results may be unstable.", flush=True)
        
    if X1.shape[0] < 3 or X2.shape[0] < 3:
        print(f"[DiffCorr] Skipping {group1} vs {group2}: fewer than 3 samples in one group.", flush=True)
        return pd.DataFrame()

    feats = list(X.columns)
    results = []

    meta_map = {}
    class_map = {}
    if feature_meta is not None and not feature_meta.empty and "UniqueID" in feature_meta.columns:
        fm = feature_meta.copy()
        fm["UniqueID"] = fm["UniqueID"].astype(str)
        if "Annotation" in fm.columns:
            meta_map = fm.drop_duplicates("UniqueID").set_index("UniqueID")["Annotation"].to_dict()
        if "Lipid Class" in fm.columns:
            class_map = fm.drop_duplicates("UniqueID").set_index("UniqueID")["Lipid Class"].to_dict()

    for i in range(len(feats)):
        f1 = feats[i]
        for j in range(i + 1, len(feats)):
            f2 = feats[j]

            r1, n1 = _pair_corr(X1[f1], X1[f2], method=method)
            r2, n2 = _pair_corr(X2[f1], X2[f2], method=method)

            z_stat, p = _fisher_z_compare(r1, n1, r2, n2)
            delta_r = np.nan
            if np.isfinite(r1) and np.isfinite(r2):
                delta_r = r1 - r2

            results.append({
                "Feature_1": f1,
                "Feature_2": f2,
                "Annotation_1": meta_map.get(str(f1), str(f1)),
                "Annotation_2": meta_map.get(str(f2), str(f2)),
                "Class_1": class_map.get(str(f1), ""),
                "Class_2": class_map.get(str(f2), ""),
                f"r_{group1}": r1,
                f"r_{group2}": r2,
                f"n_{group1}": n1,
                f"n_{group2}": n2,
                "delta_r": delta_r,
                "z_stat": z_stat,
                "p-value": p,
            })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out["p-value"] = pd.to_numeric(out["p-value"], errors="coerce")
    valid = out["p-value"].notna()
    out["FDR p-value"] = np.nan
    if valid.any():
        out.loc[valid, "FDR p-value"] = multipletests(out.loc[valid, "p-value"], method="fdr_bh")[1]

    out["abs_delta_r"] = out["delta_r"].abs()

    def _label_change(row):
        r1 = row.get(f"r_{group1}", np.nan)
        r2 = row.get(f"r_{group2}", np.nan)
        if not np.isfinite(r1) or not np.isfinite(r2):
            return "Unclear"
        if r1 >= 0.3 and r2 < 0.3:
            return f"Lost positive in {group2}"
        if r1 < 0.3 and r2 >= 0.3:
            return f"Gained positive in {group2}"
        if r1 <= -0.3 and r2 > -0.3:
            return f"Lost negative in {group2}"
        if r1 > -0.3 and r2 <= -0.3:
            return f"Gained negative in {group2}"
        return "Shifted magnitude"

    out["Change Type"] = out.apply(_label_change, axis=1)
    if out["FDR p-value"].notna().any():
        out = out.sort_values(["FDR p-value", "abs_delta_r"], ascending=[True, False], na_position="last")
    else:
        out = out.sort_values(["abs_delta_r"], ascending=[False], na_position="last")
    return out

def _plot_top_diffcorr_pairs(
    X: pd.DataFrame,
    groups: pd.Series,
    diff_df: pd.DataFrame,
    group1: str,
    group2: str,
    out_dir: Path,
    top_k: int = 10,
):
    use = diff_df.dropna(subset=["delta_r"]).sort_values("abs_delta_r", ascending=False).head(top_k)
    if use.empty:
        return

    for rank, (_, row) in enumerate(use.iterrows(), start=1):
        f1 = row["Feature_1"]
        f2 = row["Feature_2"]

        fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
        ax.set_facecolor("white")

        m1 = groups == str(group1)
        m2 = groups == str(group2)

        ax.scatter(X.loc[m1, f1], X.loc[m1, f2], s=40, alpha=0.8, label=f"{group1}")
        ax.scatter(X.loc[m2, f1], X.loc[m2, f2], s=40, alpha=0.8, label=f"{group2}")

        ax.set_xlabel(str(row.get("Annotation_1", f1)))
        ax.set_ylabel(str(row.get("Annotation_2", f2)))
        ax.set_title(
            f"#{rank} differential correlation\n"
            f"{group1}: r={row.get(f'r_{group1}', np.nan):.2f}, "
            f"{group2}: r={row.get(f'r_{group2}', np.nan):.2f}"
        )
        ax.legend(frameon=False)
        plt.tight_layout()

        safe_g1 = re.sub(r'[<>:."/\\|?*]', "_", str(group1))
        safe_g2 = re.sub(r'[<>:."/\\|?*]', "_", str(group2))
        out_png = out_dir / f"diffcorr_pair_{rank:02d}_{safe_g1}_vs_{safe_g2}.png"
        out_svg = out_dir / f"diffcorr_pair_{rank:02d}_{safe_g1}_vs_{safe_g2}.svg"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        fig.savefig(out_svg, dpi=120, bbox_inches="tight")
        plt.close(fig)
        
def _plot_lower_triangle_heatmap(C: pd.DataFrame,
                                 title: str,
                                 out_png: Path,
                                 out_svg: Path,
                                 ylabels_clean: bool = True,
                                 tick_fs: int = 8,
                                 note_text: Optional[str] = None) -> None:
    """Lower-triangle clustered heatmap with labels on left and on-diagonal rotated labels on top."""
    label_fs = tick_fs + 2
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
    ax.set_yticklabels(labels, fontsize=label_fs, rotation=0, va="center")

    # place rotated labels near diagonal
    for i, lab in enumerate(labels):
        ax.text(i + 0.5, i - 0.7, lab, rotation=90, fontsize=label_fs,
                ha="center", va="bottom", clip_on=False)
        ax.plot(i + 0.5, i - 0.125, marker="|", color="black", markersize=4,
                markeredgewidth=1, clip_on=False)

    ax.set_title(title, fontsize=14, pad=20)
    if note_text:
        plt.gcf().subplots_adjust(bottom=0.12)
        plt.gcf().text(0.5, 0.02, note_text, ha="center", va="bottom", fontsize=10, color="dimgray")
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


def _sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:."/\\|?*]', "_", str(s))


def _feature_class_map(feature_names: Iterable[str], feature_meta: Optional[pd.DataFrame]) -> pd.Series:
    feature_names = pd.Index(feature_names).astype(str)
    if feature_meta is None or feature_meta.empty or "UniqueID" not in feature_meta.columns:
        return pd.Series(index=feature_names, dtype=object)

    fm = feature_meta.copy()
    fm["UniqueID"] = fm["UniqueID"].astype(str)
    class_col = next((c for c in fm.columns if str(c).strip().lower() in {"lipid class", "lipid_class", "class"}), None)
    if class_col is None:
        return pd.Series(index=feature_names, dtype=object)

    class_map = fm.drop_duplicates("UniqueID").set_index("UniqueID")[class_col].astype(str).str.strip()
    class_map = class_map.replace({"nan": np.nan, "": np.nan})
    return class_map.reindex(feature_names)


def _corr_summary_stats(C: pd.DataFrame) -> dict:
    if C is None or C.empty or C.shape[0] < 2:
        return {
            "mean_spearman_r": np.nan,
            "median_spearman_r": np.nan,
            "abs_mean_spearman_r": np.nan,
            "n_pairs": 0,
        }
    vals = C.to_numpy(dtype=float)
    tri = vals[np.triu_indices_from(vals, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return {
            "mean_spearman_r": np.nan,
            "median_spearman_r": np.nan,
            "abs_mean_spearman_r": np.nan,
            "n_pairs": 0,
        }
    return {
        "mean_spearman_r": float(np.mean(tri)),
        "median_spearman_r": float(np.median(tri)),
        "abs_mean_spearman_r": float(np.mean(np.abs(tri))),
        "n_pairs": int(tri.size),
    }


def _plot_simple_heatmap(
    data: pd.DataFrame,
    title: str,
    out_png: Path,
    out_svg: Path,
    cmap: str = "coolwarm",
    center: float = 0.0,
    tick_fs: int = 10,
    note_text: Optional[str] = None,
) -> None:
    if data.empty:
        return
    fig_h = max(4.5, 0.34 * len(data.index) + 2.2)
    fig_w = max(6.5, 0.55 * len(data.columns) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    sns.heatmap(
        data,
        cmap=cmap,
        center=center,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean z-score" if center == 0.0 else "Value"},
        ax=ax,
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.tick_params(axis="x", labelrotation=45, labelsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    if note_text:
        fig.subplots_adjust(bottom=0.16)
        fig.text(0.5, 0.02, note_text, ha="center", va="bottom", fontsize=10, color="dimgray")
    plt.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    fig.savefig(out_svg, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _run_class_level_correlation_analysis(
    X_full: pd.DataFrame,
    groups: pd.Series,
    feature_meta: Optional[pd.DataFrame],
    ordered_groups: List[str],
    outdir: Path,
) -> None:
    class_map = _feature_class_map(X_full.columns, feature_meta)
    valid_classes = sorted([c for c in pd.unique(class_map.dropna()) if str(c).strip() and str(c).strip().lower() != "nan"])
    if not valid_classes:
        return

    class_dir = _ensure_dir(outdir / "Classwise_Spearman")
    overall_dir = _ensure_dir(class_dir / "Overall")
    groupwise_dir = _ensure_dir(class_dir / "ByGroup")
    summary_rows = []
    group_rows = []

    for lipid_class in valid_classes:
        feats = [f for f in X_full.columns if class_map.get(f) == lipid_class]
        if len(feats) < 2:
            continue

        Xc = X_full.loc[:, feats].apply(pd.to_numeric, errors="coerce")
        C = Xc.corr(method="spearman", min_periods=3)
        stats = _corr_summary_stats(C)
        summary_rows.append({
            "Lipid Class": lipid_class,
            "n_features": int(len(feats)),
            "n_samples": int(Xc.shape[0]),
            **stats,
        })

        if stats["n_pairs"] > 0:
            safe_class = _sanitize_filename(lipid_class)
            C.to_csv(overall_dir / f"{safe_class}_spearman_correlation.csv")
        _plot_lower_triangle_heatmap(
            C,
            title=f"{lipid_class}: within-class Spearman correlation",
            out_png=overall_dir / f"{safe_class}_spearman_correlation.png",
            out_svg=overall_dir / f"{safe_class}_spearman_correlation.svg",
            ylabels_clean=True,
            tick_fs=10 if len(feats) <= 20 else 8,
            note_text=(
                "Red cells indicate lipids that tend to increase and decrease together across samples.\n"
                "Blue cells indicate inverse behavior; stronger colors mean stronger coordination within the class."
            ),
        )

        for group in ordered_groups:
            mask_g = groups == group
            Xg = X_full.loc[mask_g, feats].apply(pd.to_numeric, errors="coerce")
            if Xg.shape[0] < 3:
                continue
            Cg = Xg.corr(method="spearman", min_periods=3)
            gstats = _corr_summary_stats(Cg)
            group_rows.append({
                "Lipid Class": lipid_class,
                "Group": group,
                "n_features": int(len(feats)),
                "n_samples": int(Xg.shape[0]),
                **gstats,
            })
            if gstats["n_pairs"] > 0:
                safe_class = _sanitize_filename(lipid_class)
                safe_group = _sanitize_filename(group)
                group_subdir = _ensure_dir(groupwise_dir / safe_group)
                Cg.to_csv(group_subdir / f"{safe_class}_spearman_correlation.csv")
                _plot_lower_triangle_heatmap(
                    Cg,
                    title=f"{lipid_class}: Spearman correlation in {group}",
                    out_png=group_subdir / f"{safe_class}_spearman_correlation.png",
                    out_svg=group_subdir / f"{safe_class}_spearman_correlation.svg",
                    ylabels_clean=True,
                    tick_fs=10 if len(feats) <= 20 else 8,
                    note_text=(
                        "This heatmap shows whether lipids in the same class move together within this group only.\n"
                        "Compare patterns across groups to see whether class coordination is strengthened, weakened, or rewired."
                    ),
                )

    summary_df = pd.DataFrame(summary_rows).sort_values("abs_mean_spearman_r", ascending=False, na_position="last")
    group_df = pd.DataFrame(group_rows)
    summary_df.to_csv(class_dir / "class_mean_within_correlation_overall.csv", index=False)
    group_df.to_csv(class_dir / "class_mean_within_correlation_by_group.csv", index=False)

    if not group_df.empty:
        mean_table = group_df.pivot(index="Lipid Class", columns="Group", values="mean_spearman_r").reindex(columns=ordered_groups)
        abs_table = group_df.pivot(index="Lipid Class", columns="Group", values="abs_mean_spearman_r").reindex(columns=ordered_groups)
        mean_table.to_csv(class_dir / "class_mean_within_correlation_by_group_matrix.csv")
        abs_table.to_csv(class_dir / "class_abs_mean_within_correlation_by_group_matrix.csv")
        _plot_simple_heatmap(
            mean_table,
            title="Mean within-class Spearman correlation by group",
            out_png=class_dir / "class_mean_within_correlation_by_group_heatmap.png",
            out_svg=class_dir / "class_mean_within_correlation_by_group_heatmap.svg",
            cmap="coolwarm",
            center=0.0,
            tick_fs=9,
            note_text=(
                "Positive values mean lipids within that class tend to rise and fall together in that group.\n"
                "Values near zero suggest weaker coordination within the class."
            ),
        )
        _plot_simple_heatmap(
            abs_table,
            title="Absolute mean within-class Spearman correlation by group",
            out_png=class_dir / "class_abs_mean_within_correlation_by_group_heatmap.png",
            out_svg=class_dir / "class_abs_mean_within_correlation_by_group_heatmap.svg",
            cmap="mako",
            center=None,
            tick_fs=9,
            note_text=(
                "Higher values indicate stronger within-class coupling regardless of direction.\n"
                "Use this to see which classes behave in a more tightly coordinated way within each group."
            ),
        )

    # Class trajectory scores: average within-class feature z-scores per sample
    traj_scores = pd.DataFrame(index=X_full.index)
    for lipid_class in valid_classes:
        feats = [f for f in X_full.columns if class_map.get(f) == lipid_class]
        if not feats:
            continue
        Xc = X_full.loc[:, feats].apply(pd.to_numeric, errors="coerce")
        means = Xc.mean(axis=0, skipna=True)
        stds = Xc.std(axis=0, skipna=True, ddof=1).replace(0, np.nan)
        Zc = Xc.subtract(means, axis=1).div(stds, axis=1)
        traj_scores[lipid_class] = Zc.mean(axis=1, skipna=True)

    if not traj_scores.empty:
        traj_scores.index.name = "Sample"
        traj_scores.to_csv(class_dir / "class_trajectory_scores_per_sample.csv")
        traj_group = traj_scores.copy()
        traj_group["Group"] = groups.reindex(traj_scores.index).astype(str).values
        traj_group_means = traj_group.groupby("Group").mean(numeric_only=True).reindex(ordered_groups)
        traj_group_means.to_csv(class_dir / "class_trajectory_scores_by_group.csv")
        _plot_simple_heatmap(
            traj_group_means.T,
            title="Class trajectory scores by group",
            out_png=class_dir / "class_trajectory_scores_by_group_heatmap.png",
            out_svg=class_dir / "class_trajectory_scores_by_group_heatmap.svg",
            cmap="coolwarm",
            center=0.0,
            tick_fs=9,
            note_text=(
                "Positive scores indicate that the class tends to be elevated overall in that group.\n"
                "Negative scores indicate lower overall class signal relative to the across-sample average."
            ),
        )


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
    group_order: Optional[List[str]] = None,
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
        List of top-N sizes to compute (Kruskal pre-selection).
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
        feature_meta = None
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

    # ---------------- Class-level correlation summaries ----------------
    try:
        _run_class_level_correlation_analysis(
            X_full=X_full,
            groups=groups.astype(str),
            feature_meta=feature_meta,
            ordered_groups=ordered_groups,
            outdir=outdir,
        )
    except Exception as exc:
        print(f"[Correlations] Class-level correlation block skipped: {exc}", flush=True)

    # Choose a single N for groupwise panels (smallest in the list)
    _groupwise_top_n = min(top_list) if top_list else 25
    _groupwise_feats = _kruskal_top_features(X_full, groups, top_n=_groupwise_top_n)

    # ---------------- Top-N loops ----------------
    for top_n in top_list:
        print(f"[Correlation] Top {top_n} — Kruskal+FDR pre-selection")
        # Guard for small panels
        if X_full.shape[1] == 0:
            print("No numeric features found. Skipping.")
            continue

        top_feats = _kruskal_top_features(X_full, groups, top_n=top_n)
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
            tick_fs=8 if top_n >= 50 else 10 if top_n >= 25 else 12,
            note_text=(
                "Red cells mark lipid pairs that tend to increase and decrease together across samples.\n"
                "Blue cells mark inverse relationships; clustered blocks suggest coordinated lipid modules."
            ),
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
            tick_fs=8 if Cs.shape[0] >= 50 else 10 if Cs.shape[0] >= 25 else 12,
            note_text=(
                "Warmer colors indicate samples with more similar overall lipid patterns.\n"
                "Separated blocks can reflect biological groups, batch structure, or possible outliers."
            ),
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
                    tick_fs=7 if Cg.shape[0] >= 60 else 8 if Cg.shape[0] >= 40 else 10,
                    note_text=(
                        "This shows how strongly the selected lipids move together within this group only.\n"
                        "Compare these group-specific patterns to see whether coordination strengthens or weakens between groups."
                    ),
                )
    
        # # ---------------- Differential correlation between groups ----------------
        # pairdir = _ensure_dir(outdir / f"Differential_top{top_n}")

        # diff_input_X = _impute_groupwise_min(X_full[top_feats], groups)
        # diff_input_groups = groups.loc[diff_input_X.index]

        # comp_groups = ordered_groups
        
        # for i in range(len(comp_groups)):
        #     for j in range(i + 1, len(comp_groups)):
        #         g1 = comp_groups[i]
        #         g2 = comp_groups[j]

        #         print(f"[DiffCorr] {g1} vs {g2} (top {top_n})", flush=True)

        #         diff_df = _differential_correlation(
        #             X=diff_input_X,
        #             groups=diff_input_groups,
        #             group1=g1,
        #             group2=g2,
        #             feature_meta=feature_meta,
        #             method="spearman",
        #         )

        #         if diff_df.empty:
        #             print(f"[DiffCorr] No differential-correlation output for {g1} vs {g2} at top {top_n}.", flush=True)
        #             continue

        #         safe_g1 = re.sub(r'[<>:."/\\|?*]', "_", str(g1))
        #         safe_g2 = re.sub(r'[<>:."/\\|?*]', "_", str(g2))

        #         diff_df.to_csv(
        #             pairdir / f"diffcorr_{safe_g1}_vs_{safe_g2}_top{top_n}.csv",
        #             index=False
        #         )

        #         top_changed = diff_df.head(50).copy()
        #         if not top_changed.empty:
        #             top_changed.to_csv(
        #                 pairdir / f"diffcorr_{safe_g1}_vs_{safe_g2}_top{top_n}_top50.csv",
        #                 index=False
        #             )

        #         pair_plot_dir = _ensure_dir(pairdir / f"{safe_g1}_vs_{safe_g2}_top{top_n}")
        #         _plot_top_diffcorr_pairs(
        #             X=diff_input_X,
        #             groups=diff_input_groups,
        #             diff_df=diff_df,
        #             group1=g1,
        #             group2=g2,
        #             out_dir=pair_plot_dir,
        #             top_k=10,
        #         )
    
    
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
