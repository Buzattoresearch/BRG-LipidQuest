from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests

from Stats.figure_style import get_figure_style
from Stats.utils import prepare_output_dir

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
plt.ioff()


def _load_dataset_with_meta(file_path: str, group_file: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(file_path, low_memory=False)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    meta_cols = [
        "UniqueID", "RT (min)", "m/z", "Polarity", "Annotation", "Annotation Type",
        "Headgroup", "Lipid Class", "MS/MS score", "Annotation tier", "mSigma",
        "Molecular Formula", "Plasmenyl?", "Number of carbons in fatty acyls",
        "Double bond equivalents", "Chain type", "PUFA?", "Modifications",
        "# of modifications", "Oxidized?",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    sample_cols = [c for c in df.columns if c not in meta_cols]

    if group_file is not None and os.path.exists(group_file):
        gdf = pd.read_csv(group_file, low_memory=False)
        if "Sample" not in gdf.columns or "Group" not in gdf.columns:
            raise ValueError("Group file must contain 'Sample' and 'Group'.")
    else:
        gdf = pd.DataFrame({"Sample": sample_cols, "Group": "Unknown"})

    gdf = gdf.copy()
    gdf["Sample"] = gdf["Sample"].astype(str).str.strip()
    gdf["Group"] = gdf["Group"].astype(str).str.strip()
    df_cols_lower = {str(c).lower(): c for c in sample_cols}
    matched = [df_cols_lower[s.lower()] for s in gdf["Sample"] if s.lower() in df_cols_lower]
    if not matched:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X = df[matched].T
    X.index.name = "Sample"
    feature_meta = df[meta_cols].copy()
    if "UniqueID" in feature_meta.columns:
        X.columns = feature_meta["UniqueID"].astype(str).tolist()
    else:
        X.columns = [f"Feature_{i+1}" for i in range(X.shape[1])]
    X = X.apply(pd.to_numeric, errors="coerce")

    sample_meta = gdf.drop_duplicates(subset=["Sample"], keep="first").set_index("Sample").reindex(X.index).reset_index()
    return X, sample_meta, feature_meta


def _order_groups(present: List[str], group_order: Optional[List[str]]) -> List[str]:
    present = [str(g) for g in present]
    if not group_order:
        return present
    gui = [g for g in group_order if g in present]
    rest = [g for g in present if g not in group_order]
    return gui + rest


def _choose_batch_column(sample_meta: pd.DataFrame) -> Optional[str]:
    candidates = [
        "Batch", "batch",
        "Plate", "plate",
        "RunOrder", "Run Order",
        "InjectionOrder", "Injection Order",
        "Order", "order",
    ]
    for cand in candidates:
        if cand in sample_meta.columns:
            return cand
    return None


def _build_design(sample_meta: pd.DataFrame, group_col: str = "Group", batch_col: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    design = pd.DataFrame(index=sample_meta.index)
    design["Intercept"] = 1.0

    groups = sample_meta[group_col].astype(str)
    dummies_group = pd.get_dummies(groups, prefix="Group", drop_first=True, dtype=float)
    design = pd.concat([design, dummies_group], axis=1)

    if batch_col and batch_col in sample_meta.columns:
        batch = sample_meta[batch_col]
        batch_numeric = pd.to_numeric(batch, errors="coerce")
        numeric_fraction = float(batch_numeric.notna().mean()) if len(batch_numeric) else 0.0
        if pd.api.types.is_numeric_dtype(batch) or numeric_fraction >= 0.9:
            design[f"Batch_{batch_col}"] = batch_numeric.fillna(batch_numeric.median())
        else:
            dummies_batch = pd.get_dummies(batch.astype(str), prefix="Batch", drop_first=True, dtype=float)
            design = pd.concat([design, dummies_batch], axis=1)

    return design, groups.tolist()


def _fit_feature_ols(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float, int, np.ndarray]:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    yv = y[mask]
    Xv = X[mask, :]
    n, p = Xv.shape
    if n <= p or n < 3:
        return np.full(p, np.nan), np.nan, max(n - p, 0), np.full((p, p), np.nan)

    XtX = Xv.T @ Xv
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan), np.nan, max(n - p, 0), np.full((p, p), np.nan)

    beta = XtX_inv @ Xv.T @ yv
    resid = yv - Xv @ beta
    df_resid = n - p
    if df_resid <= 0:
        return beta, np.nan, df_resid, XtX_inv
    sigma2 = float((resid @ resid) / df_resid)
    return beta, sigma2, df_resid, XtX_inv


def _moderate_variances(sigma2: np.ndarray, df_resid: np.ndarray) -> Tuple[np.ndarray, float, float]:
    valid = np.isfinite(sigma2) & (sigma2 > 0) & np.isfinite(df_resid) & (df_resid > 0)
    if valid.sum() < 10:
        s0 = float(np.nanmedian(sigma2[valid])) if valid.any() else 1.0
        d0 = 4.0
        post = np.where(valid, (d0 * s0 + df_resid * sigma2) / (d0 + df_resid), np.nan)
        return post, s0, d0

    log_s2 = np.log(sigma2[valid])
    s0 = float(np.exp(np.mean(log_s2)))
    var_log = float(np.var(log_s2, ddof=1))
    d0 = float(max(4.0, min(100.0, 2.0 / max(var_log, 1e-6))))
    post = np.where(valid, (d0 * s0 + df_resid * sigma2) / (d0 + df_resid), np.nan)
    return post, s0, d0


def _contrast_vector(design_cols: List[str], group_a: str, group_b: str, baseline_group: str) -> np.ndarray:
    c = np.zeros(len(design_cols), dtype=float)

    def _group_effect(group: str) -> np.ndarray:
        eff = np.zeros(len(design_cols), dtype=float)
        if group == baseline_group:
            return eff
        col = f"Group_{group}"
        if col in design_cols:
            eff[design_cols.index(col)] = 1.0
        return eff

    return _group_effect(group_a) - _group_effect(group_b)


def _plot_top_hits(top_df: pd.DataFrame, out_png: str, title: str, style: dict) -> None:
    if top_df.empty:
        return
    plot_df = top_df.head(20).copy()
    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(plot_df) + 2)), facecolor="white")
    ax.barh(np.arange(len(plot_df)), plot_df["abs_log2FC"], color="#4C78A8")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["Label"].tolist(), fontsize=style["tick_size"])
    ax.invert_yaxis()
    ax.set_xlabel("|log2 fold-change|", fontsize=style["label_size"], labelpad=10)
    ax.set_title(title, fontsize=style["title_size"], pad=12, fontweight="semibold")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.tick_params(axis="x", labelsize=style["tick_size"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style["line_width"])
        spine.set_color("black")
    fig.savefig(out_png, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _resolve_dashboard_context(analysis_dir: Path) -> Tuple[Path, Optional[str]]:
    if analysis_dir.parent.name == "Advanced_Differential":
        return analysis_dir.parent.parent, analysis_dir.name
    return analysis_dir.parent, None


def _candidate_analysis_paths(root_dir: Path, analysis_name: str, label: Optional[str], filename: str) -> List[Path]:
    paths = []
    if label:
        paths.append(root_dir / analysis_name / label / filename)
    paths.append(root_dir / analysis_name / filename)
    return paths


def _first_existing_path(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _build_summary_dashboard(analysis_dir: Path) -> Optional[str]:
    records = []
    root_dir, label = _resolve_dashboard_context(analysis_dir)

    for csv_path in analysis_dir.glob("*_moderated_results.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty or "FDR_BH" not in df.columns:
            continue
        top = df.sort_values(["FDR_BH", "abs_log2FC"], ascending=[True, False]).head(10)
        for _, row in top.iterrows():
            records.append({
                "Source": "Feature",
                "Analysis": csv_path.stem.replace("_moderated_results", ""),
                "ID": row.get("UniqueID", ""),
                "Label": row.get("Label", row.get("UniqueID", "")),
                "Effect": row.get("log2FC", np.nan),
                "P_value": row.get("Moderated_p_value", np.nan),
                "FDR_BH": row.get("FDR_BH", np.nan),
            })

    ratio_csv = _first_existing_path(_candidate_analysis_paths(root_dir, "Ratios", label, "ratio_statistics.csv"))
    if ratio_csv is not None and ratio_csv.exists():
        df = pd.read_csv(ratio_csv)
        if not df.empty:
            top = df.sort_values(["FDR_BH", "Kruskal_p_value"], ascending=[True, True]).head(10)
            for _, row in top.iterrows():
                records.append({
                    "Source": "Ratio",
                    "Analysis": row.get("Category", "Ratios"),
                    "ID": row.get("Ratio", ""),
                    "Label": row.get("Ratio", ""),
                    "Effect": np.nan,
                    "P_value": row.get("Kruskal_p_value", np.nan),
                    "FDR_BH": row.get("FDR_BH", np.nan),
                })

    enrich_candidates = [
        _first_existing_path(_candidate_analysis_paths(root_dir, "Enrichment", label, os.path.join("LipidClass_Enrichment", "lipid_class_enrichment_statistics.csv"))),
        _first_existing_path(_candidate_analysis_paths(root_dir, "Enrichment", label, os.path.join("ChainLength_Enrichment", "chain_length_enrichment_statistics.csv"))),
        _first_existing_path(_candidate_analysis_paths(root_dir, "Enrichment", label, os.path.join("Unsaturation_Enrichment", "unsaturation_enrichment_statistics.csv"))),
        _first_existing_path(_candidate_analysis_paths(root_dir, "Enrichment", label, os.path.join("SaturationIndex_Changes", "saturation_index_change_statistics.csv"))),
    ]
    for path in enrich_candidates:
        if path is None or not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        label_col = next((c for c in df.columns if c not in {"Kruskal_p_value", "FDR_BH", "Groups_tested"}), None)
        if not label_col:
            continue
        top = df.sort_values(["FDR_BH", "Kruskal_p_value"], ascending=[True, True]).head(10)
        for _, row in top.iterrows():
            records.append({
                "Source": "Enrichment",
                "Analysis": path.stem,
                "ID": row.get(label_col, ""),
                "Label": row.get(label_col, ""),
                "Effect": np.nan,
                "P_value": row.get("Kruskal_p_value", np.nan),
                "FDR_BH": row.get("FDR_BH", np.nan),
            })

    missingness_csv = _first_existing_path(
        _candidate_analysis_paths(root_dir, "Missingness", label, "differential_presence_absence_statistics.csv")
    )
    if missingness_csv is not None and missingness_csv.exists():
        df = pd.read_csv(missingness_csv)
        if not df.empty:
            top = df.sort_values(["FDR_BH", "Chi2_p_value"], ascending=[True, True]).head(10)
            for _, row in top.iterrows():
                feature_id = row.get("UniqueID", row.get("Feature", ""))
                records.append({
                    "Source": "Missingness",
                    "Analysis": "differential_presence_absence",
                    "ID": feature_id,
                    "Label": feature_id,
                    "Effect": row.get("Detection_rate_range", np.nan),
                    "P_value": row.get("Chi2_p_value", np.nan),
                    "FDR_BH": row.get("FDR_BH", np.nan),
                })

    if not records:
        return None

    dashboard = pd.DataFrame(records).sort_values(["FDR_BH", "P_value"], ascending=[True, True], na_position="last")
    out_csv = analysis_dir / "Summary_Dashboard_top_hits.csv"
    dashboard.to_csv(out_csv, index=False)
    return str(out_csv)


def run_from_stats(
    file_path: str,
    group_file: Optional[str],
    save_dir: str,
    group_order: Optional[List[str]] = None,
    group_colors: Optional[dict] = None,
    exclude_qc: bool = True,
    pseudocount_quantile: float = 0.01,
    dpi: int = 100,
    publication_theme: bool = False,
) -> Dict[str, str]:
    out_dir = prepare_output_dir(Path(save_dir))
    style = get_figure_style(publication_theme=publication_theme, dpi=dpi)
    print("[Advanced Differential] Running batch-aware moderated differential analysis...", flush=True)

    X, sample_meta, feature_meta = _load_dataset_with_meta(file_path, group_file)
    if X.empty or sample_meta.empty:
        raise ValueError("Dataset appears empty or malformed.")

    sample_meta = sample_meta.copy()
    sample_meta["Sample"] = sample_meta["Sample"].astype(str)
    sample_meta["Group"] = sample_meta["Group"].astype(str)
    X.index = X.index.astype(str)
    if exclude_qc:
        keep_mask = ~sample_meta["Group"].astype(str).str.contains("QC", case=False, na=False)
        sample_meta = sample_meta.loc[keep_mask].reset_index(drop=True)
        X = X.loc[sample_meta["Sample"]]

    ordered_groups = _order_groups(pd.unique(sample_meta["Group"]).tolist(), group_order)
    if len(ordered_groups) < 2:
        raise ValueError("Need at least two non-QC groups for differential analysis.")

    batch_col = _choose_batch_column(sample_meta)
    design, group_values = _build_design(sample_meta, group_col="Group", batch_col=batch_col)
    design_cols = design.columns.tolist()
    Xmat = design.to_numpy(dtype=float)
    feature_meta_lookup = feature_meta.copy()
    if "UniqueID" in feature_meta_lookup.columns:
        feature_meta_lookup["UniqueID"] = feature_meta_lookup["UniqueID"].astype(str)
        feature_meta_lookup = feature_meta_lookup.drop_duplicates(subset=["UniqueID"], keep="first").reset_index(drop=True)

    vals = X.to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals) & (vals > 0)]
    pseudocount = float(np.nanquantile(finite_vals, pseudocount_quantile)) if finite_vals.size else 1.0
    pseudocount = max(pseudocount, 1e-9)
    logX = np.log2(X + pseudocount)

    betas = []
    sigma2 = []
    df_resid = []
    stdev_unscaled = []
    for uid in logX.columns:
        beta, s2, dfr, su = _fit_feature_ols(logX[uid].to_numpy(dtype=float), Xmat)
        betas.append(beta)
        sigma2.append(s2)
        df_resid.append(dfr)
        stdev_unscaled.append(su)

    betas = np.asarray(betas, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    df_resid = np.asarray(df_resid, dtype=float)
    stdev_unscaled = np.asarray(stdev_unscaled, dtype=float)

    s2_post, s0, d0 = _moderate_variances(sigma2, df_resid)
    baseline_group = ordered_groups[0]
    outputs: Dict[str, str] = {"advanced_dir": str(out_dir)}

    for group_a, group_b in combinations(ordered_groups, 2):
        cvec = _contrast_vector(design_cols, group_a, group_b, baseline_group)
        if not np.any(cvec):
            continue

        coef = betas @ cvec
        var_unscaled = np.einsum("i,fij,j->f", cvec, stdev_unscaled, cvec)
        se = np.sqrt(s2_post * var_unscaled)
        t_stat = coef / se
        df_total = df_resid + d0
        p_values = 2.0 * (1.0 - t_dist.cdf(np.abs(t_stat), df=np.where(np.isfinite(df_total), df_total, np.nan)))
        valid = np.isfinite(p_values)
        fdr = np.full_like(p_values, np.nan, dtype=float)
        if valid.any():
            fdr[valid] = multipletests(p_values[valid], method="fdr_bh")[1]

        result = pd.DataFrame({
            "UniqueID": logX.columns.astype(str),
            "Comparison": f"{group_a}_vs_{group_b}",
            "log2FC": coef,
            "abs_log2FC": np.abs(coef),
            "Moderated_t": t_stat,
            "Moderated_p_value": p_values,
            "FDR_BH": fdr,
            "Residual_variance": sigma2,
            "Moderated_variance": s2_post,
            "Residual_df": df_resid,
            "Posterior_df": df_total,
            "BatchColumn": batch_col or "",
            "Pseudocount": pseudocount,
        })
        if not feature_meta_lookup.empty and "UniqueID" in feature_meta_lookup.columns:
            cols = [c for c in ["UniqueID", "Annotation", "Headgroup", "Lipid Class"] if c in feature_meta_lookup.columns]
            result = result.merge(feature_meta_lookup[cols], on="UniqueID", how="left")
        result["Label"] = result["Annotation"].fillna(result["UniqueID"]) if "Annotation" in result.columns else result["UniqueID"]

        out_csv = out_dir / f"{group_a}_vs_{group_b}_moderated_results.csv"
        result.sort_values(["FDR_BH", "abs_log2FC"], ascending=[True, False], na_position="last").to_csv(out_csv, index=False)
        outputs[f"{group_a}_vs_{group_b}_csv"] = str(out_csv)

        top_plot = result[result["FDR_BH"].notna()].sort_values(["FDR_BH", "abs_log2FC"], ascending=[True, False], na_position="last")
        plot_path = out_dir / f"{group_a}_vs_{group_b}_top_hits.png"
        _plot_top_hits(top_plot, str(plot_path), f"Top moderated hits: {group_a} vs {group_b}", style=style)
        outputs[f"{group_a}_vs_{group_b}_plot"] = str(plot_path)

    dashboard_csv = _build_summary_dashboard(out_dir)
    if dashboard_csv:
        outputs["summary_dashboard_csv"] = dashboard_csv

    print(f"[Advanced Differential] Completed. Results saved to: {out_dir}", flush=True)
    return outputs
