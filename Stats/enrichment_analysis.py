from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import kruskal

from Stats.figure_style import get_figure_style
from Stats.utils import prepare_output_dir, _CLASS_GROUP_MAP

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
plt.ioff()


def _load_dataset_preserve_nan(file_path: str, group_file: Optional[str]):
    df = pd.read_csv(file_path, low_memory=False)
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=str), pd.DataFrame()

    meta_cols = [
        "UniqueID", "RT (min)", "m/z", "Polarity", "Annotation",
        "Annotation Type", "Headgroup", "Lipid Class", "Δm/z (mDa)", "Δm/z (ppm)",
        "MS/MS score", "Annotation tier", "mSigma", "Molecular Formula",
        "Plasmenyl?", "Number of carbons in fatty acyls", "Double bond equivalents",
        "Chain type", "PUFA?", "Modifications", "# of modifications", "Oxidized?",
        "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    sample_cols = [c for c in df.columns if c not in meta_cols]

    if group_file is not None and os.path.exists(group_file):
        df_groups = pd.read_csv(group_file, low_memory=False)
        if "Sample" not in df_groups.columns or "Group" not in df_groups.columns:
            raise ValueError(f"[enrichment] Invalid group file format: {group_file}")
    else:
        df_groups = pd.DataFrame({"Sample": sample_cols, "Group": "Unknown"})

    df_groups["Sample"] = df_groups["Sample"].astype(str).str.strip()
    df_groups["Group"] = df_groups["Group"].astype(str).str.strip()
    df_cols_lower = {str(c).lower(): c for c in sample_cols}
    matched = [df_cols_lower[s.lower()] for s in df_groups["Sample"] if s.lower() in df_cols_lower]

    if not matched:
        return pd.DataFrame(), pd.Series(dtype=str), pd.DataFrame()

    X = df[matched].T
    X.index.name = "Sample"
    y = df_groups.set_index("Sample").loc[X.index, "Group"]
    feature_meta = df[meta_cols].copy()

    if "UniqueID" in feature_meta.columns:
        X.columns = feature_meta["UniqueID"].astype(str).tolist()
    else:
        X.columns = [f"Feature_{i+1}" for i in range(X.shape[1])]

    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y, feature_meta


def _canon_class(x: str, unknown_policy: str = "append") -> str:
    x = str(x or "").strip()
    if x in _CLASS_GROUP_MAP:
        return _CLASS_GROUP_MAP[x]
    if not x:
        return "Other"
    return "Other" if unknown_policy == "other" else x


def _order_groups(present: List[str], group_order: Optional[List[str]]) -> List[str]:
    present = [str(g) for g in present]
    if not group_order:
        return present
    gui = [g for g in group_order if g in present]
    rest = [g for g in present if g not in group_order]
    return gui + rest


def _pick_column_ci(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = [str(c) for c in df.columns]
    norm = {str(c).strip().lower(): str(c) for c in cols}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm:
            return norm[key]
    return None


def _parse_total_carbons_from_text(s: str) -> float:
    if s is None:
        return np.nan
    m = re.search(r"(\d+)\s*:\s*(\d+)", str(s))
    return float(int(m.group(1))) if m else np.nan


def _parse_total_double_bonds_from_text(s: str) -> float:
    if s is None:
        return np.nan
    m = re.search(r"(\d+)\s*:\s*(\d+)", str(s))
    return float(int(m.group(2))) if m else np.nan


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    wts = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = vals.notna() & np.isfinite(vals.to_numpy(dtype=float)) & np.isfinite(wts.to_numpy(dtype=float)) & (wts > 0)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.average(vals.loc[mask].to_numpy(dtype=float), weights=wts.loc[mask].to_numpy(dtype=float)))


def _compute_group_fraction_table(
    X: pd.DataFrame,
    y: pd.Series,
    feature_labels: pd.Series,
) -> pd.DataFrame:
    labels = feature_labels.reindex(X.columns)
    valid = labels.notna() & (labels.astype(str).str.strip() != "")
    if int(valid.sum()) == 0:
        return pd.DataFrame()

    Xv = X.loc[:, valid].apply(pd.to_numeric, errors="coerce")
    labels = labels.loc[valid].astype(str)
    per_sample = Xv.T.groupby(labels).sum(min_count=1).T
    sample_totals = per_sample.sum(axis=1, min_count=1).replace(0, np.nan)
    fractions = per_sample.div(sample_totals, axis=0)
    fractions["Group"] = y.reindex(fractions.index).astype(str).values
    return fractions.groupby("Group").mean(numeric_only=True)


def _compute_sample_fraction_table(
    X: pd.DataFrame,
    y: pd.Series,
    feature_labels: pd.Series,
) -> pd.DataFrame:
    labels = feature_labels.reindex(X.columns)
    valid = labels.notna() & (labels.astype(str).str.strip() != "")
    if int(valid.sum()) == 0:
        return pd.DataFrame()

    Xv = X.loc[:, valid].apply(pd.to_numeric, errors="coerce")
    labels = labels.loc[valid].astype(str)
    per_sample = Xv.T.groupby(labels).sum(min_count=1).T
    sample_totals = per_sample.sum(axis=1, min_count=1).replace(0, np.nan)
    fractions = per_sample.div(sample_totals, axis=0)
    fractions["Group"] = y.reindex(fractions.index).astype(str).values
    return fractions


def _compute_log2_enrichment(group_fraction_table: pd.DataFrame) -> pd.DataFrame:
    if group_fraction_table.empty:
        return pd.DataFrame()
    baseline = group_fraction_table.mean(axis=0).replace(0, np.nan)
    return np.log2(group_fraction_table.div(baseline, axis=1)).replace([np.inf, -np.inf], np.nan)


def _sort_bin_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        ordered = sorted(df.columns.tolist(), key=lambda x: float(str(x)))
        return df.reindex(columns=ordered)
    except Exception:
        return df


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return out

    order = np.argsort(valid.to_numpy(dtype=float))
    ranked = valid.iloc[order]
    n = len(ranked)
    adj = ranked.to_numpy(dtype=float) * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out.loc[ranked.index] = adj
    return out


def _compute_enrichment_stats(
    sample_fraction_table: pd.DataFrame,
    ordered_groups: List[str],
    value_name: str,
    group_col: str = "Group",
) -> pd.DataFrame:
    if sample_fraction_table.empty:
        return pd.DataFrame()

    rows = []
    numeric_cols = [c for c in sample_fraction_table.columns if c != group_col]
    for value in numeric_cols:
        group_vectors = []
        for group in ordered_groups:
            vals = pd.to_numeric(
                sample_fraction_table.loc[sample_fraction_table[group_col].astype(str) == str(group), value],
                errors="coerce",
            ).dropna()
            if len(vals) > 0:
                group_vectors.append(vals.to_numpy(dtype=float))

        if len(group_vectors) >= 2:
            try:
                _, p_value = kruskal(*group_vectors)
            except Exception:
                p_value = np.nan
        else:
            p_value = np.nan

        rows.append({
            value_name: value,
            "Kruskal_p_value": p_value,
            "Groups_tested": int(len(group_vectors)),
        })

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df["FDR_BH"] = _bh_fdr(stats_df["Kruskal_p_value"])
    return stats_df


def _compute_groupwise_value_stats(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    group_col: str,
    ordered_groups: List[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for label in sorted(df[label_col].dropna().astype(str).unique().tolist()):
        sub = df[df[label_col].astype(str) == label]
        group_vectors = []
        for group in ordered_groups:
            vals = pd.to_numeric(
                sub.loc[sub[group_col].astype(str) == str(group), value_col],
                errors="coerce",
            ).dropna()
            if len(vals) > 0:
                group_vectors.append(vals.to_numpy(dtype=float))

        if len(group_vectors) >= 2:
            try:
                _, p_value = kruskal(*group_vectors)
            except Exception:
                p_value = np.nan
        else:
            p_value = np.nan

        rows.append({
            label_col: label,
            "Kruskal_p_value": p_value,
            "Groups_tested": int(len(group_vectors)),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR_BH"] = _bh_fdr(out["Kruskal_p_value"])
    return out


def _compute_saturation_sample_table(
    X: pd.DataFrame,
    y: pd.Series,
    class_series: pd.Series,
    sat_index: pd.Series,
) -> pd.DataFrame:
    rows = []
    ordered_classes = sorted([c for c in class_series.dropna().astype(str).unique().tolist() if c and c.lower() != "nan"])
    numeric_X = X.apply(pd.to_numeric, errors="coerce")

    for sample in numeric_X.index:
        sample_values = numeric_X.loc[sample]
        for cls in ordered_classes:
            feats = class_series[class_series.astype(str) == cls].index.intersection(numeric_X.columns)
            if len(feats) == 0:
                continue
            sat_mean = _weighted_mean(sat_index.reindex(feats), sample_values.reindex(feats))
            rows.append({
                "Sample": sample,
                "Group": str(y.reindex([sample]).iloc[0]),
                "Class": cls,
                "Weighted mean DB_per_carbon": sat_mean,
            })
    return pd.DataFrame(rows)


def _plot_heatmap(
    table: pd.DataFrame,
    out_png: str,
    out_svg: str,
    title: str,
    cbar_label: str,
    note_text: Optional[str] = None,
    cmap: str = "coolwarm",
    center_zero: bool = True,
    significance_df: Optional[pd.DataFrame] = None,
    significance_label_col: Optional[str] = None,
    significance_fdr_col: str = "FDR_BH",
    significance_threshold: float = 0.05,
    style: Optional[dict] = None,
) -> None:
    if table.empty:
        return
    style = style or get_figure_style(False, 100)

    data = table.astype(float)
    fig, ax = plt.subplots(
        figsize=(max(8, 0.6 * len(data.columns) + 3), max(6, 0.45 * len(data.index) + 2)),
        facecolor="white",
    )

    vals = data.to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size == 0:
        plt.close(fig)
        return

    if center_zero:
        vmax = float(np.nanmax(np.abs(finite_vals)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))

    im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns.tolist(), rotation=45, ha="right", fontsize=style["tick_size"])
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index.tolist(), fontsize=style["tick_size"])
    ax.set_title(title, fontsize=style["title_size"], pad=14, fontweight="semibold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, labelpad=14, fontsize=style["label_size"])
    cbar.ax.tick_params(labelsize=style["tick_size"])

    if significance_df is not None and significance_label_col and significance_label_col in significance_df.columns:
        sig = significance_df.copy()
        sig[significance_label_col] = sig[significance_label_col].astype(str)
        sig_map = sig.set_index(significance_label_col)[significance_fdr_col].to_dict() if significance_fdr_col in sig.columns else {}
        for row_idx, label in enumerate(data.index.astype(str).tolist()):
            fdr_val = sig_map.get(str(label))
            if pd.notna(fdr_val) and float(fdr_val) < significance_threshold:
                ax.text(
                    len(data.columns) - 0.35,
                    row_idx,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=style["label_size"],
                    color="black",
                    fontweight="bold",
                )

    if note_text:
        fig.subplots_adjust(bottom=0.30)
        fig.text(0.5, 0.005, note_text, ha="center", va="bottom", fontsize=max(style["tick_size"] - 1, 9), color="dimgray")

    fig.savefig(out_png, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def run_from_stats(
    file_path: str,
    group_file: Optional[str],
    save_dir: str,
    group_order: Optional[List[str]] = None,
    group_colors: Optional[dict] = None,
    unknown_policy: str = "append",
    exclude_qc: bool = True,
    dpi: int = 100,
    publication_theme: bool = False,
) -> Dict[str, str]:
    out_dir = prepare_output_dir(save_dir)
    style = get_figure_style(publication_theme=publication_theme, dpi=dpi)
    print("[Enrichment] Running enrichment analysis...", flush=True)

    X, y, feature_meta = _load_dataset_preserve_nan(file_path, group_file)
    if X is None or y is None or feature_meta is None or X.empty or feature_meta.empty:
        raise ValueError("Dataset appears empty or malformed.")

    feature_meta = feature_meta.copy()
    feature_meta.columns = feature_meta.columns.astype(str).str.strip()

    uid_col = _pick_column_ci(feature_meta, ["UniqueID"])
    if uid_col is not None:
        feature_meta["_FeatureID"] = feature_meta[uid_col].astype(str)
        feature_meta = feature_meta.set_index("_FeatureID", drop=True)
    else:
        feature_meta.index = feature_meta.index.astype(str)

    X = X.copy()
    X.columns = X.columns.astype(str)
    X.index = X.index.astype(str)
    y = y.copy()
    y.index = y.index.astype(str)

    common_feats = [c for c in X.columns if c in feature_meta.index]
    X = X.loc[:, common_feats]
    feature_meta = feature_meta.loc[common_feats]

    if exclude_qc:
        keep_mask = ~y.astype(str).str.contains("QC", case=False, na=False)
        X = X.loc[keep_mask]
        y = y.loc[keep_mask]

    ordered_groups = _order_groups(pd.unique(y.astype(str)).tolist(), group_order)
    class_col = _pick_column_ci(feature_meta, ["Lipid Class", "Headgroup"])
    if class_col is None:
        raise ValueError("Could not find a class column (expected 'Lipid Class' or 'Headgroup').")

    class_series = feature_meta[class_col].astype(str).map(lambda v: _canon_class(v, unknown_policy=unknown_policy))

    carb_col = _pick_column_ci(feature_meta, ["Number of carbons in fatty acyls", "Total carbons", "Carbons"])
    carb = pd.to_numeric(feature_meta[carb_col], errors="coerce") if carb_col else pd.Series(np.nan, index=feature_meta.index)
    if float(carb.notna().mean()) < 0.05:
        text_col = _pick_column_ci(feature_meta, ["Annotation", "NoAbbrev", "Name", "Lipid"])
        if text_col:
            carb = carb.where(carb.notna(), feature_meta[text_col].apply(_parse_total_carbons_from_text))

    db_col = _pick_column_ci(feature_meta, ["Double bond equivalents", "Double bonds", "DB", "DBE"])
    db = pd.to_numeric(feature_meta[db_col], errors="coerce") if db_col else pd.Series(np.nan, index=feature_meta.index)
    if float(db.notna().mean()) < 0.05:
        text_col = _pick_column_ci(feature_meta, ["Annotation", "NoAbbrev", "Name", "Lipid"])
        if text_col:
            db = db.where(db.notna(), feature_meta[text_col].apply(_parse_total_double_bonds_from_text))

    sat_index = db / carb.replace(0, np.nan)

    class_dir = prepare_output_dir(os.path.join(out_dir, "LipidClass_Enrichment"))
    chain_dir = prepare_output_dir(os.path.join(out_dir, "ChainLength_Enrichment"))
    unsat_dir = prepare_output_dir(os.path.join(out_dir, "Unsaturation_Enrichment"))
    sat_dir = prepare_output_dir(os.path.join(out_dir, "SaturationIndex_Changes"))

    class_fraction = _compute_group_fraction_table(X, y, class_series).reindex(ordered_groups)
    class_sample_fraction = _compute_sample_fraction_table(X, y, class_series)
    class_stats = _compute_enrichment_stats(class_sample_fraction, ordered_groups, "Lipid Class")
    class_log2 = _compute_log2_enrichment(class_fraction)
    class_fraction.to_csv(os.path.join(class_dir, "lipid_class_group_mean_fraction.csv"), index=True)
    class_log2.to_csv(os.path.join(class_dir, "lipid_class_group_log2_enrichment.csv"), index=True)
    class_stats.to_csv(os.path.join(class_dir, "lipid_class_enrichment_statistics.csv"), index=False)
    _plot_heatmap(
        class_log2,
        os.path.join(class_dir, "lipid_class_log2_enrichment_heatmap.png"),
        os.path.join(class_dir, "lipid_class_log2_enrichment_heatmap.svg"),
        "Lipid class enrichment by group",
        "log2(group fraction / across-group mean fraction)",
        note_text=(
            "Positive values (red) -> class is enriched in that group relative to the across-group baseline\n"
            "Negative values (blue) -> class is depleted in that group relative to the across-group baseline\n"
            "Near zero -> class composition is close to the across-group average\n"
            "* indicates FDR < 0.05 across groups for that row"
        ),
        significance_df=class_stats,
        significance_label_col="Lipid Class",
        style=style,
    )

    carb_labels = carb.round().astype("Int64").astype(str).replace("<NA>", np.nan)
    carb_fraction = _compute_group_fraction_table(X, y, carb_labels).reindex(ordered_groups)
    carb_sample_fraction = _compute_sample_fraction_table(X, y, carb_labels)
    carb_stats = _compute_enrichment_stats(carb_sample_fraction, ordered_groups, "Total carbons")
    carb_fraction = _sort_bin_columns(carb_fraction)
    carb_log2 = _compute_log2_enrichment(carb_fraction)
    carb_fraction.to_csv(os.path.join(chain_dir, "chain_length_group_mean_fraction.csv"), index=True)
    carb_log2.to_csv(os.path.join(chain_dir, "chain_length_group_log2_enrichment.csv"), index=True)
    carb_stats.to_csv(os.path.join(chain_dir, "chain_length_enrichment_statistics.csv"), index=False)
    _plot_heatmap(
        carb_log2,
        os.path.join(chain_dir, "chain_length_log2_enrichment_heatmap.png"),
        os.path.join(chain_dir, "chain_length_log2_enrichment_heatmap.svg"),
        "Chain-length enrichment by group",
        "log2(group fraction / across-group mean fraction)",
        note_text=(
            "Positive values (red) -> that carbon bin is enriched in the group\n"
            "Negative values (blue) -> that carbon bin is depleted in the group\n"
            "Near zero -> abundance share for that carbon bin is close to the across-group average\n"
            "* indicates FDR < 0.05 across groups for that row"
        ),
        significance_df=carb_stats.assign(**{"Total carbons": carb_stats["Total carbons"].astype(str)}) if not carb_stats.empty else carb_stats,
        significance_label_col="Total carbons",
        style=style,
    )

    db_labels = db.round().astype("Int64").astype(str).replace("<NA>", np.nan)
    db_fraction = _compute_group_fraction_table(X, y, db_labels).reindex(ordered_groups)
    db_sample_fraction = _compute_sample_fraction_table(X, y, db_labels)
    db_stats = _compute_enrichment_stats(db_sample_fraction, ordered_groups, "Double bonds")
    db_fraction = _sort_bin_columns(db_fraction)
    db_log2 = _compute_log2_enrichment(db_fraction)
    db_fraction.to_csv(os.path.join(unsat_dir, "unsaturation_group_mean_fraction.csv"), index=True)
    db_log2.to_csv(os.path.join(unsat_dir, "unsaturation_group_log2_enrichment.csv"), index=True)
    db_stats.to_csv(os.path.join(unsat_dir, "unsaturation_enrichment_statistics.csv"), index=False)
    _plot_heatmap(
        db_log2,
        os.path.join(unsat_dir, "unsaturation_log2_enrichment_heatmap.png"),
        os.path.join(unsat_dir, "unsaturation_log2_enrichment_heatmap.svg"),
        "Unsaturation enrichment by group",
        "log2(group fraction / across-group mean fraction)",
        note_text=(
            "Positive values (red) -> that double-bond bin is enriched in the group\n"
            "Negative values (blue) -> that double-bond bin is depleted in the group\n"
            "Near zero -> abundance share for that unsaturation bin is close to the across-group average\n"
            "* indicates FDR < 0.05 across groups for that row"
        ),
        significance_df=db_stats.assign(**{"Double bonds": db_stats["Double bonds"].astype(str)}) if not db_stats.empty else db_stats,
        significance_label_col="Double bonds",
        style=style,
    )

    sat_rows = []
    classes = sorted([c for c in class_series.dropna().astype(str).unique().tolist() if c and c.lower() != "nan"])
    for group in ordered_groups:
        sample_mask = y.astype(str).eq(str(group))
        if int(sample_mask.sum()) == 0:
            continue
        mean_abundance = X.loc[sample_mask].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        for cls in classes:
            feats = class_series[class_series.astype(str).eq(cls)].index.intersection(X.columns)
            if len(feats) == 0:
                continue
            sat_mean = _weighted_mean(sat_index.reindex(feats), mean_abundance.reindex(feats))
            sat_rows.append({"Class": cls, "Group": group, "Weighted mean DB_per_carbon": sat_mean})

    sat_df = pd.DataFrame(sat_rows)
    if sat_df.empty:
        raise ValueError("Could not compute saturation index changes.")
    sat_df["Class mean DB_per_carbon"] = sat_df.groupby("Class")["Weighted mean DB_per_carbon"].transform("mean")
    sat_df["Saturation index change"] = sat_df["Weighted mean DB_per_carbon"] - sat_df["Class mean DB_per_carbon"]
    sat_df.to_csv(os.path.join(sat_dir, "saturation_index_changes_long.csv"), index=False)

    sat_sample_df = _compute_saturation_sample_table(X, y, class_series, sat_index)
    sat_stats = _compute_groupwise_value_stats(
        df=sat_sample_df,
        label_col="Class",
        value_col="Weighted mean DB_per_carbon",
        group_col="Group",
        ordered_groups=ordered_groups,
    )
    if not sat_stats.empty:
        sat_stats = sat_stats.rename(columns={"Class": "Lipid Class"})
    sat_stats.to_csv(os.path.join(sat_dir, "saturation_index_change_statistics.csv"), index=False)

    sat_pivot = sat_df.pivot(index="Class", columns="Group", values="Weighted mean DB_per_carbon").reindex(columns=ordered_groups)
    sat_change = sat_df.pivot(index="Class", columns="Group", values="Saturation index change").reindex(columns=ordered_groups)
    sat_pivot.to_csv(os.path.join(sat_dir, "saturation_index_by_class_group.csv"), index=True)
    sat_change.to_csv(os.path.join(sat_dir, "saturation_index_change_by_class_group.csv"), index=True)
    _plot_heatmap(
        sat_change,
        os.path.join(sat_dir, "saturation_index_change_heatmap.png"),
        os.path.join(sat_dir, "saturation_index_change_heatmap.svg"),
        "Saturation index changes by class",
        "Delta weighted mean DB/carbon",
        note_text=(
            "Positive values (red) -> higher DB/carbon than the class average across groups\n"
            "Negative values (blue) -> lower DB/carbon than the class average across groups\n"
            "Near zero -> saturation index is close to the class-wide average\n"
            "* indicates FDR < 0.05 across groups for that row"
        ),
        significance_df=sat_stats,
        significance_label_col="Lipid Class",
        style=style,
    )

    print(f"[Enrichment] Completed. Results saved to: {out_dir}", flush=True)
    return {
        "out_dir": str(out_dir),
        "lipid_class_dir": class_dir,
        "chain_length_dir": chain_dir,
        "unsaturation_dir": unsat_dir,
        "saturation_index_dir": sat_dir,
    }
