from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal

from Stats.utils import prepare_output_dir

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
plt.ioff()


def _order_groups(present: List[str], group_order: Optional[List[str]]) -> List[str]:
    present = [str(g) for g in present]
    if not group_order:
        return present
    gui = [g for g in group_order if g in present]
    rest = [g for g in present if g not in group_order]
    return gui + rest


def _load_nonimputed_dataset(file_path: str, group_file: Optional[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if group_file is None:
        raise ValueError("group_file is required for missingness-aware analysis.")

    df = pd.read_csv(file_path, low_memory=False)
    gdf = pd.read_csv(group_file, low_memory=False)

    if "Sample" not in gdf.columns or "Group" not in gdf.columns:
        raise ValueError("Group file must contain 'Sample' and 'Group' columns.")
    if "UniqueID" not in df.columns:
        raise ValueError("Stats file is missing 'UniqueID'.")

    meta_keep = [
        "UniqueID", "RT (min)", "m/z", "Polarity", "Annotation", "Annotation Type",
        "Annotation Source", "Headgroup", "Lipid Class", "MS/MS score", "Annotation tier",
        "Number of carbons in fatty acyls", "Double bond equivalents", "Plasmenyl?",
    ]
    meta_cols = [c for c in meta_keep if c in df.columns]
    sample_cols = [c for c in df.columns if c not in meta_cols]

    gdf = gdf.copy()
    gdf["Sample"] = gdf["Sample"].astype(str).str.strip()
    gdf["Group"] = gdf["Group"].astype(str).str.strip()

    sample_cols = [c for c in sample_cols if str(c).strip() in set(gdf["Sample"])]
    if not sample_cols:
        raise ValueError("No overlapping sample columns between stats file and group file.")

    feature_meta = df[meta_cols].copy()
    feature_meta["UniqueID"] = feature_meta["UniqueID"].astype(str).str.strip()

    value_df = df[["UniqueID"] + sample_cols].copy()
    value_df["UniqueID"] = value_df["UniqueID"].astype(str).str.strip()
    value_df = value_df.drop_duplicates(subset=["UniqueID"], keep="first")

    X = value_df.set_index("UniqueID")[sample_cols].transpose()
    X.index.name = "Sample"
    X = X.apply(pd.to_numeric, errors="coerce")

    gmap = gdf.drop_duplicates(subset=["Sample"], keep="first").set_index("Sample")["Group"]
    X = X.loc[[s for s in X.index if s in gmap.index]].copy()
    y = gmap.reindex(X.index)

    feature_meta = feature_meta.drop_duplicates(subset=["UniqueID"], keep="first")
    feature_meta = feature_meta.set_index("UniqueID").reindex(X.columns).reset_index()
    return X, y, feature_meta


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


def _plot_heatmap(table: pd.DataFrame, out_png: str, out_svg: str, title: str, cbar_label: str, note_text: str, cmap: str = "coolwarm", center_zero: bool = False) -> None:
    if table.empty:
        return
    fig, ax = plt.subplots(
        figsize=(max(8, 0.8 * len(table.columns) + 3), max(6, 0.35 * len(table.index) + 2.5)),
        facecolor="white",
    )
    vals = table.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        plt.close(fig)
        return

    if center_zero:
        vmax = float(np.nanmax(np.abs(finite)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))

    im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(table.columns.tolist(), rotation=45, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index.tolist(), fontsize=10)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("Group", fontsize=12, labelpad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, labelpad=14)
    fig.subplots_adjust(bottom=0.28)
    fig.text(0.5, 0.005, note_text, ha="center", va="bottom", fontsize=10.5, color="dimgray")
    fig.savefig(out_png, dpi=120, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _plot_prevalence_hist(detection_rates: pd.Series, out_png: str, out_svg: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.hist(detection_rates.dropna().to_numpy(dtype=float), bins=np.linspace(0, 1, 21), color="#4C78A8", edgecolor="white")
    ax.set_title("Feature prevalence distribution", fontsize=15, pad=12)
    ax.set_xlabel("Fraction of samples detected", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of features", fontsize=12, labelpad=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.savefig(out_png, dpi=120, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _plot_prevalence_curve(detection_rates: pd.Series, out_png: str, out_svg: str) -> None:
    thresholds = np.linspace(0, 1, 101)
    counts = [(detection_rates >= t).sum() for t in thresholds]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.plot(thresholds, counts, color="#E45756", linewidth=2)
    ax.set_title("Prevalence filtering curve", fontsize=15, pad=12)
    ax.set_xlabel("Minimum prevalence threshold", fontsize=12, labelpad=10)
    ax.set_ylabel("Features retained", fontsize=12, labelpad=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.savefig(out_png, dpi=120, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def run_from_stats(
    file_path: str,
    group_file: Optional[str],
    save_dir: str,
    group_order: Optional[List[str]] = None,
    group_colors: Optional[dict] = None,
    exclude_qc: bool = True,
    min_detected_per_group: int = 2,
) -> Dict[str, str]:
    out_dir = prepare_output_dir(save_dir)
    print("[Missingness] Running missingness-aware analysis...", flush=True)

    X, y, feature_meta = _load_nonimputed_dataset(file_path, group_file)
    if X.empty:
        raise ValueError("Dataset appears empty or malformed.")

    X.index = X.index.astype(str)
    y.index = y.index.astype(str)
    if exclude_qc:
        keep_mask = ~y.astype(str).str.contains("QC", case=False, na=False)
        X = X.loc[keep_mask]
        y = y.loc[keep_mask]

    ordered_groups = _order_groups(pd.unique(y.astype(str)).tolist(), group_order)
    detect = X.notna()

    group_presence = []
    missing_rows = []
    diff_presence_rows = []
    abundance_rows = []
    hurdle_rows = []

    for uid in X.columns:
        det_feature = detect[uid]
        value_feature = pd.to_numeric(X[uid], errors="coerce")

        detect_rates = {}
        miss_rates = {}
        contingency = []
        valid_group_names = []
        detect_vectors = []
        abundance_vectors = []

        for group in ordered_groups:
            mask = y.astype(str) == str(group)
            det_group = det_feature.loc[mask]
            val_group = value_feature.loc[mask]
            if len(det_group) == 0:
                continue
            detect_rate = float(det_group.mean())
            detect_rates[group] = detect_rate
            miss_rates[group] = 1.0 - detect_rate
            group_presence.append({"UniqueID": uid, "Group": group, "DetectionRate": detect_rate, "MissingRate": 1.0 - detect_rate})
            missing_rows.append({"UniqueID": uid, "Group": group, "MissingRate": 1.0 - detect_rate})

            detected_n = int(det_group.sum())
            missing_n = int((~det_group).sum())
            if detected_n + missing_n > 0:
                contingency.append([detected_n, missing_n])
                valid_group_names.append(group)
                detect_vectors.append(det_group.astype(int).to_numpy(dtype=int))

            detected_vals = val_group.loc[det_group].dropna()
            if len(detected_vals) >= min_detected_per_group:
                abundance_vectors.append((group, detected_vals.to_numpy(dtype=float)))

        if len(contingency) >= 2:
            try:
                chi2, p_presence, _, _ = chi2_contingency(np.asarray(contingency, dtype=float))
            except Exception:
                p_presence = np.nan
        else:
            p_presence = np.nan

        diff_presence_rows.append({"UniqueID": uid, "Presence_p_value": p_presence})

        if len(abundance_vectors) >= 2:
            try:
                _, p_abundance = kruskal(*[vals for _, vals in abundance_vectors])
            except Exception:
                p_abundance = np.nan
        else:
            p_abundance = np.nan
        abundance_rows.append({"UniqueID": uid, "DetectedOnly_p_value": p_abundance})

        hurdle_rows.append({
            "UniqueID": uid,
            "Presence_p_value": p_presence,
            "DetectedOnly_p_value": p_abundance,
        })

    presence_df = pd.DataFrame(group_presence)
    missing_df = pd.DataFrame(missing_rows)
    diff_presence_df = pd.DataFrame(diff_presence_rows)
    abundance_df = pd.DataFrame(abundance_rows)
    hurdle_df = pd.DataFrame(hurdle_rows)

    diff_presence_df["Presence_FDR_BH"] = _bh_fdr(diff_presence_df["Presence_p_value"])
    abundance_df["DetectedOnly_FDR_BH"] = _bh_fdr(abundance_df["DetectedOnly_p_value"])
    hurdle_df["Presence_FDR_BH"] = _bh_fdr(hurdle_df["Presence_p_value"])
    hurdle_df["DetectedOnly_FDR_BH"] = _bh_fdr(hurdle_df["DetectedOnly_p_value"])
    hurdle_df["Hurdle_significant"] = (
        hurdle_df["Presence_FDR_BH"].lt(0.05).fillna(False)
        | hurdle_df["DetectedOnly_FDR_BH"].lt(0.05).fillna(False)
    )

    detection_pivot = presence_df.pivot(index="UniqueID", columns="Group", values="DetectionRate").reindex(columns=ordered_groups)
    missing_pivot = missing_df.pivot(index="UniqueID", columns="Group", values="MissingRate").reindex(columns=ordered_groups)
    mean_missing_by_group = missing_df.groupby("Group")["MissingRate"].mean().reindex(ordered_groups)

    feature_detection_rates = detect.mean(axis=0)

    presence_csv = os.path.join(out_dir, "differential_presence_absence_statistics.csv")
    missing_csv = os.path.join(out_dir, "missingness_rate_by_group.csv")
    hurdle_csv = os.path.join(out_dir, "hurdle_analysis_statistics.csv")
    detection_matrix_csv = os.path.join(out_dir, "feature_detection_rate_by_group.csv")
    missing_matrix_csv = os.path.join(out_dir, "feature_missingness_rate_by_group.csv")

    diff_presence_df.to_csv(presence_csv, index=False)
    pd.DataFrame({"Group": mean_missing_by_group.index, "MeanMissingRate": mean_missing_by_group.values}).to_csv(missing_csv, index=False)
    hurdle_df.to_csv(hurdle_csv, index=False)
    detection_pivot.to_csv(detection_matrix_csv, index=True)
    missing_pivot.to_csv(missing_matrix_csv, index=True)

    _plot_heatmap(
        missing_pivot.head(100),
        os.path.join(out_dir, "missingness_rate_heatmap.png"),
        os.path.join(out_dir, "missingness_rate_heatmap.svg"),
        "Missingness rate by group",
        "Missingness rate",
        note_text="Higher values indicate more missing observations. Showing first 100 features for readability.",
        cmap="magma",
        center_zero=False,
    )
    _plot_heatmap(
        detection_pivot.head(100),
        os.path.join(out_dir, "detection_rate_heatmap.png"),
        os.path.join(out_dir, "detection_rate_heatmap.svg"),
        "Detection rate by group",
        "Detection rate",
        note_text="Higher values indicate higher prevalence across samples. Showing first 100 features for readability.",
        cmap="viridis",
        center_zero=False,
    )
    _plot_prevalence_hist(
        feature_detection_rates,
        os.path.join(out_dir, "prevalence_histogram.png"),
        os.path.join(out_dir, "prevalence_histogram.svg"),
    )
    _plot_prevalence_curve(
        feature_detection_rates,
        os.path.join(out_dir, "prevalence_filtering_curve.png"),
        os.path.join(out_dir, "prevalence_filtering_curve.svg"),
    )

    print(f"[Missingness] Completed. Results saved to: {out_dir}", flush=True)
    return {
        "out_dir": str(out_dir),
        "presence_stats_csv": presence_csv,
        "missingness_csv": missing_csv,
        "hurdle_stats_csv": hurdle_csv,
        "detection_matrix_csv": detection_matrix_csv,
        "missingness_matrix_csv": missing_matrix_csv,
    }

