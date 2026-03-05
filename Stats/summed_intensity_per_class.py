#TODO: the summed intensity plots are misleading. Each lipid class is scaled by a different internal standard that was spiked at a different concentration, 
# so the normalized intensities of different classes are on different arbitrary scales. Classes with a high-concentration IS look artificially small, and 
# classes with a low-concentration IS look artificially large. You cannot compare the absolute amounts of PC vs PE vs TG vs SM. The stacked bar height across classes is not meaningful.
# The plot is visually clean but does not represent true class composition, because each class sits on its own denominator.
# use IS concentration–scaled stacked class composition for exploratory figures, but clearly labeled “semi-quantitative” (“Class abundances adjusted by IS molar spike; inter-class comparisons remain approximate.”)
# within-class comparisons are fine!

# Stats/summed_intensity_per_class.py
from __future__ import annotations

import os, re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from Stats.utils import load_dataset, prepare_output_dir
from Stats.utils import _CLASS_ORDER, _CLASS_ORDER_BACTERIA, _CLASS_ORDER_MAMMALIAN, _CLASS_GROUP_MAP

def _canon_class(x: str, unknown_policy: str = "append") -> str:
    x = str(x or "").strip()
    if x in _CLASS_GROUP_MAP:
        return _CLASS_GROUP_MAP[x]
    if not x:
        print(x, flush=True)
        return "Other"
    return "Other" if unknown_policy == "other" else x

def _order_for_sample_type(sample_type: Optional[str]) -> list[str]:
    st = str(sample_type or "").strip().lower()
    if "bact" in st:
        return _CLASS_ORDER_BACTERIA
    if "mamm" in st or "hela" in st or "hek" in st or "human" in st:
        return _CLASS_ORDER_MAMMALIAN
    return _CLASS_ORDER  # default

def species_counts_from_collapsed(df: pd.DataFrame, unknown_policy: str = "append") -> pd.Series:
    if "Lipid Class" not in df.columns:
        raise ValueError("Collapsed file is missing 'Lipid Class'.")
    d2 = df.copy()
    d2["Lipid Class"] = d2["Lipid Class"].astype(str).map(lambda v: _canon_class(v, unknown_policy))
    s = d2.groupby("Lipid Class").size()
    s.index.name = "Lipid Class"
    return s

# =========================================
# Palette (reuse GUI colors when provided)
# =========================================

def _build_palette(labels: List[str], group_colors: Optional[dict]) -> dict:
    pal = {}
    if isinstance(group_colors, dict):
        for k, v in group_colors.items():
            if isinstance(v, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", v):
                pal[str(k)] = v.upper()

    wheel = []
    for cmap in ("tab20","tab20b","tab20c","Set3","Paired","Accent"):
        wheel += [matplotlib.colors.to_hex(c) for c in plt.get_cmap(cmap).colors]
    seen = set()
    wheel = [c.upper() for c in wheel if not (c.upper() in seen or seen.add(c.upper()))]

    i = 0
    for g in labels:
        if g not in pal:
            pal[g] = wheel[i % len(wheel)]
            i += 1
    return pal

def _build_class_colors(classes: List[str]) -> dict:
    # stable long palette for classes
    colors = []
    for cmap in ("tab20","tab20b","tab20c","Set3","Pastel2","Pastel1","Accent","Paired"):
        colors += [matplotlib.colors.to_hex(c) for c in plt.get_cmap(cmap).colors]
    # deterministic subset
    out = {}
    i = 0
    for sc in classes:
        out[sc] = colors[i % len(colors)]
        i += 1
    return out

def _order_labels(present: List[str], group_order: Optional[List[str]]) -> List[str]:
    present = [str(g) for g in present]
    if not group_order:
        return present
    gui = [g for g in group_order if g in present]
    rest = [g for g in present if g not in gui]
    return gui + rest

# =========================================
# Core aggregation
# =========================================

def totals_per_class_from_X(X: pd.DataFrame, feature_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Return: samples × classes totals
    Uses current pipeline metadata ('Lipid Class') as the class field.
    """
    meta = feature_meta.copy()

    # expected current column name(s)
    CAND = [
        "Lipid Class",   # current
        "lipid class",   # case variant
        "Class",         # occasional alternative
        "Lipid_Class"    # defensive alias
    ]
    cls_col = next((c for c in CAND if c in meta.columns), None)
    if cls_col is None:
        raise ValueError("Class field not found. Expected one of: " + ", ".join(CAND))

    if "UniqueID" not in meta.columns:
        meta["UniqueID"] = X.columns.astype(str)

    # canonicalize to your grouping map
    meta["__Class__"] = (
        meta[cls_col].astype(str)
        .str.strip()
        .map(_canon_class)   # uses _CLASS_GROUP_MAP + unknown_policy
    )

    print(f"[ClassDist] Using class column: '{cls_col}'.", flush=True)

    class_map = (
        meta[["UniqueID", "__Class__"]]
        .set_index("UniqueID")["__Class__"].astype(str)
    ).reindex(X.columns)

    # Debug: which mapped to Other?
    if "Other" in class_map.values:
        raw = meta.set_index("UniqueID")[cls_col].astype(str)
        other_uids = [uid for uid, cl in class_map.items() if cl == "Other"]
        top_raw = raw.loc[raw.index.intersection(other_uids)].value_counts().head(10).to_dict()
        print(f"[ClassDist][DEBUG] {len(other_uids)} features mapped to 'Other'. Top raw labels → counts: {top_raw}", flush=True)

    per_sample = X.groupby(by=class_map, axis=1).sum()
    per_sample.index.name = "Sample"
    return per_sample

# =========================================
# Plotting helpers (legends not clipped)
# =========================================

def _save_with_legend(fig: plt.Figure, path_png: str, path_svg: Optional[str], legends: List[matplotlib.legend.Legend]):
    extra = [leg for leg in legends if leg is not None]
    fig.savefig(path_png, dpi=100, bbox_inches="tight", bbox_extra_artists=extra)
    if path_svg:
        fig.savefig(path_svg, dpi=100, bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)

# =========================================
# Public API
# =========================================

def run_from_stats(
    file_path: str,
    group_file: Optional[str],
    save_dir: str,
    group_order: Optional[List[str]] = None,
    group_colors: Optional[dict] = None,
    sample_type: Optional[str] = None,   # ← add this
) -> Dict[str, str]:

    """
    Build per-class totals from STATS CSV (same loader as other modules),
    write CSVs, and plot:
      1) Normalized composition per sample (stacked %)
      2) Normalized composition per group median (stacked %)
      3) Absolute group median composition (stacked intensity)
    """
    out_dir = prepare_output_dir(save_dir)

    print('[SummIntensity] Running summed intensity plots...', flush = True)

    X, y, feature_meta = load_dataset(file_path, group_file)
    if X.empty or feature_meta.empty:
        raise ValueError("Dataset appears empty or malformed.")

    # Class totals per sample
    per_sample = totals_per_class_from_X(X, feature_meta)       # samples × classes (totals)
    present_classes = [sc for sc in per_sample.columns.astype(str)]
    pref_order = _order_for_sample_type(sample_type)
    ordered = [sc for sc in pref_order if sc in present_classes] + \
            [sc for sc in present_classes if sc not in pref_order]
    per_sample = per_sample[ordered]

    # Colors
    groups_present = [str(g) for g in sorted(set(y.astype(str)))]
    labels = _order_labels(groups_present, group_order)
    pal_groups = _build_palette(labels, group_colors)
    pal_sub = _build_class_colors(ordered)

    # CSV: raw per-sample class totals
    per_sample_csv = os.path.join(out_dir, "per_sample_class_totals.csv")
    per_sample.to_csv(per_sample_csv)

    # -------- 1) Normalized per sample (% per sample) --------
    norm_sample = per_sample.div(per_sample.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    norm_sample = norm_sample.fillna(0)
    norm_sample_csv = os.path.join(out_dir, "per_sample_class_percent.csv")
    norm_sample.to_csv(norm_sample_csv)

    fig, ax = plt.subplots(figsize=(16, 8))
    bottom = np.zeros(len(norm_sample.index))
    x = np.arange(len(norm_sample.index))
    ax.set_xticks(x)
    ax.set_xticklabels(norm_sample.index.astype(str), rotation=90, ha="center", fontsize=11)

    legend_handles = []
    for sc in norm_sample.columns:
        vals = norm_sample[sc].values
        bars = ax.bar(x, vals, bottom=bottom, color=pal_sub[sc], label=sc, width=0.8)
        bottom += vals
        legend_handles.append(Patch(facecolor=pal_sub[sc], edgecolor="none", label=sc))

    ax.set_title("Relative lipid class composition per sample (% total intensity)", fontsize=18, pad=12)
    ax.set_ylabel("Percentage of total intensity (%)", fontsize=14)
    ax.set_xlabel("Samples", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg_sub = ax.legend(
        handles=legend_handles,
        title="Class",
        bbox_to_anchor=(1.01, 1), loc="upper left",
        frameon=False, fontsize=11, title_fontsize=12, ncols=1
    )
    fig.subplots_adjust(right=0.82)

    out1_png = os.path.join(out_dir, "samples_relative_class_composition.png")
    out1_svg = os.path.join(out_dir, "samples_relative_class_composition.svg")
    plt.tight_layout()
    _save_with_legend(fig, out1_png, out1_svg, [leg_sub])

    # -------- 2) Group medians, normalized (% within group) --------
    df_with_group = per_sample.copy()
    df_with_group["Group"] = y.values
    med = df_with_group.groupby("Group").median(numeric_only=True)  # groups × classes
    med = med.reindex(index=_order_labels(list(med.index.astype(str)), group_order))  # reorder groups if requested
    med_norm = med.div(med.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    med_norm = med_norm.fillna(0)

    med_norm_csv = os.path.join(out_dir, "per_group_median_class_percent.csv")
    med_norm.to_csv(med_norm_csv)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(med_norm.index))
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in med_norm.index], rotation=45, ha="right", fontsize=13)

    width = 0.8
    bottom = np.zeros(len(med_norm.index))
    legend_handles = []
    for sc in med_norm.columns:
        vals = med_norm[sc].values
        ax.bar(x, vals, width=width, bottom=bottom, color=pal_sub[sc], label=sc)
        bottom += vals
        legend_handles.append(Patch(facecolor=pal_sub[sc], edgecolor="none", label=sc))

    ax.set_ylabel("Percentage of total intensity (%)", fontsize=14)
    ax.set_title("Relative lipid class composition by group (median %)", fontsize=18, pad=12)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg2 = ax.legend(
        handles=legend_handles,
        title="Class",
        bbox_to_anchor=(1.01, 1), loc="upper left",
        frameon=False, fontsize=11, title_fontsize=12
    )
    fig.subplots_adjust(right=0.82)

    out2_png = os.path.join(out_dir, "groups_relative_class_composition.png")
    out2_svg = os.path.join(out_dir, "groups_relative_class_composition.svg")
    plt.tight_layout()
    _save_with_legend(fig, out2_png, out2_svg, [leg2])

    # -------- 3) Group medians, absolute intensities --------
    med_abs_csv = os.path.join(out_dir, "per_group_median_class_intensity.csv")
    med.to_csv(med_abs_csv)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(med.index))
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in med.index], rotation=45, ha="right", fontsize=13)

    width = 0.8
    bottom = np.zeros(len(med.index))
    legend_handles = []
    for sc in med.columns:
        vals = med[sc].values
        ax.bar(x, vals, width=width, bottom=bottom, color=pal_sub[sc], label=sc)
        bottom += vals
        legend_handles.append(Patch(facecolor=pal_sub[sc], edgecolor="none", label=sc))

    ax.set_ylabel("Total intensity (median of samples)", fontsize=14)
    ax.set_title("Lipid class composition by group (median intensity)", fontsize=18, pad=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg3 = ax.legend(
        handles=legend_handles,
        title="Class",
        bbox_to_anchor=(1.01, 1), loc="upper left",
        frameon=False, fontsize=11, title_fontsize=12
    )
    fig.subplots_adjust(right=0.82)

    out3_png = os.path.join(out_dir, "groups_absolute_class_composition.png")
    out3_svg = os.path.join(out_dir, "groups_absolute_class_composition.svg")
    plt.tight_layout()
    _save_with_legend(fig, out3_png, out3_svg, [leg3])

    return {
        "per_sample_class_totals_csv": per_sample_csv,
        "per_sample_class_percent_csv": norm_sample_csv,
        "per_group_median_percent_csv": med_norm_csv,
        "per_group_median_intensity_csv": med_abs_csv,
        "samples_relative_png": out1_png,
        "groups_relative_png": out2_png,
        "groups_absolute_png": out3_png,
        "samples_relative_svg": out1_svg,
        "groups_relative_svg": out2_svg,
        "groups_absolute_svg": out3_svg,
    }
