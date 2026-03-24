#TODO: fix font sizes
#TODO: add legend with bubble size
# ------------------------------------------------------------
# Class distributions (TOTAL INTENSITY per lipid class)
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.patches import Patch
import matplotlib.lines as mlines

# >>> use the same utils as other Stats modules
from Stats.utils import load_dataset, prepare_output_dir
from Stats.utils import _CLASS_ORDER, _CLASS_ORDER_BACTERIA, _CLASS_ORDER_MAMMALIAN, _CLASS_ORDER_FUNGI, _CLASS_GROUP_MAP

# ============================================================
# Shared helpers
# ============================================================

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

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _select_confidence_feature_mask(feature_meta: pd.DataFrame, high_conf_only: bool) -> Optional[pd.Series]:
    """Return a boolean mask over features for 'High confidence' if column exists; else None."""
    if not high_conf_only:
        return None
    col = None
    for c in feature_meta.columns:
        if str(c).strip().lower() == "annotation tier":
            col = c
            break
        if str(c).strip().lower() == "identification level":
            col = c
            break
    if col is None:
        return None
    return feature_meta[col].astype(str).str.contains("High", case=False, na=False)

def _filter_high_conf_collapsed(df: pd.DataFrame, high_conf_only: bool) -> pd.DataFrame:
    if not high_conf_only:
        return df
    col = None
    for c in df.columns:
        if str(c).strip().lower() == "annotation tier":
            col = c
            break
    if col is None:
        return df
    return df[df[col].astype(str).str.contains("High", case=False, na=False)]

def species_counts_from_collapsed(df: pd.DataFrame, unknown_policy: str = "append") -> pd.Series:
    if "Lipid Class" not in df.columns:
        raise ValueError("Collapsed file is missing 'Lipid Class'.")
    d2 = df.copy()
    d2["Lipid Class"] = d2["Lipid Class"].astype(str).map(lambda v: _canon_class(v, unknown_policy))
    s = d2.groupby("Lipid Class").size()
    s.index.name = "Lipid Class"
    return s


# ============================================================
# Mode 2: COLLAPSED loader (2-Final_annotated_results_adducts_collapsed.csv)
# ============================================================

_WORD = r"[A-Za-z0-9_]"
BOUNDARY = rf"(^|(?<!{_WORD})){{}}($|(?!{_WORD}))"

def _map_columns_to_samples(collapsed: pd.DataFrame, sample_names: Iterable[str]) -> Dict[str, str]:
    """Map each sample name to a column in the collapsed file using exact or boundary match."""
    sample_names = [str(s) for s in sample_names]
    colnames = list(map(str, collapsed.columns))

    exact = {s: s for s in sample_names if s in colnames}
    mapped: Dict[str, str] = {}
    for s in sample_names:
        if s in exact:
            mapped[s] = s
            continue
        pat = re.compile(BOUNDARY.format(re.escape(s)))
        found = next((c for c in colnames if pat.search(c)), None)
        if found is not None:
            mapped[s] = found
    missing = [s for s in sample_names if s not in mapped]
    if missing:
        warnings.warn(f"Missing columns for {len(missing)} sample(s) in collapsed file; examples: {missing[:5]}")
    return mapped


def _load_collapsed_style(collapsed_csv: str, sample_groups_csv: str, exclude_qc: bool = True
                          ) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Load the collapsed debug export and align columns to Sample names from sample_groups_cleaned.csv.
    Returns: (collapsed_df_filtered, sample_names_present, sample->group)
    """
    collapsed = pd.read_csv(collapsed_csv, encoding="latin1", low_memory=False)
    if "Lipid Class" not in collapsed.columns:
        raise ValueError("Expected 'Lipid Class' in collapsed file.")

    sg = pd.read_csv(sample_groups_csv)
    for col in ("Sample", "Group"):
        if col not in sg.columns:
            raise ValueError("sample_groups_cleaned.csv must contain columns: Sample, Group")
    sg["Sample"] = sg["Sample"].astype(str)
    sg["Group"] = sg["Group"].astype(str)

    if exclude_qc:
        sg = sg[~sg["Sample"].str.startswith("QC_")].copy()

    sample_names = sg["Sample"].tolist()
    sample_to_group = dict(zip(sg["Sample"], sg["Group"]))

    col_map = _map_columns_to_samples(collapsed, sample_names)
    present_samples = [s for s in sample_names if s in col_map]

    # Build intensity matrix view
    intensity = collapsed[[col_map[s] for s in present_samples]].copy()
    intensity.columns = present_samples  # rename to clean sample names
    collapsed_view = collapsed.copy()
    for s in present_samples:
        collapsed_view[s] = intensity[s]

    return collapsed_view, present_samples, sample_to_group


# ============================================================
# Core computations
# ============================================================

def totals_per_class_from_X(X: pd.DataFrame, feature_meta: pd.DataFrame, unknown_policy: str = "append") -> pd.DataFrame:
    """
    X: samples × features (columns = UniqueID or feature ids), numeric
    feature_meta: contains 'Lipid Class' and 'UniqueID'
    Return: samples × classes totals
    """
    # Map each feature to its class (align to X columns)
    if "Lipid Class" not in feature_meta.columns:
        raise ValueError("Feature metadata is missing 'Lipid Class' column.")
    if "UniqueID" not in feature_meta.columns:
        # load_dataset normally ensures this; be defensive
        feature_meta = feature_meta.copy()
        feature_meta["UniqueID"] = X.columns.astype(str)

    class_map = (
    feature_meta[["UniqueID", "Lipid Class"]]
    .dropna(subset=["Lipid Class"])
    .assign(**{"Lipid Class": lambda d: d["Lipid Class"].astype(str).map(lambda v: _canon_class(v, unknown_policy))})
    .set_index("UniqueID")["Lipid Class"].astype(str))

    class_map = class_map.reindex(X.columns)

    # ---- DEBUG: who mapped to Other?
    if "Other" in class_map.values:
        raw = feature_meta.set_index("UniqueID")["Lipid Class"].astype(str)
        other_uids = [uid for uid, cls in class_map.items() if cls == "Other"]
        top_raw = raw.loc[raw.index.intersection(other_uids)].value_counts().head(10).to_dict()
        print(f"[ClassDist][DEBUG] {len(other_uids)} features mapped to 'Other'. Top raw labels → counts: {top_raw}. {other_uids}", flush=True)

    # Group columns by class and sum
    per_sample = X.groupby(by=class_map, axis=1).sum()

    
    per_sample.index.name = "Sample"
    return per_sample

def totals_per_class_from_collapsed(df: pd.DataFrame, sample_cols: List[str], unknown_policy: str = "append") -> pd.DataFrame:
    """Return per-sample totals per class (rows=samples, cols=classes)."""
    df2 = df[sample_cols].copy()
    df2["Lipid Class"] = df["Lipid Class"].astype(str).map(lambda v: _canon_class(v, unknown_policy))
    per_sample = df2.groupby("Lipid Class").sum(numeric_only=True).T
    per_sample.index.name = "Sample"
    return per_sample


def group_stat_from_per_sample(per_sample: pd.DataFrame, sample_to_group: Dict[str, str],
                               stat: str = "mean") -> pd.DataFrame:
    """Aggregate per-sample totals to per-group metric."""
    tmp = per_sample.copy()
    tmp["Group"] = tmp.index.map(sample_to_group).astype(str)
    tmp = tmp.dropna(subset=["Group"])
    if stat.lower() == "median":
        return tmp.groupby("Group").median(numeric_only=True)
    return tmp.groupby("Group").mean(numeric_only=True)


def species_counts_from_meta(feature_meta: pd.DataFrame, unknown_policy: str = "append") -> pd.Series:
    if "Lipid Class" not in feature_meta.columns:
        raise ValueError("Feature metadata is missing 'Lipid Class' column.")
    fm = feature_meta.copy()
    fm["Lipid Class"] = fm["Lipid Class"].astype(str).map(lambda v: _canon_class(v, unknown_policy))
    class_col = "Lipid Class"
    if "Annotation" in fm.columns:
        s = fm.groupby(class_col)["Annotation"].nunique(dropna=True)
    else:
        uid_col = "UniqueID" if "UniqueID" in fm.columns else None
        s = fm.groupby(class_col)[uid_col].nunique(dropna=True) if uid_col else fm.groupby(class_col).size()
    s.index.name = "Lipid Class"
    return s


def _order_labels(present: list[str], group_order: Optional[list[str]]) -> list[str]:
    present = [str(g) for g in present]
    if not group_order:
        return present
    # keep GUI order first, then append any groups not in order
    gui = [g for g in group_order if g in present]
    rest = [g for g in present if g not in gui]
    return gui + rest

def _build_palette(labels: list[str], group_colors: Optional[dict]) -> dict[str, str]:
    """Return {group: hex} honoring GUI colors and extending with many distinct colors."""
    palette = {}

    # 1) start with GUI palette (exact keys only)
    if isinstance(group_colors, dict):
        for k, v in group_colors.items():
            if isinstance(v, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", v):
                palette[str(k)] = v.upper()

    # 2) very long fallback wheel (60+ colors)
    long_wheel = []
    for cmap_name in ("tab10", "tab20b", "tab20c", "Set3", "Paired", "Accent"):
        long_wheel += [matplotlib.colors.to_hex(c) for c in plt.get_cmap(cmap_name).colors]
    # ensure deterministic, unique
    seen = set()
    long_wheel = [c.upper() for c in long_wheel if not (c.upper() in seen or seen.add(c.upper()))]

    # 3) assign remaining groups from long wheel
    i = 0
    for g in labels:
        if g not in palette:
            palette[g] = long_wheel[i % len(long_wheel)]
            i += 1

    return palette


def _set_bubble_x_limits(ax: plt.Axes, x_tick_positions: List[float]) -> None:
    if not x_tick_positions:
        return
    ax.set_xlim(x_tick_positions[0] - 0.28, x_tick_positions[-1] + 0.52)


def _is_semiquant_dataset(dataset_label: Optional[str], file_path: Optional[str] = None) -> bool:
    dataset_text = str(dataset_label or "").strip().lower()
    file_text = str(file_path or "").strip().lower()
    return "annotated semi-quant" in dataset_text or "semi_quant" in file_text or "semi-quant" in file_text


def _intensity_axis_label(dataset_label: Optional[str], file_path: Optional[str] = None) -> str:
    if _is_semiquant_dataset(dataset_label, file_path):
        return "Semi-quantitative class abundance\n(normalized intensity x IS concentration)"
    return "Total (summed) normalized intensity\nfor annotated lipids"


def _intensity_axis_label_log(dataset_label: Optional[str], file_path: Optional[str] = None) -> str:
    if _is_semiquant_dataset(dataset_label, file_path):
        return "Semi-quantitative class abundance\n(normalized intensity x IS concentration; log scale)"
    return "Total (summed) normalized intensity\nfor annotated lipids (log scale)"


def _percent_axis_label(dataset_label: Optional[str], file_path: Optional[str] = None) -> str:
    if _is_semiquant_dataset(dataset_label, file_path):
        return "% of semi-quantitative class abundance\n(normalized intensity x IS concentration)"
    return "% of total (summed) normalized intensity\nfor annotated lipids"


def _percent_axis_label_log(dataset_label: Optional[str], file_path: Optional[str] = None) -> str:
    if _is_semiquant_dataset(dataset_label, file_path):
        return "% of semi-quantitative class abundance\n(normalized intensity x IS concentration; log scale)"
    return "% of total (summed) normalized intensity\nfor annotated lipids (log scale)"

# ============================================================
# Plotting (Matplotlib, no seaborn)
# ============================================================

def plot_grouped_bar(per_group: pd.DataFrame, out_png: str, out_svg: Optional[str] = None,
                     title: str = "", y_label: str = "", legend_ncols: int = 2,
                     figsize=(22, 8), group_order: Optional[list] = None,
                     group_colors: Optional[dict] = None) -> None:
    _ensure_dir(os.path.dirname(out_png))
    classes = per_group.columns.tolist()
    groups_present = list(per_group.index.astype(str))
    labels = _order_labels(groups_present, group_order)
    pal = _build_palette(labels, group_colors)

    x = np.arange(len(classes))
    width = 0.8 / max(1, len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    for i, g in enumerate(labels):
        vals = per_group.loc[g, classes].values if g in per_group.index else per_group.loc[str(g), classes].values
        ax.bar(x + i * width, vals, width=width, label=str(g), color=pal[str(g)])

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel(y_label or "Total intensity", fontsize=18, labelpad=10)
    ax.set_title(title or "Total lipid-class abundances per group", fontsize=20, pad=15)
    
    # --- Axis label and tick font sizes ---
    ax.tick_params(axis="x", labelsize=16, rotation=45, labelrotation=45)
    ax.tick_params(axis="y", labelsize=16)

    # legend with consistent colors
    handles = [Patch(facecolor=pal[g], edgecolor="none", label=str(g)) for g in labels]
    ax.legend(handles=handles, ncols=legend_ncols, fontsize=16, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left", title="Groups", title_fontsize=18)

    plt.tight_layout()
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    if out_svg:
        plt.savefig(out_svg, dpi=100, bbox_inches="tight")
    plt.close()

def plot_bubble(intensity_df: pd.DataFrame, out_png: str, out_svg: Optional[str] = None,
                title: str = "", figsize=(21, 9),
                group_order: Optional[list] = None,
                group_colors: Optional[dict] = None,
                class_order: Optional[list] = None,
                y_col: str = "Total Intensity",
                y_label: str = "Total (summed) normalized intensity\nfor annotated lipids",
                y_max: Optional[float] = None) -> None:
    _ensure_dir(os.path.dirname(out_png))

    intensity_df = intensity_df.copy()
    if intensity_df.empty:
        return
    if class_order and len(class_order):
        present_classes = set(intensity_df["Lipid Class"].astype(str))
        classes = [c for c in class_order if str(c) in present_classes]
        remaining_classes = [
            c for c in intensity_df["Lipid Class"].astype(str).unique().tolist()
            if c not in classes
        ]
        classes.extend(remaining_classes)
    else:
        classes = sorted(intensity_df["Lipid Class"].astype(str).unique().tolist())
    intensity_df["Lipid Class"] = pd.Categorical(intensity_df["Lipid Class"], categories=classes, ordered=True)
    intensity_df = intensity_df.sort_values("Lipid Class")
    intensity_df["x_pos"] = intensity_df["Lipid Class"].cat.codes.astype(float)

    groups_present = sorted(set(intensity_df["Group"].astype(str)))
    labels = _order_labels(groups_present, group_order)
    pal = _build_palette(labels, group_colors)

    n_groups = len(labels)
    jitter_scale = 0.03 if n_groups < 10 else 0.015
    offsets = {g: i * jitter_scale - jitter_scale * (n_groups - 1) / 2 for i, g in enumerate(labels)}
    intensity_df["x_jittered"] = intensity_df.apply(lambda r: r["x_pos"] + offsets[str(r["Group"])], axis=1)

    smin, smax = intensity_df["Species Count"].min(), intensity_df["Species Count"].max()
    if not np.isfinite(smin):  # empty or all NaN
        smin, smax = 0.0, 0.0

    def _size_scale(s):
        if pd.isna(s): return 20.0
        if smax <= smin: return 200.0
        return 20.0 + (float(s) - smin) / (smax - smin) * (2000.0 - 20.0)

    fig, ax = plt.subplots(figsize=figsize)
    leg1 = None
    leg2 = None
    for g in labels:
        sub = intensity_df[intensity_df["Group"].astype(str) == g]
        if sub.empty:
            continue
        ax.scatter(sub["x_jittered"], sub[y_col],
                   s=sub["Species Count"].map(_size_scale),
                   alpha=0.8, linewidths=0.7,
                    edgecolors="white", color=pal[g], label=str(g))

    # Build x positions in the SAME order as `classes`
    class_to_x = (intensity_df.drop_duplicates("Lipid Class")
                .set_index("Lipid Class")["x_pos"]
                .to_dict())

    x_tick_positions = [class_to_x[c] for c in classes if c in class_to_x]

    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(classes, rotation=90, ha="center", fontsize=14)
    ax.set_xticks(x_tick_positions)
    _set_bubble_x_limits(ax, x_tick_positions)
    
    ax.tick_params(axis="y", labelsize=13)

    ymax = pd.to_numeric(intensity_df[y_col], errors="coerce").max()
    if not np.isfinite(ymax) or ymax <= 0: ymax = 1.0
    if y_max is not None and np.isfinite(y_max) and y_max > 0:
        ymax = min(float(ymax), float(y_max))
    pad = 0.06 * ymax
    ax.set_ylim(-pad, ymax * 1.1)
    # y = 0 line
    # ax.axhline(y=0, color="black", linewidth=1)

    # # Light-gray horizontal grid
    # ax.grid(
    #     axis="both",
    #     which="major",
    #     color="lightgray",
    #     linestyle="-",
    #     linewidth=0.5,
    #     alpha=0.4
    # )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.set_xlabel("Lipid Class", fontsize=20, labelpad=15)
    ax.set_ylabel(y_label, fontsize=20, labelpad=15)
    ax.set_title(title or "Lipid class abundance across groups (bubble size = # species)", fontsize=22, pad=15)
    
    # --- Axis label and tick font sizes ---
    ax.tick_params(axis="x", labelsize=20, rotation=90, labelrotation=90)
    ax.tick_params(axis="y", labelsize=20)   

    # Alternating background bands (strongest readability gain)
    for i, xpos in enumerate(x_tick_positions):
        if i % 2 == 0:
            ax.axvspan(xpos - 0.5, xpos + 0.5, alpha=0.1, zorder=0,  color="lightgray")  
    ax.set_axisbelow(True)

    # --- Group color legend ---
    handles = [Patch(facecolor=pal[g], edgecolor="none", label=str(g)) for g in labels]
    leg1 = ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.01, 1.02),
                    loc="upper left", ncols=2 if n_groups > 7 else 1,
                    fontsize=20)
    ax.add_artist(leg1)  # explicitly add so it stays when we add the next legend
    

    # --- Bubble size legend (species count) ---
    if "Species Count" in intensity_df.columns and not intensity_df["Species Count"].isna().all():
        smin, smax = intensity_df["Species Count"].min(), intensity_df["Species Count"].max()
        steps = np.linspace(smin, smax, num=3)
        example_sizes = [20.0 + (float(s) - smin) / (smax - smin) * (2000.0 - 20.0) for s in steps]
        size_handles = [plt.scatter([], [], s=s, color="gray", alpha=0.4, edgecolors="none") for s in example_sizes]
        size_labels = [f"{int(s)} species" for s in steps]
        leg2 = ax.legend(size_handles, size_labels, title="Bubble size",
                        frameon=False, bbox_to_anchor=(1.01, 0.25), loc="upper left",
                        labelspacing=1.7, fontsize=16, title_fontsize=18)
        ax.add_artist(leg2)  # add without replacing the first legend

    plt.subplots_adjust(
        left=0.12,   # give y-label breathing room
        right=0.82,  # space for legends
        bottom=0.25, # rotated class labels
        top=0.90
    )
    extras = []
    if leg1 is not None: extras.append(leg1)
    if leg2 is not None: extras.append(leg2)
    plt.savefig(out_png, dpi=100, bbox_inches="tight", bbox_extra_artists=extras)
    if out_svg:
        plt.savefig(out_svg, dpi=100, bbox_inches="tight", bbox_extra_artists=extras)
    plt.close()


def plot_bubble_log(intensity_df: pd.DataFrame, out_png: str, out_svg: Optional[str] = None,
                title: str = "", figsize=(21, 9),
                group_order: Optional[list] = None,
                group_colors: Optional[dict] = None,
                class_order: Optional[list] = None,
                y_col: str = "Total Intensity",
                y_label: str = "Total (summed) normalized intensity\nfor annotated lipids (log scale)") -> None:
    _ensure_dir(os.path.dirname(out_png))

    intensity_df = intensity_df.copy()
    intensity_df[y_col] = pd.to_numeric(intensity_df[y_col], errors="coerce")
    intensity_df = intensity_df[np.isfinite(intensity_df[y_col]) & intensity_df[y_col].gt(0)].copy()
    if intensity_df.empty:
        print(f"[ClassDist] Skipping log bubble plot for '{y_col}' because there are no positive values to display.", flush=True)
        return
    if class_order and len(class_order):
        present_classes = set(intensity_df["Lipid Class"].astype(str))
        classes = [c for c in class_order if str(c) in present_classes]
        remaining_classes = [
            c for c in intensity_df["Lipid Class"].astype(str).unique().tolist()
            if c not in classes
        ]
        classes.extend(remaining_classes)
    else:
        classes = sorted(intensity_df["Lipid Class"].astype(str).unique().tolist())
    intensity_df["Lipid Class"] = pd.Categorical(intensity_df["Lipid Class"], categories=classes, ordered=True)
    intensity_df = intensity_df.sort_values("Lipid Class")
    intensity_df["x_pos"] = intensity_df["Lipid Class"].cat.codes.astype(float)

    groups_present = sorted(set(intensity_df["Group"].astype(str)))
    labels = _order_labels(groups_present, group_order)
    pal = _build_palette(labels, group_colors)

    n_groups = len(labels)
    jitter_scale = 0.03 if n_groups < 10 else 0.015
    offsets = {g: i * jitter_scale - jitter_scale * (n_groups - 1) / 2 for i, g in enumerate(labels)}
    intensity_df["x_jittered"] = intensity_df.apply(lambda r: r["x_pos"] + offsets[str(r["Group"])], axis=1)

    smin, smax = intensity_df["Species Count"].min(), intensity_df["Species Count"].max()
    if not np.isfinite(smin):  # empty or all NaN
        smin, smax = 0.0, 0.0

    def _size_scale(s):
        if pd.isna(s): return 20.0
        if smax <= smin: return 200.0
        return 20.0 + (float(s) - smin) / (smax - smin) * (2000.0 - 20.0)

    fig, ax = plt.subplots(figsize=figsize)
    leg1 = None
    leg2 = None
    for g in labels:
        sub = intensity_df[intensity_df["Group"].astype(str) == g]
        if sub.empty:
            continue
        ax.scatter(sub["x_jittered"], sub[y_col],
                   s=sub["Species Count"].map(_size_scale),
                   alpha=0.6, linewidths=0.8,
                    edgecolors="white", color=pal[g], label=str(g))

    # Build x positions in the SAME order as `classes`
    class_to_x = (intensity_df.drop_duplicates("Lipid Class")
                .set_index("Lipid Class")["x_pos"]
                .to_dict())

    x_tick_positions = [class_to_x[c] for c in classes if c in class_to_x]

    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(classes, rotation=90, ha="center", fontsize=14)
    ax.set_xticks(x_tick_positions)
    _set_bubble_x_limits(ax, x_tick_positions)
    
    ax.tick_params(axis="y", labelsize=13)

    ymax = pd.to_numeric(intensity_df[y_col], errors="coerce").max()
    if not np.isfinite(ymax) or ymax <= 0: ymax = 1.0

    # # Light-gray horizontal grid
    # ax.grid(
    #     axis="both",
    #     which="major",
    #     color="lightgray",
    #     linestyle="-",
    #     linewidth=0.5,
    #     alpha=0.4
    # )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.set_xlabel("Lipid Class", fontsize=20, labelpad=15)
    ax.set_ylabel(y_label, fontsize=20, labelpad=15)
    ax.set_title(title or "Lipid class abundance across groups (bubble size = # species)", fontsize=22, pad=15)
    
    # --- Log scale y-axis so that most intense features don't dominate the plot too much
    ax.set_yscale("log")
    # --- Proper log scale limits ---
    vals = pd.to_numeric(intensity_df[y_col], errors="coerce")
    vals = vals[np.isfinite(vals) & (vals > 0)]

    if len(vals) == 0:
        ymin, ymax = 1e-3, 1.0
    else:
        ymin = vals.min()
        ymax = vals.max()

    # Add padding in log space
    log_min = np.floor(np.log10(ymin))
    log_max = np.ceil(np.log10(ymax))

    ymin_plot = 10 ** log_min
    ymax_plot = 10 ** (log_max + 0.2)  # small headroom

    ax.set_yscale("log")
    ax.set_ylim(ymin_plot, ymax_plot)
    
    # --- Axis label and tick font sizes ---
    ax.tick_params(axis="x", labelsize=20, rotation=90, labelrotation=90)
    ax.tick_params(axis="y", labelsize=20)   

    # Alternating background bands (strongest readability gain)
    for i, xpos in enumerate(x_tick_positions):
        if i % 2 == 0:
            ax.axvspan(xpos - 0.5, xpos + 0.5, alpha=0.1, zorder=0,  color="lightgray")  
    ax.set_axisbelow(True)

    # --- Group color legend ---
    handles = [Patch(facecolor=pal[g], edgecolor="none", label=str(g)) for g in labels]
    leg1 = ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.01, 1.02),
                    loc="upper left", ncols=2 if n_groups > 7 else 1,
                    fontsize=20)
    ax.add_artist(leg1)  # explicitly add so it stays when we add the next legend
    

    # --- Bubble size legend (species count) ---
    if "Species Count" in intensity_df.columns and not intensity_df["Species Count"].isna().all():
        smin, smax = intensity_df["Species Count"].min(), intensity_df["Species Count"].max()
        steps = np.linspace(smin, smax, num=3)
        example_sizes = [20.0 + (float(s) - smin) / (smax - smin) * (2000.0 - 20.0) for s in steps]
        size_handles = [plt.scatter([], [], s=s, color="gray", alpha=0.4, edgecolors="none") for s in example_sizes]
        size_labels = [f"{int(s)} species" for s in steps]
        leg2 = ax.legend(size_handles, size_labels, title="Bubble size",
                        frameon=False, bbox_to_anchor=(1.01, 0.25), loc="upper left",
                        labelspacing=1.7, fontsize=16, title_fontsize=18)
        ax.add_artist(leg2)  # add without replacing the first legend

    plt.subplots_adjust(
        left=0.12,   # give y-label breathing room
        right=0.82,  # space for legends
        bottom=0.25, # rotated class labels
        top=0.90
    )
    extras = []
    if leg1 is not None: extras.append(leg1)
    if leg2 is not None: extras.append(leg2)
    plt.savefig(out_png, dpi=100, bbox_inches="tight", bbox_extra_artists=extras)
    if out_svg:
        plt.savefig(out_svg, dpi=100, bbox_inches="tight", bbox_extra_artists=extras)
    print(f"[ClassDist] Saved log bubble plot: {out_png}", flush=True)
    plt.close()
    
# ============================================================
# Public API
# ============================================================

def run_from_stats(
    file_path: str,
    group_file: Optional[str],
    save_dir: str,
    group_stat: str = "mean",
    high_conf_only: bool = False,
    unknown_policy: str = "append",         
    # tolerated extras from GUI, not used here:
    group_colors: Optional[dict] = None,
    group_order: Optional[list] = None,
    **kwargs,
) -> Dict[str, str]:

    """
    Load with the shared util (same as PCA/PLS-DA/Heatmap/Volcano):
      X: samples × features (numeric)
      y: sample group labels (index-aligned with X.index)
      feature_meta: per-feature metadata (must include 'Lipid Class', 'UniqueID', optionally 'Annotation tier')
    """
    out_dir = prepare_output_dir(save_dir)
    
    print('[Class distributions] Running class distribution bar and bubble plots...', flush = True)
    
    # === Standard loader — this is how other modules get sample columns and groups
    X, y, feature_meta = load_dataset(file_path, group_file)
    print(f"Running class distribution with {file_path}.", flush = True)
    if X.empty or feature_meta.empty:
        raise ValueError("Dataset appears empty or malformed. Check file_path and group_file.", flush = True)

    # === Optional High-confidence feature gate (feature-level)
    mask = _select_confidence_feature_mask(feature_meta, high_conf_only=high_conf_only)
    if mask is not None and mask.any():
        keep_ids = set(feature_meta.loc[mask, "UniqueID"].astype(str))
        keep_ids = [c for c in X.columns if c in keep_ids]
        if keep_ids:
            X = X[keep_ids]
            feature_meta = feature_meta[feature_meta["UniqueID"].astype(str).isin(keep_ids)]

    # === Per-sample totals per lipid class
    # (build the totals FIRST, then inspect columns to know what classes actually exist)
    feature_meta = feature_meta.copy()
    feature_meta["Lipid Class"] = feature_meta["Lipid Class"].astype(str).str.strip()

    per_sample = totals_per_class_from_X(X, feature_meta, unknown_policy=unknown_policy)
    print(f"[ClassDist] per_sample-> shape={per_sample.shape}, classes={list(per_sample.columns)}", flush=True)

    ordered_pref = _order_for_sample_type(kwargs.get("sample_type", None))
    present_classes = [str(c) for c in per_sample.columns]
    # keep only classes that actually exist, in organism-preferred order first
    ordered_classes = [c for c in ordered_pref if c in present_classes] + \
                    [c for c in present_classes if c not in ordered_pref]
    if any(c not in ordered_pref for c in present_classes):
        extras = [c for c in present_classes if c not in ordered_pref]
        print(f"[ClassDist] Appended classes: {extras}", flush=True)
    # no reindexing that adds absent classes; keep per_sample as-is
    per_sample = per_sample[ordered_classes]

    # Aggregate per group
    if group_stat.lower() == "median":
        per_group = per_sample.groupby(y).median(numeric_only=True)
    else:
        per_group = per_sample.groupby(y).mean(numeric_only=True)
    per_group = per_group.reindex(columns=ordered_classes, fill_value=0)

    # Species counts (respect canonical mapping and organism order)
    scounts = species_counts_from_meta(feature_meta, unknown_policy=unknown_policy).reindex(ordered_classes).fillna(0)

    # --- Drop lipid classes with no detected intensity ---
    nonzero_classes = per_group.columns[(per_group.sum(axis=0) > 0)]

    # keep organism order but drop zeros
    ordered_classes = [c for c in ordered_classes if c in set(nonzero_classes)]

    per_sample = per_sample[ordered_classes]
    per_group  = per_group[ordered_classes]
    scounts    = scounts.reindex(ordered_classes).fillna(0)

    # === Save CSVs
    per_sample_csv = os.path.join(out_dir, "per_sample_class_totals.csv")
    per_group_csv  = os.path.join(out_dir, f"per_group_class_{group_stat}.csv")
    per_sample.to_csv(per_sample_csv)
    per_group.to_csv(per_group_csv)

    dataset_label = kwargs.get("dataset_label")
    intensity_label = _intensity_axis_label(dataset_label, file_path)
    intensity_label_log = _intensity_axis_label_log(dataset_label, file_path)
    percent_label = _percent_axis_label(dataset_label, file_path)
    percent_label_log = _percent_axis_label_log(dataset_label, file_path)

    # === Plots
    bar_png = os.path.join(out_dir, f"total_intensity_per_class_grouped_bar_{group_stat}.png")
    bar_svg = os.path.join(out_dir, f"total_intensity_per_class_grouped_bar_{group_stat}.svg")
    plot_grouped_bar(per_group, bar_png, bar_svg,
                 title=f"Total lipid-class abundances per group ({group_stat})",
                 y_label=intensity_label,
                 group_order=group_order, group_colors=group_colors)

    # Bubble input frame
    records = []
    group_labels = [str(g) for g in per_group.index]
    current_classes = per_group.columns.tolist()  # already filtered & ordered

    for g in group_labels:
        for cl in current_classes:
            records.append({
                "Lipid Class": cl,
                "Group": g,
                "Total Intensity": float(per_group.loc[g, cl]),
                "Species Count": float(scounts.get(cl, 0)),
            })
    bubble_df = pd.DataFrame(records)
    bubble_pct_df = bubble_df.copy()
    group_totals = bubble_pct_df.groupby("Group")["Total Intensity"].transform("sum").replace(0, np.nan)
    bubble_pct_df["Percent of Total Intensity"] = (
        bubble_pct_df["Total Intensity"].div(group_totals).fillna(0.0) * 100.0
    )

    bub_png = os.path.join(out_dir, f"bubble_total_intensity_per_class_{group_stat}.png")
    bub_svg = os.path.join(out_dir, f"bubble_total_intensity_per_class_{group_stat}.svg")
    plot_bubble(
        bubble_df, bub_png, bub_svg,
        title=f"Lipid class diversity and abundance across groups ({group_stat})",
        group_order=group_order, group_colors=group_colors, class_order=current_classes,
        y_label=intensity_label,
    )

    bub_pct_png = os.path.join(out_dir, f"bubble_percent_total_intensity_per_class_{group_stat}.png")
    bub_pct_svg = os.path.join(out_dir, f"bubble_percent_total_intensity_per_class_{group_stat}.svg")
    plot_bubble(
        bubble_pct_df, bub_pct_png, bub_pct_svg,
        title=f"Lipid class abundance across groups (bubble size = # species)",
        group_order=group_order,
        group_colors=group_colors,
        class_order=current_classes,
        y_col="Percent of Total Intensity",
        y_label=percent_label,
    )

    bub_pct_log_png = os.path.join(out_dir, f"bubble_log_percent_total_intensity_per_class_{group_stat}.png")
    bub_pct_log_svg = os.path.join(out_dir, f"bubble_log_percent_total_intensity_per_class_{group_stat}.svg")
    plot_bubble_log(
        bubble_pct_df, bub_pct_log_png, bub_pct_log_svg,
        title=f"Lipid class abundance across groups (log-scale % of total; bubble size = # species)",
        group_order=group_order,
        group_colors=group_colors,
        class_order=current_classes,
        y_col="Percent of Total Intensity",
        y_label=percent_label_log,
    )

    low_pct_outputs = {}
    for low_pct_threshold in (10.0, 5.0):
        bubble_pct_low_df = bubble_pct_df.loc[
            pd.to_numeric(bubble_pct_df["Percent of Total Intensity"], errors="coerce").lt(low_pct_threshold)
        ].copy()
        suffix = f"lt{int(low_pct_threshold)}"
        bub_pct_low_png = os.path.join(out_dir, f"bubble_percent_total_intensity_per_class_{suffix}_{group_stat}.png")
        bub_pct_low_svg = os.path.join(out_dir, f"bubble_percent_total_intensity_per_class_{suffix}_{group_stat}.svg")
        plot_bubble(
            bubble_pct_low_df, bub_pct_low_png, bub_pct_low_svg,
            title=f"Lipid class abundance across groups (< {int(low_pct_threshold)}% of total; bubble size = # species)",
            group_order=group_order,
            group_colors=group_colors,
            class_order=current_classes,
            y_col="Percent of Total Intensity",
            y_label=percent_label,
            y_max=low_pct_threshold,
        )
        low_pct_outputs[f"bubble_percent_{suffix}_png"] = bub_pct_low_png
        low_pct_outputs[f"bubble_percent_{suffix}_svg"] = bub_pct_low_svg
    
    bub_png_log = os.path.join(out_dir, f"bubble_log_total_intensity_per_class_{group_stat}.png")
    bub_svg_log = os.path.join(out_dir, f"bubble_log_total_intensity_per_class_{group_stat}.svg")
    
    plot_bubble_log(
        bubble_df, bub_png_log, bub_svg_log,
        title=f"Lipid class diversity and log-scale abundance across groups ({group_stat})",
        group_order=group_order, group_colors=group_colors, class_order=current_classes,
        y_label=intensity_label_log,
    )

    return {
        "per_sample_csv": per_sample_csv,
        "per_group_csv": per_group_csv,
        "bar_png": bar_png,
        "bar_svg": bar_svg,
        "bubble_png": bub_png,
        "bubble_svg": bub_svg,
        "bubble_percent_png": bub_pct_png,
        "bubble_percent_svg": bub_pct_svg,
        "bubble_percent_log_png": bub_pct_log_png,
        "bubble_percent_log_svg": bub_pct_log_svg,
        **low_pct_outputs,
    }


def run_from_collapsed(
    collapsed_csv: str,
    sample_groups_csv: str,
    save_dir: str,
    group_stat: str = "median",         # "mean" or "median"
    high_conf_only: bool = False,  
    unknown_policy: str = "append",            
    exclude_qc: bool = True,
    group_order: Optional[list] = None,
    group_colors: Optional[dict] = None,
) -> Dict[str, str]:
    """Pipeline for COLLAPSED DEBUG export + sample_groups_cleaned.csv """
    _ensure_dir(save_dir)
    csv_dir = os.path.join(save_dir, "CSV"); _ensure_dir(csv_dir)
    plot_dir = os.path.join(save_dir, "Plots"); _ensure_dir(plot_dir)

    df, samples, sample_to_group = _load_collapsed_style(collapsed_csv, sample_groups_csv, exclude_qc=exclude_qc)
    df = _filter_high_conf_collapsed(df, high_conf_only=high_conf_only)
    
    per_sample = totals_per_class_from_collapsed(df, samples, unknown_policy=unknown_policy)
    print(f"[ClassDist] per_sample(collapsed)-> shape={per_sample.shape}, classes={list(per_sample.columns)}", flush=True)

    ordered_pref = _order_for_sample_type(None)
    present_classes = [str(c) for c in per_sample.columns]
    unknowns = [c for c in present_classes if c not in ordered_pref]
    ordered_classes = list(ordered_pref) + unknowns
    if unknowns:
        print(f"[ClassDist] Appended classes after master list: {unknowns}", flush=True)

    per_sample = per_sample.reindex(columns=ordered_classes, fill_value=0)
    per_group  = group_stat_from_per_sample(per_sample, sample_to_group, stat=group_stat)
    per_group  = per_group.reindex(columns=ordered_classes, fill_value=0)

    scounts = species_counts_from_collapsed(df, unknown_policy=unknown_policy).reindex(ordered_classes).fillna(0)
    
    # --- Drop lipid classes with no detected intensity (keep organism order) ---
    nonzero_classes = set(per_group.columns[(per_group.sum(axis=0) > 0)])
    ordered_classes = [c for c in ordered_classes if c in nonzero_classes]

    per_sample = per_sample[ordered_classes]
    per_group  = per_group[ordered_classes]
    scounts    = scounts.reindex(ordered_classes).fillna(0)

    # Save
    per_sample_csv = os.path.join(csv_dir, "per_sample_class_totals.csv")
    per_group_csv = os.path.join(csv_dir, f"per_group_class_{group_stat}.csv")
    per_sample.to_csv(per_sample_csv)
    per_group.to_csv(per_group_csv)

    dataset_label = None
    intensity_label = _intensity_axis_label(dataset_label, collapsed_csv)
    percent_label = _percent_axis_label(dataset_label, collapsed_csv)
    percent_label_log = _percent_axis_label_log(dataset_label, collapsed_csv)

    # Plots
    bar_png = os.path.join(plot_dir, f"total_intensity_per_class_grouped_bar_{group_stat}.png")
    bar_svg = os.path.join(plot_dir, f"total_intensity_per_class_grouped_bar_{group_stat}.svg")
    plot_grouped_bar(
        per_group,
        bar_png,
        bar_svg,
        title=f"Total lipid-class abundances per group ({group_stat})",
        y_label=intensity_label,
        group_order=group_order,
        group_colors=group_colors,
    )

    # Bubble
    records = []
    group_labels = [str(g) for g in per_group.index]
    current_classes = per_group.columns.tolist()  # filtered & ordered

    for g in group_labels:
        for cl in current_classes:
            records.append({
                "Lipid Class": cl,
                "Group": g,
                "Total Intensity": float(per_group.loc[g, cl]),
                "Species Count": float(scounts.get(cl, 0)),
            })
    bubble_df = pd.DataFrame(records)
    bubble_pct_df = bubble_df.copy()
    group_totals = bubble_pct_df.groupby("Group")["Total Intensity"].transform("sum").replace(0, np.nan)
    bubble_pct_df["Percent of Total Intensity"] = (
        bubble_pct_df["Total Intensity"].div(group_totals).fillna(0.0) * 100.0
    )

    bub_png = os.path.join(plot_dir, f"bubble_total_intensity_per_class_{group_stat}.png")
    bub_svg = os.path.join(plot_dir, f"bubble_total_intensity_per_class_{group_stat}.svg")
    plot_bubble(
        bubble_df, bub_png, bub_svg,
        title=f"Lipid class diversity and abundance across groups ({group_stat})",
        group_order=group_order, group_colors=group_colors, class_order=current_classes,
        y_label=intensity_label,
    )

    bub_pct_png = os.path.join(plot_dir, f"bubble_percent_total_intensity_per_class_{group_stat}.png")
    bub_pct_svg = os.path.join(plot_dir, f"bubble_percent_total_intensity_per_class_{group_stat}.svg")
    plot_bubble(
        bubble_pct_df, bub_pct_png, bub_pct_svg,
        title="Lipid class abundance across groups (bubble size = # species)",
        group_order=group_order,
        group_colors=group_colors,
        class_order=current_classes,
        y_col="Percent of Total Intensity",
        y_label=percent_label,
    )

    bub_pct_log_png = os.path.join(plot_dir, f"bubble_log_percent_total_intensity_per_class_{group_stat}.png")
    bub_pct_log_svg = os.path.join(plot_dir, f"bubble_log_percent_total_intensity_per_class_{group_stat}.svg")
    plot_bubble_log(
        bubble_pct_df, bub_pct_log_png, bub_pct_log_svg,
        title="Lipid class abundance across groups (log-scale % of total; bubble size = # species)",
        group_order=group_order,
        group_colors=group_colors,
        class_order=current_classes,
        y_col="Percent of Total Intensity",
        y_label=percent_label_log,
    )

    low_pct_outputs = {}
    for low_pct_threshold in (10.0, 5.0):
        bubble_pct_low_df = bubble_pct_df.loc[
            pd.to_numeric(bubble_pct_df["Percent of Total Intensity"], errors="coerce").lt(low_pct_threshold)
        ].copy()
        suffix = f"lt{int(low_pct_threshold)}"
        bub_pct_low_png = os.path.join(plot_dir, f"bubble_percent_total_intensity_per_class_{suffix}_{group_stat}.png")
        bub_pct_low_svg = os.path.join(plot_dir, f"bubble_percent_total_intensity_per_class_{suffix}_{group_stat}.svg")
        plot_bubble(
            bubble_pct_low_df, bub_pct_low_png, bub_pct_low_svg,
            title=f"Lipid class abundance across groups (< {int(low_pct_threshold)}% of total; bubble size = # species)",
            group_order=group_order,
            group_colors=group_colors,
            class_order=current_classes,
            y_col="Percent of Total Intensity",
            y_label=percent_label,
            y_max=low_pct_threshold,
        )
        low_pct_outputs[f"bubble_percent_{suffix}_png"] = bub_pct_low_png
        low_pct_outputs[f"bubble_percent_{suffix}_svg"] = bub_pct_low_svg

    return {
        "per_sample_csv": per_sample_csv,
        "per_group_csv": per_group_csv,
        "bar_png": bar_png,
        "bar_svg": bar_svg,
        "bubble_png": bub_png,
        "bubble_svg": bub_svg,
        "bubble_percent_png": bub_pct_png,
        "bubble_percent_svg": bub_pct_svg,
        "bubble_percent_log_png": bub_pct_log_png,
        "bubble_percent_log_svg": bub_pct_log_svg,
        **low_pct_outputs,
    }


# ============================================================
# CLI
# ============================================================

def _as_bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Total intensity per lipid class (stats-style or collapsed).")
    sub = p.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("stats", help="Run from STATS-STYLE file")
    ps.add_argument("--stats_csv", required=True)
    ps.add_argument("--group_file", required=False, default=None)
    ps.add_argument("--save_dir", required=True)
    ps.add_argument("--group_stat", default="mean", choices=["mean", "median"])
    ps.add_argument("--high_conf_only", default="false")

    pc = sub.add_parser("collapsed", help="Run from COLLAPSED debug export + sample_groups_cleaned.csv")
    pc.add_argument("--collapsed_csv", required=True)
    pc.add_argument("--sample_groups_csv", required=True)
    pc.add_argument("--save_dir", required=True)
    pc.add_argument("--group_stat", default="mean", choices=["mean", "median"])
    pc.add_argument("--high_conf_only", default="false")
    pc.add_argument("--exclude_qc", default="true")

    args = p.parse_args()
    if args.mode == "stats":
        out = run_from_stats(
            file_path=args.stats_csv,
            group_file=args.group_file,
            save_dir=args.save_dir,
            group_stat=args.group_stat,
            high_conf_only=_as_bool(args.high_conf_only),
            unknown_policy="append"
        )
    else:
        out = run_from_collapsed(
            collapsed_csv=args.collapsed_csv,
            sample_groups_csv=args.sample_groups_csv,
            save_dir=args.save_dir,
            group_stat=args.group_stat,
            high_conf_only=_as_bool(args.high_conf_only),
                exclude_qc=_as_bool(args.exclude_qc),
                unknown_policy="append"
            )
    for k, v in out.items():
        print(f"{k}: {v}", flush = True)
