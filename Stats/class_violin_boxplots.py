# Stats/class_violin_boxplots.py
# ------------------------------------------------------------
# Violin and boxplots for summed intensities per lipid class
# - Loads via Stats.utils.load_dataset(file_path, group_file)
# - Aggregates per sample to class totals (reuses existing helper)
# - Produces, per class:
#     (A) Violin plot across GROUPS
#     (B) Boxplot across GROUPS
# - Writes tidy melted CSVs (per class) + per-sample class totals
# - Saves under save_dir/ClassViolinBox/
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib import collections as mcoll

from Stats.figure_style import build_group_palette as _shared_build_group_palette, get_figure_style
from Stats.utils import load_dataset, prepare_output_dir
from Stats.summed_intensity_per_class import totals_per_class_from_X

# Aesthetics consistent with your species-level plots
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
plt.ioff()
sns.set_style("white")  # we'll control grids per-axes

# =========================
# Local helpers (palette / order)
# =========================

# --- Group palette helper (shared shape across modules) ---
def _build_group_palette(groups_like, group_colors=None, group_order=None):
    return _shared_build_group_palette(groups_like, group_colors=group_colors, group_order=group_order)

def _order_labels(present: List[str], group_order: Optional[List[str]]) -> List[str]:
    present = [str(g) for g in present]
    if not group_order:
        return present
    gui = [g for g in group_order if g in present]
    rest = [g for g in present if g not in gui]
    return gui + rest

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:."/\\|?*]', "_", str(s))

def _is_semiquant_dataset(dataset_label: Optional[str], file_path: Optional[str] = None) -> bool:
    dataset_text = str(dataset_label or "").strip().lower()
    file_text = str(file_path or "").strip().lower()
    return "annotated semi-quant" in dataset_text or "semi_quant" in file_text or "semi-quant" in file_text

def _intensity_axis_label(dataset_label: Optional[str], file_path: Optional[str] = None) -> str:
    if _is_semiquant_dataset(dataset_label, file_path):
        return "Semi-quantitative abundance\n(normalized intensity x IS conc.)"
    return "Summed intensity"

# =========================
# Plotting (SEPARATE violin and boxplot)
# =========================

def _robust_ylim_from(vals: np.ndarray) -> tuple[float, float, ticker.Formatter]:
    """Compute robust y-limits and a sensible formatter."""
    finite = np.isfinite(vals)
    if not finite.any():
        return (0.0, 1.0, ticker.FormatStrFormatter('%.2f'))
    y_lo = float(np.nanpercentile(vals[finite], 1.0))
    y_hi = float(np.nanpercentile(vals[finite], 99.0))
    if not np.isfinite(y_hi):
        y_hi = float(np.nanmax(vals[finite]))
    if not np.isfinite(y_lo):
        y_lo = 0.0
    if not np.isfinite(y_hi) or y_hi <= y_lo:
        delta = abs(y_lo) if y_lo != 0 else 1.0
        y_lo, y_hi = y_lo - 0.25 * delta, y_lo + 0.75 * delta
    margin = 0.05 * max(1e-12, (y_hi - y_lo))
    ymin = 0.0 if y_lo > -0.1 * (y_hi - y_lo) else (y_lo - margin)
    ymax = y_hi + margin

    vmax = float(np.nanmax(vals[finite]))
    if vmax >= 1e4 or vmax <= 1e-3:
        fmt = ticker.ScalarFormatter(useMathText=True)
    else:
        # choose decimals based on range
        rng = ymax - ymin
        if rng < 0.01:
            fmt = ticker.FormatStrFormatter('%.4f')
        elif rng < 0.1:
            fmt = ticker.FormatStrFormatter('%.3f')
        elif rng < 1:
            fmt = ticker.FormatStrFormatter('%.2f')
        else:
            fmt = ticker.FormatStrFormatter('%.1f')
    return ymin, ymax, fmt


def _robust_ylim_no_zero_anchor(vals: np.ndarray) -> tuple[float, float, ticker.Formatter]:
    """Compute robust y-limits without forcing the lower bound to zero."""
    finite = np.isfinite(vals)
    if not finite.any():
        return (0.0, 1.0, ticker.FormatStrFormatter('%.2f'))
    y_lo = float(np.nanpercentile(vals[finite], 1.0))
    y_hi = float(np.nanpercentile(vals[finite], 99.0))
    if not np.isfinite(y_hi):
        y_hi = float(np.nanmax(vals[finite]))
    if not np.isfinite(y_lo):
        y_lo = 0.0
    if not np.isfinite(y_hi) or y_hi <= y_lo:
        delta = abs(y_lo) if y_lo != 0 else 1.0
        y_lo, y_hi = y_lo - 0.25 * delta, y_lo + 0.75 * delta
    margin = 0.05 * max(1e-12, (y_hi - y_lo))
    ymin = y_lo - margin
    ymax = y_hi + margin

    vmax = float(np.nanmax(vals[finite]))
    if vmax >= 1e4 or vmax <= 1e-3:
        fmt = ticker.ScalarFormatter(useMathText=True)
    else:
        rng = ymax - ymin
        if rng < 0.01:
            fmt = ticker.FormatStrFormatter('%.4f')
        elif rng < 0.1:
            fmt = ticker.FormatStrFormatter('%.3f')
        elif rng < 1:
            fmt = ticker.FormatStrFormatter('%.2f')
        else:
            fmt = ticker.FormatStrFormatter('%.1f')
    return ymin, ymax, fmt

def _violin_one_class(
    df: pd.DataFrame,            # columns: Sample, Group, Value
    class_name: str,
    order: List[str],
    pal: dict,
    out_png: str,
    out_svg: Optional[str] = None,
    strip: bool = True,
    jitter: bool = True,
    style: Optional[dict] = None,
    y_label: str = "Summed intensity",
):
    style = style or get_figure_style(False, 100)
    title_fs = max(style["title_size"] + 4, 24)
    label_fs = max(style["label_size"] + 3, 20)
    tick_fs = max(style["tick_size"] + 3, 18)
    fig, ax = plt.subplots(figsize=(8.4, 5.6), facecolor="white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.26)

    ax.grid(False)

    # Violin: translucent
    sns.violinplot(
        data=df, x="Group", y="Value",
        order=order, palette=[pal[g] for g in order],
        inner=None, cut=0, scale="width", linewidth=0, ax=ax
    )
    # Enforce transparency on bodies
    for pc in [c for c in ax.collections if isinstance(c, mcoll.PolyCollection)]:
        fc = pc.get_facecolor()
        if len(fc):
            r, g, b, _ = fc[0]
            pc.set_facecolor((r, g, b, 0.30))
            pc.set_edgecolor((r, g, b, 0.00))
        pc.set_zorder(1)

    # Optional points
    if strip:
        sns.stripplot(
            data=df, x="Group", y="Value",
            order=order, jitter=jitter, dodge=False, marker="o",
            palette=[pal[g] for g in order],
            alpha=0.60, size=5.0, edgecolor="white", linewidth=0.4,
            ax=ax, zorder=2
        )

    # Title/labels
    ax.set_title(f"{class_name}: summed intensity by group", fontsize=title_fs, pad=16, fontweight="semibold")
    ax.set_xlabel(None)
    ax.xaxis.label.set_visible(False)
    ax.set_ylabel(y_label, fontsize=label_fs, labelpad=14)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)

    # Y limits
    vals = pd.to_numeric(df["Value"], errors="coerce").to_numpy()
    ymin, ymax, fmt = _robust_ylim_no_zero_anchor(vals)
    ax.set_ylim(ymin, ymax)
    if isinstance(fmt, ticker.ScalarFormatter):
        ax.yaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    else:
        ax.yaxis.set_major_formatter(fmt)


    # Borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    fig.tight_layout(pad=0.8)
    fig.savefig(out_png, dpi=style["dpi"], facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05)
    if out_svg:
        fig.savefig(out_svg, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def _box_one_class(
    df: pd.DataFrame,            # columns: Sample, Group, Value
    class_name: str,
    order: List[str],
    pal: dict,
    out_png: str,
    out_svg: Optional[str] = None,
    strip: bool = True,
    jitter: bool = True,
    style: Optional[dict] = None,
    y_label: str = "Summed intensity",
):
    style = style or get_figure_style(False, 100)
    title_fs = style["title_size"]
    label_fs = style["label_size"]
    tick_fs = style["tick_size"]
    fig, ax = plt.subplots(figsize=(6.2, 5.0), facecolor="white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.32)

    ax.grid(False)

    # Boxplot with clear internal lines, no duplicate kwargs
    sns.boxplot(
        data=df, x="Group", y="Value",
        order=order,
        palette=[pal[g] for g in order],
        width=0.58,
        showfliers=False,
        linewidth=0.0,
        whiskerprops=dict(color="gray", linewidth=0.6),
        capprops=dict(color="gray", linewidth=0.6),
        medianprops=dict(color="black", linewidth=0.90),
        ax=ax,
    )
            
    # --- Make boxes semi-transparent (robust across seaborn/mpl versions) ---
    def _rgba30(c):
        r, g, b, a = mpl.colors.to_rgba(c)
        return (r, g, b, 0.30)

    # Try ax.artists first (older seaborn), then fallback to ax.patches (newer seaborn)
    boxes = []
    if getattr(ax, "artists", None):
        boxes = [art for art in ax.artists if isinstance(art, mpl.patches.PathPatch)]
    if not boxes:
        boxes = [p for p in ax.patches if isinstance(p, mpl.patches.PathPatch)]

    # Apply 30% opacity to each box face, keep whiskers/median visible
    for i, patch in enumerate(boxes):
        # Keep seaborn’s assigned color, just change alpha
        fc = patch.get_facecolor()
        try:
            # fc may be array-like of shape (4,) or (1,4)
            rgba = tuple(fc[0]) if hasattr(fc, "__len__") and len(fc) and hasattr(fc[0], "__len__") else tuple(fc)
        except Exception:
            rgba = mpl.colors.to_rgba(fc)
        patch.set_facecolor((rgba[0], rgba[1], rgba[2], 0.30))
        patch.set_edgecolor((0, 0, 0, 0))  # no border on the box fill
        patch.set_zorder(1)

    
    # Optional points
    if strip:
        sns.stripplot(
            data=df, x="Group", y="Value",
            order=order, jitter=jitter, dodge=False, marker="o",
            palette=[pal[g] for g in order],
            alpha=0.60, size=5.0, edgecolor="white", linewidth=0.4,
            ax=ax, zorder=2
        )

    # Title/labels
    ax.set_title(f"{class_name}: summed intensity by group", fontsize=title_fs, pad=16, fontweight="semibold")
    ax.set_xlabel(None)
    ax.xaxis.label.set_visible(False)
    ax.set_ylabel(y_label, fontsize=label_fs, labelpad=14)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)

    # Y limits
    vals = pd.to_numeric(df["Value"], errors="coerce").to_numpy()
    ymin, ymax, fmt = _robust_ylim_no_zero_anchor(vals)
    ax.set_ylim(ymin, ymax)
    if isinstance(fmt, ticker.ScalarFormatter):
        ax.yaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    else:
        ax.yaxis.set_major_formatter(fmt)

    # Borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style["line_width"])
        spine.set_color("black")

    fig.tight_layout(pad=1.35)
    fig.savefig(out_png, dpi=style["dpi"], facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.1)
    if out_svg:
        fig.savefig(out_svg, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# =========================
# Public API
# =========================

def run_from_stats(
    file_path: str,
    group_file: Optional[str],
    save_dir: str,
    group_order: Optional[List[str]] = None,
    group_colors: Optional[dict] = None,
    exclude_qc: bool = True,
    dpi: int = 100,
    publication_theme: bool = False,
    dataset_label: Optional[str] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Build class-level summaries from STATS CSV and produce, for each class:
      - Violin plot across groups
      - Boxplot across groups
      - Melted CSV (Sample, Group, Class, SummedIntensity)

    Returns a dict with key output paths.
    """
    base_out = prepare_output_dir(save_dir)
    style = get_figure_style(publication_theme=publication_theme, dpi=dpi)
    out_dir = _ensure_dir(os.path.join(base_out))
    intensity_label = _intensity_axis_label(dataset_label, file_path)

    print('[Class plots] Running class-level violin and boxplots...', flush=True)

    # Load standardized dataset
    X, y, feature_meta = load_dataset(file_path, group_file)
    if X.empty or feature_meta.empty:
        raise ValueError("Dataset appears empty or malformed.")

    # Aggregate to class totals (samples × classes)
    per_sample = totals_per_class_from_X(X, feature_meta)  # index = sample, columns = class
    per_sample.index = per_sample.index.astype(str)

    # Align groups
    y = y.copy()
    y.index = y.index.astype(str)
    y = y.reindex(per_sample.index)

    # Optionally remove QC
    if exclude_qc:
        mask = ~y.astype(str).str.contains("QC", case=False, na=False)
        per_sample = per_sample.loc[mask]
        y = y.loc[mask]

    # Group labels / palette (derive order now to avoid shadowing bugs)
    labels, pal_groups = _build_group_palette(
        y.dropna().astype(str),              # groups_like
        group_colors=group_colors,
        group_order=group_order
    )


    # Outputs
    per_sample_csv = os.path.join(out_dir, "per_sample_class_totals.csv")
    per_sample.to_csv(per_sample_csv, index=False)

    melted_dir = _ensure_dir(os.path.join(out_dir, "PerClass_Melted"))
    violin_dir = _ensure_dir(os.path.join(out_dir, "PerClass_Violin"))
    box_dir    = _ensure_dir(os.path.join(out_dir, "PerClass_Boxplot"))

    # Loop classes
    for cls in per_sample.columns.astype(str):
        df = pd.DataFrame({
            "Sample": per_sample.index.astype(str),
            "Group": y.astype(str).values,
            "Class": cls,
            "SummedIntensity": pd.to_numeric(per_sample[cls], errors="coerce"),
        })
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["SummedIntensity"])
        # write melted
        df.to_csv(os.path.join(melted_dir, f"melted_{_sanitize_filename(cls)}.csv"), index=False)

        # rename columns for plotting helpers
        plot_df = df.rename(columns={"SummedIntensity": "Value"})

        # plots
        safe = _sanitize_filename(cls)
        _violin_one_class(
            df=plot_df[["Sample", "Group", "Value"]],
            class_name=cls, order=labels, pal=pal_groups,
            out_png=os.path.join(violin_dir, f"violin_{safe}.png"),
            out_svg=os.path.join(violin_dir, f"violin_{safe}.svg"),
            style=style,
            y_label=intensity_label,
        )
        _box_one_class(
            df=plot_df[["Sample", "Group", "Value"]],
            class_name=cls, order=labels, pal=pal_groups,
            out_png=os.path.join(box_dir, f"boxplot_{safe}.png"),
            out_svg=os.path.join(box_dir, f"boxplot_{safe}.svg"),
            style=style,
            y_label=intensity_label,
        )
    
    return {
    "out_dir": out_dir,
    "per_sample_class_totals_csv": per_sample_csv,
    "per_class_melted_dir": melted_dir,
    "per_class_violin_dir": violin_dir,
    "per_class_box_dir": box_dir,
    }
