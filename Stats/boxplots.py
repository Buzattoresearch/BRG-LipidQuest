#TODO: add UniqueID to the file names

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.patches import PathPatch, Rectangle

from Stats.utils import load_dataset, prepare_output_dir

import warnings
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

import matplotlib as mpl
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
mpl.rcParams["mathtext.default"] = "regular" 

plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")

mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
plt.ioff()

# --- Group palette helper (shared shape across modules) ---
def _build_group_palette(groups_like, group_colors=None, group_order=None):
    """Return (order, palette_dict) for the given groups iterable/Series."""
    natural = [str(g) for g in (groups_like.tolist() if hasattr(groups_like, "tolist") else list(groups_like))]
    unique_natural = list(dict.fromkeys(natural))  # preserves first-seen order

    if group_order:
        order = [g for g in group_order if g in unique_natural] + [g for g in unique_natural if g not in group_order]
    else:
        order = unique_natural

    if group_colors:
        pal = {g: group_colors.get(g) for g in order if group_colors.get(g)}
        # fill any missing with the Matplotlib cycle
        cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
        for i, g in enumerate(order):
            if g not in pal or not pal[g]:
                pal[g] = cycle[i % len(cycle)] if cycle else "#1f77b4"
    else:
        cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
        pal = {g: (cycle[i % len(cycle)] if cycle else "#1f77b4") for i, g in enumerate(order)}

    return order, pal


def _sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:."/\\|?*]', "_", str(s))

def _nice_label(uid: str, meta_lookup: pd.DataFrame) -> str:
    """Prefer Annotation, else lipid strings from UniqueID."""
    if meta_lookup is not None and not meta_lookup.empty:
        row = meta_lookup.loc[meta_lookup["UniqueID"] == uid]
        if not row.empty:
            # Try columns in order of usefulness
            for c in ("Annotation", "Headgroup", "Lipid Class"):
                if c in row.columns:
                    val = str(row.iloc[0][c]).strip()
                    if val and val.lower() != "nan":
                        return val
    # Fallback: last token after pipe in UniqueID
    if isinstance(uid, str) and "|" in uid:
        return uid.split("|")[-1].strip()
    return str(uid)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_boxplots(file_path, group_file, save_dir,
                 dpi=100,
                 strip=True,
                 jitter=True,
                 palette="husl",
                 group_colors=None, group_order=None):
    """
    Generate one PNG + SVG boxplot per feature (lipid), grouped by sample group.
    - Uses load_dataset(file_path, group_file) like other Stats tools.
    - Skips features that are all-NaN or have <2 non-NaN values in any group.
    - Saves under <save_dir>/Boxplots/
    """
    file_path = Path(file_path)
    save_dir = prepare_output_dir(Path(save_dir))
    out_dir = _ensure_dir(save_dir)
    plt.close('all')

    print('[Boxplots] Running boxplots (species level)...', flush = True)
    # Load standardized tables
    # X: samples × features (columns = UniqueID)
    # y: sample groups (index aligned to X.index)
    # meta: feature metadata including "UniqueID", "Annotation", etc.
    X, y, feature_meta = load_dataset(file_path, group_file)

    # Minimal meta lookup (only what we need)
    meta_lookup = feature_meta if isinstance(feature_meta, pd.DataFrame) else pd.DataFrame()
    if not meta_lookup.empty and "UniqueID" in meta_lookup.columns:
        # keep only these if present
        keep = [c for c in ("UniqueID", "Annotation", "Headgroup", "Lipid Class") if c in meta_lookup.columns]
        meta_lookup = meta_lookup[keep].copy()
    else:
        meta_lookup = pd.DataFrame(columns=["UniqueID", "Annotation", "Headgroup", "Lipid Class"])

    # Palette + order from helper (respects user colors/order)
    y = y.reset_index(drop=True)
    groups, color_map = _build_group_palette(y.astype(str), group_colors=group_colors, group_order=group_order)

    # Draw each feature
    for uid in X.columns:
        series = X[uid]
        # Skip if no data
        if series.isna().all():
            continue

        # Build plotting DataFrame
        dfp = pd.DataFrame({"Group": y.values, "Value": series.values})
        # Drop rows with NaN values (typical for missing intensities)
        dfp = dfp.dropna(subset=["Value"])
        # lock the category order used by seaborn/matplotlib
        dfp["Group"] = pd.Categorical(dfp["Group"].astype(str), categories=groups, ordered=True)

        # If every group has <2 points, many boxplot stats break; guard it
        if dfp.groupby("Group")["Value"].apply(lambda s: s.notna().sum()).max() < 1:
            continue

        # Figure/axes on white
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
        ax.set_facecolor("white")
        
        # === Gridline styling ===
        # Turn on horizontal gridlines (y-axis only)
        ax.yaxis.grid(True, color="#D3D3D3", linestyle="--", linewidth=0.5, alpha=0.7)

        # # Optional: disable vertical gridlines (cleaner for grouped plots)
        # ax.xaxis.grid(False)

        # room for rotated xticks + title
        fig.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.32)

        # Boxplot  (make box fill match point colors + transparency)
        sns.boxplot(
            data=dfp, x="Group", y="Value",
            palette=[color_map[g] for g in groups],   # use your palette
            order=groups,
            showfliers=False,
            ax=ax,                            
            linewidth=0.0,                            # no dark box edges
            whiskerprops=dict(color="gray", linewidth=0.6),
            capprops=dict(color="gray", linewidth=0.6),
            medianprops=dict(color="black", linewidth=0.75),
        )

        # Apply semi-transparent facecolor per box (to match points)
        box_alpha = 0.35
        desat_factor = 0.65

        # seaborn sometimes stores boxes in ax.artists (preferred); fallback to patches.
        boxes = list(getattr(ax, "artists", []))
        if not boxes:
            boxes = [p for p in ax.patches if isinstance(p, (PathPatch, Rectangle))]

        for i, box in enumerate(boxes[:len(groups)]):
            desat = sns.desaturate(color_map[groups[i]], desat_factor)
            box.set_facecolor((*desat, box_alpha))
            box.set_edgecolor((0, 0, 0, 0))
            box.set_zorder(1)  # keep boxes behind points

        # Optional strip of individual points (white-edged for clarity)
        if strip:
            sns.stripplot(
                data=dfp, x="Group", y="Value",
                order=groups, jitter=jitter, dodge=False, marker="o",
                palette=[color_map[g] for g in groups],
                alpha=0.65, size=6, edgecolor="white", linewidth=0.4,
                ax=ax, zorder=2
            )

        title = _nice_label(uid, meta_lookup)
        ax.set_title(title, fontsize=12, pad=12, fontweight="semibold")
        ax.set_xlabel(None)                 # clear any text
        ax.xaxis.label.set_visible(False)   # force-hide the label object
        ax.set_ylabel("Normalized peak intensity", fontsize=12, labelpad=12)

        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("")                         # clear text
        ax.xaxis.label.set_visible(False)         # hide the artist

        # Y-axis formatting
        vals = pd.to_numeric(dfp["Value"], errors="coerce").to_numpy()
        finite = np.isfinite(vals)
        if not finite.any():
            plt.close(fig)
            continue

        # robust bounds
        y_lo = float(np.nanpercentile(vals[finite], 1.0))
        y_hi = float(np.nanpercentile(vals[finite], 99.0))

        # fallbacks if collapsed or non-positive range
        if not np.isfinite(y_hi):
            y_hi = float(np.nanmax(vals[finite]))
        if not np.isfinite(y_lo):
            y_lo = 0.0
        if not np.isfinite(y_hi) or y_hi <= y_lo:
            # widen a degenerate range
            delta = abs(y_lo) if y_lo != 0 else 1.0
            y_lo, y_hi = y_lo - 0.25 * delta, y_lo + 0.75 * delta

        # margin and zero-baseline policy
        margin = 0.05 * max(1e-12, (y_hi - y_lo))
        ymin = y_lo - margin
        ymax = y_hi + margin
        ax.set_ylim(ymin, ymax)

        if float(ymax) < 0.01:
            # tick formatter: scientific if needed, else fixed-point
            if np.nanmax(vals[finite]) >= 1e4 or np.nanmax(vals[finite]) <= 1e-3:
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            else:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

            # Axis border width
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                
        elif float(ymax) < 0.1:
            # tick formatter: scientific if needed, else fixed-point
            if np.nanmax(vals[finite]) >= 1e4 or np.nanmax(vals[finite]) <= 1e-3:
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            else:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

            # Axis border width
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                
        else:
            # tick formatter: scientific if needed, else fixed-point
            if np.nanmax(vals[finite]) >= 1e4 or np.nanmax(vals[finite]) <= 1e-3:
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            else:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

            # Axis border width
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

        # Safe filename
        # Combine annotation label + UniqueID to ensure uniqueness
        # ----------------------------------------------------
        annotation = None
        if uid in meta_lookup["UniqueID"].values:
            row = meta_lookup.loc[meta_lookup["UniqueID"] == uid]
            raw = str(row.iloc[0].get("Annotation", "")).strip()
            if raw and raw.lower() != "nan":
                annotation = raw

        # sanitize
        safe_ann = _sanitize_filename(annotation) if annotation else ""
        safe_uid = _sanitize_filename(uid)

        # filename logic:
        # If annotation exists → Annotation_UID
        # If not → UID only
        if safe_ann:
            base_name = f"{safe_ann}_{safe_uid}"
        else:
            base_name = safe_uid

        # safety trim
        if len(base_name) > 120:
            base_name = base_name[:120]

        fname = base_name

        ax.set_xlabel(None)
        ax.xaxis.label.set_visible(False)
       
        fig.tight_layout(pad=1.2)

        # Add full rectangular border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")

        fig.savefig(out_dir / f"{fname}_boxplot.png", dpi=dpi, facecolor=fig.get_facecolor(),
            bbox_inches="tight", pad_inches=0.1)
        # fig.savefig(out_dir / f"{fname}_boxplot_svg.svg", dpi=dpi, facecolor=fig.get_facecolor(),
        #     bbox_inches="tight", pad_inches=0.1)

        plt.close(fig)

    print(f"[Boxplots] Saved per-feature boxplots to: {out_dir}", flush=True)