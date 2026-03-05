#TODO: fix design

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib import collections as mcoll
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

def _sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:."/\\|?*]', "_", str(s))


def _nice_label(uid: str, meta_lookup: pd.DataFrame) -> str:
    if meta_lookup is not None and not meta_lookup.empty:
        row = meta_lookup.loc[meta_lookup["UniqueID"] == uid]
        if not row.empty:
            for c in ("Annotation", "Headgroup", "Lipid Class"):
                if c in row.columns:
                    val = str(row.iloc[0][c]).strip()
                    if val and val.lower() != "nan":
                        return val
    if isinstance(uid, str) and "|" in uid:
        return uid.split("|")[-1].strip()
    return str(uid)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _resolve_palette(groups_series, group_colors=None, group_order=None):
    """Return (groups_order, color_map) from a groups Series/list and optional user palette/order."""
    natural = [str(g) for g in (groups_series.tolist() if hasattr(groups_series, "tolist") else list(groups_series))]
    unique_natural = list(dict.fromkeys(natural))  # first-seen order

    if group_order:
        groups = [g for g in group_order if g in unique_natural] + [g for g in unique_natural if g not in group_order]
    else:
        groups = unique_natural

    # Build color map
    cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
    color_map = {}
    for i, g in enumerate(groups):
        if group_colors and isinstance(group_colors.get(g), str) and group_colors[g]:
            color_map[g] = group_colors[g]
        else:
            color_map[g] = cycle[i % len(cycle)] if cycle else "#1f77b4"
    return groups, color_map

def run_violinplots(file_path, group_file, save_dir,
                    dpi=100,
                    strip=True,
                    jitter=True,
                    palette="husl",
                    group_colors=None, group_order=None):

    """
    Generate one PNG + SVG violin plot per feature (lipid), grouped by sample group.
    Uses the standardized loader: load_dataset(file_path, group_file).
    Saves under <save_dir>/Violinplots/
    """
    file_path = Path(file_path)
    save_dir = prepare_output_dir(Path(save_dir))
    out_dir = _ensure_dir(save_dir)
    plt.close('all')
    
    print('[Violin plots] Running violin plots (species level)...', flush = True)

    # X: samples × features (columns = UniqueID), y: group, feature_meta: annotations
    X, y, feature_meta = load_dataset(file_path, group_file)

    # Minimal metadata lookup
    meta_lookup = feature_meta if isinstance(feature_meta, pd.DataFrame) else pd.DataFrame()
    if not meta_lookup.empty and "UniqueID" in meta_lookup.columns:
        keep = [c for c in ("UniqueID", "Annotation", "Headgroup", "Lipid Class") if c in meta_lookup.columns]
        meta_lookup = meta_lookup[keep].copy()
    else:
        meta_lookup = pd.DataFrame(columns=["UniqueID", "Annotation", "Headgroup", "Lipid Class"])

    y = y.reset_index(drop=True)
    groups, color_map = _resolve_palette(y.astype(str), group_colors=group_colors, group_order=group_order)

    for uid in X.columns:
        s = X[uid]
        if s.isna().all():
            continue

        dfp = pd.DataFrame({"Group": y.values, "Value": s.values}).dropna(subset=["Value"])
        dfp["Group"] = pd.Categorical(dfp["Group"].astype(str), categories=groups, ordered=True)
        if dfp.empty:
            continue

        # --- figure sized to number of groups (prevents crammed violins) ---
        n_groups = len(groups)
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
        ax.set_facecolor("white")
        fig.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.32)   
        
        # === Gridline styling ===
        # Turn on horizontal gridlines (y-axis only)
        ax.yaxis.grid(True, color="#D3D3D3", linestyle="--", linewidth=0.5, alpha=0.7)

        # # Optional: disable vertical gridlines (cleaner for grouped plots)
        # ax.xaxis.grid(False) 

        # --- violin plot: lighter fill, no outlines, width-scaled, no negative tails ---
        sns.violinplot(
            data=dfp, x="Group", y="Value",
            order=groups,
            palette=[color_map[g] for g in groups],
            inner=None,             # no inner bars
            cut=0,                  # don’t extend beyond data range
            scale="width",          # all violins same max width
            linewidth=0,            # no black edge on 
            ax=ax
        )

        for pc in [c for c in ax.collections if isinstance(c, mcoll.PolyCollection)]:
            pc.set_zorder(1)  # behind points
    
        # Apply real alpha to each violin (seaborn doesn’t pass alpha to facecolor)
        # Set alpha for all violin halves that seaborn creates
        for pc in [c for c in ax.collections if isinstance(c, mcoll.PolyCollection)]:
            fc = pc.get_facecolor()
            if len(fc):  # fc is an (N,4) array
                r, g, b, _ = fc[0]
                pc.set_facecolor((r, g, b, 0.30))
                pc.set_edgecolor((r, g, b, 0.00))

        # Individual points
        if strip:
            sns.stripplot(
                data=dfp, x="Group", y="Value",
                order=groups,
                jitter=jitter, dodge=False, marker="o",
                palette=[color_map[g] for g in groups],
                alpha=0.6, size=4.5, edgecolor="white", linewidth=0.4,
                ax=ax, zorder=2
            )


        # Title/labels
        title = _nice_label(uid, meta_lookup)
        ax.set_title(title, fontsize=12, pad=12, fontweight="semibold")
        ax.set_xlabel(None)                 # clear any text
        ax.xaxis.label.set_visible(False)   # force-hide the label object
        ax.set_ylabel("Normalized peak intensity", fontsize=12, labelpad=12)

        # Tick labels: set ticks explicitly, then set labels+rotation (prevents warnings)
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_xlabel("")                         # clear text
        ax.xaxis.label.set_visible(False)         # hide the artist

        # Y-axis formatting (robust, no hard clamp to 0.5)
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
                
        elif float(ymax) < 0.1:
            # tick formatter: scientific if needed, else fixed-point
            if np.nanmax(vals[finite]) >= 1e4 or np.nanmax(vals[finite]) <= 1e-3:
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            else:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
                
        else:
            # tick formatter: scientific if needed, else fixed-point
            if np.nanmax(vals[finite]) >= 1e4 or np.nanmax(vals[finite]) <= 1e-3:
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            else:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        # Subtle axis border
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

        # Combine annotation (title) + UniqueID to ensure uniqueness
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
        
        # Tight crop + save
        fig.tight_layout(pad=1.1)

        # Add full rectangular border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")

        fig.savefig(out_dir / f"{fname}_violinplot.png", dpi=dpi, facecolor=fig.get_facecolor(),
            bbox_inches="tight", pad_inches=0.1)
        # fig.savefig(out_dir / f"{fname}_violinplot_svg.svg", dpi=dpi, facecolor=fig.get_facecolor(),
        #             bbox_inches="tight", pad_inches=0.1)

        plt.close(fig)

    print(f"[Violinplots] Saved per-feature violin plots to: {out_dir}", flush=True)
