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
from scipy.stats import kruskal, mannwhitneyu

from Stats.utils import prepare_output_dir
from Stats.figure_style import build_group_palette as _shared_build_group_palette, get_figure_style

import warnings
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

import matplotlib as mpl
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["mathtext.default"] = "regular" 

plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")

mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
plt.ioff()

# --- Group palette helper (shared shape across modules) ---
def _build_group_palette(groups_like, group_colors=None, group_order=None):
    return _shared_build_group_palette(groups_like, group_colors=group_colors, group_order=group_order)


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


def _load_dataset_preserve_nan(file_path, group_file):
    df = pd.read_csv(file_path)
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
        df_groups = pd.read_csv(group_file)
        if "Sample" not in df_groups.columns or "Group" not in df_groups.columns:
            raise ValueError(f"[boxplots] Invalid group file format: {group_file}")
    else:
        df_groups = pd.DataFrame({"Sample": sample_cols, "Group": "Unknown"})

    df_groups["Sample"] = df_groups["Sample"].astype(str).str.strip()
    df_groups["Group"] = df_groups["Group"].astype(str).str.strip()
    df_cols_lower = {str(c).lower(): c for c in sample_cols}
    matched = [df_cols_lower[s.lower()] for s in df_groups["Sample"] if s.lower() in df_cols_lower]

    if len(matched) == 0:
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


def _compute_pairwise_significance(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    y_str = y.astype(str).reset_index(drop=True)
    omnibus_records = []
    pairwise_records = []

    for uid in X.columns:
        series = pd.to_numeric(X[uid], errors="coerce").reset_index(drop=True)
        dfp = pd.DataFrame({"Group": y_str.values, "Value": series.values}).dropna(subset=["Value"])
        grouped = {
            str(group): vals.to_numpy(dtype=float)
            for group, vals in dfp.groupby("Group")["Value"]
            if len(vals) > 0
        }
        if len(grouped) < 2:
            results[uid] = {"omnibus_p": np.nan, "omnibus_fdr": np.nan, "pairs": []}
            continue

        try:
            _, omnibus_p = kruskal(*grouped.values())
        except Exception:
            omnibus_p = np.nan
        results[uid] = {"omnibus_p": omnibus_p, "omnibus_fdr": np.nan, "pairs": []}
        omnibus_records.append((uid, omnibus_p))

        groups = list(grouped.keys())
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                try:
                    _, pval = mannwhitneyu(
                        grouped[g1],
                        grouped[g2],
                        alternative="two-sided",
                        method="asymptotic",
                        use_continuity=False,
                    )
                except Exception:
                    pval = np.nan
                pairwise_records.append((uid, g1, g2, pval))

    omnibus_fdr = _bh_fdr(pd.Series({uid: p for uid, p in omnibus_records}, dtype=float))
    pairwise_p = pd.Series(
        [p for _, _, _, p in pairwise_records],
        index=pd.Index(range(len(pairwise_records))),
        dtype=float,
    )
    pairwise_fdr = _bh_fdr(pairwise_p)

    for uid, omnibus_p in omnibus_records:
        results[uid]["omnibus_fdr"] = float(omnibus_fdr.get(uid, np.nan))

    for idx, (uid, g1, g2, raw_p) in enumerate(pairwise_records):
        fdr_val = float(pairwise_fdr.loc[idx]) if pd.notna(pairwise_fdr.loc[idx]) else np.nan
        omnibus_p = results[uid]["omnibus_p"]
        omnibus_fdr_val = results[uid]["omnibus_fdr"]
        results[uid]["pairs"].append({
            "Group1": g1,
            "Group2": g2,
            "Raw_p_value": raw_p,
            "FDR_BH": fdr_val,
            "Significant": bool(
                pd.notna(fdr_val)
                and fdr_val < 0.05
                and pd.notna(omnibus_fdr_val)
                and omnibus_fdr_val < 0.05
                and pd.notna(omnibus_p)
            ),
        })

    return results


def _pairwise_significance_to_frame(pairwise_significance: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for uid, payload in pairwise_significance.items():
        omnibus_p = payload.get("omnibus_p", np.nan)
        omnibus_fdr = payload.get("omnibus_fdr", np.nan)
        pairs = payload.get("pairs", [])
        if not pairs:
            rows.append({
                "UniqueID": uid,
                "Omnibus_Kruskal_p": omnibus_p,
                "Omnibus_Kruskal_FDR_BH": omnibus_fdr,
                "Group1": "",
                "Group2": "",
                "Raw_p_value": np.nan,
                "FDR_BH": np.nan,
                "Significance": "",
            })
            continue
        for pair in pairs:
            fdr_val = pair["FDR_BH"]
            if fdr_val < 0.001:
                stars = "***"
            elif fdr_val < 0.01:
                stars = "**"
            elif pd.notna(fdr_val) and fdr_val < 0.05:
                stars = "*"
            else:
                stars = ""
            rows.append({
                "UniqueID": uid,
                "Omnibus_Kruskal_p": omnibus_p,
                "Omnibus_Kruskal_FDR_BH": omnibus_fdr,
                "Group1": pair["Group1"],
                "Group2": pair["Group2"],
                "Raw_p_value": pair["Raw_p_value"],
                "FDR_BH": fdr_val,
                "Significance": stars,
            })
    return pd.DataFrame(rows)


def _add_pairwise_brackets(ax, dfp: pd.DataFrame, groups: list[str], sig_pairs: list[dict[str, object]]) -> None:
    if not sig_pairs:
        return

    vals = pd.to_numeric(dfp["Value"], errors="coerce").to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return

    significant_pairs = [pair for pair in sig_pairs if pair.get("Significant", False)]
    if not significant_pairs:
        return

    current_ymin, current_ymax = ax.get_ylim()
    data_span = max(float(np.nanmax(finite) - np.nanmin(finite)), 1e-12)
    step = max(data_span * 0.045, abs(current_ymax) * 0.015, 0.015)
    y = current_ymax + step * 0.15
    x_pos = {g: i for i, g in enumerate(groups)}

    for pair in significant_pairs:
        g1, g2, fdr_val = pair["Group1"], pair["Group2"], pair["FDR_BH"]
        if g1 not in x_pos or g2 not in x_pos:
            continue
        x1, x2 = x_pos[g1], x_pos[g2]
        if x1 > x2:
            x1, x2 = x2, x1
        if fdr_val < 0.001:
            stars = "***"
        elif fdr_val < 0.01:
            stars = "**"
        else:
            stars = "*"
        bracket_top = y + step * 0.24
        ax.plot([x1, x1, x2, x2], [y, bracket_top, bracket_top, y], color="black", linewidth=1.0, clip_on=False, zorder=5)
        ax.text((x1 + x2) / 2, bracket_top + step * 0.08, stars, ha="center", va="bottom", fontsize=12, fontweight="bold", color="crimson", clip_on=False, zorder=6)
        y += step * 0.65

    ax.set_ylim(current_ymin, y + step * 0.18)


def run_boxplots(file_path, group_file, save_dir,
                 dpi=100,
                 strip=True,
                 jitter=True,
                 palette="husl",
                 group_colors=None, group_order=None,
                 publication_theme: bool = False):
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
    style = get_figure_style(publication_theme=publication_theme, dpi=dpi)

    print('[Boxplots] Running boxplots (species level)...', flush = True)
    # Load standardized tables
    # X: samples × features (columns = UniqueID)
    # y: sample groups (index aligned to X.index)
    # meta: feature metadata including "UniqueID", "Annotation", etc.
    X, y, feature_meta = _load_dataset_preserve_nan(file_path, group_file)

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
    pairwise_significance = _compute_pairwise_significance(X, y)
    _pairwise_significance_to_frame(pairwise_significance).to_csv(
        out_dir / "boxplot_pairwise_significance.csv",
        index=False,
        float_format="%.12g",
    )

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
        fig, ax = plt.subplots(figsize=(7.2, 5.0), facecolor="white")
        ax.set_facecolor("white")
        ax.grid(False)

        # room for rotated xticks + title
        fig.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.32)

        # Boxplot  (make box fill match point colors + transparency)
        sns.boxplot(
            data=dfp, x="Group", y="Value",
            palette=[color_map[g] for g in groups],   # use your palette
            order=groups,
            showfliers=False,
            width=0.48,
            ax=ax,                            
            linewidth=0.0,                            # no dark box edges
            whiskerprops=dict(color="gray", linewidth=0.6),
            capprops=dict(color="gray", linewidth=0.6),
            medianprops=dict(color="black", linewidth=0.75),
        )

        # Apply semi-transparent facecolor per box (to match points)
        box_alpha = style["box_alpha"]
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
                alpha=0.70, size=6.5 if publication_theme else 6, edgecolor="white", linewidth=0.4,
                ax=ax, zorder=2
            )

        title = _nice_label(uid, meta_lookup)
        ax.set_title(title, fontsize=style["title_size"], pad=12, fontweight="semibold")
        ax.set_xlabel(None)                 # clear any text
        ax.xaxis.label.set_visible(False)   # force-hide the label object
        ax.set_ylabel("Normalized peak intensity", fontsize=style["label_size"], labelpad=12)

        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=style["tick_size"])
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
                spine.set_linewidth(style["line_width"])
                
        elif float(ymax) < 0.1:
            # tick formatter: scientific if needed, else fixed-point
            if np.nanmax(vals[finite]) >= 1e4 or np.nanmax(vals[finite]) <= 1e-3:
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            else:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

            # Axis border width
            for spine in ax.spines.values():
                spine.set_linewidth(style["line_width"])
                
        else:
            # tick formatter: scientific if needed, else fixed-point
            if np.nanmax(vals[finite]) >= 1e4 or np.nanmax(vals[finite]) <= 1e-3:
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            else:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

            # Axis border width
            for spine in ax.spines.values():
                spine.set_linewidth(style["line_width"])

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
       
        fig.tight_layout(pad=1.35 if publication_theme else 1.4)

        # Add full rectangular border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(style["line_width"])
            spine.set_color("black")

        _add_pairwise_brackets(ax, dfp, groups, pairwise_significance.get(uid, {}).get("pairs", []))

        fig.savefig(out_dir / f"{fname}_boxplot.png", dpi=style["dpi"], facecolor=fig.get_facecolor(),
            bbox_inches="tight", pad_inches=0.1)
        fig.savefig(out_dir / f"{fname}_boxplot.svg", facecolor=fig.get_facecolor(),
            bbox_inches="tight", pad_inches=0.1)

        plt.close(fig)

    print(f"[Boxplots] Saved per-feature boxplots to: {out_dir}", flush=True)
