# ------------------------------------------------------------
# Heatmaps for standardized Lipid workflow (UniqueID-based).
# - Loads via Stats.utils.load_dataset(file_path, group_file)
# - Autoscaling
# - Kruskal-Wallis + FDR feature ranking
# - Clustered heatmaps at multiple feature cutoffs
# - Also generates a "without outliers" version (z>4 filter)
# - Saves PNG + SVG, autoscaled data, rank table, outlier list
# ------------------------------------------------------------

import os
import warnings
from pathlib import Path
import re
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

from Stats.utils import load_dataset, prepare_output_dir
from Stats.figure_style import build_group_palette as _shared_build_group_palette, get_figure_style

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

import matplotlib as mpl
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["mathtext.default"] = "regular" 

plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")
plt.ioff()

# ==========================================================
# Utilities
# ==========================================================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _safe_savefig(fig, out_path: Path, **kwargs):
    """
    Save a figure after forcing parent directory creation and checking
    for Windows path-length problems.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_str = str(out_path)

    # Conservative limit for Windows. Real limit depends on settings,
    # but PIL often fails before you get a useful error.
    if os.name == "nt" and len(out_str) >= 240:
        raise OSError(
            f"Output path too long for Windows/PIL ({len(out_str)} chars):\n{out_str}\n"
            "Shorten the output folder path or file name."
        )

    fig.savefig(out_path, **kwargs)

def _autoscale_df(X: pd.DataFrame, save_dir: Path, filename: str) -> pd.DataFrame:
    """Autoscale (z-score across samples) and save the scaled matrix."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X.values),
        index=X.index,
        columns=X.columns,
    )
    X_scaled.to_csv(save_dir / filename, encoding="utf-8-sig")
    return X_scaled


def _kruskal_rank_features(X_scaled: pd.DataFrame, groups: pd.Series, save_dir: Path, filename: str) -> pd.DataFrame:
    """
    Kruskal-Wallis rank test for each feature across groups + FDR.
    Returns a sorted dataframe (by Adjusted_P) and writes it to disk.
    """
    groups = groups.astype(str)
    unique_groups = groups.unique()

    # If there's only 1 group, the omnibus test is not defined - return neutral p-values.
    if len(unique_groups) < 2:
        out = pd.DataFrame({"Feature": X_scaled.columns, "P_Value": 1.0})
        out["Adjusted_P_Value"] = 1.0
        out = out.sort_values("Adjusted_P_Value").reset_index(drop=True)
        out.to_csv(save_dir / filename, index=False, encoding="utf-8-sig")
        return out

    pvals = []
    for feat in X_scaled.columns:
        by_group = [
            pd.to_numeric(X_scaled.loc[groups == g, feat], errors="coerce").dropna().values
            for g in unique_groups
        ]
        valid_groups = [v for v in by_group if len(v) > 0]
        if len(valid_groups) < 2:
            p = 1.0
        else:
            try:
                p = float(kruskal(*valid_groups).pvalue)
            except Exception:
                p = 1.0
        pvals.append(p)

    rank_df = pd.DataFrame({"Feature": X_scaled.columns, "P_Value": pvals})
    rank_df["Adjusted_P_Value"] = multipletests(rank_df["P_Value"].values, method="fdr_bh")[1]
    rank_df = rank_df.sort_values("Adjusted_P_Value").reset_index(drop=True)
    rank_df.to_csv(save_dir / filename, index=False, encoding="utf-8-sig")
    return rank_df

def _sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:."/\\|?*]', "_", str(s))

def _build_uid_to_class(feature_meta: pd.DataFrame) -> dict:
    """
    Map UniqueID -> Lipid Class using feature_meta if present.
    Falls back to 'Unknown' if missing.
    """
    if not isinstance(feature_meta, pd.DataFrame) or feature_meta.empty:
        return {}

    # Try to find a lipid class column robustly
    class_col = None
    for c in feature_meta.columns:
        if str(c).strip().lower() in {"lipid class", "lipid_class", "class"}:
            class_col = c
            break
    if class_col is None or "UniqueID" not in feature_meta.columns:
        return {}

    tmp = feature_meta[["UniqueID", class_col]].copy()
    tmp["UniqueID"] = tmp["UniqueID"].astype(str).str.strip()
    tmp[class_col] = tmp[class_col].astype(str).str.strip()
    tmp[class_col] = tmp[class_col].replace({"nan": "", "NaN": "", "None": "", "<NA>": "", "NA": ""})
    tmp.loc[tmp[class_col] == "", class_col] = "Unknown"

    # Drop duplicates so first occurrence wins
    tmp = tmp.drop_duplicates("UniqueID")
    return dict(zip(tmp["UniqueID"], tmp[class_col]))

def _default_class_to_category() -> dict:
    """
    Define your lipid ontology buckets.
    Edit this list once, and all category heatmaps follow it.
    """
    return {
        # Fatty acyls
        "FA": {"CAR", "CoA", "FA", "FAG", "FAHFA", "FAL", "FOH", "HC", "NAx", "NAE", "NAT", "WE"},

        # Glycerolipids
        "GL": {"MG", "DG", "TG", "DGCC", "Hex2MG", "Hex2DG","DGTA", "DGTS", "GlcADG", "HexDG", "HexMG", "SQDG", "SQMG"},

        # Glycerophospholipids
        "GP": {"PC", "PE", "PEth", "PG", "PI", "PS", "PA", "CL", "MLCL", "DLCL",
               "LPC", "LPE", "LPG", "LPI", "LPS", "LPA",
               "BMP", "CDP-DG", "Glc-GP", "GP", "PIM", "PIP", "PnC", "PnE", "PPA"},

        # Sphingolipids
        "SL": {"Cer", "ACer", "CerP", "GlcCer", "HexCer", 
               "SM", "LSM", 
               "HexSPB", "SPB", "SBPB", "SulfateHexSPB",
               "MIPC", "M(IP)2C",
               "PE-Cer", "PI-Cer", "SCer",
               },

        # Sterols 
        "ST": {"ST", "CE"},

        # Other
        "Other": {"PK", "PR", "SL"},
    }

def _invert_class_to_category(class_to_category: dict) -> dict:
    """
    Build class -> category lookup.
    """
    out = {}
    for cat, classes in (class_to_category or {}).items():
        for cls in classes:
            out[str(cls).strip()] = str(cat).strip()
    return out

def _map_class_to_category(lipid_class: str, class_to_cat_lookup: dict) -> str:
    c = str(lipid_class).strip()
    if c in {"", "nan", "NaN", "None", "<NA>", "NA"}:
        return "Unknown"
    return class_to_cat_lookup.get(c, "Other")
def _find_meta_col(feature_meta: pd.DataFrame, candidates: set[str]) -> Optional[str]:
    if not isinstance(feature_meta, pd.DataFrame) or feature_meta.empty:
        return None
    for c in feature_meta.columns:
        if str(c).strip().lower() in candidates:
            return c
    return None

def _build_uid_to_carbons(feature_meta: pd.DataFrame) -> dict[str, int]:
    """
    Map UniqueID -> integer carbon count.
    Expects a column like "Number of carbons in fatty acyls" in feature_meta.
    """
    if not isinstance(feature_meta, pd.DataFrame) or feature_meta.empty or "UniqueID" not in feature_meta.columns:
        return {}

    carbon_col = _find_meta_col(
        feature_meta,
        {
            "number of carbons in fatty acyls",
            "carbons",
            "carbon",
            "total carbons",
            "ncarbons",
            "n_carbons",
        },
    )
    if carbon_col is None:
        return {}

    tmp = feature_meta[["UniqueID", carbon_col]].copy()
    tmp["UniqueID"] = tmp["UniqueID"].astype(str).str.strip()
    tmp[carbon_col] = pd.to_numeric(tmp[carbon_col], errors="coerce").round().astype("Int64")
    tmp = tmp.dropna(subset=[carbon_col]).drop_duplicates("UniqueID")

    return dict(zip(tmp["UniqueID"], tmp[carbon_col].astype(int)))

def _aggregate_by_carbons(
    X: pd.DataFrame,
    uid_to_carbons: dict[str, int],
    agg: str = "sum",          # "sum" or "mean"
    min_features_per_bin: int = 3,
) -> pd.DataFrame:
    """
    Return samples × carbon_bins dataframe.
    """
    # Build carbon_bin -> list of uids present in X
    bin_to_uids: dict[int, list[str]] = {}
    for uid in X.columns.astype(str):
        u = str(uid).strip()
        if u not in uid_to_carbons:
            continue
        b = int(uid_to_carbons[u])
        bin_to_uids.setdefault(b, []).append(u)

    # Aggregate per bin
    out = {}
    for b in sorted(bin_to_uids.keys()):
        uids = [u for u in bin_to_uids[b] if u in X.columns]
        if len(uids) < min_features_per_bin:
            continue
        if agg == "mean":
            out[str(b)] = X.loc[:, uids].mean(axis=1)
        else:
            out[str(b)] = X.loc[:, uids].sum(axis=1)

    if not out:
        return pd.DataFrame(index=X.index)

    Xb = pd.DataFrame(out, index=X.index)
    # Keep columns in numeric order (as strings)
    Xb = Xb.reindex(sorted(Xb.columns, key=lambda s: int(s)), axis=1)
    return Xb

def _remove_extreme_feature_outliers(X: pd.DataFrame, z_thresh: float = 4.0) -> tuple[pd.DataFrame, list]:

    """
    Remove features where ANY sample z-score exceeds |z_thresh|.
    Z-scores computed per feature (across samples).
    """
    # z-score each column by itself
    z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)

    to_drop = z.columns[(z.abs() > z_thresh).any(axis=0)].tolist()
    X_clean = X.drop(columns=to_drop) if to_drop else X.copy()
    return X_clean, to_drop

def _group_colorbar(groups: pd.Series, group_colors=None, group_order=None):
    groups = groups.astype(str)
    legend_order, color_map = _shared_build_group_palette(groups, group_colors=group_colors, group_order=group_order)
    col_colors = [color_map[g] for g in groups]
    legend_handles = [
        plt.matplotlib.patches.Patch(color=color_map[g], label=g) for g in legend_order
    ]
    return col_colors, legend_handles


def _ordered_groups_present(groups: pd.Series, group_order=None) -> list[str]:
    present = pd.unique(groups.astype(str)).tolist()
    if not group_order:
        return present
    ordered = [g for g in group_order if g in present]
    rest = [g for g in present if g not in ordered]
    return ordered + rest


def _group_boundaries_from_sequence(groups: List[str]) -> list[int]:
    labels = [str(g) for g in groups]
    boundaries: list[int] = []
    for idx in range(1, len(labels)):
        if labels[idx] != labels[idx - 1]:
            boundaries.append(idx)
    return boundaries


def _draw_group_separators(ax, boundaries: list[int], line_width: float = 1.8, alpha: float = 0.7, color: str = "#5f5f5f"):
    for boundary in boundaries:
        ax.axvline(boundary, color=color, linewidth=line_width, alpha=alpha, zorder=10)


def _order_samples_by_group(X: pd.DataFrame, y: pd.Series, group_order=None) -> tuple[pd.DataFrame, pd.Series]:
    ordered_groups = _ordered_groups_present(y, group_order=group_order)
    sample_rank = {sample: i for i, sample in enumerate(X.index.astype(str).tolist())}
    sort_df = pd.DataFrame({
        "Sample": X.index.astype(str),
        "Group": y.reindex(X.index).astype(str).values,
    })
    sort_df["_group_rank"] = sort_df["Group"].map({g: i for i, g in enumerate(ordered_groups)}).fillna(len(ordered_groups)).astype(int)
    sort_df["_sample_rank"] = sort_df["Sample"].map(sample_rank).fillna(len(sample_rank)).astype(int)
    ordered_samples = sort_df.sort_values(["_group_rank", "_sample_rank"], kind="stable")["Sample"].tolist()
    return X.loc[ordered_samples], y.loc[ordered_samples]


def _dynamic_figsize(n_features: int, n_samples: int, row_height_inch: float = 0.25, top_bottom_margin_inch: float = 2.5) -> tuple:
    """
    Compute figure size dynamically:
      - width scales with number of samples
      - height scales linearly with number of features (fixed per-row height)
    """
    width = max(6, 0.35 * n_samples)
    height = (n_features * row_height_inch) + top_bottom_margin_inch
    return (width, height)


def _annotation_width_boost(labels: list[str]) -> float:
    """Return extra figure width for long annotation labels."""
    if not labels:
        return 0.0
    max_len = max(len(str(label)) for label in labels)
    if max_len <= 24:
        return 0.0
    return min(12.0, 0.11 * (max_len - 24))


def _build_uid_to_annotation(feature_meta: pd.DataFrame) -> dict[str, str]:
    if not isinstance(feature_meta, pd.DataFrame) or feature_meta.empty or "UniqueID" not in feature_meta.columns:
        return {}
    ann_col = _find_meta_col(feature_meta, {"annotation", "name", "lipid", "noabbrev"})
    if ann_col is None:
        return {}
    tmp = feature_meta[["UniqueID", ann_col]].copy()
    tmp["UniqueID"] = tmp["UniqueID"].astype(str).str.strip()
    tmp[ann_col] = tmp[ann_col].astype(str).str.strip()
    tmp = tmp.drop_duplicates("UniqueID")
    return dict(zip(tmp["UniqueID"], tmp[ann_col]))


def get_available_annotations(file_path: str) -> List[str]:
    try:
        df = pd.read_csv(file_path, low_memory=False, usecols=["Annotation"])
    except Exception:
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception:
            return []
    if "Annotation" not in df.columns:
        return []
    annotations = df["Annotation"].dropna().astype(str).str.strip()
    annotations = annotations[annotations.ne("") & ~annotations.str.lower().eq("nan")]
    return sorted(annotations.unique().tolist(), key=str.casefold)


def _plot_selected_lipid_heatmap(
    H: pd.DataFrame,
    y_groups: pd.Series,
    out_png: Path,
    out_svg: Path,
    group_colors=None,
    group_order=None,
    style: Optional[dict] = None,
):
    style = style or get_figure_style(False, 100)
    if H.empty:
        return

    fig_w = max(7.5, 0.4 * H.shape[1] + 3.2)
    fig_h = max(4.5, 0.45 * H.shape[0] + 2.6)

    with mpl.rc_context({
        "font.family": style["font_family"],
        "font.size": style.get("base_font_size", style["label_size"]),
        "axes.titlesize": style["title_size"],
        "axes.labelsize": style["label_size"],
        "xtick.labelsize": style["tick_size"],
        "ytick.labelsize": style["tick_size"],
    }):
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        ax = fig.add_subplot(111)

        vals = H.to_numpy(dtype=float)
        vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        sns.heatmap(
            H,
            ax=ax,
            cmap=style["diverging_cmap"],
            vmin=-vmax,
            vmax=vmax,
            cbar_kws={"label": "Standardized abundance", "shrink": 0.42},
            linewidths=0.4,
            linecolor="white",
        )

        ax.set_xlabel("Samples", fontsize=style["label_size"], labelpad=12)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=55, labelsize=max(style["tick_size"] - 2, 7))
        plt.setp(ax.get_xticklabels(), ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        col_colors, _legend_handles = _group_colorbar(y_groups, group_colors=group_colors, group_order=group_order)
        ordered_groups = _ordered_groups_present(y_groups, group_order=group_order)
        group_counts = y_groups.astype(str).value_counts()
        boundary_positions = []
        centers = []
        start = 0
        for group in ordered_groups:
            count = int(group_counts.get(group, 0))
            if count <= 0:
                continue
            end = start + count
            centers.append(((start + end - 1) / 2.0, group))
            if end < H.shape[1]:
                boundary_positions.append(end)
            start = end
        _draw_group_separators(ax, boundary_positions, line_width=1.8, alpha=0.75)
        fig.subplots_adjust(left=0.24, right=0.88, bottom=0.2, top=0.82)
        fig.canvas.draw()
        heatmap_pos = ax.get_position()
        fig_height_inch = fig.get_size_inches()[1]
        label_height_rel = 0.16 / fig_height_inch
        bar_height_rel = 0.10 / fig_height_inch
        gap_rel = 0.03 / fig_height_inch
        ax_bar = fig.add_axes([
            heatmap_pos.x0,
            heatmap_pos.y1 + gap_rel,
            heatmap_pos.width,
            bar_height_rel,
        ])
        ax_labels = fig.add_axes([
            heatmap_pos.x0,
            heatmap_pos.y1 + gap_rel + bar_height_rel + gap_rel,
            heatmap_pos.width,
            label_height_rel,
        ])
        rgba_row = np.array([mpl.colors.to_rgba(c) for c in col_colors], dtype=float).reshape(1, len(col_colors), 4)
        ax_bar.imshow(rgba_row, aspect="auto", interpolation="none")
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("white")
        for i in range(1, len(col_colors)):
            ax_bar.axvline(i - 0.5, color="white", linewidth=1)
        _draw_group_separators(ax_bar, boundary_positions, line_width=2.0, alpha=0.95, color="white")
        ax_labels.set_xticks([])
        ax_labels.set_yticks([])
        ax_labels.set_frame_on(False)
        ax_labels.set_xlim(-0.5, H.shape[1] - 0.5)
        ax_bar.set_xlim(-0.5, H.shape[1] - 0.5)
        for x_pos, group in centers:
            ax_labels.text(
                x_pos,
                0.0,
                str(group),
                ha="center",
                va="bottom",
                fontsize=max(style["tick_size"] - 1, 8),
                fontweight="semibold",
            )
        _safe_savefig(fig, out_png, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.2)
        _safe_savefig(fig, out_svg, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)


def _plot_all_features_by_class_heatmap(
    X_scaled: pd.DataFrame,
    y_groups: pd.Series,
    feature_meta: pd.DataFrame,
    save_dir: Path,
    basename: str,
    group_colors=None,
    group_order=None,
    style: Optional[dict] = None,
):
    style = style or get_figure_style(False, 100)
    if X_scaled.empty:
        return

    uid_to_class = _build_uid_to_class(feature_meta)
    uid_to_annotation = _build_uid_to_annotation(feature_meta)

    ordered_samples_X, ordered_groups = _order_samples_by_group(X_scaled, y_groups, group_order=group_order)
    feature_rows = []
    for uid in ordered_samples_X.columns.astype(str):
        lipid_class = str(uid_to_class.get(str(uid).strip(), "Unknown")).strip() or "Unknown"
        annotation = str(uid_to_annotation.get(str(uid).strip(), str(uid))).strip() or str(uid)
        feature_rows.append({
            "UniqueID": str(uid),
            "Class": lipid_class,
            "Annotation": annotation,
        })
    feature_order_df = pd.DataFrame(feature_rows)
    feature_order_df["_class_sort"] = feature_order_df["Class"].astype(str).str.casefold()
    feature_order_df["_annotation_sort"] = feature_order_df["Annotation"].astype(str).str.casefold()
    feature_order_df["_uid_sort"] = feature_order_df["UniqueID"].astype(str).str.casefold()
    feature_order_df = feature_order_df.sort_values(
        ["_class_sort", "_annotation_sort", "_uid_sort"],
        kind="stable",
    ).reset_index(drop=True)

    ordered_features = feature_order_df["UniqueID"].tolist()
    H = ordered_samples_X.loc[:, ordered_features].T

    row_height = 0.065
    fig_w = max(8.5, 0.4 * H.shape[1] + 3.8)
    fig_h = max(6.0, row_height * H.shape[0] + 3.0)

    with mpl.rc_context({
        "font.family": style["font_family"],
        "font.size": style.get("base_font_size", style["label_size"]),
        "axes.titlesize": style["title_size"],
        "axes.labelsize": style["label_size"],
        "xtick.labelsize": style["tick_size"],
        "ytick.labelsize": max(style["tick_size"] - 1, 7),
    }):
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        ax = fig.add_subplot(111)

        vals = H.to_numpy(dtype=float)
        vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        sns.heatmap(
            H,
            ax=ax,
            cmap=style["diverging_cmap"],
            vmin=-vmax,
            vmax=vmax,
            cbar_kws={"label": "Standardized abundance", "shrink": 0.42},
            linewidths=0.0,
            xticklabels=True,
            yticklabels=False,
        )

        ax.set_xlabel("Samples", fontsize=style["label_size"], labelpad=12)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=55, labelsize=max(style["tick_size"] - 2, 7))
        plt.setp(ax.get_xticklabels(), ha="right")

        col_colors, _legend_handles = _group_colorbar(ordered_groups, group_colors=group_colors, group_order=group_order)
        ordered_group_names = _ordered_groups_present(ordered_groups, group_order=group_order)
        group_counts = ordered_groups.astype(str).value_counts()
        boundary_positions = []
        sample_centers = []
        start = 0
        for group in ordered_group_names:
            count = int(group_counts.get(group, 0))
            if count <= 0:
                continue
            end = start + count
            sample_centers.append(((start + end - 1) / 2.0, group))
            if end < H.shape[1]:
                boundary_positions.append(end)
            start = end
        _draw_group_separators(ax, boundary_positions, line_width=1.9, alpha=0.75)

        class_boundaries = []
        class_labels = []
        start_idx = 0
        classes = feature_order_df["Class"].tolist()
        for idx, cls in enumerate(classes):
            if idx == 0 or cls != classes[idx - 1]:
                start_idx = idx
            if idx == len(classes) - 1 or cls != classes[idx + 1]:
                end_idx = idx + 1
                class_boundaries.append(end_idx)
                class_labels.append({
                    "y": (start_idx + end_idx - 1) / 2.0,
                    "cls": cls,
                    "span": end_idx - start_idx,
                })
        for boundary in class_boundaries[:-1]:
            ax.axhline(boundary, color="black", linewidth=1.4, alpha=0.55)
        x_pos = -0.02
        filtered_labels = []
        last_y = None
        for item in class_labels:
            if item["span"] < 4:
                continue
            if last_y is not None and abs(item["y"] - last_y) < 8:
                continue
            filtered_labels.append(item)
            last_y = item["y"]
        for item in filtered_labels:
            ax.text(
                x_pos,
                item["y"] + 0.5,
                str(item["cls"]),
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=max(style["tick_size"] - 1, 8),
                fontweight="semibold",
            )
        fig.subplots_adjust(left=0.16, right=0.88, bottom=0.16, top=0.82)
        fig.canvas.draw()
        heatmap_pos = ax.get_position()
        fig_height_inch = fig.get_size_inches()[1]
        label_height_rel = 0.16 / fig_height_inch
        bar_height_rel = 0.10 / fig_height_inch
        gap_rel = 0.03 / fig_height_inch
        ax_bar = fig.add_axes([
            heatmap_pos.x0,
            heatmap_pos.y1 + gap_rel,
            heatmap_pos.width,
            bar_height_rel,
        ])
        ax_labels = fig.add_axes([
            heatmap_pos.x0,
            heatmap_pos.y1 + gap_rel + bar_height_rel + gap_rel,
            heatmap_pos.width,
            label_height_rel,
        ])
        rgba_row = np.array([mpl.colors.to_rgba(c) for c in col_colors], dtype=float).reshape(1, len(col_colors), 4)
        ax_bar.imshow(rgba_row, aspect="auto", interpolation="none")
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("white")
        for i in range(1, len(col_colors)):
            ax_bar.axvline(i - 0.5, color="white", linewidth=1)
        _draw_group_separators(ax_bar, boundary_positions, line_width=2.1, alpha=0.95, color="white")
        ax_labels.set_xticks([])
        ax_labels.set_yticks([])
        ax_labels.set_frame_on(False)
        ax_labels.set_xlim(-0.5, H.shape[1] - 0.5)
        ax_bar.set_xlim(-0.5, H.shape[1] - 0.5)
        for x_pos, group in sample_centers:
            ax_labels.text(
                x_pos,
                0.0,
                str(group),
                ha="center",
                va="bottom",
                fontsize=max(style["tick_size"] - 1, 8),
                fontweight="semibold",
            )
        fig.suptitle("All detected features by lipid class", fontsize=style["title_size"], fontweight="bold", y=0.985)
        png_path = save_dir / f"{basename}.png"
        svg_path = save_dir / f"{basename}.svg"
        order_csv = save_dir / f"{basename}_feature_order.csv"
        feature_order_df.drop(columns=["_class_sort", "_annotation_sort", "_uid_sort"]).to_csv(order_csv, index=False, encoding="utf-8-sig")
        _safe_savefig(fig, png_path, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.2)
        _safe_savefig(fig, svg_path, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)

def _plot_heatmap(
    X_scaled: pd.DataFrame,
    y_groups: pd.Series,
    feature_ids_sorted: list,
    feature_meta: pd.DataFrame,
    top_k: int,
    save_dir: Path,
    basename: str,
    group_colors=None,
    group_order=None,
    style: Optional[dict] = None,
    row_label_map: Optional[Dict[str, str]] = None,
):
    style = style or get_figure_style(False, 100)

    """
    Create clustered heatmap for the top_k features from 'feature_ids_sorted'.
    Saves:
        - one heatmap with Annotation labels ("_annotated")
        - one heatmap with UniqueID labels ("_uniqueIDs")
    """
    selected_ids = feature_ids_sorted[:top_k]
    if not selected_ids:
        return

    # Selected data (samples × features)
    Xsel = X_scaled[selected_ids]
    H = Xsel.T  # rows=features, cols=samples
    H = H + np.random.normal(0, 1e-6, H.shape)  # avoid identical-row clustering issues

    # Column colors
    col_colors, legend_handles = _group_colorbar(y_groups, group_colors=group_colors, group_order=group_order)
    for patch in legend_handles:
        patch.set_edgecolor("white")

    # ==========================================================
    # Annotation mapping (robust, like VIP code)
    # ==========================================================
    annotations = []
    row_label_map = row_label_map or {}
    if isinstance(feature_meta, pd.DataFrame) and "Annotation" in feature_meta.columns:
        uid_to_annotation = dict(zip(
            feature_meta["UniqueID"].astype(str).str.strip(),
            feature_meta["Annotation"].astype(str).str.strip()
        ))
        for uid in Xsel.columns:
            ann = row_label_map.get(str(uid), uid_to_annotation.get(str(uid).strip(), str(uid)))
            annotations.append(ann)
    else:
        annotations = [str(row_label_map.get(str(uid), str(uid))) for uid in Xsel.columns]

    # Dynamic figure sizing
    fig_w, fig_h = _dynamic_figsize(n_features=H.shape[0], n_samples=H.shape[1])
    annot_fig_w = fig_w + _annotation_width_boost(annotations)

    # ==========================================================
    # 1) Annotated version (Annotation names on Y-axis)
    # ==========================================================
    H_annot = H.copy()
    H_annot.index = annotations
    
    cg_annot = sns.clustermap(
        H_annot,
        cmap=style["diverging_cmap"],
        linewidths=0.4,
        figsize=(annot_fig_w, fig_h),
        row_cluster=True,
        col_cluster=True,
        col_colors=col_colors,  # group color bar
        method="ward",
        metric="euclidean",
        dendrogram_ratio=(0.08, 0.08),     # side, top dendograms
        cbar_kws={"shrink": 0.5, "label": "\nStandardized\nValues"},
        cbar_pos=(1.24, 0.1, 0.03, 0.3),
    )

    # --- Fix top colorbar (group bar) height ---
    fig = cg_annot.fig
    ax_top = cg_annot.ax_col_colors
    ax_heatmap = cg_annot.ax_heatmap

    fig_height_inch = fig.get_size_inches()[1]
    fixed_bar_height_inch = 0.15       # same as bottom strip, choose a constant height in inches
    fixed_bar_height_rel = fixed_bar_height_inch / fig_height_inch

    # Align the bar flush with the top of the heatmap
    heatmap_pos = ax_heatmap.get_position()
    ax_top.set_position([
        heatmap_pos.x0,
        heatmap_pos.y1 + top_k*0.00004,      # small margin (0.002 = ~1 mm)
        heatmap_pos.width,
        fixed_bar_height_rel
    ])
    
    # adjust dendrogram position
    ax_dendro = cg_annot.ax_col_dendrogram
    pos = ax_dendro.get_position()

    # Move it slightly up or down
    if top_k <=5:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.028,  # shift upward by 2.8% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k == 10:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.013,  # shift upward by 1.2% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=15 and top_k >10:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.006,  # shift upward by 0.6% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=20:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.00022,  # shift upward by 0.022% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=25:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 - 0.000017,  # shift downard by 0.001% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=30:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 - 0.007,  # shift downward by 0.7% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=40:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 - 0.011,  # shift downward by 1.1% of figure height
            pos.width,
            pos.height
        ])
    
    elif top_k >40:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 - 0.0128,  # shift downward by 1.28% of figure height
            pos.width,
            pos.height
        ])
    
    # --- Duplicate the top group colorbar exactly (with white border) ------------------------
    fig = cg_annot.fig
    ax_heatmap = cg_annot.ax_heatmap
    ax_top = cg_annot.ax_col_colors

    # Extract the top colorbar QuadMesh
    quadmesh = ax_top.collections[0]

    # Get facecolors and edge properties
    facecolors = quadmesh.get_facecolors()
    edgecolor = quadmesh.get_edgecolor()
    linewidth = quadmesh.get_linewidths()[0] if len(quadmesh.get_linewidths()) else 0.8

    # Force white borders if transparent/black
    if edgecolor is None or np.allclose(edgecolor, 0):
        edgecolor = "white"

    # Build a 1×N×4 RGBA row
    n = len(facecolors)
    rgba_row = facecolors.reshape(1, n, 4)

    # Compute position just below the heatmap
    pos = ax_heatmap.get_position()

    ax_bottom = fig.add_axes([
        ax_top.get_position().x0,
        pos.y0 - fixed_bar_height_rel - 0.001,  # spacing below heatmap
        ax_top.get_position().width,
        fixed_bar_height_rel
    ])

    # Draw identical color strip
    ax_bottom.imshow(rgba_row, aspect="auto", interpolation="none")

    # Add white border on all sides
    for spine in ax_bottom.spines.values():
        spine.set_visible(True)
        spine.set_color(edgecolor)
        spine.set_linewidth(linewidth)
        
    # Add white vertical lines between samples
    for i in range(1, n):
        ax_bottom.axvline(i - 0.5, color="white", linewidth=1)

    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])

    reordered_groups = y_groups.iloc[cg_annot.dendrogram_col.reordered_ind].astype(str).tolist()
    boundary_positions = _group_boundaries_from_sequence(reordered_groups)
    _draw_group_separators(cg_annot.ax_heatmap, boundary_positions, line_width=1.9, alpha=0.8)
    _draw_group_separators(ax_top, boundary_positions, line_width=2.1, alpha=0.95, color="white")
    _draw_group_separators(ax_bottom, boundary_positions, line_width=2.1, alpha=0.95, color="white")

    # --- Make x-ticks longer to push labels down naturally ---
    ax_heatmap.tick_params(axis="x", which="both", length=14)  # increase from default (~3-4)

    #---------------------------------------------------------------------------------------------

    # Add titles
    cg_annot.ax_heatmap.set_xlabel("Samples", fontsize=style["label_size"], labelpad=12)
    cg_annot.ax_heatmap.set_ylabel("", fontsize=14, labelpad=12)
    
    if top_k <=5:
        cg_annot.fig.suptitle(f"Clustered Heatmap (Top {top_k} by Kruskal-FDR)", fontsize=style["title_size"], weight="bold", y=1.08)
    elif top_k <=15:
        cg_annot.fig.suptitle(f"Clustered Heatmap (Top {top_k} by Kruskal-FDR)", fontsize=style["title_size"], weight="bold", y=1.04)
    elif top_k <=25:
        cg_annot.fig.suptitle(f"Clustered Heatmap (Top {top_k} by Kruskal-FDR)", fontsize=style["title_size"], weight="bold", y=1.02)
    elif top_k >25:
        cg_annot.fig.suptitle(f"Clustered Heatmap (Top {top_k} by Kruskal-FDR)", fontsize=style["title_size"], weight="bold", y=1.005)
        
    plt.setp(cg_annot.ax_heatmap.get_xticklabels(), rotation=55, ha="right", fontsize=max(style["tick_size"] - 2, 7))
    plt.setp(cg_annot.ax_heatmap.get_yticklabels(), rotation=0, ha="left", fontsize=max(style["tick_size"] - 2, 8))
    
    # --- Ensure all annotation labels are visible (Seaborn may hide half automatically) ---
    for label in cg_annot.ax_heatmap.get_yticklabels():
        label.set_visible(True)
        
    n_groups = len(legend_handles)
    ncol = 2 if n_groups > 8 else 1

    if top_k <=10:   
        if ncol == 1:
            # Attach legend to the full figure — top center, above title
            cg_annot.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.62, 1.05),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )
        if ncol == 2:
            # Attach legend to the full figure — top center, above title
            cg_annot.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.78, 1.05),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )
            
    if top_k ==15: 
        if ncol == 1:  
            # Attach legend to the full figure — top center, above title
            cg_annot.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.62, 0.98),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )
        if ncol == 2:  
            # Attach legend to the full figure — top center, above title
            cg_annot.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.78, 0.98),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )
    
    if top_k >=20:   
        if ncol == 1: 
            # Attach legend to the full figure — top center, above title
            cg_annot.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.62, 0.9),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )
        if ncol == 2: 
            # Attach legend to the full figure — top center, above title
            cg_annot.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.78, 0.9),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )

    annot_png = save_dir / f"{basename}_t{top_k}_ann.png"
    annot_svg = save_dir / f"{basename}_t{top_k}_ann.svg"

    _safe_savefig(cg_annot.fig, annot_png, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.2)
    _safe_savefig(cg_annot.fig, annot_svg, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.2)
    plt.close(cg_annot.fig)



    # ==========================================================
    # 2) UniqueID-only version
    # ==========================================================
    H_uid = H.copy()
    H_uid.index = [str(row_label_map.get(str(fid).strip(), str(fid).strip())) for fid in selected_ids]

    cg_uid = sns.clustermap(
        H_uid,
        cmap=style["diverging_cmap"],
        linewidths=0.4,
        figsize=(fig_w, fig_h),
        row_cluster=True,
        col_cluster=True,
        col_colors=col_colors,  # group colorbar
        method="ward",
        metric="euclidean",
        dendrogram_ratio=(0.10, 0.10),
        cbar_kws={"shrink": 0.5, "label": "\nStandardized\nValues"},
        cbar_pos=(1.24, 0.1, 0.03, 0.3),
    )
    
    # --- Fix top colorbar (group bar) height ---
    fig = cg_uid.fig
    ax_top = cg_uid.ax_col_colors
    ax_heatmap = cg_uid.ax_heatmap

    fig_height_inch = fig.get_size_inches()[1]
    fixed_bar_height_inch = 0.15       # same as bottom strip, choose a constant height in inches
    fixed_bar_height_rel = fixed_bar_height_inch / fig_height_inch

    # Align the bar flush with the top of the heatmap
    heatmap_pos = ax_heatmap.get_position()
    ax_top.set_position([
        heatmap_pos.x0,
        heatmap_pos.y1 + top_k*0.00004,      # small margin (0.002 = ~1 mm)
        heatmap_pos.width,
        fixed_bar_height_rel
    ])
    
    # adjust dendrogram position
    ax_dendro = cg_uid.ax_col_dendrogram
    pos = ax_dendro.get_position()

    # Move it slightly up or down
    if top_k <=5:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.028,  # shift upward by 2.8% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k == 10:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.013,  # shift upward by 1.2% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=15 and top_k >10:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.006,  # shift upward by 0.6% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=20:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.00022,  # shift upward by 0.022% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=25:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 - 0.000017,  # shift upward by 0.001% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=30:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 - 0.007,  # shift downward by 0.6% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=40:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 - 0.011,  # shift downward by 1.1% of figure height
            pos.width,
            pos.height
        ])
    
    elif top_k >40:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 - 0.0128,  # shift downward by 1.28% of figure height
            pos.width,
            pos.height
        ])
    
    # --- Duplicate the top group colorbar exactly (with white border) ------------------------
    fig = cg_uid.fig
    ax_heatmap = cg_uid.ax_heatmap
    ax_top = cg_uid.ax_col_colors

    # Extract the top colorbar QuadMesh
    quadmesh = ax_top.collections[0]

    # Get facecolors and edge properties
    facecolors = quadmesh.get_facecolors()
    edgecolor = quadmesh.get_edgecolor()
    linewidth = quadmesh.get_linewidths()[0] if len(quadmesh.get_linewidths()) else 0.8

    # Force white borders if transparent/black
    if edgecolor is None or np.allclose(edgecolor, 0):
        edgecolor = "white"

    # Build a 1×N×4 RGBA row
    n = len(facecolors)
    rgba_row = facecolors.reshape(1, n, 4)

    # Compute position just below the heatmap
    pos = ax_heatmap.get_position()

    ax_bottom = fig.add_axes([
        ax_top.get_position().x0,
        pos.y0 - fixed_bar_height_rel - 0.001,  # spacing below heatmap
        ax_top.get_position().width,
        fixed_bar_height_rel
    ])

    # Draw identical color strip
    ax_bottom.imshow(rgba_row, aspect="auto", interpolation="none")

    # Add white border on all sides
    for spine in ax_bottom.spines.values():
        spine.set_visible(True)
        spine.set_color(edgecolor)
        spine.set_linewidth(linewidth)
        
    # Add white vertical lines between samples
    for i in range(1, n):
        ax_bottom.axvline(i - 0.5, color="white", linewidth=1)

    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])

    reordered_groups = y_groups.iloc[cg_uid.dendrogram_col.reordered_ind].astype(str).tolist()
    boundary_positions = _group_boundaries_from_sequence(reordered_groups)
    _draw_group_separators(cg_uid.ax_heatmap, boundary_positions, line_width=1.9, alpha=0.8)
    _draw_group_separators(ax_top, boundary_positions, line_width=2.1, alpha=0.95, color="white")
    _draw_group_separators(ax_bottom, boundary_positions, line_width=2.1, alpha=0.95, color="white")

    # --- Make x-ticks longer to push labels down naturally ---
    ax_heatmap.tick_params(axis="x", which="both", length=14)  # increase from default (~3-4)

    #---------------------------------------------------------------------------------------------
    
    # Get handles to figure and heatmap axes
    fig = cg_uid.fig
    ax_heatmap = cg_uid.ax_heatmap
    ax_top = cg_uid.ax_col_colors
        
    # Add titles
    cg_uid.ax_heatmap.set_xlabel("Samples", fontsize=style["label_size"], labelpad=12)
    cg_uid.ax_heatmap.set_ylabel("", fontsize=14, labelpad=12)
    if top_k <=5:
        cg_uid.fig.suptitle(f"Clustered Heatmap (Top {top_k} by Kruskal-FDR)", fontsize=style["title_size"], weight="bold", y=1.08)
    elif top_k <=15:
        cg_uid.fig.suptitle(f"Clustered Heatmap (Top {top_k} by Kruskal-FDR)", fontsize=style["title_size"], weight="bold", y=1.04)
    elif top_k <=25:
        cg_uid.fig.suptitle(f"Clustered Heatmap (Top {top_k} by Kruskal-FDR)", fontsize=style["title_size"], weight="bold", y=1.02)
    elif top_k >25:
        cg_uid.fig.suptitle(f"Clustered Heatmap (Top {top_k} by Kruskal-FDR)", fontsize=style["title_size"], weight="bold", y=1.005)
          
    plt.setp(cg_uid.ax_heatmap.get_xticklabels(), rotation=55, ha="right", fontsize=max(style["tick_size"] - 2, 7))
    plt.setp(cg_uid.ax_heatmap.get_yticklabels(), rotation=0, ha="left", fontsize=max(style["tick_size"] - 2, 8))
    
    # Attach legend to the full figure — top center, above title
    n_groups = len(legend_handles)
    ncol = 2 if n_groups > 8 else 1
    
    if top_k <=10:   
        ncol = 2 if n_groups > 8 else 1
        if ncol == 1:
            # Attach legend to the full figure — top center, above title
            cg_uid.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.62, 1.05),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )
        if ncol == 2:
            # Attach legend to the full figure — top center, above title
            cg_uid.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.78, 1.05),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )
            
    if top_k ==15: 
        if ncol == 1:  
            # Attach legend to the full figure — top center, above title
            cg_uid.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.62, 0.98),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )
        if ncol == 2:  
            # Attach legend to the full figure — top center, above title
            cg_uid.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.78, 0.98),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )
    
    if top_k >=20:  
        if ncol == 1: 
            # Attach legend to the full figure — top center, above title
            cg_uid.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.62, 0.9),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )
        if ncol == 2: 
            # Attach legend to the full figure — top center, above title
            cg_uid.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.78, 0.9),  # X=right of colorbar, Y=below title
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            ncol=ncol
            )
    
    uid_png = save_dir / f"{basename}_t{top_k}_uid.png"
    uid_svg = save_dir / f"{basename}_t{top_k}_uid.svg"

    _safe_savefig(cg_uid.fig, uid_png, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.2)
    _safe_savefig(cg_uid.fig, uid_svg, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.2)
    plt.close(cg_uid.fig)

def _generate_all_heatmaps(
    X: pd.DataFrame,
    y_groups: pd.Series,
    feature_meta: pd.DataFrame,
    save_dir: Path,
    suffix_label: str,
    group_colors=None,
    group_order=None,
    per_class_heatmaps: bool = True,
    min_features_per_class: int = 5,
    per_category_heatmaps: bool = True,
    min_features_per_category: int = 15,
    class_to_category: Optional[Dict] = None,
    per_carbon_heatmaps: bool = True,
    carbon_agg: str = "sum",
    min_features_per_carbon_bin: int = 3,
    all_features_unclustered_by_class: bool = True,
    style: Optional[dict] = None,
):
    style = style or get_figure_style(False, 100)

    """
    Common routine to:
      - autoscale X
      - rank features via Kruskal-Wallis + FDR
      - plot top N heatmaps at preset cutoffs
    suffix_label is appended to output filenames/folders (e.g., "", or "Without_outliers")
    """
    # Create a subfolder for this 
    if suffix_label:
        out_dir = _ensure_dir(save_dir / (suffix_label))
    else: 
        out_dir = _ensure_dir(save_dir)
        
    # Autoscale across samples
    X_scaled = _autoscale_df(X, out_dir, "autoscaled_data.csv")

    # Rank features
    rank_df = _kruskal_rank_features(X_scaled, y_groups, out_dir, "Kruskal_results.csv")
    ranked_features = rank_df["Feature"].tolist()

    global_cutoffs = [50, 40, 30, 25, 20, 15, 10, 5]
    class_cutoffs  = [20, 15, 10, 5]
    cat_cutoffs    = [20, 15, 10, 5]

    # Heatmaps at multiple cutoffs
    for k in global_cutoffs:
        _plot_heatmap(
            X_scaled=X_scaled,
            y_groups=y_groups,
            feature_ids_sorted=ranked_features,
            feature_meta=feature_meta,
            top_k=k,
            save_dir=out_dir,
            basename="Heatmap",
            group_colors=group_colors,
            group_order=group_order,
            style=style,
        )

    if all_features_unclustered_by_class:
        _plot_all_features_by_class_heatmap(
            X_scaled=X_scaled,
            y_groups=y_groups,
            feature_meta=feature_meta,
            save_dir=out_dir,
            basename="Heatmap_all_features_unclustered_by_class",
            group_colors=group_colors,
            group_order=group_order,
            style=style,
        )

    # ---------------------------------------------------------
    # Per-class heatmaps
    # ---------------------------------------------------------
        
    if per_class_heatmaps:
        uid_to_class = _build_uid_to_class(feature_meta)

        if not uid_to_class:
            print("[Heatmap] feature_meta missing Lipid Class mapping; skipping per-class heatmaps.", flush=True)
            return

        # Build class -> list[UniqueID] limited to features present in X
        class_to_uids = {}
        for uid in X.columns.astype(str):
            u = str(uid).strip()
            cls = uid_to_class.get(u, "Unknown")
            class_to_uids.setdefault(cls, []).append(u)

        # Deterministic order
        for cls in sorted(class_to_uids.keys()):
            uids = [u for u in class_to_uids[cls] if u in X.columns]
            if len(uids) < min_features_per_class:
                continue

            class_dir = _ensure_dir(out_dir / "Class" / _sanitize_filename(cls))

            X_cls = X.loc[:, uids].copy()

            # Autoscale and rank within the class
            X_scaled_cls = _autoscale_df(X_cls, class_dir, "autoscaled_data.csv")
            rank_df_cls = _kruskal_rank_features(X_scaled_cls, y_groups, class_dir, "Kruskal_results.csv")
            ranked_cls = rank_df_cls["Feature"].tolist()

            # Use smaller cutoffs so small classes still plot
            max_k = min(len(ranked_cls), 50)
            cutoffs = [kk for kk in class_cutoffs if kk <= max_k]
            if not cutoffs:
                continue

            for kk in cutoffs:
                _plot_heatmap(
                    X_scaled=X_scaled_cls,
                    y_groups=y_groups,
                    feature_ids_sorted=ranked_cls,
                    feature_meta=feature_meta,
                    top_k=kk,
                    save_dir=class_dir,
                    basename=f"HM_{_sanitize_filename(cls)}",
                    group_colors=group_colors,
                    group_order=group_order,
                    style=style,
                )

            print(f"[Heatmap] Per-class heatmaps written: {cls} ({len(uids)} features)", flush=True)

    # ---------------------------------------------------------
    # Per-category heatmaps
    # ---------------------------------------------------------
    if per_category_heatmaps:
        uid_to_class = _build_uid_to_class(feature_meta)

        if not uid_to_class:
            print("[Heatmap] feature_meta missing Lipid Class mapping; skipping per-category heatmaps.", flush=True)
        else:
            class_to_category = class_to_category or _default_class_to_category()
            class_to_cat_lookup = _invert_class_to_category(class_to_category)

            category_to_uids = {}
            for uid in X.columns.astype(str):
                u = str(uid).strip()
                cls = uid_to_class.get(u, "Unknown")
                cat = _map_class_to_category(cls, class_to_cat_lookup)
                category_to_uids.setdefault(cat, []).append(u)

            for cat in sorted(category_to_uids.keys()):
                uids = [u for u in category_to_uids[cat] if u in X.columns]
                if len(uids) < min_features_per_category:
                    continue

                cat_dir = _ensure_dir(out_dir / "Cat" / _sanitize_filename(cat))
                X_cat = X.loc[:, uids].copy()

                X_scaled_cat = _autoscale_df(X_cat, cat_dir, "autoscaled_data.csv")
                rank_df_cat = _kruskal_rank_features(X_scaled_cat, y_groups, cat_dir, "Kruskal_results.csv")
                ranked_cat = rank_df_cat["Feature"].tolist()

                max_k = len(ranked_cat)
                cutoffs = [kk for kk in cat_cutoffs if kk <= max_k]
                if not cutoffs:
                    continue

                for kk in cutoffs:
                    _plot_heatmap(
                        X_scaled=X_scaled_cat,
                        y_groups=y_groups,
                        feature_ids_sorted=ranked_cat,
                        feature_meta=feature_meta,
                        top_k=kk,
                        save_dir=cat_dir,
                        basename=f"HM_{_sanitize_filename(cat)}",
                        group_colors=group_colors,
                        group_order=group_order,
                        style=style,
                    )

                print(f"[Heatmap] Per-category heatmaps written: {cat} ({len(uids)} features)", flush=True)
    
    # ---------------------------------------------------------
    # Per-carbon heatmaps (aggregated bins)
    # ---------------------------------------------------------
    if per_carbon_heatmaps:
        uid_to_carb = _build_uid_to_carbons(feature_meta)
        if not uid_to_carb:
            print("[Heatmap] feature_meta missing carbon-count column; skipping per-carbon heatmaps.", flush=True)
        else:
            X_carb = _aggregate_by_carbons(
                X=X,
                uid_to_carbons=uid_to_carb,
                agg=carbon_agg,
                min_features_per_bin=min_features_per_carbon_bin,
            )

            if X_carb.empty or X_carb.shape[1] < 2:
                print("[Heatmap] Not enough carbon bins to plot; skipping per-carbon heatmaps.", flush=True)
            else:
                carb_dir = _ensure_dir(out_dir / "C")
                X_scaled_carb = _autoscale_df(X_carb, carb_dir, "autoscaled_data.csv")
                rank_df_carb = _kruskal_rank_features(X_scaled_carb, y_groups, carb_dir, "Kruskal_results.csv")
                ranked_bins = rank_df_carb["Feature"].tolist()

                # Plot only 20/10/5 style cutoffs, but bins might be fewer
                for k in [20, 15, 10, 5]:
                    kk = min(k, len(ranked_bins))
                    if kk < 2:
                        continue
                    _plot_heatmap(
                        X_scaled=X_scaled_carb,
                        y_groups=y_groups,
                        feature_ids_sorted=ranked_bins,
                        feature_meta=feature_meta,   # not used for labels here, but fine
                        top_k=kk,
                        save_dir=carb_dir,
                        basename=f"HM_C_{carbon_agg}",
                        group_colors=group_colors,
                        group_order=group_order,
                        style=style,
                    )

                print(f"[Heatmap] Per-carbon heatmaps written: {X_carb.shape[1]} bins (agg={carbon_agg})", flush=True)

# ==========================================================
# Public entry point (called by the GUI)
# ==========================================================
def run_heatmap(file_path, group_file, save_dir, group_colors=None, group_order=None, dpi=100, publication_theme: bool = False):
    """
    Main entry used by StatisticsPage.
      - file_path: path to statistics/Final_Annotated*.csv variant
      - group_file: statistics/sample_groups_cleaned.csv (or None)
      - save_dir: the folder prepared by the GUI (e.g., statistics/Heatmap/VariantLabel)
    """
    file_path = Path(file_path)
    save_dir = prepare_output_dir(Path(save_dir))
    style = get_figure_style(publication_theme=publication_theme, dpi=dpi)

    print('[Heatmaps] Running heatmaps...', flush = True)
    print(f"[Heatmap] Starting for: {file_path.name}", flush = True)

    # Load standardized dataset:
    #   X: samples × features (columns = UniqueID)
    #   y: sample groups
    #   feature_meta: includes Annotation, UniqueID, etc.
    X, y, feature_meta = load_dataset(file_path, group_file)

    # Guardrails
    if X.empty or len(X.columns) == 0:
        print("[Heatmap] No data found — skipping.", flush = True)
        return
    if y.nunique() < 2:
        print("[Heatmap] Only one group detected — clustering will run, Kruskal ranking becomes neutral.", flush = True)
    print(f"[Heatmap] Data: {X.shape[0]} samples × {X.shape[1]} features, groups={y.nunique()}", flush = True)

    # ---------- Standard (all features) ----------
    _generate_all_heatmaps(
        X=X,
        y_groups=y,
        feature_meta=feature_meta,
        save_dir=save_dir,
        suffix_label="",
        group_colors=group_colors,
        group_order=group_order,
        per_class_heatmaps=True, min_features_per_class=5,
        per_category_heatmaps=True, min_features_per_category=15,
        class_to_category=None,  # or pass your explicit dict
        per_carbon_heatmaps=True,
        carbon_agg="sum",                 # or "mean"
        min_features_per_carbon_bin=3,
        style=style,
    )

    # # ---------- Without Outliers ----------
    # X_no, dropped = _remove_extreme_feature_outliers(X, z_thresh=4.0)
    # dropped = dropped or []
    # outlier_dir = _ensure_dir(save_dir / "Without_outliers")
    # pd.DataFrame({"Removed_Features": dropped}).to_csv(
    #     outlier_dir / "outlier_features.csv", index=False, encoding="utf-8-sig"
    # )

    # if X_no.shape[1] == 0:
    #     print("[Heatmap] All features were flagged as outliers; skipping 'Without_outliers' run.", flush = True)
    #     return

    # _generate_all_heatmaps(
    #     X=X_no,
    #     y_groups=y,
    #     feature_meta=feature_meta,
    #     save_dir=save_dir,
    #     suffix_label="Without_outliers",  # writes to save_dir / "Without_outliers"
    #     group_colors=group_colors,
    #     group_order=group_order,
    #     per_class_heatmaps=True, min_features_per_class=5,
    #     per_category_heatmaps=True, min_features_per_category=15,
    #     class_to_category=None,  # or pass your explicit dict
    #     per_carbon_heatmaps=True,
    #     carbon_agg="sum",                 # or "mean"
    #     min_features_per_carbon_bin=3,
    #     style=style,
    # )

    # print(f"[Heatmap] Completed. Output in: {save_dir}\n", flush = True)


def run_selected_lipid_heatmap(
    file_path,
    group_file,
    save_dir,
    selected_annotations: Optional[List[str]] = None,
    selected_annotation_groups: Optional[Dict[str, List[str]]] = None,
    group_colors=None,
    group_order=None,
    dpi=100,
    publication_theme: bool = False,
):
    file_path = Path(file_path)
    normalized_groups: dict[str, list[str]] = {}
    for group_name, annotations in (selected_annotation_groups or {}).items():
        clean_group = str(group_name).strip() or "Selected heatmap"
        clean_annotations = [str(x).strip() for x in (annotations or []) if str(x).strip()]
        if clean_annotations:
            normalized_groups[clean_group] = clean_annotations
    if not normalized_groups:
        selected_annotations = [str(x).strip() for x in (selected_annotations or []) if str(x).strip()]
        if selected_annotations:
            normalized_groups["Selected heatmap"] = selected_annotations
    if not normalized_groups:
        raise ValueError("No annotations were selected for the heatmap.")
    save_dir = prepare_output_dir(Path(save_dir))
    style = get_figure_style(publication_theme=publication_theme, dpi=dpi)

    print("[Selected Heatmap] Running selected lipid heatmap...", flush=True)
    X, y, feature_meta = load_dataset(file_path, group_file)
    if X.empty or feature_meta.empty:
        raise ValueError("Dataset appears empty or malformed.")

    if "UniqueID" not in feature_meta.columns:
        raise ValueError("Feature metadata is missing 'UniqueID'.")

    ann_col = None
    for c in feature_meta.columns:
        if str(c).strip().lower() == "annotation":
            ann_col = c
            break
    if ann_col is None:
        raise ValueError("Feature metadata is missing 'Annotation'.")

    feature_meta = feature_meta.copy()
    feature_meta["UniqueID"] = feature_meta["UniqueID"].astype(str).str.strip()
    feature_meta[ann_col] = feature_meta[ann_col].astype(str).str.strip()

    uid_to_annotation = feature_meta.drop_duplicates("UniqueID").set_index("UniqueID")[ann_col].to_dict()
    annotation_to_uids: dict[str, list[str]] = {}
    for uid in X.columns.astype(str):
        ann = str(uid_to_annotation.get(str(uid).strip(), "")).strip()
        if ann:
            annotation_to_uids.setdefault(ann, []).append(str(uid).strip())

    X_ordered, y_ordered = _order_samples_by_group(X, y, group_order=group_order)

    out_dir = prepare_output_dir(save_dir / "Selected_Lipid_Heatmap")
    sample_order_csv = out_dir / "sample_order.csv"
    pd.DataFrame({"Sample": X_ordered.index.astype(str), "Group": y_ordered.astype(str).values}).to_csv(sample_order_csv, index=False, encoding="utf-8-sig")

    plot_outputs: dict[str, dict[str, str]] = {}
    any_found = False
    for group_name, selected_annotations in normalized_groups.items():
        rows = []
        found_annotations = []
        missing_annotations = []
        for annotation in selected_annotations:
            uids = [uid for uid in annotation_to_uids.get(annotation, []) if uid in X_ordered.columns]
            if not uids:
                missing_annotations.append(annotation)
                continue
            signal = X_ordered.loc[:, uids].sum(axis=1)
            rows.append(signal.rename(annotation))
            found_annotations.append(annotation)

        group_dir = prepare_output_dir(out_dir / _sanitize_filename(group_name))
        selected_csv = group_dir / "selected_annotations_order.csv"
        pd.DataFrame({"Annotation": selected_annotations, "Found": [a in found_annotations for a in selected_annotations]}).to_csv(
            selected_csv,
            index=False,
            encoding="utf-8-sig",
        )

        if not rows:
            missing_csv = group_dir / "selected_lipids_missing_annotations.csv"
            pd.DataFrame({"MissingAnnotation": missing_annotations}).to_csv(missing_csv, index=False, encoding="utf-8-sig")
            plot_outputs[group_name] = {
                "out_dir": str(group_dir),
                "selected_annotations_csv": str(selected_csv),
                "missing_annotations_csv": str(missing_csv),
            }
            continue

        any_found = True
        heatmap_df = pd.DataFrame(rows)
        heatmap_df.index = found_annotations
        heatmap_df.columns = X_ordered.index.astype(str)

        scaler = StandardScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(heatmap_df.T).T,
            index=heatmap_df.index,
            columns=heatmap_df.columns,
        )

        raw_csv = group_dir / "selected_lipids_raw_abundance.csv"
        scaled_csv = group_dir / "selected_lipids_autoscaled.csv"
        missing_csv = group_dir / "selected_lipids_missing_annotations.csv"
        png_path = group_dir / "selected_lipid_heatmap.png"
        svg_path = group_dir / "selected_lipid_heatmap.svg"

        heatmap_df.to_csv(raw_csv, encoding="utf-8-sig")
        scaled.to_csv(scaled_csv, encoding="utf-8-sig")
        pd.DataFrame({"MissingAnnotation": missing_annotations}).to_csv(missing_csv, index=False, encoding="utf-8-sig")

        _plot_selected_lipid_heatmap(
            scaled,
            y_ordered,
            png_path,
            svg_path,
            group_colors=group_colors,
            group_order=group_order,
            style=style,
        )
        plot_outputs[group_name] = {
            "out_dir": str(group_dir),
            "raw_csv": str(raw_csv),
            "scaled_csv": str(scaled_csv),
            "selected_annotations_csv": str(selected_csv),
            "missing_annotations_csv": str(missing_csv),
            "heatmap_png": str(png_path),
            "heatmap_svg": str(svg_path),
        }

    if not any_found:
        raise ValueError("None of the selected annotations were found in the dataset.")

    print(f"[Selected Heatmap] Completed. Output in: {out_dir}", flush=True)
    return {
        "out_dir": str(out_dir),
        "sample_order_csv": str(sample_order_csv),
        "plots": plot_outputs,
    }


# Optional local test
if __name__ == "__main__":
    # Example:
    # python Stats/heatmap_analysis.py path/to/Final_Annotated.csv path/to/sample_groups.csv ./out/Heatmap
    import sys

    fpath = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "statistics" / "Final_Annotated.csv"
    gpath = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "statistics" / "sample_groups_cleaned.csv"
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.cwd() / "statistics" / "Heatmap" / "ManualTest"

    run_heatmap(fpath, gpath if gpath.exists() else None, outdir)
