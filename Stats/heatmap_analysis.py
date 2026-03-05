# ------------------------------------------------------------
# Heatmaps for standardized Lipid workflow (UniqueID-based).
# - Loads via Stats.utils.load_dataset(file_path, group_file)
# - Autoscaling
# - ANOVA + FDR feature ranking
# - Clustered heatmaps at multiple feature cutoffs
# - Also generates a "without outliers" version (z>4 filter)
# - Saves PNG + SVG, autoscaled data, ANOVA table, outlier list
# ------------------------------------------------------------

import os
import warnings
from pathlib import Path
import re
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

from Stats.utils import load_dataset, prepare_output_dir

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

import matplotlib as mpl
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
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


def _anova_rank_features(X_scaled: pd.DataFrame, groups: pd.Series, save_dir: Path, filename: str) -> pd.DataFrame:
    """
    One-way ANOVA for each feature across groups + FDR.
    Returns a sorted dataframe (by Adjusted_P) and writes it to disk.
    """
    groups = groups.astype(str)
    unique_groups = groups.unique()

    # If there's only 1 group, ANOVA is not defined—return neutral p-values
    if len(unique_groups) < 2:
        out = pd.DataFrame({"Feature": X_scaled.columns, "P_Value": 1.0})
        out["Adjusted_P_Value"] = 1.0
        out = out.sort_values("Adjusted_P_Value").reset_index(drop=True)
        out.to_csv(save_dir / filename, index=False, encoding="utf-8-sig")
        return out

    pvals = []
    for feat in X_scaled.columns:
        # collect values by group for this feature
        by_group = [X_scaled.loc[groups == g, feat].values for g in unique_groups]
        # must have at least 2 points per group for a valid F-test; otherwise set high p
        if all(len(v) > 1 for v in by_group):
            p = f_oneway(*by_group).pvalue
        else:
            p = 1.0
        pvals.append(p)

    anova_df = pd.DataFrame({"Feature": X_scaled.columns, "P_Value": pvals})
    # FDR (Benjamini-Hochberg)
    anova_df["Adjusted_P_Value"] = multipletests(anova_df["P_Value"].values, method="fdr_bh")[1]
    anova_df = anova_df.sort_values("Adjusted_P_Value").reset_index(drop=True)
    anova_df.to_csv(save_dir / filename, index=False, encoding="utf-8-sig")
    return anova_df

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
        "FA": {"CAR", "CoA", "FA", "FAG", "FAHFA", "FAL", "FOH", "HC", "NA", "NAE", "NAT", "WE"},

        # Glycerolipids
        "GL": {"MG", "DG", "TG", "DGCC", "DGMG", "DGDG","DGTA", "DGTS", "GlcADG", "MGDG", "MGMG", "SQDG", "SQMG"},

        # Glycerophospholipids
        "GP": {"PC", "PE", "PG", "PI", "PS", "PA", "CL",
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
    unique_in_order = list(pd.unique(groups))

    # Separate QC (case-insensitive)
    qc_labels  = [g for g in unique_in_order if g.lower() == "qc"]
    non_qc     = [g for g in unique_in_order if g.lower() != "qc"]

    # Legend order: user order first, then remaining, QC last
    if group_order:
        ordered_non_qc = [g for g in group_order if g in non_qc] + [g for g in non_qc if g not in group_order]
    else:
        ordered_non_qc = non_qc[:]
    legend_order = ordered_non_qc + qc_labels

    # Fallback cycle
    cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", []) or ["#1f77b4"]

    # Build color map honoring user palette first
    color_map = {}
    # non-QC first
    for i, g in enumerate(ordered_non_qc):
        c = (group_colors or {}).get(g)
        color_map[g] = c if c else cycle[i % len(cycle)]
    # QC color(s)
    for g in qc_labels:
        c = (group_colors or {}).get(g)
        color_map[g] = c if c else "#000000"

    # Ensure every group seen gets a color (safety backfill)
    for g in unique_in_order:
        if g not in color_map:
            color_map[g] = cycle[len(color_map) % len(cycle)]

    # Per-sample colors in sample order
    col_colors = [color_map[g] for g in groups]

    legend_handles = [
        plt.matplotlib.patches.Patch(color=color_map[g], label=g) for g in legend_order
    ]
    return col_colors, legend_handles


def _dynamic_figsize(n_features: int, n_samples: int, row_height_inch: float = 0.25, top_bottom_margin_inch: float = 2.5) -> tuple:
    """
    Compute figure size dynamically:
      - width scales with number of samples
      - height scales linearly with number of features (fixed per-row height)
    """
    width = max(6, 0.35 * n_samples)
    height = (n_features * row_height_inch) + top_bottom_margin_inch
    return (width, height)

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
):

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

    # Dynamic figure sizing
    fig_w, fig_h = _dynamic_figsize(n_features=H.shape[0], n_samples=H.shape[1])

    # ==========================================================
    # Annotation mapping (robust, like VIP code)
    # ==========================================================
    annotations = []
    if isinstance(feature_meta, pd.DataFrame) and "Annotation" in feature_meta.columns:
        uid_to_annotation = dict(zip(
            feature_meta["UniqueID"].astype(str).str.strip(),
            feature_meta["Annotation"].astype(str).str.strip()
        ))
        for uid in Xsel.columns:
            ann = uid_to_annotation.get(str(uid).strip(), str(uid))
            annotations.append(ann)
    else:
        annotations = [str(uid) for uid in Xsel.columns]

    # ==========================================================
    # 1) Annotated version (Annotation names on Y-axis)
    # ==========================================================
    H_annot = H.copy()
    H_annot.index = annotations
    
    cg_annot = sns.clustermap(
        H_annot,
        cmap="coolwarm",
        linewidths=0.4,
        figsize=(fig_w, fig_h),
        row_cluster=True,
        col_cluster=True,
        col_colors=col_colors,  # group color bar
        method="ward",
        metric="euclidean",
        dendrogram_ratio=(0.08, 0.08),     # side, top dendograms
        cbar_kws={"shrink": 0.5, "label": "\nStandardized\nValues"},
        cbar_pos=(1.04, 0.1, 0.03, 0.3),
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

    # --- Make x-ticks longer to push labels down naturally ---
    ax_heatmap.tick_params(axis="x", which="both", length=14)  # increase from default (~3-4)

    #---------------------------------------------------------------------------------------------

    # Add titles
    cg_annot.ax_heatmap.set_xlabel("Samples (Groups)", fontsize=14, labelpad=12)
    cg_annot.ax_heatmap.set_ylabel("", fontsize=14, labelpad=12)
    
    if top_k <=5:
        cg_annot.fig.suptitle(f"Clustered Heatmap (Top {top_k} Lipids by ANOVA-FDR)", fontsize=14, weight="bold", y=1.08)
    elif top_k <=15:
        cg_annot.fig.suptitle(f"Clustered Heatmap (Top {top_k} Lipids by ANOVA-FDR)", fontsize=14, weight="bold", y=1.04)
    elif top_k <=25:
        cg_annot.fig.suptitle(f"Clustered Heatmap (Top {top_k} Lipids by ANOVA-FDR)", fontsize=14, weight="bold", y=1.02)
    elif top_k >25:
        cg_annot.fig.suptitle(f"Clustered Heatmap (Top {top_k} Lipids by ANOVA-FDR)", fontsize=14, weight="bold", y=1.005)
        
    plt.setp(cg_annot.ax_heatmap.get_xticklabels(), rotation=55, ha="right", fontsize=11)
    plt.setp(cg_annot.ax_heatmap.get_yticklabels(), rotation=0, ha="left", fontsize=10)
    
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
            bbox_to_anchor=(1.32, 1.05),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )
        if ncol == 2:
            # Attach legend to the full figure — top center, above title
            cg_annot.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.42, 1.05),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )
            
    if top_k ==15: 
        if ncol == 1:  
            # Attach legend to the full figure — top center, above title
            cg_annot.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.32, 0.98),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )
        if ncol == 2:  
            # Attach legend to the full figure — top center, above title
            cg_annot.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.42, 0.98),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )
    
    if top_k >=20:   
        if ncol == 1: 
            # Attach legend to the full figure — top center, above title
            cg_annot.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.32, 0.9),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )
        if ncol == 2: 
            # Attach legend to the full figure — top center, above title
            cg_annot.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.42, 0.9),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )

    plt.savefig(save_dir / f"{basename}_top{top_k}_annotated.png", dpi=100, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(save_dir / f"{basename}_top{top_k}_annotated.svg", dpi=100, bbox_inches="tight", pad_inches=0.2)
    plt.close()



    # ==========================================================
    # 2) UniqueID-only version
    # ==========================================================
    H_uid = H.copy()
    H_uid.index = [str(fid).strip() for fid in selected_ids]

    cg_uid = sns.clustermap(
        H_uid,
        cmap="coolwarm",
        linewidths=0.4,
        figsize=(fig_w, fig_h),
        row_cluster=True,
        col_cluster=True,
        col_colors=col_colors,  # group colorbar
        method="ward",
        metric="euclidean",
        dendrogram_ratio=(0.10, 0.10),
        cbar_kws={"shrink": 0.5, "label": "\nStandardized\nValues"},
        cbar_pos=(1.04, 0.1, 0.03, 0.3),
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

    # --- Make x-ticks longer to push labels down naturally ---
    ax_heatmap.tick_params(axis="x", which="both", length=14)  # increase from default (~3-4)

    #---------------------------------------------------------------------------------------------
    
    # Get handles to figure and heatmap axes
    fig = cg_uid.fig
    ax_heatmap = cg_uid.ax_heatmap
    ax_top = cg_uid.ax_col_colors
        
    # Add titles
    cg_uid.ax_heatmap.set_xlabel("Samples (Groups)", fontsize=14, labelpad=12)
    cg_uid.ax_heatmap.set_ylabel("", fontsize=14, labelpad=12)
    if top_k <=5:
        cg_uid.fig.suptitle(f"Clustered Heatmap (Top {top_k} Lipids by ANOVA-FDR)", fontsize=14, weight="bold", y=1.08)
    elif top_k <=15:
        cg_uid.fig.suptitle(f"Clustered Heatmap (Top {top_k} Lipids by ANOVA-FDR)", fontsize=14, weight="bold", y=1.04)
    elif top_k <=25:
        cg_uid.fig.suptitle(f"Clustered Heatmap (Top {top_k} Lipids by ANOVA-FDR)", fontsize=14, weight="bold", y=1.02)
    elif top_k >25:
        cg_uid.fig.suptitle(f"Clustered Heatmap (Top {top_k} Lipids by ANOVA-FDR)", fontsize=14, weight="bold", y=1.005)
          
    plt.setp(cg_uid.ax_heatmap.get_xticklabels(), rotation=55, ha="right", fontsize=11)
    plt.setp(cg_uid.ax_heatmap.get_yticklabels(), rotation=0, ha="left", fontsize=10)
    
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
            bbox_to_anchor=(1.32, 1.05),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )
        if ncol == 2:
            # Attach legend to the full figure — top center, above title
            cg_uid.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.42, 1.05),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )
            
    if top_k ==15: 
        if ncol == 1:  
            # Attach legend to the full figure — top center, above title
            cg_uid.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.32, 0.98),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )
        if ncol == 2:  
            # Attach legend to the full figure — top center, above title
            cg_uid.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.42, 0.98),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )
    
    if top_k >=20:  
        if ncol == 1: 
            # Attach legend to the full figure — top center, above title
            cg_uid.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.32, 0.9),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )
        if ncol == 2: 
            # Attach legend to the full figure — top center, above title
            cg_uid.fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.42, 0.9),  # X=right of colorbar, Y=below title
            fontsize=12,
            title_fontsize=12,
            frameon=False,
            ncol=ncol
            )
    
    plt.savefig(save_dir / f"{basename}_top{top_k}_uniqueIDs.png", dpi=100, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(save_dir / f"{basename}_top{top_k}_uniqueIDs.svg", dpi=100, bbox_inches="tight", pad_inches=0.2)
    plt.close()

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
):

    """
    Common routine to:
      - autoscale X
      - rank features via ANOVA + FDR
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
    anova_df = _anova_rank_features(X_scaled, y_groups, out_dir, "ANOVA_results.csv")
    ranked_features = anova_df["Feature"].tolist()

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

            class_dir = _ensure_dir(out_dir / "Per_class" / _sanitize_filename(cls))

            X_cls = X.loc[:, uids].copy()

            # Autoscale and rank within the class
            X_scaled_cls = _autoscale_df(X_cls, class_dir, "autoscaled_data.csv")
            anova_df_cls = _anova_rank_features(X_scaled_cls, y_groups, class_dir, "ANOVA_results.csv")
            ranked_cls = anova_df_cls["Feature"].tolist()

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
                    basename=f"Heatmap_{_sanitize_filename(cls)}",
                    group_colors=group_colors,
                    group_order=group_order,
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

                cat_dir = _ensure_dir(out_dir / "Per_category" / _sanitize_filename(cat))
                X_cat = X.loc[:, uids].copy()

                X_scaled_cat = _autoscale_df(X_cat, cat_dir, "autoscaled_data.csv")
                anova_df_cat = _anova_rank_features(X_scaled_cat, y_groups, cat_dir, "ANOVA_results.csv")
                ranked_cat = anova_df_cat["Feature"].tolist()

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
                        basename=f"Heatmap_{_sanitize_filename(cat)}",
                        group_colors=group_colors,
                        group_order=group_order,
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
                carb_dir = _ensure_dir(out_dir / "Per_carbons")
                X_scaled_carb = _autoscale_df(X_carb, carb_dir, "autoscaled_data.csv")
                anova_df_carb = _anova_rank_features(X_scaled_carb, y_groups, carb_dir, "ANOVA_results.csv")
                ranked_bins = anova_df_carb["Feature"].tolist()

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
                        basename=f"Heatmap_Carbons_{carbon_agg}",
                        group_colors=group_colors,
                        group_order=group_order,
                    )

                print(f"[Heatmap] Per-carbon heatmaps written: {X_carb.shape[1]} bins (agg={carbon_agg})", flush=True)

# ==========================================================
# Public entry point (called by the GUI)
# ==========================================================
def run_heatmap(file_path, group_file, save_dir, group_colors=None, group_order=None):
    """
    Main entry used by StatisticsPage.
      - file_path: path to statistics/Final_Annotated*.csv variant
      - group_file: statistics/sample_groups_cleaned.csv (or None)
      - save_dir: the folder prepared by the GUI (e.g., statistics/Heatmap/VariantLabel)
    """
    file_path = Path(file_path)
    save_dir = prepare_output_dir(Path(save_dir))

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
        print("[Heatmap] Only one group detected — clustering will run, ANOVA becomes neutral.", flush = True)
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
    )

    # ---------- Without Outliers ----------
    X_no, dropped = _remove_extreme_feature_outliers(X, z_thresh=4.0)
    dropped = dropped or []
    outlier_dir = _ensure_dir(save_dir / "Without_outliers")
    pd.DataFrame({"Removed_Features": dropped}).to_csv(
        outlier_dir / "outlier_features.csv", index=False, encoding="utf-8-sig"
    )

    if X_no.shape[1] == 0:
        print("[Heatmap] All features were flagged as outliers; skipping 'Without_outliers' run.", flush = True)
        return

    _generate_all_heatmaps(
        X=X_no,
        y_groups=y,
        feature_meta=feature_meta,
        save_dir=save_dir,
        suffix_label="Without_outliers",  # writes to save_dir / "Without_outliers"
        group_colors=group_colors,
        group_order=group_order,
        per_class_heatmaps=True, min_features_per_class=5,
        per_category_heatmaps=True, min_features_per_category=15,
        class_to_category=None,  # or pass your explicit dict
        per_carbon_heatmaps=True,
        carbon_agg="sum",                 # or "mean"
        min_features_per_carbon_bin=3,
    )

    print(f"[Heatmap] Completed. Output in: {save_dir}\n", flush = True)


# Optional local test
if __name__ == "__main__":
    # Example:
    # python Stats/heatmap_analysis.py path/to/Final_Annotated.csv path/to/sample_groups.csv ./out/Heatmap
    import sys

    fpath = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "statistics" / "Final_Annotated.csv"
    gpath = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "statistics" / "sample_groups_cleaned.csv"
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.cwd() / "statistics" / "Heatmap" / "ManualTest"

    run_heatmap(fpath, gpath if gpath.exists() else None, outdir)