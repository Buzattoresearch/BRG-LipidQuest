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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

from Stats.utils import load_dataset, prepare_output_dir

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12


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


def _feature_labels_from_annotations(feature_meta: pd.DataFrame, feature_ids: list, with_ids: bool) -> list:
    """
    Make y-axis labels using Annotation. Fall back to the ID if Annotation missing.
    When with_ids=True, uses "UniqueID|Annotation".
    """
    ann = {}
    if isinstance(feature_meta, pd.DataFrame) and not feature_meta.empty:
        if {"UniqueID", "Annotation"}.issubset(feature_meta.columns):
            # map UniqueID (as str) → Annotation (str)
            ann = dict(
                zip(
                    feature_meta["UniqueID"].astype(str).str.strip(),
                    feature_meta["Annotation"].astype(str).str.strip(),
                )
            )

    labels = []
    for fid in feature_ids:
        fid_str = str(fid).strip()
        a = ann.get(fid_str, fid_str) or fid_str
        if with_ids:
            # Always include the ID, then '|' + annotation if distinct
            if a == fid_str:
                labels.append(fid_str)
            else:
                labels.append(f"{fid_str}|{a}")
        else:
            labels.append(a)
    return labels


def _group_colorbar(groups: pd.Series):
    """Return a list of per-sample colors and legend handles.
    - Many distinct hues for arbitrary #groups
    - QC (any case) is always black and not counted against the palette
    """
    groups = groups.astype(str)
    unique_in_order = list(pd.unique(groups))  # preserves sample order
    # Separate QC so we don't waste a palette slot
    non_qc = [g for g in unique_in_order if g.lower() != "qc"]
    has_qc = any(g.lower() == "qc" for g in unique_in_order)

    # Build a large, distinct palette for non-QC groups
    n = len(non_qc)
    if n <= 10:
        base = sns.color_palette("tab10", n_colors=n)
    elif n <= 20:
        base = sns.color_palette("tab20", n_colors=n)
    elif n <= 32:
        base = (
            sns.color_palette("tab20", 20)
            + sns.color_palette("tab20b", 20)[:6]
            + sns.color_palette("tab20c", 20)[:6]
        )
        base = base[:n]
    else:
        # Arbitrarily many — HUSL keeps them reasonably distinct
        base = sns.husl_palette(n, s=.90, l=.55)

    # Map non-QC groups to colors (deterministic order)
    color_map = {g: base[i] for i, g in enumerate(non_qc)}
    if has_qc:
        color_map.update({g: "#000000" for g in unique_in_order if g.lower() == "qc"})

    # Per-sample colors in the same order as columns/samples
    col_colors = groups.map(color_map).tolist()

    # Legend handles (QC shown in black)
    legend_order = non_qc + ([g for g in unique_in_order if g.lower() == "qc"] if has_qc else [])
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
    col_colors, legend_handles = _group_colorbar(y_groups)
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
        cbar_kws={"shrink": 0.6, "label": "\nStandardized Values"},
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
            pos.y0 + 0.027,  # shift upward by 2.7% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k == 10:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.012,  # shift upward by 1.2% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=15 and top_k >10:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.004,  # shift upward by 0.4% of figure height
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
            pos.y0 - 0.00001,  # shift upward by 0.0025% of figure height
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
        
    # Attach legend to the full figure — top center, above title
    cg_annot.fig.legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(1.15, 0.9),  # X=right of colorbar, Y=below title
    fontsize=12,
    title_fontsize=12,
    frameon=False,
    )

    plt.savefig(save_dir / f"{basename}_top{top_k}_annotated.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(save_dir / f"{basename}_top{top_k}_annotated.svg", dpi=300, bbox_inches="tight", pad_inches=0.2)
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
        cbar_kws={"shrink": 0.6, "label": "\nStandardized Values"},
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
            pos.y0 + 0.027,  # shift upward by 2.7% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k == 10:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.012,  # shift upward by 1.2% of figure height
            pos.width,
            pos.height
        ])
        
    elif top_k <=15 and top_k >10:
        ax_dendro.set_position([
            pos.x0,
            pos.y0 + 0.004,  # shift upward by 0.4% of figure height
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
            pos.y0 - 0.00001,  # shift upward by 0.001% of figure height
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
    cg_uid.fig.legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(1.15, 0.9),  # X=right of colorbar, Y=below title
    fontsize=12,
    title_fontsize=12,
    frameon=False,
    )
    
    plt.savefig(save_dir / f"{basename}_top{top_k}_uniqueIDs.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(save_dir / f"{basename}_top{top_k}_uniqueIDs.svg", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()

def _generate_all_heatmaps(
    X: pd.DataFrame,
    y_groups: pd.Series,
    feature_meta: pd.DataFrame,
    save_dir: Path,
    suffix_label: str,
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

    # Heatmaps at multiple cutoffs
    for k in [50, 40, 30, 25, 20, 15, 10, 5]:
        _plot_heatmap(
            X_scaled=X_scaled,
            y_groups=y_groups,
            feature_ids_sorted=ranked_features,
            feature_meta=feature_meta,
            top_k=k,
            save_dir=out_dir,
            basename="Heatmap",
        )


# ==========================================================
# Public entry point (called by the GUI)
# ==========================================================
def run_heatmap(file_path, group_file, save_dir):
    """
    Main entry used by StatisticsPage.
      - file_path: path to statistics/Final_Annotated*.csv variant
      - group_file: statistics/sample_groups_cleaned.csv (or None)
      - save_dir: the folder prepared by the GUI (e.g., statistics/Heatmap/VariantLabel)
    """
    file_path = Path(file_path)
    save_dir = prepare_output_dir(Path(save_dir))

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
        suffix_label="",  # writes to save_dir / "Standard"
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
