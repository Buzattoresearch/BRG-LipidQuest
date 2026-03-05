# TODO: check if sample type is being passed from the GUI
# TODO: Annotation type and headgroup are not showing in the Excel and CSV files
# TODO: confirm that the Summary_Volcano file is actually correct

# ------------------------------------------------------------
# Volcano analysis for standardized Lipid workflow (LipidQuest)
# - Loads via Stats.utils.load_dataset(file_path, group_file)
# - Pairwise group comparisons (t-test or Mann–Whitney)
# - FDR correction (Benjamini–Hochberg)
# - Volcano plots (PNG+SVG), per-comparison CSVs
# - Summary CSV + structured Excel
# - Class bar plots + two bubble-plot designs
# - Saves all results under save_dir/VolcanoFDR/
# ------------------------------------------------------------

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, mannwhitneyu
from pandas import IndexSlice
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

from Stats.utils import load_dataset, prepare_output_dir
from Stats.utils import _CLASS_ORDER, _CLASS_ORDER_BACTERIA, _CLASS_ORDER_MAMMALIAN, _CLASS_GROUP_MAP

import warnings
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
# Utilities (general)
# ==========================================================
def _sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:."/\\|?*]', "_", str(s))

def _add_jitter(values, jitter_strength=0.025):
    j = values + np.random.uniform(-jitter_strength, jitter_strength, size=len(values))
    return np.maximum(0.0, j)   # clip at 0

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ==========================================================
# Volcano core computation + plotting
# ==========================================================
def _compute_volcano(g1_name, g2_name, X, y, meta_lookup,
                     method, test_type, p_thresh, fdr_thresh, fc_thresh):
    """Perform one pairwise comparison and return volcano DataFrame."""
    # --- Select groups and common features ---
    y_str = y.astype(str)
    group1 = X.loc[y_str == str(g1_name)]
    group2 = X.loc[y_str == str(g2_name)]

    common = group1.columns.intersection(group2.columns)
    group1 = group1[common]
    group2 = group2[common]

    # --- Remove features that are entirely empty in BOTH groups ---
    nonzero_any = (group1.notna().any(axis=0) & group2.notna().any(axis=0))
    group1 = group1.loc[:, nonzero_any]
    group2 = group2.loc[:, nonzero_any]
    valid_UniqueIDs = list(group1.columns)

    # --- p-values on filtered features only ---
    pvals = []
    for UniqueID in valid_UniqueIDs:
        x1 = group1[UniqueID].replace([np.inf, -np.inf], np.nan).dropna()
        x2 = group2[UniqueID].replace([np.inf, -np.inf], np.nan).dropna()

        # Need at least 2 per group for variance-based tests; otherwise neutral p
        if (len(x1) < 2) or (len(x2) < 2):
            p = 1.0
        else:
            # If both groups effectively constant, neutral p
            if (np.std(x1, ddof=1) < 1e-12) and (np.std(x2, ddof=1) < 1e-12):
                p = 1.0
            else:
                try:
                    if test_type == "non-parametric":
                        _, p = mannwhitneyu(x1, x2, alternative="two-sided")
                    else:
                        _, p = ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
                except Exception:
                    p = 1.0
        pvals.append(float(p))

    pvals = np.asarray(pvals, dtype=float)
    fdr_corrected = multipletests(pvals, alpha=0.05, method=method)[1]

    # --- Robust FC using medians + per-feature pseudocount ---
    g1_med = group1.replace(0, np.nan).median()
    g2_med = group2.replace(0, np.nan).median()

    pooled_med = pd.concat([group1, group2]).replace(0, np.nan).median()
    eps = (0.01 * pooled_med).fillna(0) + 1e-9  # 1% of pooled nonzero median, with floor

    g1m = (g1_med.fillna(0) + eps)
    g2m = (g2_med.fillna(0) + eps)

    fc = (g1m / g2m).clip(lower=1e-6, upper=1e6)
    fc_log2 = np.log2(fc)

    df = pd.DataFrame({
        "UniqueID": valid_UniqueIDs,
        "Fold Change": np.round(fc.values, 18),
        "log2(Fold Change)": np.round(fc_log2.values, 10),
        "p-value": np.round(pvals, 18),
        "FDR p-value": np.round(fdr_corrected, 18),
    })

    # Clean non-finite and compute -log10(FDR)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log2(Fold Change)", "FDR p-value"])
    df["-log10(FDR p-value)"] = -np.log10(np.clip(df["FDR p-value"], 1e-300, 1.0))

    # Significance status
    log2fc_thresh = np.log2(fc_thresh)
    df["Significance"] = "Not Significant"
    df.loc[
        (df["log2(Fold Change)"] >= log2fc_thresh) &
        (df["p-value"] < p_thresh) &
        (df["FDR p-value"] < fdr_thresh),
        "Significance"
    ] = "Up"
    df.loc[
        (df["log2(Fold Change)"] <= -log2fc_thresh) &
        (df["p-value"] < p_thresh) &
        (df["FDR p-value"] < fdr_thresh),
        "Significance"
    ] = "Down"

    df["UniqueID"] = df["UniqueID"].astype(str).str.strip()

    # ---- Merge Annotation, Headgroup, and Class metadata ----
    if meta_lookup is not None and not meta_lookup.empty:
        df = df.merge(meta_lookup, how="left", on="UniqueID")
    else:
        df["Annotation"] = "Unknown"
        df["Annotation Type"] = "Unknown"
        df["Headgroup"] = "Unknown"
        df["Lipid Class"] = "Unknown"

    # ---- Reorder columns for tidy output ----
    preferred_order = [
        "UniqueID", "Annotation", "Annotation Type", "Headgroup", "Lipid Class",
        "Fold Change", "log2(Fold Change)",
        "p-value", "FDR p-value", "-log10(FDR p-value)", "Significance",
    ]
    existing = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    df = df[existing + remaining]

    return df

def _plot_volcano(df, g1, g2, save_dir: Path, p_thresh, fdr_thresh, fc_thresh):
    plt.close('all')

    # Handle empty comparisons: still save a placeholder plot
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=False, facecolor="white")
        ax.set_facecolor("white")
        fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.20)

        ax.text(
            0.5, 0.5,
            f"No valid features to plot\n({g1} vs {g2})",
            ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title(f"Volcano Plot: {g1} vs {g2}", fontsize=15, pad=15, fontweight="semibold")
        ax.set_xlabel("log2(Fold Change)")
        ax.set_ylabel("-log10(FDR p-value)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")

        fname_base = f"VolcanoFDR_{_sanitize_filename(g1)}_vs_{_sanitize_filename(g2)}_FC{_sanitize_filename(str(fc_thresh))}"
        fig.savefig(save_dir / f"{fname_base}.png", dpi=100, facecolor=fig.get_facecolor())
        fig.savefig(save_dir / f"{fname_base}.svg", dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    # Keep only finite rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["log2(Fold Change)", "-log10(FDR p-value)"]
    )

    # --- create a Figure and an Axes; draw everything on "ax" ---
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=False, facecolor="white")
    ax.set_facecolor("white")
    # Reserve bottom space for the legend; do this ONCE.
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.30)

    log2fc_thresh = np.log2(fc_thresh)

    # Axis limits — use true min/max only (no percentiles)
    fc_line = np.log2(fc_thresh)

    # X limits
    x = df["log2(Fold Change)"].to_numpy()
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    x_span = max(1e-6, x_max - x_min)
    x_pad  = max(0.5, 0.05 * x_span)
    ax.set_xlim(
        min(x_min - x_pad, -fc_line - 0.1),
        max(x_max + x_pad,  fc_line + 0.1),
    )

    # Y limits
    y = df["-log10(FDR p-value)"].to_numpy()
    y_max = float(np.nanmax(y)) if y.size else 0.0
    y_cut = -np.log10(fdr_thresh)
    y_pad = max(0.1, 0.08 * max(1.0, y_max))
    ax.set_ylim(0.0, max(y_max + y_pad, y_cut + 0.2, 1.25))

    yvals = df["-log10(FDR p-value)"].to_numpy()
    y_max = float(np.nanmax(yvals)) if yvals.size else 0.0
    y_cut = -np.log10(fdr_thresh)
    norm = Normalize(vmin=0.0, vmax=max(y_max, y_cut))

    cmap_up = plt.colormaps.get_cmap("Reds")
    cmap_down = plt.colormaps.get_cmap("Blues")
    gray_cmap = LinearSegmentedColormap.from_list("custom_gray", ["#d4d4d4", "#292929"], N=256)

    up = df[df["Significance"] == "Up"]
    down = df[df["Significance"] == "Down"]
    ns = df[df["Significance"] == "Not Significant"]

    # --- all scatters drawn on ax (no stray parens) ---
    ax.scatter(
        ns["log2(Fold Change)"],
        _add_jitter(ns["-log10(FDR p-value)"]),
        c=gray_cmap(norm(ns["-log10(FDR p-value)"])),
        edgecolors="none", alpha=0.8, s=25, linewidth=0.0,
        label="Not Significant", zorder=1
    )
    ax.scatter(
        down["log2(Fold Change)"],
        _add_jitter(down["-log10(FDR p-value)"]),
        c=cmap_down(norm(down["-log10(FDR p-value)"])),
        edgecolors="black", linewidth=0.25, alpha=0.95, s=36,
        label="Down", zorder=3
    )
    ax.scatter(
        up["log2(Fold Change)"],
        _add_jitter(up["-log10(FDR p-value)"]),
        c=cmap_up(norm(up["-log10(FDR p-value)"])),
        edgecolors="black", linewidth=0.25, alpha=0.95, s=36,
        label="Up", zorder=4
    )


    # Threshold lines
    ax.axhline(y=-np.log10(fdr_thresh), color="gray", linestyle="--", linewidth=0.6, zorder=2)
    ax.axvline(x= log2fc_thresh,        color="gray", linestyle="--", linewidth=0.6)
    ax.axvline(x=-log2fc_thresh,        color="gray", linestyle="--", linewidth=0.6)

    # Labels / title
    ax.set_title(f"Volcano Plot: {g1} vs {g2}", fontsize=15, pad=15, fontweight="semibold")
    ax.set_xlabel("log2(Fold Change)", fontsize=12, labelpad=6)
    ax.set_ylabel("-log10(FDR p-value)", fontsize=12, labelpad=6)

    ax.tick_params(labelsize=10)

    # Legend (bottom center)
    legend_labels = [
        f"Significantly Increased (FC {g1}/{g2} ≥ {fc_thresh:.2f}, p < {p_thresh}, FDR < {fdr_thresh}): {len(up)}",
        f"Significantly Decreased (FC {g1}/{g2} ≤ {1/fc_thresh:.2f}, p < {p_thresh}, FDR < {fdr_thresh}): {len(down)}",
        "Not Significant",
    ]
    handles = [
        Line2D([], [], marker='o', color='none', markerfacecolor='#aa1515', markeredgecolor='#aa1515', markersize=8, label=legend_labels[0]),
        Line2D([], [], marker='o', color='none', markerfacecolor='#1e4b9e', markeredgecolor='#1e4b9e', markersize=8, label=legend_labels[1]),
        Line2D([], [], marker='o', color='none', markerfacecolor='#818182', markeredgecolor='#818182', markersize=8, label=legend_labels[2]),
    ]
    # legend anchored below the axes but inside the reserved bottom margin
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),  # within the figure canvas
        ncol=1,
        frameon=True,
        fontsize=11
    )

    # Add full rectangular border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # Save
    fname_base = f"VolcanoFDR_{_sanitize_filename(g1)}_vs_{_sanitize_filename(g2)}_FC{_sanitize_filename(str(fc_thresh))}"
    fig.savefig(save_dir / f"{fname_base}.png", dpi=120, facecolor=fig.get_facecolor())
    fig.savefig(save_dir / f"{fname_base}.svg", dpi=120, facecolor=fig.get_facecolor())
    
    plt.close(fig)
    
# ==========================================================
# Summary tables (CSV + Excel)
# ==========================================================
def _save_summary_tables(csv_dir: Path, output_csv: Path, output_excel: Path,
                         fc_thresh: float, fdr_thresh: float, p_thresh: float):
    files = sorted([f for f in os.listdir(csv_dir) if f.endswith("_FDR.csv")])
    if not files:
        print("[Volcano] No _FDR.csv files found.", flush=True)
        return

    meta_cols = ["UniqueID", "Annotation", "Annotation Type", "Headgroup", "Lipid Class"]
    numeric_cols = ["Fold Change", "log2(Fold Change)", "p-value", "FDR p-value"]

    combined_meta = None
    combined_numeric = None
    summaries = []

    for fname in files:
        comp = fname.replace("_FDR.csv", "")
        df = pd.read_csv(csv_dir / fname)

        if "UniqueID" not in df.columns:
            print(f"[Volcano] Skipping {fname}: missing UniqueID.", flush=True)
            continue

        df["UniqueID"] = df["UniqueID"].astype(str).str.strip()

        # Ensure all metadata columns exist (even if empty)
        for c in meta_cols:
            if c not in df.columns:
                df[c] = ""

        # Ensure numeric columns exist (even if empty)
        for c in numeric_cols:
            if c not in df.columns:
                df[c] = np.nan

        # ---- Summary counts (robust) ----
        up = (df["Fold Change"] > fc_thresh) & (df["FDR p-value"] < fdr_thresh) & (df["p-value"] < p_thresh)
        down = (df["Fold Change"] < 1 / fc_thresh) & (df["FDR p-value"] < fdr_thresh) & (df["p-value"] < p_thresh)
        summaries.append((comp, int(np.nansum(up)), int(np.nansum(down))))

        # ---- Meta table (one time, keyed by UniqueID) ----
        this_meta = df[meta_cols].drop_duplicates("UniqueID").set_index("UniqueID")

        if combined_meta is None:
            combined_meta = this_meta
        else:
            # fill missing meta only
            combined_meta = combined_meta.combine_first(this_meta)

        # ---- Per-comparison numeric table keyed by UniqueID ----
        use = df[["UniqueID"] + numeric_cols].copy()

        use.rename(columns={
            "Fold Change": "FoldChange",
            "log2(Fold Change)": "log2FC",
            "p-value": "pval",
            "FDR p-value": "FDR_p",
        }, inplace=True)

        for c in ["FoldChange", "log2FC", "pval", "FDR_p"]:
            use[c] = pd.to_numeric(use[c], errors="coerce")

        use["Sig_FDR_and_p"] = np.select(
            [
                (use["FDR_p"] < fdr_thresh) & (use["pval"] < p_thresh) & (use["FoldChange"] > 1),
                (use["FDR_p"] < fdr_thresh) & (use["pval"] < p_thresh) & (use["FoldChange"] < 1),
            ],
            ["Significantly increased", "Significantly decreased"],
            default="Not significant"
        )
        use["Sig_p_only"] = np.select(
            [
                (use["pval"] < p_thresh) & (use["FoldChange"] > 1),
                (use["pval"] < p_thresh) & (use["FoldChange"] < 1),
            ],
            ["Significantly increased", "Significantly decreased"],
            default="Not significant"
        )

        use = use.set_index("UniqueID")

        # MultiIndex columns: (comp, variable)
        use.columns = pd.MultiIndex.from_product([[comp], list(use.columns)])

        if combined_numeric is None:
            combined_numeric = use
        else:
            combined_numeric = combined_numeric.join(use, how="outer")

    if combined_meta is None:
        print("[Volcano] No valid files for summary.", flush=True)
        return

    # Final combined table: meta first, then per-comparison multiindex numeric
    # Keep UniqueID as the index so we can safely concat with MultiIndex columns.
    if combined_meta.index.name != "UniqueID":
        combined_meta.index.name = "UniqueID"

    if combined_numeric is None:
        combined_wide = combined_meta.copy()
    else:
        # Both have index=UniqueID; combined_numeric has MultiIndex columns
        combined_wide = pd.concat([combined_meta, combined_numeric], axis=1, join="outer")

    # ---- Write CSV (must be flat columns) ----
    combined_csv = combined_wide.reset_index()  # UniqueID back as a column

    def _flatten_col(c):
        if isinstance(c, tuple):
            # (comparison, variable) -> "comparison__variable"
            return f"{c[0]}__{c[1]}"
        return str(c)

    combined_csv.columns = [_flatten_col(c) for c in combined_csv.columns]
    combined_csv.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Keep these for the Excel writer below
    combined_meta_reset = combined_meta.reset_index()
    combined_numeric_reset = combined_numeric.reset_index() if combined_numeric is not None else None

    # ---------------- Excel ----------------
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Volcano Data"

    # Build 2-row headers: Row1 comparison, Row2 variable
    # Meta columns get blank comparison header
    meta_headers_row1 = [""] * len(meta_cols)
    meta_headers_row2 = meta_cols

    if combined_numeric is None:
        ws1.append(meta_headers_row1)
        ws1.append(meta_headers_row2)
        for _, r in combined_meta_reset[meta_cols].iterrows():
            ws1.append(r.tolist())
    else:
        # Determine numeric columns from combined_numeric
        numeric_multi = combined_numeric.columns  # MultiIndex
        row1 = meta_headers_row1 + [c[0] for c in numeric_multi]
        rename_map = {
            "FoldChange": "Fold Change",
            "log2FC": "log₂(Fold Change)",
            "pval": "raw p-value",
            "FDR_p": "FDR-p",
            "Sig_FDR_and_p": "Significant (FDR-p & raw p)",
            "Sig_p_only": "Significant (raw p only)"
        }
        row2 = meta_headers_row2 + [rename_map.get(c[1], c[1]) for c in numeric_multi]

        ws1.append(row1)
        ws1.append(row2)

        # Write data
        # Reconstruct in the same order: meta then numeric
        meta_part = combined_meta_reset[meta_cols].copy()
        numeric_part = combined_numeric_reset.drop(columns=["UniqueID"]).copy()
        full = pd.concat([meta_part, numeric_part], axis=1)

        for _, r in full.iterrows():
            ws1.append(r.tolist())

        # Merge header cells for each comparison (Row 1 only, numeric region)
        from openpyxl.utils import get_column_letter
        start_col = len(meta_cols) + 1
        last = None
        run_start = start_col
        for j in range(start_col, start_col + len(numeric_multi)):
            comp = row1[j - 1]
            if last is None:
                last = comp
                run_start = j
            elif comp != last:
                if last:
                    ws1.merge_cells(start_row=1, start_column=run_start, end_row=1, end_column=j - 1)
                last = comp
                run_start = j
        if last:
            ws1.merge_cells(start_row=1, start_column=run_start, end_row=1, end_column=start_col + len(numeric_multi) - 1)

    # Format headers
    header_font = Font(bold=True)
    center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for cell in ws1[1] + ws1[2]:
        cell.font = header_font
        cell.alignment = center_align

    # Autosize
    from openpyxl.utils import get_column_letter
    for i, column_cells in enumerate(ws1.columns, start=1):
        max_length = 0
        for cell in column_cells:
            if cell.value is None:
                continue
            max_length = max(max_length, len(str(cell.value)))
        ws1.column_dimensions[get_column_letter(i)].width = min(max_length + 2, 40)

    # Summary Counts sheet
    ws2 = wb.create_sheet("Summary Counts")
    ws2.append([
        "Comparison",
        f"Up (FC ≥ {fc_thresh}, raw p < {p_thresh}, FDR < {fdr_thresh})",
        f"Down (FC ≤ 1/{fc_thresh}, raw p < {p_thresh}, FDR < {fdr_thresh})"
    ])
    for comp, up_cnt, down_cnt in summaries:
        ws2.append([comp, up_cnt, down_cnt])

    for i, column_cells in enumerate(ws2.columns, start=1):
        max_length = 0
        for cell in column_cells:
            if cell.value is None:
                continue
            max_length = max(max_length, len(str(cell.value)))
        ws2.column_dimensions[get_column_letter(i)].width = min(max_length + 2, 40)

    wb.save(output_excel)
    print(f"[Volcano] Saved Excel with two sheets: {output_excel}", flush=True)

# ==========================================================
# Class extraction / grouping (for bar + bubble plots)
# ==========================================================
def _extract_Class(name):
    """Extract lipid Class token from your 'UniqueID' naming."""
    if pd.isna(name):
        return "Unknown"
    name = str(name)
    if "|" in name:
        lipid_part = name.split("|")[-1].strip()
    else:
        lipid_part = name.strip()
    m = re.match(r"^([A-Z]+(?: O)?(?:-)?[A-Z]*)", lipid_part)
    return m.group(1) if m else "Unknown"

# Master class order
_CLASS_ORDER_DEFAULT = _CLASS_ORDER

def _resolve_class_order(sample_type: str) -> list[str]:
    st = (sample_type or "").strip().lower()
    if st.startswith("bact"):
        return _CLASS_ORDER_BACTERIA
    elif st.startswith("mamm"):
        return _CLASS_ORDER_MAMMALIAN
    return _CLASS_ORDER_DEFAULT

# ==========================================================
# Class bar plots (Up black / Down gray)
# ==========================================================
def _generate_class_barplots_from_csv(input_dir: Path, output_dir: Path, sample_type: str,
                                      p_value_threshold: float, fdr_threshold: float, fold_change_threshold: float):
    print("\n[Volcano] Generating class bar plots", flush = True)
    plt.close('all')
    csv_input = input_dir / "CSV_files"
    if not csv_input.exists():
        print(f"[Volcano] Skipping plots: missing folder {csv_input}", flush=True)
        return
    files = [p.name for p in csv_input.glob("*_FDR.csv")]

    # class order by sample_type
    class_order = _resolve_class_order(sample_type)

    _ensure_dir(output_dir)

    for file in files:
        df = pd.read_csv(csv_input / file)

        # --- Robust checks and normalization ---
        if "UniqueID" not in df.columns or "Significance" not in df.columns:
            print(f"[Volcano] Skipping {file} (missing columns)", flush = True)
            continue

        # Normalize 'Significance' labels (avoid trailing spaces or lowercase)
        df["Significance"] = df["Significance"].astype(str).str.strip().str.capitalize()

        if "Significance" in df.columns:
            df["Significance"] = (
                df["Significance"]
                .astype(str)
                .str.strip()          # remove leading/trailing spaces
                .str.replace(r"\s+", " ", regex=True)  # collapse weird whitespace
                .str.capitalize()     # unify case ("up"→"Up", "down"→"Down")
            )
    
        # Ensure expected values exist
        valid_sig = df["Significance"].isin(["Up", "Down"])
        print(f"[Volcano] {file}: {valid_sig.sum()} significant entries detected", flush = True)
        if valid_sig.sum() == 0:
            print(f"[Volcano] {file}: 0 significant entries; producing empty plot", flush=True)
            # Create an empty but labeled figure so folders are consistent
            fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
            ax.set_facecolor("white")
            ax.text(0.5, 0.5, "No significant features\n(at current thresholds)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            outname = output_dir / file.replace("_FDR.csv", "_EMPTY.png")
            fig.savefig(outname, dpi=120, facecolor=fig.get_facecolor())
            plt.close(fig)
            # and continue to next file
            continue

        # --- Prefer 'Lipid Class' column from metadata ---
        if "Lipid Class" in df.columns:
            df["Class_Group"] = (
                df["Lipid Class"]
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
                .replace("nan", "Unknown")
            )
        else:
            # fallback to parsed class from UniqueID
            df["Class"] = df["UniqueID"].apply(_extract_Class)
            df["Class_Group"] = df["Class"].map(_CLASS_GROUP_MAP).fillna(df["Class"])
        # -------------------------------------------------

        # --- Stable counts (no NaNs, fixed order) ---
        # --- Stable counts even if one column is missing ---
        count_data = (
            df.groupby(["Class_Group", "Significance"])
            .size()
            .unstack(fill_value=0)
        )
        # Ensure both columns exist
        for col in ("Up", "Down"):
            if col not in count_data.columns:
                count_data[col] = 0

        # Order columns and classes
        count_data = count_data[["Up", "Down"]].reindex(class_order, fill_value=0)

        # Up positive, Down negative for diverging bars
        count_data["Up"] = count_data["Up"].astype(int)
        count_data["Down"] = -count_data["Down"].astype(int)

        # --- Plot with robust limits ---
        # --- Figure/axes with explicit white backgrounds ---
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
        ax.set_facecolor("white")

        x = np.arange(len(count_data))
        ax.bar(x, count_data["Up"].to_numpy(), color="black", width=0.8)
        ax.bar(x, count_data["Down"].to_numpy(), color="gray", width=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(count_data.index)
        # anchor and pad so 90° labels don’t get clipped
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=90, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=8)

        comparison = file.replace("_FDR.csv", "")
        ax.set_title(f"Class distribution for significantly altered lipids ({comparison})", pad=28)

        black_label = rf'raw p < {p_value_threshold}, FDR < {fdr_threshold}, FC ≥ {fold_change_threshold:.2f}'
        gray_label  = rf'raw p < {p_value_threshold}, FDR < {fdr_threshold}, FC ≤ {1/fold_change_threshold:.2f}'

        handles = [
            plt.matplotlib.patches.Patch(color="black", label=black_label),
            plt.matplotlib.patches.Patch(color="gray",  label=gray_label),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),          # relative to the *figure*
            bbox_transform=fig.transFigure,
            ncol=2,
            frameon=False,
            fontsize=10
        )

        ax.axhline(0, color="black", linewidth=0.8)

        # Robust y-limits
        abs_counts = np.abs(count_data[["Up", "Down"]].to_numpy().ravel())
        ymax = int(np.nanpercentile(abs_counts, 99)) if abs_counts.size else 0
        ymax = max(ymax, int(abs_counts.max()) if abs_counts.size else 0, 1)
        ax.set_ylim(-ymax - 1, ymax + 1)

        ax.set_ylabel("Number of Significantly Altered Lipids")

        # Keep room for tick labels and legend; save on white background (no transparency)
        fig.subplots_adjust(bottom=0.28, top=0.84)

        # Add full rectangular border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")

        outpath = output_dir / f"{comparison}_class_distribution_barplot.png"
        fig.savefig(outpath, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

# ==========================================================
# Bubble plot (design 1): y = Class, color by log2FC (seismic)
# ==========================================================
def _generate_bubble_plot_from_csv(input_dir: Path, output_dir: Path, sample_type: str):
    plt.rcParams["font.size"] = 24
    print("\n[Volcano] Generating bubble plots (design 1)", flush = True)
    plt.close('all')
    csv_input = input_dir / "CSV_files"
    if not csv_input.exists():
        print(f"[Volcano] Skipping plots: missing folder {csv_input}", flush=True)
        return
    files = [f for f in os.listdir(csv_input) if f.endswith("_FDR.csv")]

    ordered_classes = _resolve_class_order(sample_type)

    _ensure_dir(output_dir)

    for file in files:
        df = pd.read_csv(csv_input / file)

        # --- Robust checks and normalization ---
        if "UniqueID" not in df.columns or "Significance" not in df.columns:
            print(f"[Volcano] Skipping {file} (missing columns)", flush = True)
            continue

        # Normalize 'Significance' labels (avoid trailing spaces or lowercase)
        df["Significance"] = df["Significance"].astype(str).str.strip().str.capitalize()

        # Ensure expected values exist
        valid_sig = df["Significance"].isin(["Up", "Down"])
        print(f"[Volcano] {file}: {valid_sig.sum()} significant entries detected", flush = True)
        if valid_sig.sum() == 0:
            print(f"[Volcano] {file}: 0 significant entries; producing empty plot", flush=True)
            # Create an empty but labeled figure so folders are consistent
            plt.figure(figsize=(10, 4))
            plt.text(0.5, 0.5, "No significant features\n(at current thresholds)",
                    ha="center", va="center")
            plt.axis("off")
            outname = output_dir / file.replace("_FDR.csv", "_EMPTY.png")
            plt.savefig(outname, dpi=120)
            plt.close()
            # and continue to next file
            continue

        if "Lipid Class" in df.columns:
            df["Class_Group"] = (
                df["Lipid Class"]
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
                .replace("nan", "Unknown")
            )
        else:
            df["Class"] = df["UniqueID"].apply(_extract_Class)
            df["Class_Group"] = df["Class"].map(_CLASS_GROUP_MAP).fillna(df["Class"])

        sig_df = df[df["Significance"].isin(["Up", "Down"])].copy()
        sig_df = sig_df[
            sig_df["log2(Fold Change)"].notna() &
            sig_df["FDR p-value"].notna() &
            sig_df["Class_Group"].notna()
        ]
        sig_df = sig_df[sig_df["Class_Group"].apply(lambda x: isinstance(x, str))]
        if sig_df.empty:
            print(f"[Volcano] {file}: no significant lipids to plot.", flush = True)
            continue

        # Map Classes to indices for y-axis
        y_mapping = {cls: i for i, cls in enumerate(ordered_classes)}
        sig_df["y"] = sig_df["Class_Group"].map(y_mapping).astype(float)

        # Colors by log2FC; use diverging only if data crosses 0
        fc_vals = sig_df["log2(Fold Change)"].to_numpy()
        if np.isfinite(fc_vals).sum() == 0:
            continue

        lo, hi = np.nanpercentile(fc_vals, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -1.0, 1.0

        span = max(0.5, hi - lo)
        vmin = lo - 0.05 * span
        vmax = hi + 0.05 * span

        if lo < 0 < hi:
            # Data spans both sides of zero → diverging norm around 0
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            # All-positive or all-negative → sequential norm
            from matplotlib.colors import Normalize
            if vmin >= vmax:  # guard against identical values
                vmin, vmax = vmin - 0.5, vmax + 0.5
            norm = Normalize(vmin=vmin, vmax=vmax)

        # --- Bubble sizes from FDR with percentile caps ---
        fdr_log10 = -np.log10(sig_df["FDR p-value"].clip(lower=1e-300, upper=1.0)).to_numpy()
        s_lo, s_hi = np.nanpercentile(fdr_log10, [5, 95])
        if not np.isfinite(s_lo): s_lo = 0.0
        if not np.isfinite(s_hi) or s_hi <= s_lo: s_hi = max(s_lo + 1.0, 2.0)
        fdr_capped = np.clip(fdr_log10, s_lo, s_hi)
        # Map to a consistent on-screen size range
        bubble_sizes = np.interp(fdr_capped, [s_lo, s_hi], [30, 600])

        fig, ax = plt.subplots(figsize=(12, max(12, 0.4 * len(ordered_classes))))
        sc = ax.scatter(
            sig_df["log2(Fold Change)"].to_numpy(), sig_df["y"].to_numpy(),
            s=bubble_sizes, c=fc_vals, cmap="seismic", norm=norm,
            edgecolor="black", linewidth=0.4, alpha=0.85
        )

        ax.set_yticks(list(y_mapping.values()))
        ax.set_yticklabels(list(y_mapping.keys()), fontsize = 18)

        # Robust x-limits
        ax.set_xlim(vmin, vmax)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.tick_params(axis='x', labelsize=18)

        ax.set_xlabel("log2(Fold Change)", fontsize=22, labelpad=10)
        ax.set_ylabel("Lipid Class", fontsize=22, labelpad=10)
        ax.set_title(f"Fold-change distribution for significantly altered lipids\n({file.replace('_FDR.csv', '')})", pad=20, fontsize=22)

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, label="log2(Fold Change)", shrink=0.4, aspect=20)
        cbar.ax.tick_params(labelsize=16)              # tick labels
        cbar.set_label("log2(Fold Change)", fontsize=20)  # colorbar label

        # Bubble size legend (two reference sizes based on the mapping)
        legend_sizes = [np.interp(v, [s_lo, s_hi], [30, 600]) for v in [s_lo, s_hi]]
        legend_labels = [f"FDR≈{10**(-s_lo):.2g}", f"FDR≈{10**(-s_hi):.2g}"]
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", label=legend_labels[i],
                   markerfacecolor="gray", markeredgecolor="black", markersize=np.sqrt(s/np.pi))
            for i, s in enumerate(legend_sizes)
        ]
        ax.legend(handles=legend_handles, title="FDR (size)\n", fontsize = 16, bbox_to_anchor=(1.28, 1),
                  loc="upper left", borderaxespad=0.0, frameon=False)

        plt.tight_layout()

        # Add full rectangular border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")
            
        outname = output_dir / file.replace("_FDR.csv", "_bubble_plot.png")
        fig.savefig(outname, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

# ==========================================================
# Bubble plot (design 2): x = Class (jitter), color by Class palette
# ==========================================================
def _generate_bubble_plot_from_csv_design2(input_dir: Path, output_dir: Path, sample_type: str):
    plt.rcParams["font.size"] = 24
    print("\n[Volcano] Generating bubble plots (design 2)", flush = True)
    plt.close('all')
    csv_input = input_dir / "CSV_files"
    if not csv_input.exists():
        print(f"[Volcano] Skipping plots: missing folder {csv_input}", flush=True)
        return
    files = [f for f in os.listdir(csv_input) if f.endswith("_FDR.csv")]

    ordered_classes = _resolve_class_order(sample_type)

    _ensure_dir(output_dir)

    for file in files:
        df = pd.read_csv(csv_input / file)

        # --- Robust checks and normalization ---
        if "UniqueID" not in df.columns or "Significance" not in df.columns:
            print(f"[Volcano] Skipping {file} (missing columns)", flush = True)
            continue

        # Normalize 'Significance' labels (avoid trailing spaces or lowercase)
        df["Significance"] = df["Significance"].astype(str).str.strip().str.capitalize()

        # Ensure expected values exist
        valid_sig = df["Significance"].isin(["Up", "Down"])
        print(f"[Volcano] {file}: {valid_sig.sum()} significant entries detected", flush = True)
        if valid_sig.sum() == 0:
            print(f"[Volcano] {file}: 0 significant entries; producing empty plot", flush=True)
            # Create an empty but labeled figure so folders are consistent
            plt.figure(figsize=(12, 4))
            plt.text(0.5, 0.5, "No significant features\n(at current thresholds)",
                    ha="center", va="center")
            plt.axis("off")
            outname = output_dir / file.replace("_FDR.csv", "_EMPTY.png")
            plt.savefig(outname, dpi=120)
            plt.close()
            # and continue to next file
            continue

        if "Lipid Class" in df.columns:
            df["Class_Group"] = (
                df["Lipid Class"]
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
                .replace("nan", "Unknown")
            )
        else:
            df["Class"] = df["UniqueID"].apply(_extract_Class)
            df["Class_Group"] = df["Class"].map(_CLASS_GROUP_MAP).fillna(df["Class"])
        
        sig_df = df[df["Significance"].isin(["Up", "Down"])].copy()
        sig_df = sig_df[
            sig_df["log2(Fold Change)"].notna() &
            sig_df["FDR p-value"].notna() &
            sig_df["Class_Group"].notna()
        ]
        sig_df = sig_df[sig_df["Class_Group"].apply(lambda x: isinstance(x, str))]
        if sig_df.empty:
            print(f"[Volcano] {file}: no significant lipids to plot.", flush = True)
            continue

               # X positions (categorical → numeric + jitter)
        x_mapping = {cls: i for i, cls in enumerate(ordered_classes)}
        sig_df["x"] = sig_df["Class_Group"].map(x_mapping).astype(float)
        # Keep jitter tight so points don't get clipped at figure edges
        sig_df["x_jittered"] = sig_df["x"] + np.random.uniform(-0.22, 0.22, size=len(sig_df))

        # Robust bubble sizes from FDR (percentile caps)
        fdr_log10 = -np.log10(sig_df["FDR p-value"].clip(lower=1e-300, upper=1.0)).to_numpy()
        s_lo, s_hi = np.nanpercentile(fdr_log10, [5, 95])
        if not np.isfinite(s_lo): s_lo = 0.0
        if not np.isfinite(s_hi) or s_hi <= s_lo: s_hi = max(s_lo + 1.0, 2.0)
        fdr_capped = np.clip(fdr_log10, s_lo, s_hi)
        bubble_sizes = np.interp(fdr_capped, [s_lo, s_hi], [30, 600])

        # Colors by Class (Set2), fallback gray
        palette = sns.color_palette("Set2", len(ordered_classes))
        color_dict = dict(zip(ordered_classes, palette))
        colors = [color_dict.get(cls, (0.5, 0.5, 0.5)) for cls in sig_df["Class_Group"]]

        fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * len(ordered_classes))))
        ax.scatter(
            sig_df["x_jittered"].to_numpy(), sig_df["log2(Fold Change)"].to_numpy(),
            s=bubble_sizes, c=colors, edgecolor="black", linewidth=0.4, alpha=0.7
        )

        # Robust symmetric y-limits from percentiles of |log2FC|
        fc_vals = sig_df["log2(Fold Change)"].to_numpy()
        lo, hi = np.nanpercentile(np.abs(fc_vals), [1, 99])
        ymax = max(0.5, hi)
        ax.set_ylim(-ymax * 1.1, ymax * 1.1)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
               
        ax.set_xlabel("log2(Fold Change)", fontsize=22, labelpad=10)
        ax.set_ylabel("Lipid Class", fontsize=22, labelpad=10)
        ax.set_title(
            f"Fold-change distribution for significantly altered lipids\n({file.replace('_FDR.csv', '')})",
            fontsize=22,
            pad=20,)

        ax.set_xticks(list(x_mapping.values()))
        ax.set_xticklabels(list(x_mapping.keys()), rotation=90, fontsize = 16)
        ax.tick_params(axis='y', labelsize=18)

        # Size legend consistent with mapping
        legend_sizes = [np.interp(v, [s_lo, s_hi], [30, 600]) for v in [s_lo, s_hi]]
        legend_labels = [f"FDR≈{10**(-s_lo):.2g}", f"FDR≈{10**(-s_hi):.2g}"]
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", label=legend_labels[i],
                   markerfacecolor="gray", markeredgecolor="black", markersize=np.sqrt(s/np.pi))
            for i, s in enumerate(legend_sizes)
        ]
        ax.legend(handles=legend_handles, title="FDR (size)\n", fontsize = 16, 
                  bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

        plt.tight_layout()

        # Add full rectangular border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")
            
        outname = output_dir / file.replace("_FDR.csv", "_bubble_plot_v2.png")
        fig.savefig(outname, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

# ==========================================================
# Public entry point (called by GUI)
# ==========================================================
def run_volcano(file_path, group_file, save_dir,
                method="fdr_bh", test_type="parametric",
                run_bar_plots=True, run_bubble_plots=True,
                sample_type="Mammalians",
                p_value_threshold=0.05, fdr_threshold=0.10, fold_change_threshold=1.5, 
                group_colors=None, group_order=None):
    """
    Main entry point for Volcano + summaries + optional bar/bubble plots.

    Args mirror your previous script; outputs and folder structure remain the same.
    """
    file_path = Path(file_path)
    save_dir = prepare_output_dir(Path(save_dir))
    volcano_dir = _ensure_dir(save_dir)
    csv_dir = _ensure_dir(volcano_dir / "CSV_files")
    plt.close('all')

    print(f"[Volcano] Starting for: {file_path.name}", flush = True)
    print(f"[Volcano] sample_type received: {sample_type}", flush=True)

    # Load standardized dataset
    # X: samples × features (columns = UniqueID or UniqueIDs)
    # y: sample groups
    # Load data
    X, y, feature_meta = load_dataset(file_path, group_file)
    
    # Ensure alignment and string dtype for group ops
    X = X.reset_index(drop=True)
    y = y.astype(str).str.strip().reset_index(drop=True)

    # -------------------------------------------------------
    # Build metadata lookup table from feature_meta
    # -------------------------------------------------------
    meta_lookup = None
    if isinstance(feature_meta, pd.DataFrame) and not feature_meta.empty:
        meta_lookup = feature_meta.copy()

        # Canonicalize column names
        rename_cols = {}
        for c in meta_lookup.columns:
            cl = str(c).strip().lower()
            if cl in {"uniqueid", "unique id", "feature id", "id"}:
                rename_cols[c] = "UniqueID"
            elif cl in {"annotation"}:
                rename_cols[c] = "Annotation"
            elif cl in {"annotation type", "annotation_type"}:
                rename_cols[c] = "Annotation Type"
            elif cl in {"headgroup", "head group"}:
                rename_cols[c] = "Headgroup"
            elif cl in {"lipid class", "lipid_class", "class"}:
                rename_cols[c] = "Lipid Class"
        if rename_cols:
            meta_lookup = meta_lookup.rename(columns=rename_cols)

        # Split combined column if present
        combo = [c for c in meta_lookup.columns
                if str(c).strip().lower() in {"annotation type headgroup", "annotation type and headgroup"}]
        if combo and ("Annotation Type" not in meta_lookup.columns or "Headgroup" not in meta_lookup.columns):
            col = combo[0]
            s = meta_lookup[col].astype(str).fillna("")
            parts = s.str.split(r"\s*[|;:]\s*", n=1, expand=True)
            if parts.shape[1] == 1:
                parts = s.str.split(r"\s{2,}", n=1, expand=True)
            meta_lookup["Annotation Type"] = parts[0].astype(str).str.strip()
            meta_lookup["Headgroup"] = (parts[1].astype(str).str.strip() if parts.shape[1] > 1 else "")

        # Ensure required columns exist
        if "UniqueID" not in meta_lookup.columns:
            raise ValueError("[Volcano] feature_meta lacks a UniqueID column (or known alias).")

        for c in ["Annotation", "Annotation Type", "Headgroup", "Lipid Class"]:
            if c not in meta_lookup.columns:
                meta_lookup[c] = ""

        # Clean
        meta_lookup["UniqueID"] = meta_lookup["UniqueID"].astype(str).str.strip()
        for c in ["Annotation", "Annotation Type", "Headgroup", "Lipid Class"]:
            meta_lookup[c] = (
                meta_lookup[c]
                .astype(str)
                .replace({"nan": "", "NaN": "", "None": "", "<NA>": "", "NA": ""})
                .str.strip()
            )

        meta_lookup = meta_lookup[["UniqueID", "Annotation", "Annotation Type", "Headgroup", "Lipid Class"]].drop_duplicates("UniqueID")
    else:
        meta_lookup = pd.DataFrame(columns=["UniqueID", "Annotation", "Annotation Type", "Headgroup", "Lipid Class"])
    # -------------------------------------------------------

    # --- Skip QC samples completely --------------------------------------
    mask_non_qc = ~y.str.lower().str.contains("qc", na=False)
    X = X.loc[mask_non_qc].reset_index(drop=True)
    y = y.loc[mask_non_qc].reset_index(drop=True)
    # ---------------------------------------------------------------------

    # Drop empty / nan-like group labels
    bad = y.str.lower().isin(["", "nan", "none", "<na>", "na", "null"])
    X = X.loc[~bad].reset_index(drop=True)
    y = y.loc[~bad].reset_index(drop=True)

    groups = sorted(y.unique().tolist())
    print(f"[Volcano] Groups (excluding QC): {groups}", flush=True)

    #    # Pairwise comparisons (do BOTH directions: A vs B and B vs A)
    for g1 in groups:
        for g2 in groups:
            if str(g1) == str(g2):
                continue

            print(f"[Volcano] Comparing {g1} vs {g2}", flush=True)
            df_volc = _compute_volcano(
                g1, g2, X, y, meta_lookup,
                method=method, test_type=test_type,
                p_thresh=p_value_threshold, fdr_thresh=fdr_threshold,
                fc_thresh=fold_change_threshold
            )

            # Sanitize before saving/plotting
            df_volc = df_volc.replace([np.inf, -np.inf], np.nan)
            df_volc = df_volc.dropna(subset=["Fold Change", "log2(Fold Change)", "p-value", "FDR p-value"])

            # --- log counts clearly ---
            up_n = int((df_volc["Significance"] == "Up").sum())
            down_n = int((df_volc["Significance"] == "Down").sum())
            print(f"[Volcano] {g1}_vs_{g2}: {up_n} Up, {down_n} Down (FC≥{fold_change_threshold:.2f}, "
                  f"p<{p_value_threshold}, FDR<{fdr_threshold})", flush=True)

            csv_out = csv_dir / f"{_sanitize_filename(g1)}_vs_{_sanitize_filename(g2)}_FDR.csv"

            # Always write a file so summaries see every comparison
            if df_volc.empty:
                empty_cols = [
                    "UniqueID", "Annotation", "Annotation Type", "Headgroup", "Lipid Class",
                    "Fold Change", "log2(Fold Change)", "p-value", "FDR p-value",
                    "-log10(FDR p-value)", "Significance",
                ]
                pd.DataFrame(columns=empty_cols).to_csv(csv_out, index=False)
            else:
                df_volc.to_csv(csv_out, index=False)

            # Always plot (function will render an empty placeholder if needed)
            _plot_volcano(
                df_volc, g1, g2, volcano_dir,
                p_thresh=p_value_threshold, fdr_thresh=fdr_threshold, fc_thresh=fold_change_threshold
            )

    # Summary tables
    _save_summary_tables(
        csv_dir=csv_dir,
        output_csv=volcano_dir / "Summary_table_raw.csv",
        output_excel=volcano_dir / "Summary_Volcano.xlsx",
        fc_thresh=fold_change_threshold, fdr_thresh=fdr_threshold, p_thresh=p_value_threshold
    )

    # class bar plots
    if run_bar_plots:
        bar_dir = _ensure_dir(volcano_dir / "Bar_plots")
        _generate_class_barplots_from_csv(
            input_dir=volcano_dir, output_dir=bar_dir, sample_type=sample_type,
            p_value_threshold=p_value_threshold, fdr_threshold=fdr_threshold, fold_change_threshold=fold_change_threshold
        )

    # bubble plots (both designs)
    if run_bubble_plots:
        bubble_dir = _ensure_dir(volcano_dir / "Bubble_plots")
        _generate_bubble_plot_from_csv(
            input_dir=volcano_dir, output_dir=bubble_dir, sample_type=sample_type
        )
        _generate_bubble_plot_from_csv_design2(
            input_dir=volcano_dir, output_dir=bubble_dir, sample_type=sample_type
        )

    print(f"[Volcano] Completed. Output in: {volcano_dir}\n", flush = True)


# Optional local test
if __name__ == "__main__":
    import sys
    fpath = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "statistics" / "Final_Annotated.csv"
    gpath = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "statistics" / "sample_groups_cleaned.csv"
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.cwd() / "statistics" / "Volcano" / "ManualTest"

    run_volcano(
        file_path=fpath,
        group_file=gpath if gpath.exists() else None,
        save_dir=outdir,
        method="fdr_bh",
        test_type="parametric",
        run_bar_plots=True,
        run_bubble_plots=True,
        sample_type="Mammalians",
        p_value_threshold=0.05,
        fdr_threshold=0.10,
        fold_change_threshold=1.5,
    )
