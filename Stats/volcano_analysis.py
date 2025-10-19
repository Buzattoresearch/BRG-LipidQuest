# TODO: check if sample type is being passed from the GUI
# TODO: Add FC, p thresholds to GUI
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

import warnings
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 13
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")

# ==========================================================
# Utilities (general)
# ==========================================================
def _sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:."/\\|?*]', "_", str(s))

def _add_jitter(values, jitter_strength=0.025):
    return values + np.random.uniform(-jitter_strength, jitter_strength, size=len(values))

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ==========================================================
# Volcano core computation + plotting
# ==========================================================
def _compute_volcano(g1_name, g2_name, X, y, meta_lookup,
                     method, test_type, p_thresh, fdr_thresh, fc_thresh):
    """Perform one pairwise comparison and return volcano DataFrame."""
    group1 = X[y == g1_name]
    group2 = X[y == g2_name]
    common = group1.columns.intersection(group2.columns)
    group1 = group1[common]
    group2 = group2[common]

    pvals, valid_UniqueIDs = [], []
    for UniqueID in common:
        x1, x2 = group1[UniqueID], group2[UniqueID]
        if np.std(x1, ddof=1) < 1e-8 or np.std(x2, ddof=1) < 1e-8:
            p = 1.0
        else:
            try:
                if test_type == "non-parametric":
                    _, p = mannwhitneyu(x1, x2, alternative="two-sided")
                else:
                    _, p = ttest_ind(x1, x2, equal_var=False)
            except Exception:
                p = 1.0
        pvals.append(p)
        valid_UniqueIDs.append(UniqueID)

    pvals = np.array(pvals)
    fdr_corrected = multipletests(pvals, alpha=0.05, method=method)[1]
    g1_means = group1[valid_UniqueIDs].mean()
    g2_means = group2[valid_UniqueIDs].mean()

    # Fold change and log2 FC
    fc_abs = np.clip(g1_means / g2_means, 1e-12, None)
    fc_log2 = np.log2(fc_abs)
    
    df = pd.DataFrame({
        "UniqueID": valid_UniqueIDs,
        "Fold Change": np.round(fc_abs, 18),
        "log2(Fold Change)": np.round(fc_log2, 10),
        "p-value": np.round(pvals, 18),
        "FDR p-value": np.round(fdr_corrected, 18),
    })
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

    # ---- Merge Annotation, Headgroup, and Class metadata ----
    if meta_lookup is not None and not meta_lookup.empty:
        # Merge based on UniqueID
        df = df.merge(meta_lookup, how="left", on="UniqueID")
    else:
        df["Annotation"] = "Unknown"
        df["Annotation Type"] = "Unknown"
        df["Headgroup"] = "Unknown"
        df["Lipid Class"] = "Unknown"
    # ---------------------------------------------------------
    
    # ---- Reorder columns for tidy output ----
    preferred_order = [
        "UniqueID",
        "Annotation",
        "Annotation Type", 
        "Headgroup",
        "Lipid Class",
        "Fold Change",
        "log2(Fold Change)",
        "p-value",
        "FDR p-value",
        "-log10(FDR p-value)",
        "Significance",
    ]

    # Only include columns that actually exist (avoids KeyErrors)
    existing = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]

    df = df[existing + remaining]
    # ------------------------------------------

    return df

def _plot_volcano(df, g1, g2, save_dir: Path, p_thresh, fdr_thresh, fc_thresh):
    plt.figure(figsize=(10, 8))
    log2fc_thresh = np.log2(fc_thresh)

    # Color maps
    cmap_up = plt.colormaps.get_cmap("Reds")
    cmap_down = plt.colormaps.get_cmap("Blues")
    gray_cmap = LinearSegmentedColormap.from_list("custom_gray", ["#d4d4d4", "#292929"], N=256)

    norm = Normalize(vmin=df["-log10(FDR p-value)"].min(), vmax=df["-log10(FDR p-value)"].max())

    up = df[df["Significance"] == "Up"]
    down = df[df["Significance"] == "Down"]
    ns = df[df["Significance"] == "Not Significant"]

    plt.scatter(
        up["log2(Fold Change)"], _add_jitter(up["-log10(FDR p-value)"]),
        color=cmap_up(norm(up["-log10(FDR p-value)"])), alpha=0.8, s=30, linewidth=0.8, label="Up"
    )
    plt.scatter(
        down["log2(Fold Change)"], _add_jitter(down["-log10(FDR p-value)"]),
        color=cmap_down(norm(down["-log10(FDR p-value)"])), alpha=0.8, s=30, linewidth=0.8, label="Down"
    )
    plt.scatter(
        ns["log2(Fold Change)"], _add_jitter(ns["-log10(FDR p-value)"]),
        color=gray_cmap(norm(ns["-log10(FDR p-value)"])), alpha=0.8, s=25, linewidth=0.8, label="Not Significant"
    )

    # Threshold lines
    plt.axhline(y=-np.log10(fdr_thresh), color="gray", linestyle="--", linewidth=0.6)
    plt.axvline(x=log2fc_thresh, color="gray", linestyle="--", linewidth=0.6)
    plt.axvline(x=-log2fc_thresh, color="gray", linestyle="--", linewidth=0.6)

    # Titles/labels
    plt.title(f"Volcano Plot: {g1} vs {g2}\n", fontsize=16, loc="center")
    plt.xlabel("log₂(Fold Change)", fontsize=13)
    plt.ylabel("-log₁₀(FDR p-value)", fontsize=13)

    # Legend
    legend_labels = [
        f"Significantly Increased (FC {g1}/{g2} ≥ {fc_thresh:.2f}, p < {p_thresh}, FDR < {fdr_thresh}): {len(up)}",
        f"Significantly Decreased (FC {g1}/{g2} ≤ {1/fc_thresh:.2f}, p < {p_thresh}, FDR < {fdr_thresh}): {len(down)}",
        "Not Significant",
    ]
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor="#aa1515", markeredgecolor="#aa1515", markersize=8, label=legend_labels[0]),
        Line2D([], [], marker="o", color="none", markerfacecolor="#1e4b9e", markeredgecolor="#1e4b9e", markersize=8, label=legend_labels[1]),
        Line2D([], [], marker="o", color="none", markerfacecolor="#818182", markeredgecolor="#818182", markersize=8, label=legend_labels[2]),
    ]
    plt.legend(handles=handles, bbox_to_anchor=(0.5, -0.15), loc="upper center", fontsize=11, frameon=True, ncol=1)

    plt.tight_layout(rect=[0.1, 0.05, 1, 1])

    # Save
    fname_base = f"VolcanoFDR_{_sanitize_filename(g1)}_vs_{_sanitize_filename(g2)}_FC{_sanitize_filename(str(fc_thresh))}"
    plt.savefig(save_dir / f"{fname_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / f"{fname_base}_svg.svg", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


# ==========================================================
# Summary tables (CSV + Excel)
# ==========================================================
def _save_summary_tables(csv_dir: Path, output_csv: Path, output_excel: Path,
                         fc_thresh: float, fdr_thresh: float, p_thresh: float):
    files = sorted([f for f in os.listdir(csv_dir) if f.endswith("_FDR.csv")])
    if not files:
        print("[Volcano] No _FDR.csv files found.", flush=True)
        return

    all_data, summaries = [], []
    
    for fname in files:
        comp = fname.replace("_FDR.csv", "")
        df = pd.read_csv(csv_dir / fname)

        # --- Columns to keep ---
        meta_cols = ["UniqueID", "Annotation", "Annotation Type", "Headgroup", "Lipid Class"]
        numeric_cols = ["Fold Change", "log2(Fold Change)", "p-value", "FDR p-value"]

        existing_meta = [c for c in meta_cols if c in df.columns]
        existing_numeric = [c for c in numeric_cols if c in df.columns]

        use = df[existing_meta + existing_numeric].copy()

        # Rename numeric columns
        use.rename(
            columns={
                "FDR p-value": "FDR_p",
                "p-value": "pval",
                "Fold Change": "FoldChange",
                "log2(Fold Change)": "log2FC",
            },
            inplace=True,
        )

        # Descriptive significance text
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

        # --- Build MultiIndex safely ---
        existing_meta = [c for c in ["UniqueID", "Annotation", "Annotation Type", "Headgroup", "Lipid Class"] if c in use.columns]
        numeric_cols_final = [c for c in use.columns if c not in existing_meta]

        if not all_data:
            # First file → include metadata once
            multi_cols = [("", c) for c in existing_meta] + [(comp, c) for c in numeric_cols_final]
        else:
            # Later files → numeric columns only
            use = use[numeric_cols_final].copy()
            multi_cols = [(comp, c) for c in numeric_cols_final]

        # Assign columns safely (match length)
        if len(use.columns) != len(multi_cols):
            multi_cols = multi_cols[:len(use.columns)]
        use.columns = pd.MultiIndex.from_tuples(multi_cols)

        all_data.append(use)

        # --- Summary counts ---
        up = (df["Fold Change"] > fc_thresh) & (df["FDR p-value"] < fdr_thresh) & (df["p-value"] < p_thresh)
        down = (df["Fold Change"] < 1 / fc_thresh) & (df["FDR p-value"] < fdr_thresh) & (df["p-value"] < p_thresh)
        summaries.append((comp, int(up.sum()), int(down.sum())))

    # Combine all comparison tables
    combined = pd.concat(all_data, axis=1).sort_index()
    # Drop duplicate columns (keeps metadata only once)
    combined = pd.concat(all_data, axis=1)
    # Save to CSV
    combined = combined.loc[:, ~combined.columns.duplicated()].sort_index()
    combined.to_csv(output_csv, encoding="utf-8-sig")

    # ======================================================
    # Excel output with two tabs
    # ======================================================
    wb = Workbook()

    # --- Sheet 1: Detailed Volcano data ---
    ws1 = wb.active
    ws1.title = "Volcano Data"

    # ✅ Keep MultiIndex intact (comparison on top, variable names below)
    if not isinstance(combined.columns, pd.MultiIndex):
        combined.columns = pd.MultiIndex.from_tuples([("", c) for c in combined.columns])

    # Reset index so UniqueID becomes first column
    excel_combined = combined.copy()

    # Apply human-readable renames on the *second-level* column names only
    rename_map = {
        "FoldChange": "Fold Change",
        "log2FC": "log₂(Fold Change)",
        "pval": "raw p-value",
        "FDR_p": "FDR-p",
        "Sig_FDR_and_p": "Significant (FDR-p & raw p)",
        "Sig_p_only": "Significant (raw p only)"
    }
    excel_combined.columns = pd.MultiIndex.from_tuples([
        (a, rename_map.get(b, b)) for a, b in excel_combined.columns
    ])
        
    # --- Write two header rows ---
    # Row 1: comparison names (first cell blank)
    first_row = []
    for i, (a, b) in enumerate(excel_combined.columns):
        if i == 0:
            first_row.append("")  # keep first cell empty
        else:
            first_row.append(str(a) if a else "")
    ws1.append(first_row)

    # Row 2: variable names (including UniqueID)
    second_row = []
    for i, (a, b) in enumerate(excel_combined.columns):
        if b:
            second_row.append(str(b))
        else:
            second_row.append(str(a))
    ws1.append(second_row)

    # --- Write data rows ---
    for _, row in excel_combined.iterrows():
        ws1.append(row.tolist())
        
    # --- Merge top-row cells for each comparison ---
    col_idx = 1
    last_comp = first_row[0]
    merge_start = 1

    for j, comp in enumerate(first_row, start=1):
        if comp != last_comp:
            if last_comp not in ("", None):
                ws1.merge_cells(
                    start_row=1, start_column=merge_start,
                    end_row=1, end_column=j - 1
                )
            merge_start = j
            last_comp = comp

    # Merge the last group
    if last_comp not in ("", None):
        ws1.merge_cells(
            start_row=1, start_column=merge_start,
            end_row=1, end_column=len(first_row)
        )
    
    # --- Format header rows ---
    header_font = Font(bold=True)
    center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for cell in ws1[1] + ws1[2]:
        cell.font = header_font
        cell.alignment = center_align

    # --- Auto-fit column widths based on content ---
    from openpyxl.utils import get_column_letter

    for i, column_cells in enumerate(ws1.columns, start=1):
        # get max text length across cells in this column
        max_length = 0
        for cell in column_cells:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except Exception:
                pass
        adjusted_width = max_length + 2  # small padding
        ws1.column_dimensions[get_column_letter(i)].width = min(adjusted_width, 40)

    # --- Sheet 2: Summary Counts ---
    ws2 = wb.create_sheet("Summary Counts")

    # Header
    ws2.append([
        "Comparison",
        f"Up (FC ≥ {fc_thresh}, raw p < {p_thresh}, FDR < {fdr_thresh})",
        f"Down (FC ≤ 1/{fc_thresh}, raw p < {p_thresh}, FDR < {fdr_thresh})"
    ])

    # Data rows
    for comp, up_cnt, down_cnt in summaries:
        ws2.append([comp, up_cnt, down_cnt])

    # Auto-fit columns for Sheet 2
    for i, column_cells in enumerate(ws2.columns, start=1):
        max_length = 0
        for cell in column_cells:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except Exception:
                pass
        ws2.column_dimensions[get_column_letter(i)].width = min(max_length + 2, 40)
    
    # Save workbook 
    wb.save(output_excel) 
    print(f"[Volcano] Saved Excel with two sheets: {output_excel}", flush = True)

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


# Master class order (same as your previous script)
_CLASS_ORDER = [
    "CAR", "CoA", "FA", "FAHFA", "FAG, FAG, FOH, HC", "NA, NAE, NAT", "WE",
    "MG", "DG", "TG", "DGDG, MGDG, DGMG, MGMG", "DGTA, DGTS, DGCC, GlcADG", "SQDG, SQMG",
    "PA", "LPA", "PC", "LPC", "PE", "LPE", "PG", "LPG", "PI", "PIP", "LPI", "PS, PS-NAc", "LPS",
    "CL", "BMP", "PIM", "GP, Glc-GP",
    "ACer", "Cer", "CerP", "HexCer, GlcCer", "LSM", "MIPC, M(IP)2C", "PE-Cer, PI-Cer",
    "SCer", "SHexCer", "SM", "SPB, HexSPB, SPBP",
    "CE", "ST",
    "PK", "PR", "SL", "Other"
]

# If you want different orders by sample_type, reuse the same for now
_CLASS_ORDER_BACTERIA = _CLASS_ORDER
_CLASS_ORDER_MAMMALIAN = _CLASS_ORDER

# Class → group mapping (as in your code, condensed)
_Class_GROUP_MAP = {
    "CAR":"CAR","Car":"CAR","FA":"FA",
    "FAL":"FAG, FAG, FOH, HC","FAG":"FAG, FAG, FOH, HC","FOH":"FAG, FAG, FOH, HC","HCH":"FAG, FAG, FOH, HC",
    "NA":"NA, NAE, NAT","NAE":"NA, NAE, NAT","NAT":"NA, NAE, NAT",
    "DGDG":"DGDG, MGDG, DGMG, MGMG","MGDG":"DGDG, MGDG, DGMG, MGMG","DGMG":"DGDG, MGDG, DGMG, MGMG","MGMG":"DGDG, MGDG, DGMG, MGMG",
    "DGTA":"DGTA, DGTS, DGCC, GlcADG","DGTS":"DGTA, DGTS, DGCC, GlcADG","DGCC":"DGTA, DGTS, DGCC, GlcADG","GlcADG":"DGTA, DGTS, DGCC, GlcADG","G":"DGTA, DGTS, DGCC, GlcADG",
    "SQDG":"SQDG, SQMG","SQMG":"SQDG, SQMG",
    "BMP":"BMP","LBPA":"BMP",
    "GP":"GP, Glc-GP","Glc-GP":"GP, Glc-GP","CDP-DG":"GP, Glc-GP","PT":"GP, Glc-GP","LPT":"GP, Glc-GP",
    "PS":"PS, PS-NAc","PS-NAc":"PS, PS-NAc","PS ":"PS, PS-NAc",
    "PC O-":"PC","PC":"PC","PnC":"PC",
    "PE O-":"PE","PE":"PE","PnE":"PE",
    "PG O-":"PG","PG":"PG",
    "PI O-":"PI","PI":"PI",
    "PS O-":"PS, PS-NAc",
    "PA O-":"PA","PA":"PA","PPA":"PA",
    "LPC O-":"LPC","LPC":"LPC",
    "LPE O-":"LPE","LPE":"LPE",
    "C":"Cer","Cer":"Cer",
    "MLCL":"CL",
    "ST":"ST","SFE":"ST",
    "ACer":"ACer","AC":"ACer",
    "HexCer":"HexCer, GlcCer","GlcCer":"HexCer, GlcCer","H":"HexCer, GlcCer",
    "MIPC":"MIPC, M(IP)2C","M(IP)2C":"MIPC, M(IP)2C","IPC":"MIPC, M(IP)2C",
    "PE-Cer":"PE-Cer, PI-Cer","PI-Cer":"PE-Cer, PI-Cer","CerPE":"PE-Cer, PI-Cer","CerPI":"PE-Cer, PI-Cer",
    "HexSPB":"SPB, HexSPB, SPBP","SPB":"SPB, HexSPB, SPBP","SPBP":"SPB, HexSPB, SPBP",
    "TG":"TG","TG O-":"TG",
    "N":"Other","": "Other"
}


# ==========================================================
# Class bar plots (Up black / Down gray)
# ==========================================================
def _generate_class_barplots_from_csv(input_dir: Path, output_dir: Path, sample_type: str,
                                      p_value_threshold: float, fdr_threshold: float, fold_change_threshold: float):
    print("\n[Volcano] Generating class bar plots", flush = True)
    csv_input = input_dir / "CSV_files"
    files = [f for f in os.listdir(csv_input) if f.endswith("_FDR.csv")]

    # class order by sample_type
    if sample_type == "Bacteria":
        class_order = _CLASS_ORDER_BACTERIA
    elif sample_type == "Mammalians":
        class_order = _CLASS_ORDER_MAMMALIAN
    else:
        class_order = _CLASS_ORDER

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
            print(f"[Volcano] Skipping {file} (no significant entries)", flush = True)
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
            df["Class_Group"] = df["Class"].map(_Class_GROUP_MAP).fillna(df["Class"])
        # -------------------------------------------------

        count_data = df.groupby(["Class_Group", "Significance"]).size().unstack(fill_value=0)
        count_data["Up"] = count_data.get("Up", 0)
        count_data["Down"] = -count_data.get("Down", 0)
        count_data = count_data.reindex(class_order, fill_value=0)

        plt.figure(figsize=(12, 6))
        x = np.arange(len(count_data))
        plt.bar(x, count_data["Up"], color="black", width=0.8)
        plt.bar(x, count_data["Down"], color="gray", width=0.8)
        plt.xticks(x, count_data.index, rotation=90)

        comparison = file.replace("_FDR.csv", "")
        plt.title(f"Class distribution for significantly altered lipids ({comparison})", pad=30)

        black_label = rf'raw p < {p_value_threshold}, FDR < {fdr_threshold}, FC ≥ {fold_change_threshold:.2f}'
        gray_label  = rf'raw p < {p_value_threshold}, FDR < {fdr_threshold}, FC ≤ {1/fold_change_threshold:.2f}'

        plt.legend(
            handles=[
                plt.matplotlib.patches.Patch(color="black", label=black_label),
                plt.matplotlib.patches.Patch(color="gray", label=gray_label)
            ],
            loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=10
        )

        plt.axhline(0, color="black", linewidth=0.8)
        try:
            max_up = max(int(count_data["Up"].max()), 0)
            max_dn = max(int(-count_data["Down"].min()), 0)
            ylim = max(max_up, max_dn)
            buffer = max(1, int(np.ceil(ylim * 0.05)))
            plt.ylim(-ylim - buffer, ylim + buffer)
        except ValueError:
            pass

        plt.ylabel("Number of Significantly Altered Lipids")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        outpath = output_dir / f"{comparison}_class_distribution_barplot.png"
        plt.savefig(outpath, dpi=300)
        plt.close()


# ==========================================================
# Bubble plot (design 1): y = Class, color by log2FC (seismic)
# ==========================================================
def _generate_bubble_plot_from_csv(input_dir: Path, output_dir: Path, sample_type: str):
    print("\n[Volcano] Generating bubble plots (design 1)", flush = True)
    csv_input = input_dir / "CSV_files"
    files = [f for f in os.listdir(csv_input) if f.endswith("_FDR.csv")]

    if sample_type == "Bacteria":
        ordered_classes = _CLASS_ORDER_BACTERIA
    elif sample_type == "Mammalians":
        ordered_classes = _CLASS_ORDER_MAMMALIAN
    else:
        ordered_classes = _CLASS_ORDER

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
            print(f"[Volcano] Skipping {file} (no significant entries)", flush = True)
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
            df["Class_Group"] = df["Class"].map(_Class_GROUP_MAP).fillna(df["Class"])

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

        # Colors by log2FC with center=0
        fc_vals = sig_df["log2(Fold Change)"]
        fc_abs_max = np.ceil(np.max(np.abs(fc_vals))) if len(fc_vals) else 1
        norm = TwoSlopeNorm(vmin=-fc_abs_max, vcenter=0, vmax=fc_abs_max)

        # Bubble sizes by FDR significance
        bubble_sizes = np.clip((-np.log10(sig_df["FDR p-value"])) ** 2 * 20, 10, 1200)

        fig, ax = plt.subplots(figsize=(12, max(12, 0.4 * len(ordered_classes))))
        sc = ax.scatter(
            fc_vals, sig_df["y"],
            s=bubble_sizes, c=fc_vals, cmap="seismic", norm=norm,
            edgecolor="black", linewidth=0.4, alpha=0.85
        )

        ax.set_yticks(list(y_mapping.values()))
        ax.set_yticklabels(list(y_mapping.keys()))

        x_max = np.ceil(np.abs(sig_df["log2(Fold Change)"]).max()) if len(sig_df) else 1
        ax.set_xlim(-x_max, x_max)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)

        ax.set_xlabel("log₂(Fold Change)")
        ax.set_ylabel("Lipid Class")
        ax.set_title(f"Fold-change distribution for significantly altered lipids\n({file.replace('_FDR.csv', '')})", pad=20)

        # Colorbar
        plt.colorbar(sc, ax=ax, label="log₂(Fold Change)", shrink=0.4, aspect=20)

        # Bubble size legend
        fdr_examples = [0.05, 0.001]
        def _size_scale(fdr): return np.clip((-np.log10(fdr)) ** 2 * 40, 10, 1500)
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", label=f"FDR = {fdr:.3f}",
                   markerfacecolor="gray", markersize=np.sqrt(_size_scale(fdr)))
            for fdr in fdr_examples
        ]
        ax.legend(
            handles=legend_handles, title="FDR significance",
            bbox_to_anchor=(1.28, 1), loc="upper left", borderaxespad=0.0, frameon=False
        )

        plt.subplots_adjust(bottom=0.25)
        plt.tight_layout()

        outname = output_dir / file.replace("_FDR.csv", "_bubble_plot.png")
        plt.savefig(outname, dpi=300)
        plt.close()


# ==========================================================
# Bubble plot (design 2): x = Class (jitter), color by Class palette
# ==========================================================
def _generate_bubble_plot_from_csv_design2(input_dir: Path, output_dir: Path, sample_type: str):
    print("\n[Volcano] Generating bubble plots (design 2)", flush = True)
    csv_input = input_dir / "CSV_files"
    files = [f for f in os.listdir(csv_input) if f.endswith("_FDR.csv")]

    if sample_type == "Bacteria":
        ordered_classes = _CLASS_ORDER_BACTERIA
    elif sample_type == "Mammalians":
        ordered_classes = _CLASS_ORDER_MAMMALIAN
    else:
        ordered_classes = _CLASS_ORDER

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
            print(f"[Volcano] Skipping {file} (no significant entries)", flush = True)
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
            df["Class_Group"] = df["Class"].map(_Class_GROUP_MAP).fillna(df["Class"])
        
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
        sig_df["x_jittered"] = sig_df["x"] + np.random.uniform(-0.25, 0.25, size=len(sig_df))

        # Bubble sizes
        bubble_sizes = np.clip((-np.log10(sig_df["FDR p-value"])) ** 2 * 20, 10, 1200)

        # Colors by Class (Set2)
        palette = sns.color_palette("Set2", len(ordered_classes))
        color_dict = dict(zip(ordered_classes, palette))
        colors = [color_dict.get(cls, (0.5, 0.5, 0.5)) for cls in sig_df["Class_Group"]]

        fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * len(ordered_classes))))
        ax.scatter(
            sig_df["x_jittered"], sig_df["log2(Fold Change)"],
            s=bubble_sizes, c=colors, edgecolor="black", linewidth=0.4, alpha=0.6
        )

        # Symmetric y-limit with padding
        y_max = np.ceil(np.abs(sig_df["log2(Fold Change)"]).max())
        y_pad = y_max * 0.1
        ax.set_ylim(-y_max - y_pad, y_max + y_pad)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)

        # Axis styling
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)

        ax.set_ylabel("log₂(Fold Change)")
        ax.set_xlabel("Lipid Class")
        ax.set_title(f"Fold-change distribution for significantly altered lipids\n({file.replace('_FDR.csv', '')})", pad=20)

        ax.set_xticks(list(x_mapping.values()))
        ax.set_xticklabels(list(x_mapping.keys()), rotation=90)

        # Bubble size legend
        fdr_examples = [0.05, 0.001]
        def _size_scale2(fdr): return np.clip((-np.log10(fdr)) ** 2 * 50, 10, 1500)
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", label=f"FDR = {fdr:.3f}",
                   markerfacecolor="gray", markersize=np.sqrt(_size_scale2(fdr)))
            for fdr in fdr_examples
        ]
        ax.legend(
            handles=legend_handles, title="FDR significance",
            bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, frameon=False
        )

        plt.tight_layout()
        outname = output_dir / file.replace("_FDR.csv", "_bubble_plot_v2.png")
        plt.savefig(outname, dpi=300)
        plt.close()


# ==========================================================
# Public entry point (called by GUI)
# ==========================================================
def run_volcano(file_path, group_file, save_dir,
                method="fdr_bh", test_type="parametric",
                run_bar_plots=True, run_bubble_plots=True,
                sample_type="Mammalians",
                p_value_threshold=0.05, fdr_threshold=0.10, fold_change_threshold=1.5):
    """
    Main entry point for Volcano + summaries + optional bar/bubble plots.

    Args mirror your previous script; outputs and folder structure remain the same.
    """
    file_path = Path(file_path)
    save_dir = prepare_output_dir(Path(save_dir))
    volcano_dir = _ensure_dir(save_dir)
    csv_dir = _ensure_dir(volcano_dir / "CSV_files")

    print(f"[Volcano] Starting for: {file_path.name}", flush = True)

    # Load standardized dataset
    # X: samples × features (columns = UniqueID or UniqueIDs)
    # y: sample groups
    # Load data
    X, y, feature_meta = load_dataset(file_path, group_file)
    
    # --- Build metadata lookup table from original file ---
    if isinstance(feature_meta, pd.DataFrame) and not feature_meta.empty:
        meta_lookup = feature_meta.copy()

        # Keep only relevant metadata columns if they exist
        keep_cols = ["UniqueID", "Annotation", "Annotation Type", "Headgroup", "Lipid Class"]
        keep_cols = [c for c in keep_cols if c in meta_lookup.columns]
        meta_lookup = meta_lookup[keep_cols].astype(str)

        # Clean up spaces and empty entries
        meta_lookup["UniqueID"] = meta_lookup["UniqueID"].astype(str).str.strip()
        for col in meta_lookup.columns:
            meta_lookup[col] = meta_lookup[col].replace("nan", "").replace("None", "").replace("NaN", "").str.strip()
    else:
        meta_lookup = pd.DataFrame(columns=["UniqueID", "Annotation", "Annotation Type", "Headgroup", "Lipid Class"])
    # -------------------------------------------------------

    # --- Skip QC samples completely --------------------------------------
    mask_non_qc = ~y.str.lower().str.contains("qc", na=False)
    X = X.loc[mask_non_qc]
    y = y.loc[mask_non_qc]
    # ---------------------------------------------------------------------

    groups = y.astype(str).unique()
    print(f"[Volcano] Groups (excluding QC): {list(groups)}", flush = True)

    # Pairwise comparisons (A vs B, no reversed duplicate)
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            print(f"[Volcano] Comparing {g1} vs {g2}", flush = True)
            df_volc = _compute_volcano(
                g1, g2, X, y, meta_lookup,
                method=method, test_type=test_type,
                p_thresh=p_value_threshold, fdr_thresh=fdr_threshold,
                fc_thresh=fold_change_threshold
            )

            # Save per-comparison CSV
            csv_out = csv_dir / f"{_sanitize_filename(g1)}_vs_{_sanitize_filename(g2)}_FDR.csv"
            df_volc.to_csv(csv_out, index=False)

            # Plot Volcano
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
