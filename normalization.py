
# ---------------------------------------------------------------------
# Class-matched internal standard normalization for lipidomics datasets
# (Updated logic v3.2 — 2025-12-04)
# ---------------------------------------------------------------------

'''
This script performs class-matched internal standard normalization for a processed lipidomics dataset, evaluates its performance, and generates diagnostic outputs.
It begins by loading the lipid feature table, the internal standards table, and a class-to-internal-standard mapping file, along with sample group information to identify QC samples. 
It first filters the internal standards to retain only reliable ones, removing standards with excessive variability, missing values, or very low signal. 
If multiple adducts exist for the same internal standard, it keeps the most stable and intense version.

For each lipid feature, the script selects the most appropriate internal standard from the same ionization polarity. 
It first tries the preferred standards listed for that lipid class (same class or most similar headgroup structure). 
If none meet QC reproducibility criteria, it scans all standards of the same polarity and selects the one that best improves, or least worsens, QC variability within an allowed threshold. 
If no suitable candidate passes the threshold, it falls back to the standard that gives the lowest QC variability overall. 
If QC data are unavailable, it selects the most intense standard of the same polarity. 
As a final step, it enforces that all features within the same lipid class and polarity use the same denominator for consistency.

The selected internal standard is then used to normalize feature intensities by division, sample by sample. 
The script recalculates QC and sample variability to assess whether normalization improved reproducibility. 
It saves normalized annotated and unannotated feature tables separately and writes detailed logs documenting internal standard selection and filtering decisions.

UNDER CONSTRUCTION: Optional semi-quantification can be applied by scaling normalized intensities using the concentration of the matched internal standard and a specified dilution factor, producing values proportional to internal standard amounts.

Finally, the script generates quality control plots, including QC variability distributions, PCA before and after normalization, summed intensity comparisons across samples and groups, and total signal per lipid class before and after normalization.
'''

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union
import warnings
import re

warnings.filterwarnings("ignore", message=".*is_sparse is deprecated.*", category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

'''
IS normalization

Every feature is divided by a carefully chosen IS intensity.

1. Filter the internal standards (IS)

    Remove unreliable IS peaks. Keep only IS rows where:
        • RSD across QCs < 35%
        • RSD across samples < 70%
        • no missing or zero values in any sample column.
    If multiple adducts exist for the same IS, it picks the adduct with the highest mean QC intensity.


2. Match each feature to an IS

    2.1. Stage 1 - Appendix options: look up the preferred IS list for each lipid class (from appendix CSV). Select the first one that:
        • belongs to the same polarity (pos/neg),
        • improves QC RSD (follwoing the threshold MAX_QC_RSD_WORSEN_PCT).
        If that works, it uses that IS.
        
    2.2. Stage 2 - Same-polarity scan: If none of the appendix IS help, it scans all IS of the same polarity and picks the one giving the largest RSD improvement.
         Stage 2 returns no winner if every same-polarity IS causes QC RSD to worsen by more than the allowed % (or if base/after RSD is NaN due to missing QC values, zeros, etc.).
         
    2.3. Fallback - If no IS passes the allowed QC RSD %, pick the IS with minimum rsd_after (best absolute QC RSD), even if it worsens beyond the threshold.
         If QC RSD is not computable for any candidate, it selects the max-mean IS in that polarity.  
         A “hard safety append None” (no IS selected) is triggered only when there are no same-polarity IS vectors.
       
3. Once the best denominator is chosen (the best IS), the feature's raw intensities are divided by that denominator for each sample column.    
   After normalization, it recomputes:
    • RSD across QCs (to check improvement),
    • RSD across each sample group.
    • How many features improved, stayed the same, or worsened.
    
It saves:
    4-Final_annotated_results_normalized.csv (normalized data),
    5-Final_unknowns_normalized.csv (unannotated features),
    summary plots (RSD boxplot, histogram, and PCA before/after) under debug/normalization/.
    
'''


# =======================
# Configuration switches
# =======================
qc_rsd_limit = 35
sample_rsd_limit = 70
MAX_QC_RSD_WORSEN_PCT = 20.0     # Allow QC RSD to increase by at most this % (relative) after normalization

def _safe_rsd_series(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf, 0], np.nan).dropna().to_numpy(float)
    if v.size <= 1:
        return np.nan
    m = float(np.nanmean(v))
    if not np.isfinite(m) or m == 0.0:
        return np.nan
    return float(np.nanstd(v, ddof=1) / m * 100.0)


def _corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 5:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])

def _qc_rsd_within_worsen_limit(base_rsd: float, rsd_after: float, max_worsen_pct: float) -> bool:
    """
    Returns True if rsd_after does not exceed base_rsd by more than max_worsen_pct (relative).
    Example: base=20, max_worsen=10% -> allow up to 22.
    """
    if not (np.isfinite(base_rsd) and np.isfinite(rsd_after)):
        return False
    if base_rsd <= 0:
        return False
    return rsd_after <= (base_rsd * (1.0 + float(max_worsen_pct) / 100.0))

def _canonical_is_name(s: str) -> str:
    """
    Normalize internal standard names so matching is robust.

    Examples:
        "[D3]FA 20:0 (sapon.)" -> "[D3]FA 20:0"
        "[D7]PC 15:0_18:1 extra" -> "[D7]PC 15:0_18:1"

    Rules:
        - Remove anything in parentheses
        - Trim whitespace
    """
    if pd.isna(s):
        return ""
    s = str(s).strip()

    # Remove parenthetical parts
    s = re.sub(r"\s*\(.*?\)", "", s)

    return s.strip()

def calculate_qc_rsd_post_norm(normalized_csv, sample_groups_csv, output_folder="results"):
    normalized_csv = Path(normalized_csv)
    df = pd.read_csv(normalized_csv, low_memory=False)
    group_df = pd.read_csv(sample_groups_csv, low_memory=False)
    outdir = Path(output_folder) / "debug" / "normalization"
    outdir.mkdir(parents=True, exist_ok=True)

    qc_samples = group_df.loc[group_df["Group"].astype(str).str.upper().str.strip() == "QC","Sample"].tolist()
    if not qc_samples:
        print("\n\n ------- WARNING: No QC columns found. -------", flush= True)

    qc_cols = []
    for sample in qc_samples:
        if sample in df.columns:
            qc_cols.append(sample)
        else:
            qc_cols.extend([c for c in df.columns if sample in c])
    qc_cols = list(dict.fromkeys(qc_cols))
    if not qc_cols:
        print("\n\n ------- WARNING: No matching QC columns found in normalized file for assigned QC samples.---------- \n\n", flush= True)

    if qc_cols:
        qc_rsd = []
        for _, row in df.iterrows():
            vals = pd.to_numeric(row[qc_cols], errors="coerce").replace(0, np.nan).dropna()
            rsd = (np.std(vals, ddof=1) / np.mean(vals) * 100) if len(vals) > 1 else np.nan
            qc_rsd.append(rsd)
        df["RSD QCs (%) post-norm"] = qc_rsd
    else:
        print("[WARNING] No QC samples/columns found. Writing 'RSD QCs (%) post-norm' as NaN.", flush=True)
        df["RSD QCs (%) post-norm"] = np.nan

    # Use this QC column list everywhere below (selection, overrides, reporting)
    qc_cols_use = qc_cols[:]  # shallow copy

    out_path = outdir / f"{normalized_csv.stem}_with_QC_RSD.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    if qc_cols:
        median_rsd = float(np.nanmedian(df["RSD QCs (%) post-norm"]))
        print(f"[INFO] Median QC RSD after normalization: {median_rsd:.2f}%", flush=True)
        print(f"[INFO] Added 'RSD QCs (%) post-norm' column and saved to: {out_path}", flush = True)
    else:
        print("[INFO] Median QC RSD after normalization: n/a (no QC).", flush=True)

    return out_path, df


def plot_rsd_distributions(imputed_filtered_csv, normalized_with_rsd_csv, output_folder="results"):
    df_before = pd.read_csv(imputed_filtered_csv, low_memory=False)
    df_after = pd.read_csv(normalized_with_rsd_csv, low_memory=False)
    outdir = Path(output_folder) / "debug" / "normalization"
    outdir.mkdir(parents=True, exist_ok=True)

    df_before.columns = df_before.columns.str.strip().str.replace("\\xa0", " ", regex=False)
    df_after.columns = df_after.columns.str.strip().str.replace("\\xa0", " ", regex=False)

    before_col = next((c for c in df_before.columns if "rsd" in c.lower() and "qc" in c.lower()), None)
    after_col = next((c for c in df_after.columns if "rsd" in c.lower() and "qc" in c.lower() and "post-norm" in c.lower()), None)

    rsd_before = pd.to_numeric(df_before[before_col], errors="coerce").dropna() if before_col else pd.Series([], dtype=float)
    rsd_after  = pd.to_numeric(df_after[after_col],  errors="coerce").dropna() if after_col  else pd.Series([], dtype=float)

    if rsd_before.empty and rsd_after.empty:
        print("[WARNING] No valid RSD data found for plotting.", flush = True); return

    data = pd.DataFrame({
        "RSD (%)": np.concatenate([rsd_before.values, rsd_after.values]),
        "Condition": ["Before normalization"] * len(rsd_before) + ["After normalization"] * len(rsd_after)
    })

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=data, x="Condition", y="RSD (%)", hue="Condition", palette="Set2", showfliers=False, legend=False)
    plt.title("QC RSD Distribution Before vs After Normalization")
    plt.tight_layout(); plt.savefig(outdir / "RSD_boxplot_before_after.png", dpi=100); plt.close()

    plt.figure(figsize=(7, 5))
    sns.histplot(data=data, x="RSD (%)", hue="Condition", bins=40, kde=True, palette="Set2")
    plt.title("QC RSD Histogram Before vs After Normalization")
    plt.tight_layout(); plt.savefig(outdir / "RSD_hist_before_after.png", dpi=100); plt.close()

    print(f"[INFO] Saved RSD boxplot and histogram in: {outdir}", flush = True)


def plot_pca_before_after(imputed_filtered_csv, normalized_with_rsd_csv, sample_groups_csv, output_folder="results"):
    df_before = pd.read_csv(imputed_filtered_csv, low_memory=False)
    df_after  = pd.read_csv(normalized_with_rsd_csv, low_memory=False)
    group_df  = pd.read_csv(sample_groups_csv, low_memory=False)
    output_folder = Path(output_folder) / "debug" / "normalization"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Normalize column names (fix NBSP, stray spaces)
    df_before.columns = df_before.columns.str.strip().str.replace("\xa0", " ", regex=False)
    df_after.columns  = df_after.columns.str.strip().str.replace("\xa0", " ", regex=False)

    sample_cols = [c for c in df_before.columns if c.startswith("P_") or c.startswith("N_")]
    if not sample_cols:  # tolerate files missing the closing bracket
        sample_cols = [c for c in df_before.columns if c.startswith("P_") or c.startswith("N_")]

    sample_cols = [c for c in sample_cols if c in df_after.columns]

    if not sample_cols:
        print("[WARNING] No sample columns found for PCA after header normalization; skipping PCA plots.")
        print(f"[DEBUG] Example 'before' columns head: {list(df_before.columns[:10])}")
        print(f"[DEBUG] Example 'after'  columns head: {list(df_after.columns[:10])}")
        return

    sample_labels = []
    for c in sample_cols:
        grp = group_df.loc[group_df["Sample"] == c, "Group"]
        sample_labels.append(grp.values[0] if not grp.empty else "Unknown")

    def run_pca_and_plot(df, label, suffix):
        X = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        X_log = np.log2(X + 1)
        X_scaled = StandardScaler().fit_transform(X_log.T)
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame({"PC1": pca_res[:, 0], "PC2": pca_res[:, 1], "Group": sample_labels})

        plt.figure(figsize=(7, 6))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Group", s=70, palette="Set2", alpha=0.8)
        plt.title(f"PCA {label}\\n({pca.explained_variance_ratio_[0]*100:.1f}% + {pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        plt.tight_layout(); plt.savefig(output_folder / f"PCA_{suffix}.png", dpi=100); plt.close()

    run_pca_and_plot(df_before, "Before normalization", "before")
    run_pca_and_plot(df_after,  "After normalization",  "after")
    print(f"[INFO] PCA plots saved in: {output_folder}", flush = True)

# ---------------- Back-compat export for other modules ----------------
def evaluate_normalization_performance(imputed_filtered_csv, normalized_csv, sample_groups_csv, output_folder="results"):
    """
    Backward-compatible wrapper used by other modules.
    Computes post-norm QC RSD, prints medians, and writes the comparison plots.
    Returns the path to the '..._with_QC_RSD.csv' file.
    """
    print("\n[STEP] Evaluating normalization performance...", flush=True)
    try:
        normalized_with_rsd_csv, df_after = calculate_qc_rsd_post_norm(
            normalized_csv, sample_groups_csv, output_folder
        )

        # Load before-normalization data
        df_before = pd.read_csv(imputed_filtered_csv, low_memory=False)
        df_before.columns = df_before.columns.str.strip().str.replace("\xa0", " ", regex=False)
        df_after.columns  = df_after.columns.str.strip().str.replace("\xa0", " ", regex=False)

        # Flexible lookup for RSD columns
        before_col = next((c for c in df_before.columns if "rsd" in c.lower() and "qc" in c.lower()), None)
        after_col  = next((c for c in df_after.columns  if "rsd" in c.lower() and "qc" in c.lower() and "post-norm" in c.lower()), None)

        if before_col:
            rsd_before = pd.to_numeric(df_before[before_col], errors="coerce").dropna()
            print(f"[INFO] Using '{before_col}' as pre-normalization RSD column.", flush=True)
        else:
            print("[WARNING] Could not find pre-normalization RSD column.", flush=True)
            rsd_before = pd.Series([], dtype=float)

        if after_col:
            rsd_after = pd.to_numeric(df_after[after_col], errors="coerce").dropna()
            print(f"[INFO] Using '{after_col}' as post-normalization RSD column.", flush=True)
        else:
            print("[WARNING] Could not find post-normalization RSD column.", flush=True)
            rsd_after = pd.Series([], dtype=float)

        if not rsd_before.empty and not rsd_after.empty:
            med_before = float(np.nanmedian(rsd_before)); med_after = float(np.nanmedian(rsd_after))
            print(f"[INFO] Median QC RSD before normalization: {med_before:.2f}%")
            print(f"[INFO] Median QC RSD after normalization:  {med_after:.2f}%")
            print(f"[INFO] Delta-RSD (after - before): {med_after - med_before:+.2f}%")
        else:
            print("[WARNING] Could not compute RSD improvement — missing RSD data.", flush=True)

        plot_rsd_distributions(str(imputed_filtered_csv), str(normalized_with_rsd_csv), str(output_folder))
        plot_pca_before_after(str(imputed_filtered_csv), str(normalized_with_rsd_csv), str(sample_groups_csv), str(output_folder))

        print(f"[DONE] Normalization performance evaluation complete. "
              f"Plots saved under {Path(output_folder)/'normalization'}\n", flush=True)

        return normalized_with_rsd_csv

    except Exception as e:
        import traceback
        print(f"[ERROR] Normalization evaluation failed: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return None
    

# ---------------------------------------------------------------------
#   NORMALIZATION PIPELINE
# ---------------------------------------------------------------------

def normalize_by_internal_standards(
    features_csv,
    internal_standards_csv,
    class_to_is_csv,
    output_folder="results",
    is_dilution_factor: float = 1.0,
    is_mix_file: str = None,
    is_mix_type: str = "Avanti Splash Lipidomix",
):

    print('\nStarting IS normalization...', flush = True)
    features_csv = Path(features_csv); internal_standards_csv = Path(internal_standards_csv)
    class_to_is_csv = Path(class_to_is_csv)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(features_csv, low_memory=False).reset_index(drop=True)
    is_df = pd.read_csv(internal_standards_csv, low_memory=False).reset_index(drop=True)
    is_df["Annotation"] = is_df["Annotation"].apply(_canonical_is_name)
    class_map = pd.read_csv(class_to_is_csv, low_memory=False)
    
    # Preserve QC detectability columns from the imputed feature table
    qc_meta_cols = [c for c in ["QC detected count"] if c in features_df.columns]
    if qc_meta_cols:
        print(f"[INFO] Preserving QC metadata columns: {qc_meta_cols}", flush=True)
    else:
        print("[INFO] No QC metadata columns found in input features file.", flush=True)

    sample_cols = [c for c in features_df.columns if c.startswith("P_") or c.startswith("N_")]
    if not sample_cols:
        raise ValueError("No sample columns found. Expected columns starting with P_ or N_.")

    # --- Determine polarity tag for output filenames ---
    pol_tag = ""
    if "Polarity" in features_df.columns:
        pol_series = features_df["Polarity"].dropna().astype(str).str.lower()
        if not pol_series.empty:
            first_pol = pol_series.iloc[0]
            if "pos" in first_pol:
                pol_tag = "Pos_"
            elif "neg" in first_pol:
                pol_tag = "Neg_"
    if not pol_tag:
        has_pos = any(str(c).startswith("P_") for c in sample_cols)
        has_neg = any(str(c).startswith("N_") for c in sample_cols)
        if has_pos and not has_neg:
            pol_tag = "Pos_"
        elif has_neg and not has_pos:
            pol_tag = "Neg_"

    group_file = Path(output_folder).parent / "sample_groups.csv"
    if not group_file.exists(): raise FileNotFoundError("sample_groups.csv not found in output folder.")
    group_df = pd.read_csv(group_file, low_memory=False)

    qc_samples = group_df.loc[group_df["Group"].astype(str).str.upper().str.strip() == "QC","Sample"].tolist()

    qc_cols = []
    for s in qc_samples:
        if s in features_df.columns:
            qc_cols.append(s)
        else:
            qc_cols.extend([c for c in features_df.columns if s in c])
    qc_cols = list(dict.fromkeys(qc_cols))

    if qc_cols:
        print(f"[INFO] Found {len(qc_cols)} QC columns for QC-based filtering/scoring.", flush=True)
    else:
        print("\n--------------------------- WARNING: NO QC SAMPLES FOUND. SKIPPING QC FILTERING/SCORING. ---------------------------\n", flush=True)
        
    # Use this QC column list everywhere below (selection, overrides, reporting)
    qc_cols_use = qc_cols[:]  # shallow copy

    # ---------------------------------------------------------
    # Identify blank samples so they do not penalize IS filters
    # ---------------------------------------------------------
    blank_samples = group_df.loc[
        group_df["Group"].astype(str).str.strip().str.lower().str.contains("blank", na=False),
        "Sample"
    ].tolist()

    blank_cols = []
    for s in blank_samples:
        if s in features_df.columns:
            blank_cols.append(s)
        else:
            blank_cols.extend([c for c in features_df.columns if s in c])
    blank_cols = list(dict.fromkeys(blank_cols))

    if blank_cols:
        print(f"[INFO] Found {len(blank_cols)} blank columns. They will be excluded from IS missingness and sample-RSD filtering.", flush=True)
    else:
        print("[INFO] No blank columns detected for IS filtering.", flush=True)

    # ---------------------------------------------------------
    # Select internal standards metadata file based on user GUI
    # ---------------------------------------------------------
    is_meta_path = None

    # If user selected "Other", use their chosen file
    if is_mix_file:
        cand = Path(is_mix_file)
        if cand.exists():
            is_meta_path = cand
        else:
            print(f"[WARNING] User-selected IS file does not exist: {cand}", flush=True)

    # Otherwise choose one of the built-in Appendix files
    if is_meta_path is None:
        p = Path(__file__).resolve()
        appendix_dir = None

        # Search upward for Appendix folder
        for _ in range(8):
            cand = p.parent / "Appendix"
            if cand.exists():
                appendix_dir = cand
                break
            p = p.parent

        if appendix_dir is None:
            print("[WARNING] Could not locate Appendix folder. Semi-quantification may fail.", flush=True)
        else:
            if is_mix_type == "Avanti Splash Lipidomix":
                cand = appendix_dir / "Internal Standards - Avanti Splash Lipidomix.xlsx"
            elif is_mix_type == "BRG Internal Standard Mix":
                cand = appendix_dir / "Internal Standards - BRG IS Mix.xlsx"
            else:  # fallback template
                cand = appendix_dir / "Internal Standards - Template.xlsx"

            if cand.exists():
                is_meta_path = cand
            else:
                print(f"[WARNING] Could not find expected IS metadata file: {cand}", flush=True)
                is_meta_path = None

    def norm_pol(x):
        s = str(x).lower().strip()
        if s.startswith("pos"): return "pos"
        if s.startswith("neg"): return "neg"
        return None

    features_df["Polarity_norm"] = features_df["Polarity"].apply(norm_pol)
    is_df["Polarity_norm"] = is_df["Polarity"].apply(norm_pol)

    for col in ["RSD QCs (%)", "RSD Samples (%)"]:
        if col in is_df.columns: is_df[col] = pd.to_numeric(is_df[col], errors="coerce")

    sample_cols_is = [c for c in is_df.columns if c.startswith("P_") or c.startswith("N_")]
    if not sample_cols_is:
        raise ValueError("Internal standards file has no sample intensity columns (P_/N_).")

    # Exclude blanks from strict IS reliability filters
    nonblank_sample_cols_is = [c for c in sample_cols_is if c not in blank_cols]
    if not nonblank_sample_cols_is:
        print("[WARNING] No non-blank IS sample columns found. Falling back to all sample columns for IS filtering.", flush=True)
        nonblank_sample_cols_is = sample_cols_is[:]


    if "RSD QCs (%)" not in is_df.columns: is_df["RSD QCs (%)"] = np.nan
    is_df["RSD QCs (%)"] = is_df["RSD QCs (%)"].astype(float)

    def _compute_is_qc_rsd_row(row):
        if not qc_cols:
            return np.nan
        rsd = row.get("RSD QCs (%)", np.nan)
        if pd.isna(rsd):
            rsd = _safe_rsd_series(pd.to_numeric(row[qc_cols], errors="coerce"))
        return rsd

    qc_rsd_vals = []; sample_rsd_vals = []; no_missing_flags = []
    for _, row in is_df.iterrows():
        qc_rsd_vals.append(_compute_is_qc_rsd_row(row))

        # Recompute sample RSD using non-blank samples only
        rsd_samples = row.get("RSD Samples (%)", np.nan)
        if pd.isna(rsd_samples):
            rsd_samples = _safe_rsd_series(pd.to_numeric(row[nonblank_sample_cols_is], errors="coerce"))
        sample_rsd_vals.append(float(rsd_samples) if pd.notna(rsd_samples) else np.nan)

        # Missingness / low-signal rule should ignore blanks
        vals = pd.to_numeric(row[nonblank_sample_cols_is], errors="coerce").to_numpy(float)
        valid_vals = np.isfinite(vals) & (vals != 0) & (vals >= 5000)
        no_missing_flags.append(bool(np.all(valid_vals)))

    is_df["__QC_RSD__computed__"] = qc_rsd_vals
    is_df["__RSD_Samples__"] = sample_rsd_vals
    is_df["__NoMissing__"] = no_missing_flags

    # ---------------------------------------------------------
    # DEBUG: Record why each IS passed/failed strict filters
    # ---------------------------------------------------------
    try:
        dbg_dir = output_folder / "debug" / "normalization"
        dbg_dir.mkdir(parents=True, exist_ok=True)

        is_dbg = is_df.copy()

        # Boolean pass flags
        is_dbg["pass_NoMissing"] = is_dbg["__NoMissing__"].astype(bool)
        if qc_cols:
            is_dbg["pass_QC_RSD"] = pd.to_numeric(is_dbg["__QC_RSD__computed__"], errors="coerce") < float(qc_rsd_limit)
        else:
            is_dbg["pass_QC_RSD"] = True  # no QC, cannot evaluate this criterion
        is_dbg["pass_Sample_RSD"] = pd.to_numeric(is_dbg["__RSD_Samples__"], errors="coerce") < float(sample_rsd_limit)

        # Human-readable failure reason(s)
        reasons = []
        for _, r in is_dbg.iterrows():
            rs = []
            if not bool(r.get("pass_NoMissing", False)):
                rs.append("missing_or_low_intensity(<5000)_in_some_samples")
            if qc_cols and not bool(r.get("pass_QC_RSD", False)):
                rs.append(f"QC_RSD>={qc_rsd_limit}")
            if not bool(r.get("pass_Sample_RSD", False)):
                rs.append(f"Sample_RSD>={sample_rsd_limit}")
            reasons.append(";".join(rs) if rs else "PASS")

        is_dbg["strict_filter_status"] = reasons

        keep_cols = [
            "UniqueID", "Annotation", "Polarity", "Polarity_norm",
            "__QC_RSD__computed__", "__RSD_Samples__", "__NoMissing__",
            "pass_NoMissing", "pass_QC_RSD", "pass_Sample_RSD",
            "strict_filter_status"
        ]
        keep_cols = [c for c in keep_cols if c in is_dbg.columns]

        out_is_filter = dbg_dir / f"{pol_tag}IS_filter_summary.csv"
        is_dbg[keep_cols].to_csv(out_is_filter, index=False, encoding="utf-8-sig")
        print(f"[DEBUG] Wrote: {out_is_filter}", flush=True)

    except Exception as e:
        print(f"[WARNING] Failed to write IS_filter_summary.csv: {e}", flush=True)

    pre_filter_len = len(is_df)
    is_df_all = is_df.copy()

    if qc_cols:
        is_df = is_df[
            (is_df["__NoMissing__"]) &
            (is_df["__QC_RSD__computed__"] < qc_rsd_limit) &
            (is_df["__RSD_Samples__"] < sample_rsd_limit)
        ].copy()
    else:
        is_df = is_df[
            (is_df["__NoMissing__"]) &
            (is_df["__RSD_Samples__"] < sample_rsd_limit)
        ].copy()

    print(f"[INFO] Internal standards passing strict filters: {len(is_df)}/{pre_filter_len}", flush=True)

    # Fallback: if nothing passes, relax only the missingness rule after excluding blanks
    if is_df.empty:
        print("[WARNING] No internal standards passed strict filters. Applying fallback filter without the '__NoMissing__' requirement.", flush=True)

        if qc_cols:
            is_df = is_df_all[
                (is_df_all["__QC_RSD__computed__"] < qc_rsd_limit) &
                (is_df_all["__RSD_Samples__"] < sample_rsd_limit)
            ].copy()
        else:
            is_df = is_df_all[
                (is_df_all["__RSD_Samples__"] < sample_rsd_limit)
            ].copy()

        print(f"[INFO] Internal standards passing fallback filters: {len(is_df)}/{pre_filter_len}", flush=True)

    # Final hard fallback: if still empty, keep all IS and let downstream selection choose the least bad option
    if is_df.empty:
        print("[WARNING] Still no internal standards available after fallback. Retaining all internal standards for downstream ranking.", flush=True)
        is_df = is_df_all.copy()

    # Collapse adducts
    collapsed_rows = []
    for (annot, pol), group in is_df.groupby(["Annotation", "Polarity_norm"], dropna=True):
        group = group.copy()
        if qc_cols:
            group["__MEAN__"] = group[qc_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=False)
        else:
            # No QC: pick the adduct with highest mean across all sample intensity columns
            sample_cols_is_local = [c for c in group.columns if str(c).startswith("P_") or str(c).startswith("N_")]
            group["__MEAN__"] = group[sample_cols_is_local].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=False)

        best_row = group.sort_values("__MEAN__", ascending=False).iloc[0]
        collapsed_rows.append(best_row)
    is_df = pd.DataFrame(collapsed_rows).reset_index(drop=True)
    print(f"[INFO] Collapsed IS (unique Annotation + polarity): {len(is_df)}", flush = True)

    is_numeric = is_df.copy()
    for c in [col for col in is_numeric.columns if col.startswith("P_") or col.startswith("N_")]:
        is_numeric[c] = pd.to_numeric(is_numeric[c], errors="coerce")
    is_numeric["RSD QCs (%)"] = is_numeric["__QC_RSD__computed__"].astype(float)

    # Appendix mapping
    class_map_dict = {}
    cols_pos = [c for c in class_map.columns if "(main)" in c and "pos" in c.lower()] + [c for c in class_map.columns if "(option" in c.lower() and "pos" in c.lower()]
    cols_neg = [c for c in class_map.columns if "(main)" in c and "neg" in c.lower()] + [c for c in class_map.columns if "(option" in c.lower() and "neg" in c.lower()]
    for _, row in class_map.iterrows():
        lipid_class = str(row["Class"]).strip()
        if not lipid_class or lipid_class.lower() == "nan": continue
        pos_list = [
            _canonical_is_name(row[c])
            for c in cols_pos
            if pd.notna(row[c]) and str(row[c]).strip()
        ]
        neg_list = [
            _canonical_is_name(row[c])
            for c in cols_neg
            if pd.notna(row[c]) and str(row[c]).strip()
        ]
        pos_list = [nm for nm in pos_list if not is_numeric[(is_numeric["Annotation"].str.strip() == nm) & (is_numeric["Polarity_norm"] == "pos")].empty]
        neg_list = [nm for nm in neg_list if not is_numeric[(is_numeric["Annotation"].str.strip() == nm) & (is_numeric["Polarity_norm"] == "neg")].empty]
        class_map_dict[(lipid_class, "pos")] = pos_list; class_map_dict[(lipid_class, "neg")] = neg_list
    print(f"[DEBUG] Loaded {len(class_map_dict)} class→IS lists (after strict IS filtering).", flush = True)

    feat_sample_cols = sample_cols
    LOW_N = (len(feat_sample_cols) < 5)
    is_sample_cols = [c for c in is_numeric.columns if c in feat_sample_cols]

    def qc_rsd_improvement_by_den(row_feat: pd.Series, den_series: Union[pd.Series, np.ndarray], qc_cols: List[str]) -> Tuple[float, float, float]:
        qc_feat = pd.to_numeric(row_feat[qc_cols], errors="coerce")
        den_qc = pd.to_numeric(den_series[qc_cols], errors="coerce").replace(0, np.nan) if isinstance(den_series, pd.Series) else pd.Series(den_series, index=qc_cols)
        base_rsd = _safe_rsd_series(qc_feat); rsd_after = _safe_rsd_series(qc_feat / den_qc)
        impr = (base_rsd - rsd_after) if (np.isfinite(base_rsd) and np.isfinite(rsd_after)) else np.nan
        return base_rsd, rsd_after, impr

    chosen_den_name = []; chosen_reason = []; chosen_den_vector = []
    IS_VECS = {
        (_canonical_is_name(r["Annotation"]), r["Polarity_norm"]):
            pd.to_numeric(r[feat_sample_cols], errors="coerce")
        for _, r in is_numeric.iterrows()
    }

    # ---------------------------------------------------------
    # DEBUG: Candidate audit log (per feature, per candidate IS)
    # ---------------------------------------------------------
    candidate_audit = []

    def _audit(idx, row, stage, cand_name, base_rsd, rsd_after, pass_worsen, pass_vec, chosen, discard_reason):
        candidate_audit.append({
            "FeatureIndex": int(idx),
            "UniqueID": row.get("UniqueID", ""),
            "FeatureAnnotation": str(row.get("Annotation", "")).strip(),
            "LipidClass": str(row.get("Lipid Class", "")).strip(),
            "Polarity_norm": str(row.get("Polarity_norm", "")).strip(),
            "Stage": stage,
            "CandidateIS": str(cand_name).strip(),
            "base_qc_rsd": base_rsd,
            "post_qc_rsd": rsd_after,
            "pass_worsen": bool(pass_worsen),
            "pass_vec": bool(pass_vec),
            "chosen": bool(chosen),
            "discard_reason": str(discard_reason).strip(),
        })

    for idx, row in features_df.iterrows():
        start_len = len(chosen_den_name)

        lipid_class = str(row.get("Lipid Class", "")).strip()
        annotation  = str(row.get("Annotation", "")).strip()
        pol         = row["Polarity_norm"]

        # Require a valid polarity; if missing, we cannot normalize.
        if pol not in ("pos", "neg"):
            chosen_den_name.append(np.nan)
            chosen_reason.append("Missing polarity → cannot normalize")
            chosen_den_vector.append(None)
            continue

        feat_all = pd.to_numeric(row[feat_sample_cols], errors="coerce").to_numpy(float)

        # Stage 1 (class list) only if we have a usable class name
        have_class = bool(lipid_class) and str(lipid_class).lower() != "nan"
        win = None
        if have_class:
            cand_list = class_map_dict.get((lipid_class, pol), [])
            for cand in cand_list:
                vec = IS_VECS.get((cand, pol))
                if vec is None:
                    _audit(idx, row, "Stage1_Appendix", cand, np.nan, np.nan, False, False, False, "vec_missing_or_wrong_polarity")
                    continue
                r = _corr_safe(feat_all, vec.to_numpy(float))
                base_rsd, rsd_after, impr = qc_rsd_improvement_by_den(
                    row,
                    is_numeric[(is_numeric["Annotation"].str.strip()==cand)&(is_numeric["Polarity_norm"]==pol)].iloc[0],
                    qc_cols
                )
                
                # Decide pass/fail flags for audit
                if qc_cols:
                    den_row = is_numeric[
                        (is_numeric["Annotation"].str.strip() == cand) &
                        (is_numeric["Polarity_norm"] == pol)
                    ].iloc[0]
                    base_rsd, rsd_after, impr = qc_rsd_improvement_by_den(row, den_row, qc_cols)
                    pass_worsen = _qc_rsd_within_worsen_limit(base_rsd, rsd_after, MAX_QC_RSD_WORSEN_PCT)
                    ok = pass_worsen
                else:
                    # No QC: Stage-1 cannot score “improvement”, so accept first appendix candidate that exists.
                    base_rsd, rsd_after, impr = (np.nan, np.nan, np.nan)
                    pass_worsen = True
                    ok = True

                if ok:
                    _audit(idx, row, "Stage1_Appendix", cand, base_rsd, rsd_after, pass_worsen, True, True, "")
                    if qc_cols:
                        win = ("IS:"+cand,
                            f"Appendix pass: RSD {base_rsd:.2f}%→{rsd_after:.2f}% (Δ={impr:.2f} pp, max worsen {MAX_QC_RSD_WORSEN_PCT:.1f}%)",
                            vec)
                    else:
                        win = ("IS:"+cand, "Appendix pass (no QC)", vec)
                    break
                else:
                    rs = []
                    if qc_cols and not pass_worsen:
                        rs.append(f"QC_RSD_worsened>{MAX_QC_RSD_WORSEN_PCT}%")
                    _audit(idx, row, "Stage1_Appendix", cand, base_rsd, rsd_after, pass_worsen, True, False,
                        f"QC_RSD_worsened>{MAX_QC_RSD_WORSEN_PCT}%" if (qc_cols and not pass_worsen) else "failed_criteria")
                    
        # If Stage-1 found a winner, use it and continue; otherwise proceed to Stage-2+
        if win is not None:
            chosen_den_name.append(win[0])
            chosen_reason.append(win[1])
            chosen_den_vector.append(win[2])
            continue

        # Stage 2
        # Do NOT restrict by class here; scan all IS in the same polarity.
        # Stage 2 returns no winner if every same-polarity IS causes QC RSD to worsen by more than the allowed % (or if base/after RSD is NaN due to missing QC values, zeros, etc.).
        
        restrict_to = None
            
        if qc_cols:
            best = None
            
            for (name, pol_k), vec in IS_VECS.items():
                if pol_k != pol:
                    _audit(idx, row, "Stage2_Scan", name, np.nan, np.nan, False, True, False, "wrong_polarity")
                    continue
                base_rsd, rsd_after, impr = qc_rsd_improvement_by_den(
                    row,
                    is_numeric[(is_numeric["Annotation"].str.strip() == name) & (is_numeric["Polarity_norm"] == pol)].iloc[0],
                    qc_cols
                )
                pass_worsen = _qc_rsd_within_worsen_limit(base_rsd, rsd_after, MAX_QC_RSD_WORSEN_PCT)
                if not pass_worsen:
                    _audit(
                        idx, row, "Stage2_Scan", name,
                        base_rsd, rsd_after,
                        False,  # pass_worsen
                        True,   # pass_vec (vec exists)
                        False,  # chosen
                        f"QC_RSD_worsened>{MAX_QC_RSD_WORSEN_PCT}%"
                    )
                    continue

                r = _corr_safe(feat_all, vec.to_numpy(float))  # keep for logging only (not used as a filter)
                _audit(
                    idx, row, "Stage2_Scan", name,
                    base_rsd, rsd_after,
                    True,   # pass_worsen
                    True,   # pass_vec
                    False,  # chosen (not chosen yet)
                    "tested_pass_worsen"
                )
                item = (impr, name, vec, base_rsd, rsd_after)
                if (best is None) or (item > best):
                    best = item
            
            if best is not None:
                chosen_den_name.append("IS:" + best[1])
                chosen_reason.append(
                    f"Polarity-wide best under worsen≤{MAX_QC_RSD_WORSEN_PCT:.1f}%: ΔRSD={best[0]:.2f} pp"
                )
                chosen_den_vector.append(best[2])
                _audit(
                    idx, row, "Stage2_Scan", best[1],
                    best[3], best[4],
                    True,   # pass_worsen
                    True,   # pass_vec
                    True,   # chosen
                    "selected_best_improvement"
                )
                continue

            # -------------------------------
            # Stage 2 fallback (QC branch)
            # -------------------------------
            # Fallback A: choose the IS that yields the lowest post-normalization QC RSD (absolute), even if it violates the worsen threshold.
            
            fallback_best = None  # (rsd_after, name, vec, base_rsd, rsd_after)
            saw_same_pol = False

            for (name, pol_k), vec in IS_VECS.items():
                if pol_k != pol:
                    continue
                saw_same_pol = True

                den_row = is_numeric[
                    (is_numeric["Annotation"].str.strip() == name) &
                    (is_numeric["Polarity_norm"] == pol)
                ]
                if den_row.empty:
                    continue
                den_row = den_row.iloc[0]

                base_rsd, rsd_after, impr = qc_rsd_improvement_by_den(row, den_row, qc_cols)

                if not (np.isfinite(base_rsd) and np.isfinite(rsd_after)):
                    _audit(
                        idx, row, "Stage2_Fallback_min_post_RSD", name,
                        base_rsd, rsd_after,
                        False,  # pass_worsen
                        True,   # pass_vec
                        False,
                        "qc_rsd_not_finite"
                    )
                    continue

                _audit(
                    idx, row, "Stage2_Fallback_min_post_RSD", name,
                    base_rsd, rsd_after,
                    False,  # pass_worsen (not enforced here)
                    True,
                    False,
                    "tested_min_post_rsd"
                )

                item = (rsd_after, name, vec, base_rsd, rsd_after)
                if (fallback_best is None) or (item < fallback_best):
                    fallback_best = item

            if fallback_best is not None:
                chosen_den_name.append("IS:" + fallback_best[1])
                chosen_reason.append(
                    f"Stage2 fallback: min post-QC RSD (worsen gate bypassed): {fallback_best[3]:.2f}%→{fallback_best[4]:.2f}%"
                )
                chosen_den_vector.append(fallback_best[2])
                _audit(
                    idx, row, "Stage2_Fallback_min_post_RSD", fallback_best[1],
                    fallback_best[3], fallback_best[4],
                    False,
                    True,
                    True,
                    "selected_min_post_rsd"
                )
                continue

            # Fallback B: if QC RSD cannot be computed for any candidate, pick max-mean IS in this polarity.
            if saw_same_pol:
                mean_best = None  # (mean_intensity, name, vec)
                for (name, pol_k), vec in IS_VECS.items():
                    if pol_k != pol:
                        continue
                    m = float(np.nanmean(vec.to_numpy(float)))
                    item = (m, name, vec)
                    if (mean_best is None) or (item > mean_best):
                        mean_best = item

                if mean_best is not None and np.isfinite(mean_best[0]):
                    chosen_den_name.append("IS:" + mean_best[1])
                    chosen_reason.append(f"Stage2 fallback: QC RSD unavailable, used polarity max-mean IS: mean={mean_best[0]:.2g}")
                    chosen_den_vector.append(mean_best[2])
                    _audit(
                        idx, row, "Stage2_Fallback_max_mean", mean_best[1],
                        np.nan, np.nan,
                        False,
                        True,
                        True,
                        "selected_max_mean_no_qc_rsd"
                    )
                    continue
        else:
            # No QC: pick the IS with highest mean intensity in this polarity
            best = None
            for (name, pol_k), vec in IS_VECS.items():
                if pol_k != pol:
                    continue
                m = float(np.nanmean(vec.to_numpy(float)))
                item = (m, name, vec)
                if (best is None) or (item > best):
                    best = item
            if best is not None:
                chosen_den_name.append("IS:" + best[1])
                chosen_reason.append(f"Polarity-wide max-mean IS (no QC): mean={best[0]:.2g}")
                chosen_den_vector.append(best[2])
                continue

            chosen_den_name.append(None)
            chosen_reason.append("no IS available")
            chosen_den_vector.append(None)
            continue

        # ---- HARD SAFETY: ensure 1 denominator decision per feature ----
        if len(chosen_den_name) == start_len:
            chosen_den_name.append(None)
            chosen_reason.append("no IS available (fallback append)")
            chosen_den_vector.append(None)
            
    # ---------------------------------------------------------
    # DEBUG: Write candidate audit table
    # ---------------------------------------------------------
    try:
        dbg_dir = output_folder / "debug" / "normalization"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        audit_df = pd.DataFrame(candidate_audit)
        out_audit = dbg_dir / f"{pol_tag}IS_candidate_audit.csv"
        audit_df.to_csv(out_audit, index=False, encoding="utf-8-sig")
        print(f"[DEBUG] Wrote: {out_audit}", flush=True)
    except Exception as e:
        print(f"[WARNING] Failed to write IS_candidate_audit.csv: {e}", flush=True)

    # ------------------------------------------------------------
    # CLASS-LEVEL OVERRIDE: enforce one denominator per class/polarity
    # ------------------------------------------------------------
    class_counts: dict[tuple[str, str], dict[str, int]] = {}
    # First pass: count which denominator is most common in each (class, polarity)
    for i, (_, row) in enumerate(features_df.iterrows()):
        lipid_class = str(row.get("Lipid Class", "")).strip()
        pol = row.get("Polarity_norm", None)
        den_name = chosen_den_name[i] if i < len(chosen_den_name) else None
        den_vec  = chosen_den_vector[i] if i < len(chosen_den_vector) else None

        if not lipid_class or lipid_class.lower() == "nan":
            continue
        if pol not in ("pos", "neg"):
            continue
        if den_vec is None or not isinstance(den_name, str) or not den_name:
            continue

        key = (lipid_class, pol)
        if key not in class_counts:
            class_counts[key] = {}
        class_counts[key][den_name] = class_counts[key].get(den_name, 0) + 1

    # For each class/polarity, find the most frequent denominator and its vector
    class_default: dict[tuple[str, str], tuple[str, pd.Series | np.ndarray]] = {}
    for key, counts in class_counts.items():
        # pick denominator name with highest count
        den_name_best = max(counts.items(), key=lambda kv: kv[1])[0]
        den_vec_best = None
        # find any row that used this denominator for this class/polarity
        for idx, row in features_df.iterrows():
            lipid_class = str(row.get("Lipid Class", "")).strip()
            pol = row.get("Polarity_norm", None)

            if idx >= len(chosen_den_name) or idx >= len(chosen_den_vector):
                continue

            if (lipid_class, pol) == key and chosen_den_name[idx] == den_name_best:
                den_vec_best = chosen_den_vector[idx]
                break
        if den_vec_best is not None:
            class_default[key] = (den_name_best, den_vec_best)

    # Second pass: override per-feature denominators so all features in a class share the same one
    overridden = 0
    total_class_rows = 0
    for idx, row in features_df.iterrows():
        lipid_class = str(row.get("Lipid Class", "")).strip()
        pol = row.get("Polarity_norm", None)
        if not lipid_class or lipid_class.lower() == "nan" or pol not in ("pos", "neg"):
            continue

        key = (lipid_class, pol)
        if key not in class_default:
            continue

        total_class_rows += 1
        best_name, best_vec = class_default[key]

        # If this feature already uses the class-major denominator, nothing to do
        if chosen_den_vector[idx] is not None and chosen_den_name[idx] == best_name:
            continue

        # Override denominator choice for this feature
        chosen_den_name[idx] = best_name
        chosen_den_vector[idx] = best_vec


        old_reason = chosen_reason[idx] if idx < len(chosen_reason) else ""
        if old_reason:
            chosen_reason[idx] = f"{old_reason} | overridden to class-major denominator {best_name} (class={lipid_class})"
        else:
            chosen_reason[idx] = f"class-major denominator {best_name} (class={lipid_class})"

        overridden += 1

    print(
        f"[INFO] Class-level denominator enforcement: overridden {overridden} feature(s) "
        f"across {len(class_default)} class/polarity group(s).",
        flush=True,
    )

    # Apply normalization
    norm_df = features_df.copy()
    # Remove carried-over legacy RSD columns that must be recomputed after normalization
    legacy_rsd_cols = [c for c in ["RSD QCs (%)", "RSD Samples (%)"] if c in norm_df.columns]
    if legacy_rsd_cols:
        norm_df.drop(columns=legacy_rsd_cols, inplace=True)
    
    for idx, row in features_df.iterrows():
        den = chosen_den_vector[idx]
        if den is None: continue
        den_vals = pd.to_numeric(den[feat_sample_cols], errors="coerce").replace(0, np.nan)
        for col in feat_sample_cols:
            fval = pd.to_numeric([row[col]], errors="coerce")[0] if col in features_df.columns else np.nan
            dval = pd.to_numeric([den_vals[col]], errors="coerce")[0]
            norm_df.at[idx, col] = (fval / dval) if (np.isfinite(fval) and np.isfinite(dval) and dval != 0) else np.nan

    norm_df["Matched IS"] = chosen_den_name; norm_df["Matched IS Reason"] = chosen_reason

    # ---------------------------------------------------------
    # DEBUG: Summarize which IS is used per lipid class/polarity
    # ---------------------------------------------------------
    try:
        dbg_dir = output_folder / "debug" / "normalization"
        dbg_dir.mkdir(parents=True, exist_ok=True)

        tmp = norm_df.copy()

        tmp["Matched IS_clean"] = tmp["Matched IS"].astype(str).str.strip()
        tmp["Matched IS_clean"] = tmp["Matched IS_clean"].str.replace(r"^IS:", "", regex=True)
        tmp["Matched IS_clean"] = tmp["Matched IS_clean"].apply(_canonical_is_name)

        tmp["Lipid Class"] = tmp["Lipid Class"].astype(str).str.strip()
        tmp["Polarity_norm"] = tmp["Polarity_norm"].astype(str).str.strip()

        tmp2 = tmp[
            tmp["Lipid Class"].notna() &
            (tmp["Lipid Class"].str.lower() != "nan") &
            tmp["Matched IS_clean"].notna() &
            (tmp["Matched IS_clean"] != "") &
            (tmp["Matched IS_clean"].str.lower() != "nan") &
            tmp["Polarity_norm"].isin(["pos", "neg"])
        ].copy()

        class_summary = (
            tmp2.groupby(["Lipid Class", "Polarity_norm", "Matched IS_clean"], dropna=False)
                .size()
                .reset_index(name="n_features")
        )

        totals = (
            class_summary.groupby(["Lipid Class", "Polarity_norm"])["n_features"]
                         .sum()
                         .reset_index(name="n_features_in_class")
        )
        class_summary = class_summary.merge(totals, on=["Lipid Class", "Polarity_norm"], how="left")
        class_summary["fraction_of_class"] = class_summary["n_features"] / class_summary["n_features_in_class"]

        class_summary = class_summary.sort_values(
            ["Lipid Class", "Polarity_norm", "n_features"],
            ascending=[True, True, False]
        )

        out_by_class = dbg_dir / f"{pol_tag}IS_selected_by_class.csv"
        class_summary.to_csv(out_by_class, index=False, encoding="utf-8-sig")
        print(f"[DEBUG] Wrote: {out_by_class}", flush=True)

    except Exception as e:
        print(f"[WARNING] Failed to write IS_selected_by_class.csv: {e}", flush=True)

    # Recompute RSDs and report
    print("[STEP] Recalculating RSDs after normalization...", flush = True)
    group_map = { g.strip(): [s for s in group_df.loc[group_df["Group"] == g, "Sample"].tolist()] for g in group_df["Group"].dropna().unique() }
    qc_groups = [g for g in group_map if g.upper() == "QC"]
    qc_samples_map = [s for g in qc_groups for s in group_map[g]]
    non_qc_groups = [g for g in group_map if g.upper() != "QC"]

    # qc_cols_use already defined earlier from sample_groups.csv mapping
    qc_cols_use = [c for c in qc_cols if c in sample_cols]

    non_qc_cols = [c for c in sample_cols if c not in qc_cols_use]
    has_qc = bool(qc_cols_use)
    if not has_qc:
        # No QC in this dataset, treat all samples as non-QC for RSD Samples (%)
        non_qc_cols = sample_cols[:]

    rsd_qc_vals, rsd_sample_vals = [], []
    improved = neutral = worse_allowed = worse_exceeded = 0
    for i, row in norm_df.iterrows():
        if has_qc:
            qc_vals_after = pd.to_numeric(row[qc_cols_use], errors="coerce").replace(0, np.nan).dropna()
            rsd_qc = (qc_vals_after.std(ddof=1) / qc_vals_after.mean() * 100) if len(qc_vals_after) > 1 else np.nan
        else:
            rsd_qc = np.nan
        sample_vals = pd.to_numeric(row[non_qc_cols], errors="coerce").replace(0, np.nan).dropna()
        rsd_samples = (sample_vals.std(ddof=1) / sample_vals.mean() * 100) if len(sample_vals) > 1 else np.nan
        rsd_qc_vals.append(rsd_qc); rsd_sample_vals.append(rsd_samples)

        if has_qc:
            b = _safe_rsd_series(pd.to_numeric(features_df.loc[i, qc_cols_use], errors="coerce"))
            a = rsd_qc
            if np.isfinite(b) and np.isfinite(a):
                if a < b:
                    improved += 1
                elif a == b:
                    neutral += 1
                else:
                    if a <= b * (1.0 + MAX_QC_RSD_WORSEN_PCT / 100.0):
                        worse_allowed += 1
                    else:
                        worse_exceeded += 1


    norm_df["RSD QCs (%)"] = rsd_qc_vals
    norm_df["RSD Samples (%)"] = rsd_sample_vals

    stale_group_rsd_cols = [c for c in norm_df.columns if c.startswith("RSD_")]
    if stale_group_rsd_cols:
        norm_df.drop(columns=stale_group_rsd_cols, inplace=True)
    
    for g in non_qc_groups:
        if str(g).strip().upper() == "QC":
            continue

        cols = [c for c in sample_cols if any((s == c) or (s in c) for s in group_map[g])]
        rsd_vals = []
        for _, row in norm_df.iterrows():
            vals = pd.to_numeric(row[cols], errors="coerce").replace(0, np.nan).dropna()
            rsd = (vals.std(ddof=1) / vals.mean() * 100) if len(vals) > 1 else np.nan
            rsd_vals.append(rsd)
        norm_df["RSD_{0} [%]".format(g)] = rsd_vals
   
    # Save
    print("[STEP] Reordering columns before saving...", flush=True)

    core_cols = [
        "UniqueID", "RT (min)", "m/z", "Neutral mass", "Adducts", "Polarity", "Internal Standard",
        "RSD QCs (%)", "RSD Samples (%)", "QC detected count", "RSD QCs observed before imputation (%)",
        "MS/MS available?", "Annotation", "Annotation Type",
        "Metaboscape Annotation Status", "Annotation Source", "Headgroup", "Lipid Class",
        "Δm/z (mDa)", "Δm/z (ppm)", "MS/MS score", "Annotation tier", "mSigma",
        "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]",
        "Molecular Formula",
        "Plasmenyl?", "Number of carbons in fatty acyls", "Double bond equivalents",
        "Chain type", "PUFA?", "Modifications", "# of modifications", "Oxidized?",
        "Carbons / double bond equivalent ratio", "Average Intensity (all samples)",
        "Minimum Intensity (all samples)", "Maximum Intensity (all samples)",
        "Matched IS", "Matched IS Reason", "Polarity_norm"
    ]

    group_rsd_cols = [
        c for c in norm_df.columns
        if c.startswith("RSD_") and c not in ["RSD QCs (%)", "RSD Samples (%)"]
    ]
    group_rsd_cols = sorted(group_rsd_cols)

    ordered_cols = []
    for c in core_cols:
        if c in norm_df.columns:
            ordered_cols.append(c)
        if c == "RSD Samples (%)":
            ordered_cols.extend([x for x in group_rsd_cols if x not in ordered_cols])

    ordered_cols.extend([c for c in sample_cols if c not in ordered_cols])
    norm_df = norm_df[[c for c in ordered_cols if c in norm_df.columns]]

    out_annotated = output_folder / "debug" / f"{pol_tag}4-Final_annotated_results_normalized.csv"
    out_unknown   = output_folder / "debug" / f"{pol_tag}5-Final_unknowns_normalized.csv"
    (output_folder / "debug").mkdir(parents=True, exist_ok=True)

    if has_qc:
        print(
            f"\n[SUMMARY] Features vs pre-normalization (QC RSD): "
            f"improved={improved}, neutral={neutral}, worse_allowed(≤{MAX_QC_RSD_WORSEN_PCT:.1f}%)={worse_allowed}, worse_exceeded={worse_exceeded}\n",
            flush=True)
    else:
        print("\n[SUMMARY] QC RSD comparison skipped (no QC).\n", flush=True)
        
    if "Annotation" in norm_df.columns:
        unknown_mask = norm_df["Annotation"].astype(str).str.strip().isin(["", "nan", "NaN", "N/A", "Unassigned", "No match", "None", "_", "Unknown"])
        unknown_df = norm_df[unknown_mask].copy(); annotated_df = norm_df[~unknown_mask].copy()
        unknown_df.to_csv(out_unknown, index=False, encoding="utf-8-sig")
        annotated_df.to_csv(out_annotated, index=False, encoding="utf-8-sig")
        print(f"[INFO] Separated {len(unknown_df)} unannotated features = {out_unknown}", flush = True)
        print(f"[INFO] Saved {len(annotated_df)} annotated features = {out_annotated}", flush = True)
        out_path = out_annotated
    else:
        norm_df.to_csv(out_annotated, index=False, encoding="utf-8-sig")
        out_path = out_annotated
        print("[WARNING] 'Annotation' column missing — saved all to annotated file.", flush = True)

    # ------------------------------------------------------------------
    # Semi-quantification: 4b-Final_annotated_results_norm_semi-quant.csv
    # ------------------------------------------------------------------
    try:
        # Locate Appendix/Internal Standards.xlsx (relative search up to a few levels)
        print(f'[INFO] Starting semi-quantification...', flush=True)
        # Use the metadata file already selected in the GUI or fallbacks
        # (is_meta_path was determined earlier in the function)
        if is_meta_path is None:
            print("[WARNING] Semi-quantification skipped: no valid IS metadata file available.", flush=True)

        elif "Annotation" not in norm_df.columns:
            print("[WARNING] Semi-quantification skipped: no 'Annotation' column.", flush=True)
        else:
            # Use only the annotated features file we just wrote
            annotated_for_sq = annotated_df.copy() if "annotated_df" in locals() else norm_df.copy()

            # Sample columns
            sample_cols_sq = [c for c in annotated_for_sq.columns if c.startswith("P_") or c.startswith("N_")]
            if not sample_cols_sq:
                print("[WARNING] Semi-quantification skipped: no sample columns found.", flush=True)
            else:
                # Read IS metadata with stock concentrations
                is_meta = pd.read_excel(is_meta_path)
                if "Standard" not in is_meta.columns or "Concentration (stock, umol/L)" not in is_meta.columns:
                    print("[WARNING] Semi-quantification skipped: 'Standard' or 'Concentration (stock, umol/L)' missing in Internal Standards.xlsx", flush=True)
                else:
                    stock_map = (
                        is_meta[["Standard", "Concentration (stock, umol/L)"]]
                        .dropna(subset=["Standard"])
                    )
                    stock_map["Standard"] = stock_map["Standard"].apply(_canonical_is_name)
                    stock_map = stock_map.set_index("Standard")["Concentration (stock, umol/L)"].to_dict()

                    # Guard against invalid dilution
                    if not isinstance(is_dilution_factor, (int, float)) or is_dilution_factor <= 0:
                        effective_dil = 1.0
                    else:
                        effective_dil = float(is_dilution_factor)

                    applied = 0
                    annotated_semi = annotated_for_sq.copy()

                    for idx, row in annotated_semi.iterrows():
                        m_is = str(row.get("Matched IS", "") or "").strip()
                        if not m_is:
                            continue
                        # strip 'IS:' prefix used in normalization
                        if m_is.upper().startswith("IS:"):
                            base_name = _canonical_is_name(m_is.split(":", 1)[1].strip())
                        else:
                            base_name = m_is

                        c_stock = stock_map.get(base_name, None)
                        try:
                            c_stock = float(c_stock)
                        except Exception:
                            continue
                        if not np.isfinite(c_stock):
                            continue

                        c_final = float(c_stock) / effective_dil  # µg/mL or µmol/L in the working solution

                        # Multiply the normalized intensities by c_final to get semi-quant values
                        vals = pd.to_numeric(
                            annotated_semi.loc[idx, sample_cols_sq],
                            errors="coerce"
                        )
                        annotated_semi.loc[idx, sample_cols_sq] = vals * c_final
                        applied += 1

                    out_annotated_semi = output_folder / "debug" / f"{pol_tag}4b-Final_annotated_results_norm_semi-quant.csv"
                    annotated_semi.to_csv(out_annotated_semi, index=False, encoding="utf-8-sig")
                    print(f"[INFO] Semi-quantification applied to {applied} feature(s).", flush=True)
                    print(f"[INFO] Semi-quantified annotated file saved: {out_annotated_semi}", flush=True)

    except Exception as e:
        print(f"[WARNING] Semi-quantification step failed: {e}", flush=True)


    # Evaluate
    try:
        print("\\n[STEP] Evaluating normalization performance...", flush = True)
        group_file = Path(output_folder).parent / "sample_groups.csv"
        normalized_with_rsd_csv, df_after = calculate_qc_rsd_post_norm(out_path, group_file, output_folder)
        if "RSD QCs (%) post-norm" in df_after.columns and df_after["RSD QCs (%) post-norm"].isna().all():
            print("[INFO] No QC detected in evaluation outputs. Skipping QC-based plots.", flush=True)
        else:
            plot_rsd_distributions(features_csv, normalized_with_rsd_csv, output_folder)
            plot_pca_before_after(features_csv, normalized_with_rsd_csv, group_file, output_folder)
            evaluate_normalization_performance(str(features_csv), str(out_path), str(group_file), str(output_folder))
        df_before = pd.read_csv(features_csv, low_memory=False)
        df_before.columns = df_before.columns.str.strip().str.replace("\\xa0", " ", regex=False)
        df_after.columns  = df_after.columns.str.strip().str.replace("\\xa0", " ", regex=False)

        before_col = next((c for c in df_before.columns if "rsd" in c.lower() and "qc" in c.lower()), None)
        after_col  = next((c for c in df_after.columns  if "rsd" in c.lower() and "qc" in c.lower() and "post-norm" in c.lower()), None)

        rsd_before = pd.to_numeric(df_before[before_col], errors="coerce").dropna() if before_col else pd.Series([], dtype=float)
        rsd_after  = pd.to_numeric(df_after[after_col],  errors="coerce").dropna()
        if not rsd_before.empty and not rsd_after.empty:
            med_before = float(np.nanmedian(rsd_before)); med_after = float(np.nanmedian(rsd_after))
            print(f"[INFO] Median QC RSD before normalization: {med_before:.2f}%", flush = True)
            print(f"[INFO] Median QC RSD after normalization:  {med_after:.2f}%", flush = True)
            print(f"[INFO] Delta-RSD (after - before): {med_after - med_before:+.2f}%", flush = True)

        print(f"[DONE] Normalization performance evaluation complete. Plots saved under {Path(output_folder)/'normalization'}\\n", flush = True)
    except Exception as e:
        print(f"[WARNING] Normalization evaluation skipped due to error: {e}", flush = True)
   

    # ------------------------------------------------------------------
    # Plot summed normalized intensities per sample and QC
    # ------------------------------------------------------------------
    try:
        # Load the annotated normalized CSV
        df_before_path = Path(features_csv)
        df_before = pd.read_csv(df_before_path, low_memory=False)
        df_norm = pd.read_csv(out_path, low_memory=False)

        # Try loading semi-quantified intensities
        df_semi = None
        semi_path = output_folder / "debug" / f"{pol_tag}4b-Final_annotated_results_norm_semi-quant.csv"
        if semi_path.exists():
            df_semi = pd.read_csv(semi_path, low_memory=False)
            df_semi.columns = df_semi.columns.str.strip().str.replace("\xa0", " ", regex=False)
            print(f"[INFO] Loaded semi-quantified file: {semi_path}", flush=True)
        else:
            print("[INFO] No semi-quantified file found. Skipping semi-quant plots.", flush=True)

        # create the debug / normalization folder
        plot_dir = Path(output_folder) / "debug" / "normalization"
        plot_dir.mkdir(parents=True, exist_ok=True)

        print("[STEP] Plotting summed normalized intensities per sample/QC...", flush=True)

        df_norm.columns = df_norm.columns.str.strip().str.replace("\xa0", " ", regex=False)
        norm_cols = [c for c in df_norm.columns if c.startswith("P_") or c.startswith("N_")]

        if not norm_cols:
            print("[WARNING] No normalized intensity columns found.", flush=True)
        else:
            # Convert to float
            numeric_norm = df_norm[norm_cols].apply(pd.to_numeric, errors="coerce")

            # Literal Excel-like sum (no hidden rows)
            vals = numeric_norm.to_numpy(float)
            summed_vals = pd.Series(np.nansum(vals, axis=0), index=numeric_norm.columns)

            # Use sample_groups.csv ordering
            group_df = pd.read_csv(group_file)
            nonqc = group_df[group_df["Group"].str.upper() != "QC"]["Sample"].tolist()
            qc     = group_df[group_df["Group"].str.upper() == "QC"]["Sample"].tolist()

            def expand(names):
                out = []
                for nm in names:
                    out.extend([col for col in summed_vals.index if nm in col])
                return out

            ordered_cols = expand(nonqc) + expand(qc)
            ordered_cols = [c for i,c in enumerate(ordered_cols) if c in summed_vals.index and ordered_cols.index(c) == i]

            summed_vals = summed_vals.loc[ordered_cols]

            # Plot
            plt.figure(figsize=(12,5))
            plt.bar(summed_vals.index, summed_vals.values)
            plt.xticks(rotation=90)
            plt.ylabel("Summed normalized intensity")
            plt.title(f"{pol_tag}Summed normalized intensities per sample/QC")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{pol_tag}summed_normalized_intensities.png", dpi=100)
            plt.close()

         # --------------------------------------------
        # Summed intensities (SEMI-QUANT)
        # --------------------------------------------
        if df_semi is not None:
            semi_cols = [c for c in df_semi.columns if c.startswith("P_") or c.startswith("N_")]
            numeric_semi = df_semi[semi_cols].apply(pd.to_numeric, errors="coerce")
            vals = numeric_semi.to_numpy(float)
            summed_vals_semi = pd.Series(np.nansum(vals, axis=0), index=numeric_semi.columns)

            ordered_cols_semi = expand(nonqc) + expand(qc)
            ordered_cols_semi = [c for c in ordered_cols_semi if c in summed_vals_semi.index]

            summed_vals_semi = summed_vals_semi.loc[ordered_cols_semi]

            plt.figure(figsize=(12,5))
            plt.bar(summed_vals_semi.index, summed_vals_semi.values)
            plt.xticks(rotation=90)
            plt.ylabel("Summed semi-quant intensity (umol/L)")
            plt.title(f"{pol_tag}Summed semi-quantified intensities per sample/QC")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{pol_tag}summed_intensities_semi_quant.png", dpi=100)
            plt.close()

            print("[INFO] Summed intensity (semi-quant) plot saved.", flush=True)


            print("[INFO] Summed normalized intensity plot saved.", flush=True)

        print("[STEP] Plotting summed intensities per sample/QC before normalization...", flush=True)

        df_before.columns = df_before.columns.str.strip().str.replace("\xa0", " ", regex=False)
        before_cols = [c for c in df_before.columns if c.startswith("P_") or c.startswith("N_")]

        if not before_cols:
            print("[WARNING] No normalized intensity columns found.", flush=True)
        else:
            # Convert to float
            numeric_before = df_before[before_cols].apply(pd.to_numeric, errors="coerce")

            # Literal Excel-like sum (no hidden rows)
            vals = numeric_before.to_numpy(float)
            summed_vals_before = pd.Series(np.nansum(vals, axis=0), index=numeric_before.columns)

            def expand(names):
                out = []
                for nm in names:
                    out.extend([col for col in summed_vals_before.index if nm in col])
                return out

            ordered_cols_before = expand(nonqc) + expand(qc)
            ordered_cols_before = [c for i,c in enumerate(ordered_cols_before) if c in summed_vals_before.index and ordered_cols_before.index(c) == i]

            summed_vals_before = summed_vals_before.loc[ordered_cols_before]

            # Plot
            plt.figure(figsize=(12,5))
            plt.bar(summed_vals_before.index, summed_vals_before.values)
            plt.xticks(rotation=90)
            plt.ylabel("Summed intensity before normalization")
            plt.title(f"{pol_tag}Summed intensities before normalization per sample/QC")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{pol_tag}summed_intensities_before_norm.png", dpi=100)
            plt.close()

            print("[INFO] Summed intensity (before normalization) plot saved.", flush=True)

    except Exception as e:
        print(f"[WARNING] Failed to plot summed intensities: {e}", flush=True)

    # ------------------------------------------------------------------
    # Distribution histograms before and after normalization
    # ------------------------------------------------------------------
    print("\n[STEP] Plotting intensity distributions before/after normalization...\n", flush=True)

    try:

        # ----- BEFORE normalization -----
        # Flatten to 1D vector
        vals_before = numeric_before.to_numpy(dtype=float).reshape(-1)

        # Raw histogram (before)
        plt.figure(figsize=(7,4))
        plt.hist(vals_before, bins=80, alpha=0.7)
        plt.xlabel("Intensity (raw)")
        plt.ylabel("Count")
        plt.title(f"{pol_tag}Intensity Distribution BEFORE normalization")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{pol_tag}hist_before_raw.png", dpi=100)
        plt.close()

        # Log-scaled histogram (before)
        plt.figure(figsize=(7,4))
        plt.hist(np.log10(vals_before + 1e-12), bins=80, alpha=0.7)
        plt.xlabel("log10(Intensity)")
        plt.ylabel("Count")
        plt.title(f"{pol_tag}Intensity Distribution BEFORE normalization (log10)")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{pol_tag}hist_before_log.png", dpi=100)
        plt.close()

        # ----- AFTER normalization -----
        vals_after = numeric_norm.to_numpy(dtype=float).reshape(-1)

        # Raw histogram (after)
        plt.figure(figsize=(7,4))
        plt.hist(vals_after, bins=80, alpha=0.7)
        plt.xlabel("Intensity (raw)")
        plt.ylabel("Count")
        plt.title(f"{pol_tag}Intensity Distribution AFTER normalization")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{pol_tag}hist_after_raw.png", dpi=100)
        plt.close()

        # Log-scaled histogram (after)
        plt.figure(figsize=(7,4))
        plt.hist(np.log10(vals_after + 1e-12), bins=80, alpha=0.7)
        plt.xlabel("log10(Intensity)")
        plt.ylabel("Count")
        plt.title(f"{pol_tag}Intensity Distribution AFTER normalization (log10)")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{pol_tag}hist_after_log.png", dpi=100)
        plt.close()

        print("[INFO] Intensity distribution histograms saved.", flush=True)
    
    except Exception as e:
        print(f"\n[WARNING] Failed to plot histograms: {e}\n", flush=True)

    # ------------------------------------------------------------------
    # SEPARATE BOXPLOTS: summed intensities per group (AFTER normalization)
    # ------------------------------------------------------------------
    print("[STEP] Boxplot of summed intensities per group (AFTER normalization)...", flush=True)
    
    try:
        
        group_df = pd.read_csv(group_file)
        group_names = group_df["Group"].unique().tolist()

        # Map sample→group AFTER (keep sample names)
        group_sums_after = {g: [] for g in group_names}
        for sample in summed_vals.index:
            grp = group_df.loc[group_df["Sample"] == sample, "Group"]
            if not grp.empty:
                group_sums_after[grp.values[0]].append((sample, float(summed_vals[sample])))


        group_sums_after = {g: v for g,v in group_sums_after.items() if len(v) > 0}
        labelsA = sorted(group_sums_after.keys())

        # Colors per group
        cmap = plt.cm.tab20(np.linspace(0,1,len(labelsA)))
        color_mapA = dict(zip(labelsA, cmap))

        # Prepare data
        dataA = [[v for (_, v) in group_sums_after[g]] for g in labelsA]

        fig, ax = plt.subplots(figsize=(10,5))
        bp = ax.boxplot(
            dataA,
            labels=labelsA,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )

        # Apply fill colors
        for patch, g in zip(bp["boxes"], labelsA):
            patch.set_facecolor(color_mapA[g])
            patch.set_edgecolor("black")

        # Scatter jittered points + label outliers
        x_positions = np.arange(1, len(labelsA) + 1)
        for i, g in enumerate(labelsA):
            pairs = group_sums_after[g]  # list of (sample, value)
            sample_names = [s for (s, _) in pairs]
            vals = np.array([v for (_, v) in pairs], dtype=float)

            xjit = np.random.normal(x_positions[i], 0.06, len(vals))
            ax.scatter(xjit, vals, s=25, color="black", alpha=0.75, zorder=4)

            # Outlier thresholds (1.2*IQR)
            if len(vals) >= 3:
                q1, q3 = np.percentile(vals, [25, 75])
                iqr = q3 - q1
                lo = q1 - 1.2 * iqr
                hi = q3 + 1.2 * iqr
            else:
                lo, hi = -np.inf, np.inf

            # Label ONLY outliers, using sample name
            for xv, yv, sname in zip(xjit, vals, sample_names):
                if yv < lo or yv > hi:
                    ax.text(xv, yv, sname, fontsize=7, color="red", ha="left", va="bottom")

        ax.set_ylabel("Summed normalized intensity")
        ax.set_title(f"{pol_tag} Summed intensities per group (AFTER normalization)")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()

        outA = plot_dir / f"{pol_tag}summed_intensity_groups_AFTER.png"
        fig.savefig(outA, dpi=100)
        plt.close()
        print(f"[INFO] Saved: {outA}", flush=True)

        # ----------------------------------------------------------
        # Boxplot: summed intensities per group (SEMI-QUANT)
        # ----------------------------------------------------------
        if df_semi is not None:
            print("[STEP] Boxplot of summed intensities per group (SEMI-QUANT)...", flush=True)

            semi_cols = [c for c in df_semi.columns if c.startswith("P_") or c.startswith("N_")]
            numeric_semi = df_semi[semi_cols].apply(pd.to_numeric, errors="coerce")
            vals_semi = numeric_semi.to_numpy(float)
            summed_vals_semi = pd.Series(np.nansum(vals_semi, axis=0), index=numeric_semi.columns)

            group_sums_semi = {g: [] for g in group_names}
            for sample in summed_vals_semi.index:
                grp = group_df.loc[group_df["Sample"] == sample, "Group"]
                if not grp.empty:
                    group_sums_semi[grp.values[0]].append(float(summed_vals_semi[sample]))

            group_sums_semi = {g: v for g, v in group_sums_semi.items() if len(v) > 0}
            labelsS = sorted(group_sums_semi.keys())

            cmapS = plt.cm.tab20(np.linspace(0,1,len(labelsS)))
            color_mapS = dict(zip(labelsS, cmapS))

            dataS = [group_sums_semi[g] for g in labelsS]

            fig, ax = plt.subplots(figsize=(10,5))
            bp = ax.boxplot(
                dataS,
                labels=labelsS,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.4),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
            )

            for patch, g in zip(bp["boxes"], labelsS):
                patch.set_facecolor(color_mapS[g])
                patch.set_edgecolor("black")

            x_positions = np.arange(1, len(labelsS)+1)
            for i, g in enumerate(labelsS):
                vals = np.array(group_sums_semi[g], dtype=float)
                xjit = np.random.normal(x_positions[i], 0.06, len(vals))
                ax.scatter(xjit, vals, s=25, color="black", alpha=0.75, zorder=4)

            ax.set_ylabel("Summed semi-quant intensity (umol/L)")
            ax.set_title(f"{pol_tag} Summed intensities per group (SEMI-QUANT)")
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()

            outS = plot_dir / f"{pol_tag}summed_intensity_groups_SEMIQUANT.png"
            fig.savefig(outS, dpi=100)
            plt.close()
            print(f"[INFO] Saved: {outS}", flush=True)


        # ------------------------------------------------------------------
        # SEPARATE BOXPLOTS: summed intensities per group (BEFORE normalization)
        # ------------------------------------------------------------------
        print("[STEP] Boxplot of summed intensities per group (BEFORE normalization)...", flush=True)

        # Map sample→group BEFORE
        # Map sample→group BEFORE (keep sample names)
        group_sums_before = {g: [] for g in group_names}
        for sample in summed_vals_before.index:
            grp = group_df.loc[group_df["Sample"] == sample, "Group"]
            if not grp.empty:
                group_sums_before[grp.values[0]].append((sample, float(summed_vals_before[sample])))

        group_sums_before = {g: v for g,v in group_sums_before.items() if len(v) > 0}
        labelsB = sorted(group_sums_before.keys())

        # Colors per group (same palette)
        cmapB = plt.cm.tab20(np.linspace(0,1,len(labelsB)))
        color_mapB = dict(zip(labelsB, cmapB))

        # Prepare data
        dataB = [[v for (_, v) in group_sums_before[g]] for g in labelsB]

        fig, ax = plt.subplots(figsize=(10,5))
        bp = ax.boxplot(
            dataB,
            labels=labelsB,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )

        # Colors
        for patch, g in zip(bp["boxes"], labelsB):
            patch.set_facecolor(color_mapB[g])
            patch.set_edgecolor("black")

        # Jittered points + outliers (label with sample name)
        x_positions = np.arange(1, len(labelsB) + 1)
        for i, g in enumerate(labelsB):
            pairs = group_sums_before[g]  # list of (sample, value)
            sample_names = [s for (s, _) in pairs]
            vals = np.array([v for (_, v) in pairs], dtype=float)

            xjit = np.random.normal(x_positions[i], 0.06, len(vals))
            ax.scatter(xjit, vals, s=25, color="black", alpha=0.75, zorder=4)

            if len(vals) >= 3:
                q1, q3 = np.percentile(vals, [25, 75])
                iqr = q3 - q1
                lo = q1 - 1.2 * iqr
                hi = q3 + 1.2 * iqr
            else:
                lo, hi = -np.inf, np.inf

            for xv, yv, sname in zip(xjit, vals, sample_names):
                if yv < lo or yv > hi:
                    ax.text(xv, yv, sname, fontsize=7, color="red", ha="left", va="bottom")


        ax.set_ylabel("Summed raw intensity")
        ax.set_title(f"{pol_tag} Summed intensities per group (BEFORE normalization)")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()

        outB = plot_dir / f"{pol_tag}summed_intensity_groups_BEFORE.png"
        fig.savefig(outB, dpi=100)
        plt.close()
        print(f"[INFO] Saved: {outB}", flush=True)

    except Exception as e:
        print(f"\n[WARNING] Failed to plot boxplots: {e}\n", flush=True)

    # --------------------------------------------------------------
    # BAR PLOTS: summed intensities per lipid class (AFTER norm)
    # --------------------------------------------------------------
    print("[STEP] Creating bar plot of summed class intensities (AFTER normalization)...", flush=True)

    try:

        # ---------------------------------------------------------
        # Consistent class color mapping (shared BEFORE and AFTER)
        # ---------------------------------------------------------
        all_classes = set()

        # collect classes from BEFORE
        if "Lipid Class" in df_before.columns:
            all_classes.update(df_before["Lipid Class"].dropna().unique().tolist())

        # collect classes from AFTER
        if "Lipid Class" in df_norm.columns:
            all_classes.update(df_norm["Lipid Class"].dropna().unique().tolist())

        # sorted for stability
        all_classes = sorted(all_classes)

        # assign one color per class (global mapping)
        cmap_global = plt.cm.tab20(np.linspace(0, 1, len(all_classes)))
        class_color = dict(zip(all_classes, cmap_global))


        # Identify sample columns
        sample_cols_norm = [c for c in df_norm.columns if c.startswith("P_") or c.startswith("N_")]

        # Compute summed intensity per class
        if "Lipid Class" not in df_norm.columns:
            print("[WARNING] 'Lipid Class' column missing in normalized file; skipping class barplots.")
        else:
            # Convert sample columns to numeric once
            numeric_norm = df_norm[sample_cols_norm].apply(pd.to_numeric, errors="coerce")

            # Add a row-wise sum
            df_norm["_row_sum_norm"] = numeric_norm.sum(axis=1)

            # Clean grouped sum
            class_sums_after = (
                df_norm.groupby("Lipid Class")["_row_sum_norm"]
                .sum()
                .sort_values(ascending=False)
            )

            # Plot
            plt.figure(figsize=(12,6))
            plt.bar(
                class_sums_after.index,
                class_sums_after.values,
                color=[class_color[c] for c in class_sums_after.index]
            )
            plt.yscale("log")
            plt.xticks(rotation=90)
            plt.ylabel("Summed normalized intensity (log scale)")
            plt.title(f"{pol_tag} Summed intensity per lipid class (AFTER normalization)")
            plt.tight_layout()

            out_after_class = plot_dir / f"{pol_tag}summed_intensity_per_class_AFTER.png"
            plt.savefig(out_after_class, dpi=100)
            plt.close()
            print(f"[INFO] Saved: {out_after_class}", flush=True)

        # --------------------------------------------------------------
        # BAR PLOTS: summed intensities per lipid class (SEMI-QUANT)
        # --------------------------------------------------------------
        if df_semi is not None and "Lipid Class" in df_semi.columns:
            print("[STEP] Creating bar plot of summed class intensities (SEMI-QUANT)...", flush=True)

            semi_cols = [c for c in df_semi.columns if c.startswith("P_") or c.startswith("N_")]
            numeric_semi = df_semi[semi_cols].apply(pd.to_numeric, errors="coerce")
            df_semi["_row_sum_semi"] = numeric_semi.sum(axis=1)

            class_sums_semi = (
                df_semi.groupby("Lipid Class")["_row_sum_semi"]
                .sum()
                .sort_values(ascending=False)
            )

            plt.figure(figsize=(12,6))
            plt.bar(
                class_sums_semi.index,
                class_sums_semi.values,
                color=[class_color[c] for c in class_sums_semi.index]
            )
            plt.yscale("log")
            plt.xticks(rotation=90)
            plt.ylabel("Summed semi-quantified intensity (umol/L) (log scale)")
            plt.title(f"{pol_tag} Summed intensity per lipid class (SEMI-QUANT)")
            plt.tight_layout()

            out_semi_class = plot_dir / f"{pol_tag}summed_intensity_per_class_SEMIQUANT.png"
            plt.savefig(out_semi_class, dpi=100)
            plt.close()
            print(f"[INFO] Saved: {out_semi_class}", flush=True)


        # --------------------------------------------------------------
        # BAR PLOTS: summed intensities per lipid class (BEFORE norm)
        # --------------------------------------------------------------
        print("[STEP] Creating bar plot of summed class intensities (BEFORE normalization)...", flush=True)

        sample_cols_before = [c for c in df_before.columns if c.startswith("P_") or c.startswith("N_")]

        if "Lipid Class" not in df_before.columns:
            print("[WARNING] 'Lipid Class' column missing in input file; skipping BEFORE class barplot.")
        else:
            numeric_before = df_before[sample_cols_before].apply(pd.to_numeric, errors="coerce")
            df_before["_row_sum_before"] = numeric_before.sum(axis=1)

            class_sums_before = (
                df_before.groupby("Lipid Class")["_row_sum_before"]
                .sum()
                .sort_values(ascending=False)
            )

            plt.figure(figsize=(12,6))
            plt.bar(
                class_sums_before.index,
                class_sums_before.values,
                color=[class_color[c] for c in class_sums_before.index]
            )
            plt.yscale("log")
            plt.xticks(rotation=90)
            plt.ylabel("Summed raw intensity (log scale)")
            plt.title(f"{pol_tag} Summed intensity per lipid class (BEFORE normalization)")
            plt.tight_layout()

            out_before_class = plot_dir / f"{pol_tag}summed_intensity_per_class_BEFORE.png"
            plt.savefig(out_before_class, dpi=100)
            plt.close()
            print(f"[INFO] Saved: {out_before_class}", flush=True)

    except Exception as e:
        print(f"\n[WARNING] Failed to plot bar plots per class: {e}\n", flush=True)

    print(f"Normalization complete. Saved normalized results to: {out_path}.\n\n", flush = True)


    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Normalize lipidomics data by class-matched internal standards (updated logic v3.2, with semi-quantification)."
    )
    parser.add_argument("--features", required=True, help="Path to Final_annotated_results_imputed_filtered.csv")
    parser.add_argument("--isfile", required=True, help="Path to Internal_standards.csv")
    parser.add_argument("--classmap", required=True, help="Path to Class_to_internal_standards.csv")
    parser.add_argument("--out", default="results", help="Output folder")
    parser.add_argument(
        "--is_dilution_factor",
        type=float,
        default=1.0,
        help="Dilution factor applied to IS stock solution (e.g., 20 for a 1:20 dilution).",
    )
    args = parser.parse_args()

    normalize_by_internal_standards(
        args.features,
        args.isfile,
        args.classmap,
        args.out,
        is_dilution_factor=args.is_dilution_factor,
    )
