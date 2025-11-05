
# ---------------------------------------------------------------------
# Class-matched internal standard normalization for lipidomics datasets
# ---------------------------------------------------------------------
import re
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')  # ensures full UTF-8 output even on Windows

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*is_sparse is deprecated.*",
    category=FutureWarning
)

# ---------------------------------------------------------------------
# Normalization evaluation utilities
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def calculate_qc_rsd_post_norm(normalized_csv, sample_groups_csv, output_folder="results"):
    """
    Calculate QC RSD (%) for each feature after normalization
    using the same QC column logic as in the normalization step.
    """
    print(f'\nStarting internal standard normalization...\n')
    df = pd.read_csv(normalized_csv, low_memory=False)
    group_df = pd.read_csv(sample_groups_csv, low_memory=False)
    output_folder = Path(output_folder) / "debug" /"normalization"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Identify QC samples from sample_groups.csv
    qc_samples = group_df.loc[group_df["Group"].str.upper().str.strip() == "QC", "Sample"].tolist()
    if not qc_samples:
        raise ValueError("No QC samples found in sample_groups.csv.", flush = True)

    # Identify columns in df corresponding to QC samples (exact or substring match)
    qc_cols = []
    for sample in qc_samples:
        if sample in df.columns:
            qc_cols.append(sample)
        else:
            qc_cols.extend([c for c in df.columns if sample in c])

    qc_cols = list(set(qc_cols))
    if not qc_cols:
        raise ValueError("No matching QC columns found in normalized file for assigned QC samples.", flush = True)

    print(f"[INFO] Found {len(qc_cols)} QC columns for post-normalization RSD calculation.", flush=True)

    # Compute RSD for each feature
    qc_rsd = []
    for _, row in df.iterrows():
        vals = row[qc_cols]

        # Always treat as Series
        if not isinstance(vals, pd.Series):
            vals = pd.Series([vals])

        vals = pd.to_numeric(vals, errors="coerce").dropna()

        if len(vals) > 1:
            rsd = (np.std(vals, ddof=1) / np.mean(vals)) * 100
        else:
            rsd = np.nan

        qc_rsd.append(rsd)

    df["RSD QCs (%) post-norm"] = qc_rsd

    out_path = output_folder / "Final_annotated_results_normalized_with_QC_RSD.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    median_rsd = np.nanmedian(df["RSD QCs (%) post-norm"])
    print(f"[INFO] Median QC RSD after normalization: {median_rsd:.2f}%", flush = True)
    print(f"[INFO] Added 'RSD QCs (%) post-norm' column and saved to: {out_path}", flush = True)
    return out_path, df

def plot_rsd_distributions(imputed_filtered_csv, normalized_with_rsd_csv, output_folder="results"):
    """
    Plot boxplot and histogram comparing RSD QCs before and after normalization.
    Handles column name inconsistencies robustly.
    """
    df_before = pd.read_csv(imputed_filtered_csv, low_memory=False)
    df_after = pd.read_csv(normalized_with_rsd_csv, low_memory=False)
    output_folder = Path(output_folder) /"debug" / "normalization"
    output_folder.mkdir(parents=True, exist_ok=True)

    # --- Normalize column names ---
    df_before.columns = df_before.columns.str.strip().str.replace("\xa0", " ", regex=False)
    df_after.columns = df_after.columns.str.strip().str.replace("\xa0", " ", regex=False)

    # --- Find RSD columns (case-insensitive, flexible order of words) ---
    before_col = next(
        (c for c in df_before.columns if "rsd" in c.lower() and "qc" in c.lower()),
        None)
    after_col = next(
        (c for c in df_after.columns if "rsd" in c.lower() and "qc" in c.lower() and "post-norm" in c.lower()),
        None)

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

    if rsd_before.empty and rsd_after.empty:
        print("[WARNING] No valid RSD data found for plotting.", flush=True)
        return

    print(f"[INFO] Plotting RSD distributions (Before: {len(rsd_before)}, After: {len(rsd_after)})...", flush=True)

    data = pd.DataFrame({
        "RSD (%)": np.concatenate([rsd_before.values, rsd_after.values]),
        "Condition": ["Before normalization"] * len(rsd_before) + ["After normalization"] * len(rsd_after)
    })

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=data, x="Condition", y="RSD (%)", hue="Condition",
                palette="Set2", showfliers=False, legend=False)
    plt.title("QC RSD Distribution Before vs After Normalization")
    plt.tight_layout()
    plt.savefig(output_folder / "RSD_boxplot_before_after.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.histplot(data=data, x="RSD (%)", hue="Condition", bins=40, kde=True, palette="Set2")
    plt.title("QC RSD Histogram Before vs After Normalization")
    plt.tight_layout()
    plt.savefig(output_folder / "RSD_hist_before_after.png", dpi=300)
    plt.close()

    print(f"[INFO] Saved RSD boxplot and histogram in: {output_folder}", flush=True)


def plot_pca_before_after(imputed_filtered_csv, normalized_with_rsd_csv, sample_groups_csv, output_folder="results"):
    """
    PCA visualization of samples before and after normalization.
    """
    df_before = pd.read_csv(imputed_filtered_csv, low_memory=False)
    df_after = pd.read_csv(normalized_with_rsd_csv, low_memory=False)
    group_df = pd.read_csv(sample_groups_csv, low_memory=False)
    output_folder = Path(output_folder) / "debug" / "normalization"
    output_folder.mkdir(parents=True, exist_ok=True)

    sample_cols = [c for c in df_before.columns if c.startswith("[POS") or c.startswith("[NEG]")]
    sample_cols = [c for c in sample_cols if c in df_after.columns]

    if not sample_cols:
        print("[WARNING] No sample columns found for PCA.")
        return

    sample_labels = []
    for c in sample_cols:
        grp = group_df.loc[group_df["Sample"] == c, "Group"]
        sample_labels.append(grp.values[0] if not grp.empty else "Unknown")

    def run_pca_and_plot(df, label, suffix):
        X = df[sample_cols].apply(pd.to_numeric, errors="coerce")
        if X.isna().all(axis=None):
            print(f"[WARNING] All intensities missing for {label}. Skipping PCA.", flush = True)
            return

        X = X.fillna(0)
        X_log = np.log2(X + 1)
        X_scaled = StandardScaler().fit_transform(X_log.T)
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame({
            "PC1": pca_res[:, 0],
            "PC2": pca_res[:, 1],
            "Group": sample_labels
        })

        plt.figure(figsize=(7, 6))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Group", s=70, palette="Set2", alpha=0.8)
        plt.title(f"PCA {label}\n({pca.explained_variance_ratio_[0]*100:.1f}% + {pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        plt.tight_layout()
        plt.savefig(output_folder / f"PCA_{suffix}.png", dpi=300)
        plt.close()

    run_pca_and_plot(df_before, "Before normalization", "before")
    run_pca_and_plot(df_after, "After normalization", "after")
    print(f"[INFO] PCA plots saved in: {output_folder}", flush = True)


def evaluate_normalization_performance(imputed_filtered_csv, normalized_csv, sample_groups_csv, output_folder="results"):
    """
    Compute QC RSD after normalization and generate evaluation plots,
    using flexible column detection for both before and after.
    """
    print("\n[STEP] Evaluating normalization performance...", flush=True)
    try:
        normalized_with_rsd_csv, df_after = calculate_qc_rsd_post_norm(
            normalized_csv, sample_groups_csv, output_folder
        )

        # Load before-normalization data
        df_before = pd.read_csv(imputed_filtered_csv, low_memory=False)
        df_before.columns = df_before.columns.str.strip().str.replace("\xa0", " ", regex=False)
        df_after.columns = df_after.columns.str.strip().str.replace("\xa0", " ", regex=False)

        # --- Flexible lookup for RSD columns ---
        before_col = next((c for c in df_before.columns if "rsd" in c.lower() and "qc" in c.lower()), None)
        after_col = next(
        (c for c in df_after.columns if "rsd" in c.lower() and "qc" in c.lower() and "post-norm" in c.lower()),
        None)
        
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

        # --- Median RSD comparison ---
        if not rsd_before.empty and not rsd_after.empty:
            med_before = np.nanmedian(rsd_before)
            med_after = np.nanmedian(rsd_after)
            print(f"[INFO] Median QC RSD before normalization: {med_before:.2f}%", flush=True)
            print(f"[INFO] Median QC RSD after normalization:  {med_after:.2f}%", flush=True)
            print(f"[INFO] Delta-RSD (after - before): {med_after - med_before:+.2f}%", flush=True)
        else:
            print("[WARNING] Could not compute RSD improvement — missing RSD data.", flush=True)

        # --- Generate plots ---
        plot_rsd_distributions(imputed_filtered_csv, normalized_with_rsd_csv, output_folder)
        plot_pca_before_after(imputed_filtered_csv, normalized_with_rsd_csv, sample_groups_csv, output_folder)

        print(f"[DONE] Normalization performance evaluation complete. "
              f"Plots saved under {Path(output_folder)/'normalization'}\n", flush=True)

    except Exception as e:
        import traceback
        print(f"[ERROR] Normalization evaluation failed: {e}", flush=True)
        print(traceback.format_exc(), flush=True)


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize_by_internal_standards(
    features_csv,
    internal_standards_csv,
    class_to_is_csv,
    output_folder="results"
):
    features_csv = Path(features_csv)
    internal_standards_csv = Path(internal_standards_csv)
    class_to_is_csv = Path(class_to_is_csv)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    features_df = pd.read_csv(features_csv, low_memory=False)
    is_df = pd.read_csv(internal_standards_csv, low_memory=False)
    class_map = pd.read_csv(class_to_is_csv, low_memory=False)

    # --- Identify sample columns (intensity columns that start with [POS or [NEG) ---
    sample_cols = [c for c in features_df.columns if c.startswith("[POS") or c.startswith("[NEG")]
    if not sample_cols:
        raise ValueError("No sample columns found. Expected columns starting with [POS or [NEG].")

    # --- Load sample groups to identify QC samples using GUI assignments ---
    group_file = Path(output_folder) / "sample_groups.csv"
    if not group_file.exists():
        raise FileNotFoundError("sample_groups.csv not found in output folder.")
    group_df = pd.read_csv(group_file, low_memory=False)

    # Identify QC samples from GUI assignment
    qc_samples = group_df.loc[group_df["Group"].str.upper().str.strip() == "QC", "Sample"].tolist()
    if not qc_samples:
        raise ValueError("No QC samples found in sample_groups.csv.")

    # Identify columns in features_df corresponding to QC samples
    qc_cols = []
    for sample in qc_samples:
        # Exact match to the column name in the dataset
        if sample in features_df.columns:
            qc_cols.append(sample)
        else:
            # If the full name doesn’t match (due to format like “[POS]...[NEG]...”),
            # include columns that *contain* that substring
            qc_cols.extend([c for c in features_df.columns if sample in c])

    qc_cols = list(set(qc_cols))
    if not qc_cols:
        raise ValueError("No matching QC columns found in features file for assigned QC samples.", flush = True)

    print(f"[INFO] Found {len(qc_cols)} QC columns for RSD filtering of internal standards.", flush = True)

    # --- Normalize polarity text ---
    def norm_pol(x):
        s = str(x).lower().strip()
        if s.startswith("pos"):
            return "pos"
        elif s.startswith("neg"):
            return "neg"
        return None

    # Apply polarity normalization for datasets that actually have the column
    features_df["Polarity_norm"] = features_df["Polarity"].apply(norm_pol)
    is_df["Polarity_norm"] = is_df["Polarity"].apply(norm_pol)
    
    # -----------------------------------------------------------------
    # Collapse duplicate internal standards (same Annotation + Polarity)
    # Keep the one with the highest mean intensity if:
    #   (1) RSD QCs (%) < 25, and
    #   (2) no missing values across all sample columns.
    # If not satisfied, try the next highest intensity.
    # If none meet criteria, keep the highest-intensity feature anyway.
    # -----------------------------------------------------------------
    collapsed_is = []
    sample_cols_is = [c for c in is_df.columns if c.startswith("[POS") or c.startswith("[NEG]")]

    for (annot, pol), group in is_df.groupby(["Annotation", "Polarity_norm"]):
        if group.empty:
            continue

        group = group.copy()
        group["mean_intensity"] = group[sample_cols_is].mean(axis=1, skipna=True)
        group_sorted = group.sort_values("mean_intensity", ascending=False)

        best_row = None
        for _, row in group_sorted.iterrows():
            rsd_val = pd.to_numeric(row.get("RSD QCs (%)", np.nan), errors="coerce")
            vals = row[sample_cols_is].astype(float)
            has_missing = vals.isna().any() or (vals == 0).any()

            if (not has_missing) and pd.notna(rsd_val) and rsd_val < 25:
                best_row = row
                break  # ideal candidate found, stop loop

        # If no row meets both conditions, fallback to the highest mean_intensity
        if best_row is None:
            best_row = group_sorted.iloc[0]

        collapsed_is.append(best_row)

    is_df = pd.DataFrame(collapsed_is).reset_index(drop=True)
    print(f"[INFO] Collapsed internal standards: now {len(is_df)} unique IS entries.", flush=True)

    # --- Build mapping from Class_to_internal_standards.csv ---
    # This supports multiple fallback IS per polarity (main → option 1 → option 2 → option 3)
    class_map_dict = {}

    # Pre-sort columns to ensure "main" → "option 1" → "option 2" → "option 3"
    cols_pos = [c for c in class_map.columns if "(main)" in c and "pos" in c.lower()] + \
                [c for c in class_map.columns if "(option" in c.lower() and "pos" in c.lower()]
    cols_neg = [c for c in class_map.columns if "(main)" in c and "neg" in c.lower()] + \
                [c for c in class_map.columns if "(option" in c.lower() and "neg" in c.lower()]

    for _, row in class_map.iterrows():
        lipid_class = str(row["Class"]).strip()
        if not lipid_class or lipid_class.lower() == "nan":
            continue  # skip invalid or empty class rows
        # Keep ordered list of IS candidates for each polarity
        pos_list = [str(row[c]).strip() for c in cols_pos if pd.notna(row[c]) and str(row[c]).strip()]
        neg_list = [str(row[c]).strip() for c in cols_neg if pd.notna(row[c]) and str(row[c]).strip()]
        class_map_dict[(lipid_class, "pos")] = pos_list
        class_map_dict[(lipid_class, "neg")] = neg_list

    print(f"[DEBUG] Loaded {len(class_map_dict)} class-to-IS mappings with priority order.")

    # --- Match each feature to best available internal standard ---
    matched_is = []
    match_reasons = [] 
    for idx, row in features_df.iterrows():
        lipid_class = str(row.get("Lipid Class", "")).strip()
        annotation = str(row.get("Annotation", "")).strip()

        # Skip features with no annotation or lipid class
        if not lipid_class or lipid_class.lower() == "nan" or not annotation or annotation.lower() == "nan":
            matched_is.append(np.nan)
            match_reasons.append("Skipped — no annotation or lipid class")
            continue

        key = (lipid_class, row["Polarity_norm"])
        is_candidates = class_map_dict.get(key, [])
        
        matched = np.nan
        best_rsd = np.inf
        fallback_candidates = []
        match_reason = "No valid IS found"

        # --- Step 1: try IS list in priority order (main → option 1 → option 2 → ...) ---
        for i, cand in enumerate(is_candidates):
            match_rows = is_df[
                (is_df["Annotation"].str.strip() == cand) &
                (is_df["Polarity_norm"] == row["Polarity_norm"])
            ]
            if match_rows.empty:
                continue

            # Skip IS if missing or zero values
            is_vals = match_rows[sample_cols].astype(float).values.flatten()
            if np.any(np.isnan(is_vals)) or np.any(is_vals == 0):
                continue

            # Use existing RSD QCs column if available
            if "RSD QCs (%)" in match_rows.columns and pd.notna(match_rows["RSD QCs (%)"].values[0]):
                rsd_qc = float(match_rows["RSD QCs (%)"].values[0])
            else:
                # Compute QC RSD only if precomputed column missing
                rsd_qc = np.inf
                if qc_cols:
                    qc_vals = match_rows[qc_cols].astype(float).values.flatten()
                    qc_vals = qc_vals[~np.isnan(qc_vals)]
                    if len(qc_vals) > 1:
                        rsd_qc = (np.std(qc_vals, ddof=1) / np.mean(qc_vals)) * 100

            # Pass if within RSD limits
            if 2 <= rsd_qc <= 25:
                matched = cand
                match_reason = f"Used priority option {i} ({cand})"
                break
            else:
                # Save for fallback
                if np.isfinite(rsd_qc):
                    fallback_candidates.append((cand, rsd_qc))

        # --- Step 2: fallback to lowest-RSD valid IS of same polarity ---
        if pd.isna(matched) and fallback_candidates:
            matched, best_rsd = min(fallback_candidates, key=lambda x: x[1])
            match_reason = f"Fallback to lowest RSD IS ({matched}, RSD={best_rsd:.2f}%)"

        # --- Step 3: no valid IS found ---
        if pd.isna(matched):
            match_reason = "No valid IS passed filters"

        matched_is.append(matched)
        match_reasons.append(match_reason)


    features_df["Matched IS"] = matched_is

    # --- Apply normalization ---
    norm_df = features_df.copy()
    for idx, row in features_df.iterrows():
        is_name = row["Matched IS"]
        pol = row["Polarity_norm"]

        if pd.isna(is_name):
            continue

        # Get IS row in internal standards file
        match_is = is_df[(is_df["Annotation"].str.strip() == is_name) & (is_df["Polarity_norm"] == pol)]
        if match_is.empty:
            continue

        is_intensities = match_is[sample_cols].astype(float).values.flatten()
        # Replace zeros with NaN to avoid division errors
        is_intensities[is_intensities == 0] = np.nan

        # Normalize each sample
        for col in sample_cols:
            feature_val = row[col]
            is_val = match_is[col].values[0] if col in match_is.columns else np.nan
            if pd.notna(feature_val) and pd.notna(is_val) and is_val != 0:
                norm_df.loc[idx, col] = feature_val / is_val
            else:
                norm_df.loc[idx, col] = np.nan

    norm_df["Matched IS"] = matched_is
    norm_df["Matched IS Reason"] = match_reasons

    # --- Move the columns right before sample columns (for clarity) ---
    first_cols = list(norm_df.columns)
    for col in ["Matched IS", "Matched IS Reason"]:
        if col in first_cols:
            first_cols.remove(col)
    # Insert before first sample column
    insert_pos = first_cols.index(sample_cols[0]) if sample_cols[0] in first_cols else len(first_cols)
    first_cols[insert_pos:insert_pos] = ["Matched IS", "Matched IS Reason", "Polarity_norm"]
    norm_df = norm_df[first_cols]

    # -----------------------------------------------------------------
    # Recalculate RSDs after normalization using sample_groups.csv
    # -----------------------------------------------------------------
    print("[STEP] Recalculating RSDs after normalization...", flush=True)
    group_map = {
        g.strip(): [s for s in group_df.loc[group_df["Group"] == g, "Sample"].tolist()]
        for g in group_df["Group"].dropna().unique()
    }
    qc_groups = [g for g in group_map if g.upper() == "QC"]
    qc_samples = [s for g in qc_groups for s in group_map[g]]
    non_qc_groups = [g for g in group_map if g.upper() != "QC"]

    qc_cols = [c for c in sample_cols if any(s == c or s in c for s in qc_samples)]
    non_qc_cols = [c for c in sample_cols if c not in qc_cols]

    rsd_qc_vals, rsd_sample_vals = [], []
    for _, row in norm_df.iterrows():
        qc_vals = pd.to_numeric(row[qc_cols], errors="coerce").replace(0, np.nan).dropna()
        rsd_qc = (qc_vals.std(ddof=1) / qc_vals.mean() * 100) if len(qc_vals) > 1 else np.nan
        sample_vals = pd.to_numeric(row[non_qc_cols], errors="coerce").replace(0, np.nan).dropna()
        rsd_samples = (sample_vals.std(ddof=1) / sample_vals.mean() * 100) if len(sample_vals) > 1 else np.nan
        rsd_qc_vals.append(rsd_qc)
        rsd_sample_vals.append(rsd_samples)
    norm_df["RSD QCs (%)"] = rsd_qc_vals
    norm_df["RSD Samples (%)"] = rsd_sample_vals

    for g in non_qc_groups:
        cols = [c for c in sample_cols if any(s == c or s in c for s in group_map[g])]
        rsd_vals = []
        for _, row in norm_df.iterrows():
            vals = pd.to_numeric(row[cols], errors="coerce").replace(0, np.nan).dropna()
            rsd = (vals.std(ddof=1) / vals.mean() * 100) if len(vals) > 1 else np.nan
            rsd_vals.append(rsd)
        norm_df[f"RSD_{g} [%]"] = rsd_vals

    # -----------------------------------------------------------------
    # Reorder columns before saving
    # -----------------------------------------------------------------
    print("[STEP] Reordering columns before saving...", flush=True)
    core_cols = [
        "UniqueID", "RT (min)", "m/z", "Neutral mass", "Adducts", "Polarity", "Internal Standard",
        "RSD QCs (%)", "RSD Samples (%)", "RSD_12x [%]", "RSD_15x [%]", "RSD_5x [%]", "RSD_8x [%]",
        "RSD_QC [%]", "MS/MS available?", "Annotation", "Annotation Type",
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
    ordered_cols = [c for c in core_cols if c in norm_df.columns] + sample_cols
    norm_df = norm_df[[c for c in ordered_cols if c in norm_df.columns]]
    
    # -----------------------------------------------------------------
    # Separate unknown (unannotated) features before saving
    # -----------------------------------------------------------------
    if "Annotation" in norm_df.columns:
        # Identify empty or NaN annotations
        unknown_mask = norm_df["Annotation"].astype(str).str.strip().isin(
            ["", "nan", "NaN", "N/A", "Unassigned", "No match", "None", "_", "Unknown"])
        unknown_df = norm_df[unknown_mask].copy()
        annotated_df = norm_df[~unknown_mask].copy()

        unknown_path = output_folder / "debug" / "6-Final_unknowns.csv"
        annotated_path = output_folder / "debug" / "5-Final_annotated_results_normalized.csv"

        unknown_df.to_csv(unknown_path, index=False, encoding="utf-8-sig")
        annotated_df.to_csv(annotated_path, index=False, encoding="utf-8-sig")

        print(f"[INFO] Separated {len(unknown_df)} unannotated features = {unknown_path}", flush=True)
        print(f"[INFO] Saved {len(annotated_df)} annotated features = {annotated_path}", flush=True)

        out_path = annotated_path
        norm_df = annotated_df
    else:
        print("[WARNING] 'Annotation' column missing — skipping unknown separation.", flush=True)

    # -----------------------------------------------------------------
    # Evaluate normalization performance automatically
    # -----------------------------------------------------------------
    try:
        print("\n[STEP] Evaluating normalization performance...")
        evaluate_normalization_performance(
            features_csv,            # before normalization (imputed_filtered)
            out_path,                # after normalization
            output_folder / "sample_groups.csv",
            output_folder
        )
        print("[DONE] Normalization performance evaluation complete.\n")
    except Exception as e:
        print(f"[WARNING] Normalization evaluation skipped due to error: {e}", flush=True)

    print(f"Normalization complete. Saved normalized results to: {out_path}")
    return out_path



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize lipidomics data by class-matched internal standards.")
    parser.add_argument("--features", required=True, help="Path to Final_annotated_results_imputed_filtered.csv")
    parser.add_argument("--isfile", required=True, help="Path to Internal_standards.csv")
    parser.add_argument("--classmap", required=True, help="Path to Class_to_internal_standards.csv")
    parser.add_argument("--out", default="results", help="Output folder")
    args = parser.parse_args()

    normalize_by_internal_standards(args.features, args.isfile, args.classmap, args.out)
    
