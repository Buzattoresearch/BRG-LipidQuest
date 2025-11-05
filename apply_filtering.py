"""
    Lipid filtering pipeline:
    1. Load raw search results
    2. Apply scoring
    3. Filter by biological plausibility (inline here)
    4. Collapse duplicates
    5. Apply minimum score cutoff
    6. Save outputs
    7. Plot results
    """
    
import pandas as pd
import numpy as np
import copy
from pathlib import Path
import importlib
import re
from internal_standard_plots import plot_internal_standards
from handle_adducts import handle_adducts
from generate_plots import plot_results, plot_kendrick_mass_vs_defect

# -----------------------------------------------------------------

#                      HELPER FUNCTIONS

# -----------------------------------------------------------------

def collapse_duplicates(df):
    """
    Enforce true uniqueness of UniqueID.
    For each UniqueID:
      1) If there are annotated rows, keep exactly one annotated row
         selected by highest priority then highest MS Score.
      2) If there are no annotated rows, keep exactly one unassigned row
         selected by highest MS Score.
    Rows without a UniqueID are preserved.
    """
    # Preconditions
    if "UniqueID" not in df.columns or "Annotation" not in df.columns:
        return df.copy()

    # Make a working copy
    dfx = df.copy()

    # Normalize helper columns
    ann = dfx["Annotation"].astype(str).str.strip()
    is_unassigned = ann.eq("") | ann.eq("nan") | dfx["Annotation"].isna()

    # Priority for Annotation Type
    # Lower number means higher priority
    
    def anno_type_priority(s):
        s = str(s).strip().upper()
        if s == "IS":
            return 0
        if s == "MS/MS MATCH":
            return 1
        if s == "MS MATCH":
            return 2
        return 3

    # Build a selection key
    # 1) has_annotation: 1 if annotated, 0 if unassigned
    # 2) type priority as above
    # 3) MS Score descending
    has_ann = (~is_unassigned).astype(int)
    type_pri = dfx.get("Annotation Type", "").apply(anno_type_priority)
    ms_score = pd.to_numeric(dfx.get("MS Score", 0), errors="coerce").fillna(0)

    dfx["_has_ann"] = has_ann
    dfx["_type_pri"] = type_pri
    dfx["_ms_score"] = ms_score

    # Split rows with and without UniqueID
    with_uid = dfx[dfx["UniqueID"].notna()].copy()
    without_uid = dfx[dfx["UniqueID"].isna()].copy()

    # For each UniqueID, pick the single best row
    # Sort so the first row per group is the keeper
    with_uid_sorted = (
        with_uid
        .sort_values(
            by=["_has_ann", "_type_pri", "_ms_score"],
            ascending=[False, True, False]
        )
        .drop_duplicates(subset=["UniqueID"], keep="first")
    )

    # Clean helper cols
    with_uid_sorted = with_uid_sorted.drop(columns=["_has_ann", "_type_pri", "_ms_score"], errors="ignore")
    without_uid = without_uid.drop(columns=["_has_ann", "_type_pri", "_ms_score"], errors="ignore")

    # Recombine
    out = pd.concat([with_uid_sorted, without_uid], ignore_index=True)

    # Guarantee that each UniqueID appears at most once
    # Defensive assertion in case of unexpected input
    # If you prefer silent behavior, comment this out
    # dup_check = out["UniqueID"].dropna()
    # assert not dup_check.duplicated().any(), "Duplicate UniqueID after collapsing"

    return out

def count_unassigned(df):
    if "Annotation" not in df.columns:
        return 0
    ann = df["Annotation"].astype(str).str.strip()
    return int(
        (ann.eq("") | ann.eq("nan") | ann.eq("Unassigned") | df["Annotation"].isna()).sum()
    )

# === Helper to reorder columns consistently ===
def reorder_columns(df):
    sample_cols = [c for c in df.columns if str(c).startswith(("[POS", "[NEG", "P_", "N_"))]
    rsd_cols = [c for c in df.columns if re.match(r"RSD.*\[%\]", c)]
    preferred_order = [
        "UniqueID", "RT (min)", "m/z", "Neutral mass", "Adducts", "Polarity",
        "Internal Standard", "RSD QCs (%)", "RSD Samples (%)"
    ]
    group_rsd_cols = sorted([c for c in rsd_cols if c not in ("RSD QCs (%)", "RSD Samples (%)")])
    metadata_following = [
            "MS/MS available?", "Annotation", "Annotation Type",
            "Metaboscape Annotation Status", "Annotation Source", "Headgroup", "Lipid Class",
            "Δm/z (mDa)", "Δm/z (ppm)", "MS/MS score", "Annotation tier", "mSigma",
            "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]",
            "Molecular Formula", "Plasmenyl?", "Number of carbons in fatty acyls",
            "Double bond equivalents", "Number of carbons in fatty acyl 1", "Double bonds in fatty acyl 1",
            "Number of carbons in fatty acyl 2", "Double bonds in fatty acyl 2",
            "Number of carbons in fatty acyl 3", "Double bonds in fatty acyl 3",
            "Number of carbons in fatty acyl 4", "Double bonds in fatty acyl 4",
            "Chain type", "PUFA?", "Modifications", "# of modifications",
            "Oxidized?", "Carbons / double bond equivalent ratio"
    ]
    intensity_cols = [
            "Average Intensity (all samples, from MetaboScape)",
            "Average Intensity (all samples)", "Minimum Intensity (all samples)",
            "Maximum Intensity (all samples)"
    ]
    flags_cols = ["Relative Stdev", "Flags", "Flag type"]

    new_order = (
            [c for c in preferred_order if c in df.columns] +
            [c for c in group_rsd_cols if c in df.columns] +
            [c for c in metadata_following if c in df.columns] +
            [c for c in intensity_cols if c in df.columns] +
            [c for c in flags_cols if c in df.columns] +
            sample_cols
    )
    new_order = [c for c in new_order if c in df.columns]
    return df[new_order]

# ---------------------------------------------------------------------
# Kendrick Mass Defect (KMD) filtering for lipidomics datasets
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Kendrick mass calculation
# ---------------------------------------------------------------------
def calculate_kendrick_mass_defect(
    mass,
    base_unit_exact=14.01565,
    base_unit_nominal=14.00000
):
    """Return Kendrick mass and Kendrick mass defect (KMD)."""
    kendrick_mass = mass * base_unit_nominal / base_unit_exact
    kendrick_nominal = np.round(kendrick_mass)
    kendrick_defect = kendrick_mass - kendrick_nominal
    return kendrick_mass, kendrick_defect

def apply_kendrick_filter(
    df,
    mass_column="Neutral mass",
    subclass_column="Lipid Class",
    kmd_deviation=0.06,
    min_class_size=5,
    output_folder=None
):
    """
    Filter features based on Kendrick Mass Defect (KMD) consistency within each lipid class.

    Keeps rows where KMD is within ±kmd_deviation of the class median.
    Classes with ≤ min_class_size entries are left unfiltered.
    """
    print(f'\nApplying Kendrick Mass Defect filtering... \n')

    df = df.copy()
    removed_rows = []
    kept_rows = []

    # Ensure required columns exist
    if mass_column not in df.columns:
        raise ValueError(f"Missing mass column: '{mass_column}'")
    if subclass_column not in df.columns:
        raise ValueError(f"Missing subclass column: '{subclass_column}'")

    # Compute Kendrick Mass Defect for all rows
    df["Kendrick Mass"], df["KMD"] = zip(*df[mass_column].astype(float).map(calculate_kendrick_mass_defect))

    # Compute class medians
    kmd_medians = (
        df.groupby(subclass_column)["KMD"]
        .agg(["count", "median"])
        .reset_index()
    )
    kmd_medians = kmd_medians[kmd_medians["count"] > min_class_size]
    median_map = dict(zip(kmd_medians[subclass_column], kmd_medians["median"]))

    # Apply filtering
    for _, row in df.iterrows():
        subclass = row.get(subclass_column)
        kmd = row.get("KMD", np.nan)

        if subclass in median_map:
            deviation = abs(kmd - median_map[subclass])
            if deviation <= kmd_deviation:
                kept_rows.append(row)
            else:
                r = copy.deepcopy(row)
                r["removed_reason"] = f"KMD deviation {deviation:.4f} > {kmd_deviation}"
                removed_rows.append(r)
        else:
            # Keep small subclasses unchanged
            kept_rows.append(row)

    kept_df = pd.DataFrame(kept_rows)
    removed_df = pd.DataFrame(removed_rows)

    print(f"[INFO] Kendrick filter removed {len(removed_df)} features; kept {len(kept_df)}.")
    print(f"[INFO] Median KMDs calculated for {len(median_map)} classes (min size = {min_class_size}).")

    # Optional output logging
    if output_folder:
        import os
        from pathlib import Path
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        if not removed_df.empty:
            removed_df.to_csv(Path(output_folder) / "debug" / "Removed_by_Kendrick.csv", index=False, encoding="utf-8-sig")
        kept_df.to_csv(Path(output_folder) / "debug" / "Kept_after_Kendrick.csv", index=False, encoding="utf-8-sig")

    return kept_df, removed_df

def run_pipeline(input_csv, output_folder, min_score=70, scoring_module="scoring_mammalians", plausibility_module="plausability_filtering_mammalians"):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(input_csv, low_memory=False)

    # Dynamically import scoring & plausibility logic
    scoring = importlib.import_module(scoring_module)
    plausibility = importlib.import_module(plausibility_module)

    print(f'\nFiltering annotations... \n')
    
    """
    Lipid filtering pipeline with debug printouts of unassigned counts.
    """
    input_path = Path(input_csv)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print('\n----- Starting MS filtering ----- \n')
    # Step 1: Load
    df = pd.read_csv(input_path, low_memory=False)
    print(f'Before filtering and scoring: {len(df)}, unassigned: {count_unassigned(df)}', flush=True)

    # Step 2: Apply scoring
    print(f"[INFO] Applying scoring using {scoring_module}")
    df_scored = scoring.apply_scoring(df, output_folder)
    print(f'After scoring and RT filter: {len(df_scored )}, unassigned: {count_unassigned(df_scored )}', flush=True)

    # ------------------------------------------------------------
    # Step 3: Separate internal standards before filtering
    # ------------------------------------------------------------
    if "Annotation Type" in df_scored.columns:
        is_mask = df_scored["Annotation Type"].astype(str).str.upper().str.strip().eq("IS")
        df_is = df_scored[is_mask].copy()
        df_nonis = df_scored[~is_mask].copy()
        print(f"[INFO] Detected {len(df_is)} internal standards. These will be excluded from filtering steps.", flush=True)
    else:
        df_is = pd.DataFrame()
        df_nonis = df_scored.copy()

    # Step 3: Apply plausibility filter (only to non-IS)
    print(f"[INFO] Applying plausibility filtering using {plausibility_module} (excluding internal standards)")
    df_filtered = plausibility.apply_plausability_filter(df_nonis, output_folder)
    print(f'After plausibility filter: {len(df_filtered)}, unassigned: {count_unassigned(df_filtered)}', flush=True)
    
    # ------------------------------------------------------------
    # Step 4: Apply Kendrick Mass Defect filter
    # ------------------------------------------------------------
    try:
        df_kmd_kept, df_kmd_removed = apply_kendrick_filter(
            df_filtered,
            mass_column="Neutral mass",
            subclass_column="Lipid Class",
            kmd_deviation=0.06,  # decrease to make it more strict
            output_folder=output_folder
        )
        df_filtered = df_kmd_kept
        print(f"[INFO] After Kendrick filter: {len(df_filtered)} kept; {len(df_kmd_removed)} removed.", flush=True)
    except Exception as e:
        print(f"[WARNING] Kendrick filter skipped: {e}", flush=True)


    # Step 5: Collapse duplicates (still only non-IS)
    df_collapsed = collapse_duplicates(df_filtered)
    print(f'After collapse duplicates: {len(df_collapsed)}, unassigned: {count_unassigned(df_collapsed)}', flush=True)

    # Step 6: Apply cutoff to scored rows, keep unassigned (non-IS)
    work = df_collapsed
    if "Annotation" in work.columns:
        ann = work["Annotation"].astype(str).str.strip()
        unassigned = ann.eq("") | ann.eq("nan") | ann.eq("Unassigned") | work["Annotation"].isna()
    else:
        unassigned = pd.Series([True] * len(work), index=work.index)

    df_final_nonis = work[(work["MS Score"] >= min_score) | (unassigned)].reset_index(drop=True)
    print(f'After cutoff (non-IS only): {len(df_final_nonis)}, unassigned: {count_unassigned(df_final_nonis)}', flush=True)

    # Step 7: Recombine IS with processed lipids
    df_final = pd.concat([df_final_nonis, df_is], ignore_index=True)
    print(f"[INFO] Recombined final table with {len(df_final)} total rows (including internal standards).", flush=True)


    # --- Compute RSD QCs (%) and RSD Samples (%) for all features ---
    group_file = Path(output_folder) / "sample_groups.csv"
    if group_file.exists():
        group_df = pd.read_csv(group_file, low_memory=False)
        qc_samples = group_df.loc[group_df["Group"].str.upper().str.strip() == "QC", "Sample"].tolist()

        # Build group → sample mapping
        group_map = {
            g.strip(): [s for s in group_df.loc[group_df["Group"] == g, "Sample"].tolist()]
            for g in group_df["Group"].unique()
        }

        # Identify all sample columns
        sample_cols = [c for c in df_final.columns if c.startswith("[POS") or c.startswith("[NEG")]

        # Match QC columns
        qc_cols = []
        for sample in qc_samples:
            if sample in df_final.columns:
                qc_cols.append(sample)
            else:
                qc_cols.extend([c for c in df_final.columns if sample in c])
        qc_cols = list(set(qc_cols))
        print(f"[INFO] Found {len(qc_cols)} QC columns for RSD calculation.", flush = True)

        # Compute per-row RSDs
        rsd_qc_vals, rsd_sample_vals = [], []
        for _, row in df_final.iterrows():
            # QC RSD
            if qc_cols:
                qc_vals = row[qc_cols].astype(float).replace(0, np.nan).dropna()
                rsd_qc = (qc_vals.std(ddof=1) / qc_vals.mean()) * 100 if len(qc_vals) > 1 else np.nan
            else:
                rsd_qc = np.nan

            # Group RSD (across all non-QC samples)
            non_qc_cols = [c for c in sample_cols if c not in qc_cols]
            vals = row[non_qc_cols].astype(float).replace(0, np.nan).dropna()
            rsd_samples = (vals.std(ddof=1) / vals.mean()) * 100 if len(vals) > 1 else np.nan

            rsd_qc_vals.append(rsd_qc)
            rsd_sample_vals.append(rsd_samples)

        df_final["RSD QCs (%)"] = rsd_qc_vals
        df_final["RSD Samples (%)"] = rsd_sample_vals
        print("[INFO] Added RSD QCs (%) and RSD Samples (%) columns to final results.", flush = True)
    else:
        print("[WARNING] sample_groups.csv not found; RSD QCs and Samples left blank.", flush = True)

    # --- Apply unified column ordering ---
    df_final = reorder_columns(df_final)
    df_scored = reorder_columns(df_scored)

    # Step 6: Save outputs
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    debug_folder = output_path / "debug"
    debug_folder.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_csv)
    scored_name = f"{input_path.stem}_scored.csv"   # gives raw_ms_search_results_scored.csv
    scored_path = debug_folder / scored_name
    final_path = debug_folder / "1-Final_MS_results.csv"

    df_scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    df_final.to_csv(final_path, index=False, encoding="utf-8-sig")
    
    # Step 7: Generate Internal Standards table
    if "Annotation Type" in df_final.columns:
        internal_standards_df = df_final[df_final["Annotation Type"].astype(str).str.upper() == "IS"].copy()
        internal_standards_df = reorder_columns(internal_standards_df)
        if not internal_standards_df.empty:
            internal_standards_path = output_path / "Internal_standards.csv"
            internal_standards_df.to_csv(internal_standards_path, index=False, encoding="utf-8-sig")
            print(f"Internal standards table saved to: {internal_standards_path}", flush = True)
        else:
            print("No internal standards detected in 1-Final_MS_results.", flush = True)
            
        # --- Generate internal standard plots ---
        try:
            plot_internal_standards(internal_standards_csv=internal_standards_path, output_folder=output_path)
            print(f"\n ----- Internal standard plots saved to ({output_folder}) ----- \n", flush = True)
        except Exception as e:
            print(f"\n\n ======= Warning: could not generate internal standard plots ({e}) ========\n\n", flush = True)
    
    else:
        print("Warning: 'Annotation Type' column not found; skipping internal standards export.", flush = True)
        
    # ----------------------------------------------
    #                HANDLE ADDUCTS
    # ----------------------------------------------
    
    kept_path, removed_path, summary_path = handle_adducts(input_csv=final_path, output_folder=output_folder, rt_tolerance_seconds=6)

    # ----------------------------------------------
    #      PLOT RESULTS (from generate_plots.py)
    # ----------------------------------------------
        
    print("[INFO] Plotting annotation results.", flush = True)
    try:
        plot_results(input_csv = kept_path, output_folder=output_folder)
    except:
        print("\n\n ======= Plot results failed. ========\n\n", flush = True)
    # try:
    plot_kendrick_mass_vs_defect(input_csv = kept_path, results_folder = output_folder)
    # except:
    #     print("Plot Kendrick Mass Defect failed.", flush = True)

    return scored_path, kept_path
