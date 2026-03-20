'''
Sample loading and initial cleanup

This step loads a MetaboScape results file, assigns a unique identifier to every feature, and removes unused export columns. 
It standardizes lipid names by applying targeted corrections and removing positional tags. 
The software extracts the headgroup from each annotation, maps it to a defined lipid class using an internal reference file (Appendix folder), and explicitly tags plasmalogens. 
It assigns polarity based on adduct information and verifies that the file contains only one ionization mode. 
Each feature is labeled with an annotation type and confidence tier based on MS/MS score and source metadata. 
Fatty acyl chains are parsed from the lipid name to compute total carbons, double bonds, chain parity, PUFA status, modification counts, and oxidation flags.

The software then calculates per-feature intensity statistics and relative standard deviations across all samples and within experimental groups defined in the sample mapping file. 
It generates quick quality control summaries and RSD distribution plots. 
Automated data cleansing is applied to remove known contaminants, flat baseline or saturated signals, duplicated low-intensity masses, highly unstable QC features with rescue rules, 
and features that fail minimum detection thresholds within groups. 
Finally, it outputs standardized sanitized and cleaned datasets, logs removed features with reasons, and saves diagnostic plots summarizing intensity distributions and sample-level 
signal balance before and after filtering.
'''

import re
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from lipid_utils import (
    parse_fatty_acyls, classify_chain_type, classify_pufa,
    extract_modifications, count_modifications, is_oxidized)
from data_cleansing import apply_data_cleansing

def read_metaboscape_table(path, sheet_name=None, index_col=None):
    path = Path(path)
    xls = pd.ExcelFile(path)
    s = sheet_name if sheet_name is not None else xls.sheet_names[0]

    df = pd.read_excel(xls, sheet_name=s, index_col=index_col, header=0)
    df.insert(0, "UniqueID", range(1, len(df) + 1))

    # Drop unwanted columns if they exist
    for col in ["Flags", "Boxplot", "AQ"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def load_headgroup_to_class(mapping_path):
    """Load the Headgroup_to_class mapping file and return a dict {headgroup: subclass}."""
    print(f'\nLoading the results file...\n')
    df_map = pd.read_csv(mapping_path, encoding="latin1", low_memory=False)
    mapping = {}

    for _, row in df_map.iterrows():
        lipid_class = str(row["Lipid Class"]).strip()
        for col in df_map.columns[1:]:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                mapping[str(val).strip()] = lipid_class
    return mapping


def sanitize_file(
    path,
    output_folder=None,
    mz_tol_ppm=None,
    rsd_thresh=None,
    min_int=None,
    rsd_qc_thresh=None,
    min_detect_in_group=None,
    max_group_rsd_thresh=None
    ):

    path = Path(path)
    df = read_metaboscape_table(path)

    # -------------------------------
    # Name corrections before parsing
    # -------------------------------
    name_corrections = {
        "cholestenone": "ST 27:2;O (Cholestenone)",
        "Cholestenone": "ST 27:2;O (Cholestenone)",
    }

    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str).str.strip()
        df["Name"] = df["Name"].replace(name_corrections)

    # Remove "-SN2" tags from names
        df["Name"] = df["Name"].str.replace("-SN2", "", regex=False).str.strip()
        df["Name"] = df["Name"].str.replace("-SN1", "", regex=False).str.strip()

    # -------------------------------
    # Add Headgroup
    # -------------------------------
    if "Name" in df.columns:
        df["Headgroup"] = df["Name"].fillna("").apply(
            lambda x: str(x).split(" ")[0] if str(x).strip() else ""
        )

    # Remove isotope/deuterium labels
    df["Headgroup"] = df["Headgroup"].str.replace(r"\[D\d+\]", "", regex=True).str.strip()

    # Load mapping from Appendix
    program_folder = Path(__file__).resolve().parent
    mapping_path = program_folder / "Appendix" / "Headgroup_to_class.csv"
    headgroup_map = load_headgroup_to_class(mapping_path)

    # Map Headgroup → Lipid Class
    df["Lipid Class"] = df["Headgroup"].map(headgroup_map)
    
    df.loc[df["Headgroup"].isna(), "Lipid Class"] = ""  # True NaN headgroup → empty
    df.loc[df["Headgroup"].astype(str).str.strip() == "", "Lipid Class"] = "" # Blank headgroup → empty
    df.loc[df["Headgroup"].astype(str).str.strip() == "nan", "Lipid Class"] = "" # nan headgroup → empty
    df.loc[df["Headgroup"].astype(str).str.strip() == "NA", "Lipid Class"] = "NA" # Literal "NA" headgroup → "NA"
    # Only set "Other" when unmapped AND headgroup is not NA/blank
    df.loc[
        df["Lipid Class"].isna()
        & df["Headgroup"].notna()
        & (df["Headgroup"].str.strip() != "")
        & (df["Headgroup"].str.strip() != "NA"),
        "Lipid Class"] = "Other"
    
    df.loc[df["Headgroup"].isna() | (df["Headgroup"].str.strip() == ""), "Lipid Class"] = ""

    # Handle plasmalogens
    mask = df["Name"].str.contains(r"\bO-", na=False)
    df.loc[mask, "Headgroup"] = df.loc[mask, "Headgroup"] + " O-"
    df["Plasmenyl?"] = df["Name"].str.contains(r"\bO-", na=False).map({True: "Yes", False: "No"})

    def assign_annotation_type(name, annotation_status, annotation_source, msms_score):
        # IS has priority
        if isinstance(name, str) and re.search(r"\[D\d+\]", name):
            return "IS"
        # Otherwise, MS/MS match if score > 0
        if pd.notna(msms_score) and float(msms_score) > 0:
            return "MS/MS match"
        elif ("TARGET" in str(annotation_status).upper() or "TARGET" in str(annotation_source).upper()):
            return "Target List match (MS, RT, CCS)"
        return ""

    df["Annotation Type"] = df.apply(
        lambda row: assign_annotation_type(row.get("Name", ""), row.get("Annotations", ""), row.get("Annotation Source", ""), row.get("MS/MS score", "")),
        axis=1
    )
    
    

    # --- Polarity ---
    def detect_polarity(ions):
        if not isinstance(ions, str) or not ions.strip():
            return ""
        first_ion = ions.split(",")[0].strip()
        if "]+" in first_ion or "]2+" in first_ion:
            return "Pos"
        if "]-" in first_ion or "]2-" in first_ion:
            return "Neg"
        return ""
    df["Polarity"] = df["Ions"].apply(detect_polarity)

    # Parse fatty acyls
    fa_info = df["Name"].apply(parse_fatty_acyls)
    for i in range(4):
        df[f"Number of carbons in fatty acyl {i+1}"] = fa_info.apply(lambda x: x[i][0] if len(x) > i else "")
        df[f"Double bonds in fatty acyl {i+1}"] = fa_info.apply(lambda x: x[i][1] if len(x) > i else "")

    df["Number of carbons in fatty acyls"] = fa_info.apply(lambda x: sum(c for c, _ in x) if x else "")
    df["Double bond equivalents"] = fa_info.apply(lambda x: sum(d for _, d in x) if x else "")

    # Derived annotations
    def _classify_chain_type(row):
        vals = [
            row.get("Number of carbons in fatty acyls"),
            row.get("Number of carbons in first fatty acyl"),
            row.get("Number of carbons in second fatty acyl"),
        ]
        vals = [v for v in vals if pd.notna(v) and v != ""]
        if not vals:
            return ""
        if any(int(v) % 2 != 0 for v in vals):
            return "odd"
        return "even"

    df["Chain type"] = df.apply(_classify_chain_type, axis=1)
    
    df["PUFA?"] = df["Double bond equivalents"].apply(classify_pufa)
    df["Modifications"] = df["Name"].apply(extract_modifications)
    df["# of modifications"] = df["Modifications"].apply(count_modifications)

    # Ratio safely
    df["Carbons / double bond equivalent ratio"] = (
        pd.to_numeric(df["Number of carbons in fatty acyls"], errors="coerce")
        / pd.to_numeric(df["Double bond equivalents"], errors="coerce")
    )
    df["Carbons / double bond equivalent ratio"] = (
        df["Carbons / double bond equivalent ratio"]
        .replace([float("inf"), -float("inf")], pd.NA)
        .fillna("")
    )

    # Oxidized?
    df["Oxidized?"] = df.apply(
        lambda row: is_oxidized(row.get("Modifications", ""), row.get("Lipid Class", "")), axis=1
    )

    # --- Annotation tier ---
    def assign_tier(msms_score, annotations):

        ann = "" if pd.isna(annotations) else str(annotations)

        # robust numeric parsing
        score = pd.to_numeric(msms_score, errors="coerce")

        # 1) Target list always high confidence
        if "Target List" in ann:
            return "High confidence"
        if "TargetList" in ann:
            return "High confidence"
        if "Target list" in ann:
            return "High confidence"

        # 2) MS/MS score based
        if pd.notna(score) and score > 400:
            return "High confidence"
        if pd.notna(score) and 1 < score <= 400:
            return "Low confidence"

        # 3) No MS/MS score, rely on source string
        if ("Lipid Species" in ann) or ("Spectral Library" in ann) or ("SpectralLibrary" in ann):
            return "Low confidence"
        
        return ""

    df["Annotation tier"] = df.apply(lambda row: assign_tier(
        row.get("MS/MS score", ""),
        row.get("Annotations", "")
    ), axis=1)

    # --- Compute intensity-based metrics ---
    sample_cols = [c for c in df.columns if str(c).startswith("P_") or str(c).startswith("N_")]

    # Determine polarity from sample naming convention
    has_pos = any(c.startswith("P_") for c in sample_cols)
    has_neg = any(c.startswith("N_") for c in sample_cols)

    if has_pos and has_neg:
        # Fatal error: mixed polarities in the same file
        print("**** Mixed polarities detected in sample columns (P_ and N_) ****", flush=True)
        raise ValueError(
            "Input file contains mixed polarities (P_ and N_). "
            "Each file must contain only POS (P_) or only NEG (N_) samples."
        )

    elif has_pos:
        detected_pol = "POS"

    elif has_neg:
        detected_pol = "NEG"

    else:
        detected_pol = ""

    # Tag for POS/NEG-specific outputs
    if detected_pol == "POS":
        pol_tag = "Pos_"
        prefix_for_cleansing = "P_"
    elif detected_pol == "NEG":
        pol_tag = "Neg_"
        prefix_for_cleansing = "N_"
    else:
        pol_tag = ""
        prefix_for_cleansing = None

    df["Detected Polarity (from samples)"] = detected_pol
    print(f"\n\n ----------- Loading data for detected polarity: {detected_pol} ---------------- \n\n", flush=True)

    if sample_cols:
        numeric_intensities = df[sample_cols].apply(pd.to_numeric, errors="coerce")

        # Average, min, max across all samples
        df["Average Intensity (all samples)"] = numeric_intensities.mean(axis=1)
        df["Minimum Intensity (all samples)"] = numeric_intensities.min(axis=1)
        df["Maximum Intensity (all samples)"] = numeric_intensities.max(axis=1)

        # --- Load group assignments ---
        group_file = Path(output_folder.parent) / "sample_groups.csv"
        print(f'Loading sample_groups.csv from: {group_file}', flush = True)
        if group_file.exists():
            try:
                group_df = pd.read_csv(group_file, low_memory=False)
                group_map = dict(zip(group_df["Sample"], group_df["Group"]))
            except Exception as e:
                print(f"[WARNING] Failed to load sample_groups.csv: {e}", flush = True)
                group_map = {}
        else:
            group_map = {}

        # Split QC vs. sample columns based on group assignment
        qc_cols = [c for c in sample_cols if group_map.get(c, "").strip().lower() == "qc"]
        sample_only_cols = [c for c in sample_cols if c not in qc_cols]
        print(f"\nDETECTED SAMPLE MAPPING: ", flush = True)
        for s in sample_cols:
            print(f"{s}: group = '{group_map.get(s, None)}'", flush = True)


        # Define RSD function
        def rsd(series):
            vals = pd.to_numeric(series, errors="coerce").dropna()
            return (vals.std() / vals.mean() * 100) if len(vals) > 1 and vals.mean() != 0 else np.nan

        # Compute QC and Sample RSDs
        if qc_cols:
            df["QC RSD [%]"] = df[qc_cols].apply(rsd, axis=1)
        else:
            df["QC RSD [%]"] = np.nan

        if sample_only_cols:
            df["Sample RSD [%]"] = df[sample_only_cols].apply(rsd, axis=1)
        else:
            df["Sample RSD [%]"] = np.nan

        # --- Compute and print per-group RSD summary ---
        if group_map:
            group_rsd_summary = {}
            for group_name in sorted(set(group_map.values())):
                group_cols = [s for s, g in group_map.items() if g == group_name and s in sample_cols]
                if not group_cols:
                    continue
                df[f"RSD_{group_name} [%]"] = df[group_cols].apply(rsd, axis=1)
                median_val = np.nanmedian(df[f"RSD_{group_name} [%]"])
                group_rsd_summary[group_name] = median_val

            print("\n=== RSD Summary by Group (median RSDs for all features) ===", flush = True)
            for g, val in group_rsd_summary.items():
                print(f"  {g:15s}: {val:6.2f}%")
            print("============================\n", flush = True)

        # Summary print for QC vs samples
        try:
            qc_median = np.nanmedian(df["QC RSD [%]"]) if "QC RSD [%]" in df else np.nan
            sample_median = np.nanmedian(df["Sample RSD [%]"]) if "Sample RSD [%]" in df else np.nan
            print(f"Median QC RSD = {qc_median:.2f}%, Median Sample RSD = {sample_median:.2f}%", flush = True)
        except:
            print(f"Error when calculating the median RSDs for QCs and samples.", flush = True)

        # --- Plot RSD distributions per group ---
        try:
            
            # Collect data for plotting
            rsd_data = []
            for g in group_rsd_summary:
                colname = f"RSD_{g} [%]"
                if colname in df.columns:
                    vals = pd.to_numeric(df[colname], errors="coerce").dropna()
                    rsd_data.append((g, vals))

            if rsd_data:
                fig, ax = plt.subplots(figsize=(8, 5))
                labels = [g for g, _ in rsd_data]
                data = [v for _, v in rsd_data]

                # Violin plots for RSD distributions per group
                parts = ax.violinplot(data, showmeans=True, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor('#87CEFA')
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
                ax.set_ylabel("Feature RSD (%)")
                ax.set_title("Within-Group RSD Distribution")

                # Add median values on top
                for i, (g, vals) in enumerate(rsd_data, start=1):
                    med = np.nanmedian(vals)
                    ax.text(i, med + 0.5, f"{med:.1f}%", ha='center', va='bottom', fontsize=8, color='black')

                plt.tight_layout()
                
                # Save figure
                fig_path = Path(output_folder) / "debug" / f"{pol_tag}RSD_by_group.png"
                # Ensure debug folder exists before saving
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(fig_path, dpi=100)
                plt.close(fig)

                print(f"Saved RSD summary plot → {fig_path}")

                # Also save median values as CSV
                summary_df = pd.DataFrame(
                    [(g, np.nanmedian(v)) for g, v in rsd_data],
                    columns=["Group", "Median RSD (%)"]
                )
                summary_csv_path = Path(output_folder) / "debug" / f"{pol_tag}RSD_summary_by_group.csv"
                summary_df.to_csv(summary_csv_path, index=False)
                print(f"Saved RSD summary table → {summary_csv_path}")

        except Exception as e:
            print(f"[WARNING] Failed to generate RSD summary plot: {e}")


    else:
        print("No P_ or N_ sample columns found. Skipping intensity metrics.")

        
    # Apply data cleansing 
    # Default QC RSD threshold for the rough filter
    rsd_qc_thresh_rough = 75.0 if rsd_qc_thresh is None else float(rsd_qc_thresh)
    df_clean, removed_df, baseline_df = apply_data_cleansing(
        df,
        output_folder,
        contaminant_file="Appendix/Contaminants.csv",
        ppm_tolerance=mz_tol_ppm,
        min_int=min_int,
        prefix=prefix_for_cleansing,
        rsd_qc_thresh_rough=rsd_qc_thresh_rough,
        min_detect_in_group=min_detect_in_group
    )

    # --- Reorder columns logically ---
    # Rename
    rename_map = {
        "RT [min]": "RT (min)", "m/z meas.": "m/z", "M meas.": "Neutral mass",
        "Ions": "Adducts", "QC RSD [%]": "RSD QCs (%)", "Samples RSD [%]": "RSD Samples (%)",
        "MS/MS": "MS/MS available?", "Name": "Annotation", "Annotations": "Metaboscape Annotation Status", 
        "Δm/z [mDa]": "Δm/z (mDa)", "Δm/z [ppm]": "Δm/z (ppm)", "_RelStd": "Relative Stdev", "_Flag": "Flags", "_FlagType": "Flag type",
        "_MeanIntensity": "Average Intensity (all samples, from MetaboScape)"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df_clean = df_clean.rename(columns={k: v for k, v in rename_map.items() if k in df_clean.columns})

    # Ensure identical column structure
    missing_in_clean = [c for c in df.columns if c not in df_clean.columns]
    for col in missing_in_clean:
        df_clean[col] = np.nan

    df_clean = df_clean[df.columns]  # identical order and structure

    # Drop old or redundant RSD columns if both exist
    for dup_col in ["Sample RSD (%)", "RSD Sample (%)", "RSD QCs (%)"]:
        duplicates = [c for c in df.columns if c.strip().lower() == dup_col.strip().lower()]
        if len(duplicates) > 1:
            df = df.drop(columns=duplicates[1:])  # keep first occurrence
            df_clean = df_clean.drop(columns=duplicates[1:])  # keep first occurrence

    # --- Reorder columns logically ---
    sample_cols = [c for c in df.columns if str(c).startswith(("P_", "N_"))]
    rsd_cols = [c for c in df.columns if re.match(r"RSD.*\[%\]", c)]
    sample_cols_clean = [c for c in df_clean.columns if str(c).startswith(("P_", "N_"))]
    rsd_cols_clean = [c for c in df_clean.columns if re.match(r"RSD.*\[%\]", c)]

    # Always keep core metadata first
    preferred_order = [
        "UniqueID", "RT (min)", "m/z", "Neutral mass", "Adducts", "Polarity",
        "Internal Standard", "RSD QCs (%)", "RSD Samples (%)"
    ]

    # Place per-group RSD columns right after the two main RSDs
    group_rsd_cols = sorted([c for c in rsd_cols if c not in ("RSD QCs (%)", "RSD Samples (%)")])
    group_rsd_cols_clean = sorted([c for c in rsd_cols_clean if c not in ("RSD QCs (%)", "RSD Samples (%)")])

    # Continue with biological / annotation columns
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

    # Assemble final order
    new_order = (
        [c for c in preferred_order if c in df_clean.columns] +
        [c for c in group_rsd_cols_clean if c in df_clean.columns] +
        [c for c in metadata_following if c in df_clean.columns] +
        [c for c in intensity_cols if c in df_clean.columns] +
        [c for c in flags_cols if c in df_clean.columns] +
        sample_cols_clean
    )

    # Apply the final column order
    df = df[[c for c in new_order if c in df.columns]]
    df_clean = df_clean[[c for c in new_order if c in df_clean.columns]]

    # --- Save ---

    # --- Build base stem without duplicated polarity tag ---
    base_stem = path.stem
    upper_stem = base_stem.upper()
    if upper_stem.startswith("POS_") or upper_stem.startswith("NEG_"):
        # Strip the leading polarity tag (Pos_/Neg_) once
        base_stem = base_stem.split("_", 1)[1]

    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        debug_folder = output_folder / "debug"
        debug_folder.mkdir(parents=True, exist_ok=True)

        output_path = debug_folder / f"{pol_tag}{base_stem}_sanitized.csv"
        output_path_clean = debug_folder / f"{pol_tag}{base_stem}_clean.csv"
        output_path_cleansed = debug_folder / f"{pol_tag}{base_stem}_removed_contaminants.csv"
        output_path_baseline = debug_folder / f"{pol_tag}{base_stem}_removed_baseline.csv"

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        df_clean.to_csv(output_path_clean, index=False, encoding="utf-8-sig")
        removed_df.to_csv(output_path_cleansed, index=False, encoding="utf-8-sig")
        baseline_df.to_csv(output_path_baseline, index=False, encoding="utf-8-sig")

    else:
        output_path = path.with_name(f"/debug/{pol_tag}{base_stem}_sanitized.csv")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        output_path_clean = path.with_name(f"/debug/{pol_tag}{base_stem}_clean.csv")
        df_clean.to_csv(output_path_clean, index=False, encoding="utf-8-sig")

    # --- Plot histograms ---   
    plot_dir = output_folder / "debug"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # AFTER — log10 histograms (keep zeros, add tiny pseudocount)
    X = df[sample_cols].apply(pd.to_numeric, errors="coerce").to_numpy().reshape(-1)
    X_clean = df_clean[sample_cols].apply(pd.to_numeric, errors="coerce").to_numpy().reshape(-1)

    epsilon = 1e-10  # avoids -inf in log10

    plt.figure(figsize=(7,4))
    plt.hist(np.log10(X + epsilon), bins=100, alpha=0.5, label="original dataset")
    plt.hist(np.log10(X_clean + epsilon), bins=100, alpha=0.5, label="after data cleansing")
    plt.legend()
    plt.xlabel("log10(intensity)")
    plt.ylabel("count")
    plt.title("Log-intensity distribution before vs after data cleansing")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{pol_tag}histogram_loaded_dataset.png", dpi=100)
    plt.close()


    # --- Plot summed intensities per sample (before vs after cleansing) ---
    try:
        # Sum intensities across features for each sample column
        summed_before = df[sample_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)
        summed_after  = df_clean[sample_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)

        # Build side-by-side bar plot
        x = np.arange(len(sample_cols))
        width = 0.4

        plt.figure(figsize=(10, 5))
        plt.bar(x - width/2, summed_before, width, label="original dataset", alpha=0.7)
        plt.bar(x + width/2, summed_after,  width, label="data cleansing", alpha=0.7)

        plt.xticks(x, sample_cols, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Summed intensity\n(all detected features)")
        # x-axis labels
        plt.title("Summed intensities per sample (before vs after data cleansing)")
        plt.legend()
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.28)
        plt.savefig(plot_dir / f"{pol_tag}summed_intensities_before_after.png", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Saved summed intensity plot → {plot_dir / f'{pol_tag}summed_intensities_before_after_data_cleansing.png'}")

    except Exception as e:
        print(f"[WARNING] Failed to generate summed intensity plot: {e}")

    # ------------------------------------------------------------
    # BOXPLOT: Summed intensities per group
    #  - one color per group
    #  - QC last
    #  - show all points ON TOP of boxes
    #  - label ONLY outliers
    # ------------------------------------------------------------
    try:
        # Group → values mapping
        group_sums = {}

        non_qc_groups = sorted({group_map.get(s, "ungrouped") for s in sample_only_cols})

        for g in non_qc_groups:
            members = sorted([s for s in sample_only_cols if group_map.get(s, "ungrouped") == g])
            if members:
                group_sums[g] = [summed_before[m] for m in members]

        if qc_cols:
            group_sums["QC"] = [summed_before[s] for s in sorted(qc_cols)]

        labels = list(group_sums.keys())
        data = [group_sums[g] for g in labels]

        # Distinct colors per group
        cmap = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        group_colors = dict(zip(labels, cmap))

        fig, ax = plt.subplots(figsize=(8, 5))

        # -------------------------
        # Draw boxplot
        # -------------------------
        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showfliers=False,  
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )

        # Color boxes
        for patch, g in zip(bp['boxes'], labels):
            patch.set_facecolor(group_colors[g])
            patch.set_edgecolor("black")

        # -------------------------
        # Draw ALL points ON TOP OF BOXES
        # -------------------------
        for i, g in enumerate(labels):
            vals = group_sums[g]
            xjit = np.random.normal(i + 1, 0.06, size=len(vals))

            ax.scatter(
                xjit,
                vals,
                s=32,
                color="black",
                alpha=0.70,
                zorder=4  # ensure points are on top
            )

        # -------------------------
        # Label outliers ONLY
        # Tukey rule (1.5 × IQR)
        # -------------------------
        for i, g in enumerate(labels):
            vals = np.array(group_sums[g])
            if len(vals) < 4:
                continue

            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            iqr = q3 - q1
            lower_cut = q1 - 1.5 * iqr
            upper_cut = q3 + 1.5 * iqr

            outlier_indices = np.where((vals < lower_cut) | (vals > upper_cut))[0]

            # Label each outlier
            for idx in outlier_indices:
                # match sample → value
                for s in (sample_only_cols + qc_cols):
                    if (group_map.get(s, "ungrouped") == g or (g == "QC" and s in qc_cols)) \
                    and summed_before[s] == vals[idx]:
                        sample_name = s
                        break

                ax.text(
                    i + 1.10,
                    vals[idx],
                    sample_name,
                    fontsize=8,
                    va="center",
                    color="black",
                    zorder=5
                )

        # -------------------------
        # Final formatting
        # -------------------------
        ax.set_ylabel("Summed intensity")
        ax.set_title(f"Summed intensities per group — {pol_tag.replace('_','')}")
        # Rotate x-axis labels
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        out_path = plot_dir / f"{pol_tag}summed_intensities_group_boxplot.png"
        plt.savefig(out_path, dpi=100)
        plt.close()

        print(f"Saved summed-intensity boxplot → {out_path}")

    except Exception as e:
        print(f"[WARNING] Failed to generate summed-intensity boxplot: {e}")




    # --- Plot summed intensities per sample (group-ordered, QC last, colored) ---
    try:
        # Sum intensities across features
        summed_before = df[sample_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)

        # Ensure we have a group map
        if "group_map" not in locals():
            group_map = {}

        # Build ordered list of sample-only by group:
        #   1. group names sorted alphabetically
        #   2. samples within each group sorted alphabetically
        grouped_samples = []
        unique_groups = sorted({group_map.get(s, "ungrouped") for s in sample_only_cols})

        for g in unique_groups:
            members = sorted([s for s in sample_only_cols if group_map.get(s, "ungrouped") == g])
            grouped_samples.extend(members)

        # Now append QCs at the end (order QCs alphabetically)
        ordered_samples = grouped_samples + sorted(qc_cols)

        # Recompute unique groups including "QC" as its own color group
        full_groups = unique_groups + ["QC"] if qc_cols else unique_groups

        # Color map
        colormap = plt.cm.tab20(np.linspace(0, 1, len(full_groups)))
        group_to_color = dict(zip(full_groups, colormap))

        # Build color vector
        colors = []
        for s in ordered_samples:
            g = group_map.get(s, "ungrouped")
            if s in qc_cols:
                colors.append(group_to_color["QC"])
            else:
                colors.append(group_to_color[g])

        # Summed-intensity values in ordered sample order
        y = [summed_before[s] for s in ordered_samples]

        # ------------------------------------------------------------
        #  ADD BAR PLOTS PER GROUP
        # ------------------------------------------------------------

        plt.figure(figsize=(11, 5))
        plt.bar(range(len(ordered_samples)), y, color=colors)

        plt.xticks(range(len(ordered_samples)), ordered_samples, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Summed intensity\n(all detected features)")
        plt.title(f"Summed intensities for {pol_tag}ion")

        # Legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=group_to_color[g]) for g in full_groups]
        plt.legend(handles, full_groups, title="Groups",
                   bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(plot_dir / f"{pol_tag}summed_intensities_group.png", dpi=100)
        plt.close()

        print(f"Saved grouped summed intensity plot → "
              f"{plot_dir / f'{pol_tag}summed_intensities_group.png'}")

    except Exception as e:
        print(f"[WARNING] Failed to generate grouped summed-intensity plot: {e}")

    return output_path, output_path_clean, df, pol_tag
