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

    df = pd.read_excel(xls, sheet_name=s, index_col=index_col, header=0, low_memory=False)
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

    # Add Headgroup
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
    df["Lipid Class"] = df["Lipid Class"].fillna("Other")
    df.loc[df["Headgroup"].isna() | (df["Headgroup"].str.strip() == ""), "Lipid Class"] = ""

    # Handle plasmalogens
    mask = df["Name"].str.contains(r"\bO-", na=False)
    df.loc[mask, "Headgroup"] = df.loc[mask, "Headgroup"] + " O-"
    df["Plasmenyl?"] = df["Name"].str.contains(r"\bO-", na=False).map({True: "Yes", False: "No"})

    def assign_annotation_type(name, msms_score):
        # IS has priority
        if isinstance(name, str) and re.search(r"\[D\d+\]", name):
            return "IS"
        # Otherwise, MS/MS match if score > 0
        if pd.notna(msms_score) and float(msms_score) > 0:
            return "MS/MS match"
        return ""

    df["Annotation Type"] = df.apply(
        lambda row: assign_annotation_type(row.get("Name", ""), row.get("MS/MS score", "")),
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
    df["Annotation tier"] = df["MS/MS score"].apply(
        lambda x: "High confidence" if pd.notna(x) and x > 400 else ""
    )

    # --- Compute intensity-based metrics ---
    sample_cols = [c for c in df.columns if str(c).startswith("[POS") or str(c).startswith("[NEG") or str(c).startswith("P_") or str(c).startswith("N_")]
    if sample_cols:
        numeric_intensities = df[sample_cols].apply(pd.to_numeric, errors="coerce")

        # Average, min, max across all samples
        df["Average Intensity (all samples)"] = numeric_intensities.mean(axis=1)
        df["Minimum Intensity (all samples)"] = numeric_intensities.min(axis=1)
        df["Maximum Intensity (all samples)"] = numeric_intensities.max(axis=1)

        # --- Load group assignments ---
        group_file = Path(output_folder) / "sample_groups.csv"
        if group_file.exists():
            try:
                group_df = pd.read_csv(group_file, low_memory=False)
                group_map = dict(zip(group_df["Sample"], group_df["Group"]))
            except Exception as e:
                print(f"[WARNING] Failed to load sample_groups.csv: {e}")
                group_map = {}
        else:
            group_map = {}

        # Split QC vs. sample columns based on group assignment
        qc_cols = [c for c in sample_cols if group_map.get(c, "").strip().lower() == "qc"]
        sample_only_cols = [c for c in sample_cols if c not in qc_cols]

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
                
                # Ensure debug folder exists before saving
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                # Save figure
                fig_path = Path(output_folder) / "debug" / "RSD_by_group.png"
                plt.savefig(fig_path, dpi=150)
                plt.close(fig)

                print(f"Saved RSD summary plot → {fig_path}")

                # Also save median values as CSV
                summary_df = pd.DataFrame(
                    [(g, np.nanmedian(v)) for g, v in rsd_data],
                    columns=["Group", "Median RSD (%)"]
                )
                summary_df.to_csv(Path(output_folder) / "debug" / "RSD_summary_by_group.csv", index=False)
                print(f"Saved RSD summary table → {output_folder}/debug/RSD_summary_by_group.csv")

        except Exception as e:
            print(f"[WARNING] Failed to generate RSD summary plot: {e}")


    else:
        print("No [POS]/[NEG] sample columns found. Skipping intensity metrics.")

        
    # Apply data cleansing 
    df_clean, removed_df, baseline_df = apply_data_cleansing(
    df,
    output_folder,
    contaminant_file="Appendix/Contaminants.csv",
    ppm_tolerance=mz_tol_ppm,
    min_int=min_int,
    prefix="[POS",
    rsd_qc_thresh=rsd_qc_thresh,
    min_detect_in_group=min_detect_in_group,
    max_group_rsd_thresh=max_group_rsd_thresh
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
    sample_cols = [c for c in df.columns if str(c).startswith(("[POS", "[NEG", "P_", "N_"))]
    rsd_cols = [c for c in df.columns if re.match(r"RSD.*\[%\]", c)]
    sample_cols_clean = [c for c in df_clean.columns if str(c).startswith(("[POS", "[NEG", "P_", "N_"))]
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
        [c for c in preferred_order if c in df.columns] +
        [c for c in group_rsd_cols if c in df.columns] +
        [c for c in metadata_following if c in df.columns] +
        [c for c in intensity_cols if c in df.columns] +
        [c for c in flags_cols if c in df.columns] +
        sample_cols
    )
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
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        debug_folder = output_folder / "debug"
        debug_folder.mkdir(parents=True, exist_ok=True)

        output_path = debug_folder / f"{path.stem}_sanitized.csv"
        output_path_clean = debug_folder / f"{path.stem}_clean.csv"
        output_path_cleansed = debug_folder / f"{path.stem}_removed_contaminants.csv"
        output_path_baseline = debug_folder / f"{path.stem}_removed_baseline.csv"

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        df_clean.to_csv(output_path_clean, index=False, encoding="utf-8-sig")
        removed_df.to_csv(output_path_cleansed, index=False, encoding="utf-8-sig")
        baseline_df.to_csv(output_path_baseline, index=False, encoding="utf-8-sig")
    else:
        output_path = path.with_name(f"/debug/{path.stem}_sanitized.csv")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        output_path_clean = path.with_name(f"/debug/{path.stem}_clean.csv")
        df_clean.to_csv(output_path_clean, index=False, encoding="utf-8-sig")

    return output_path, output_path_clean, df
