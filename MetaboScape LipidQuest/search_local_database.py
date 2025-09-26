import os, re
import numpy as np
import pandas as pd
from pathlib import Path
from lipid_utils import (
    parse_fatty_acyls, classify_chain_type, classify_pufa,
    extract_modifications, count_modifications, is_oxidized
)
from load_file import load_headgroup_to_class

# Adduct definitions
adducts = {
    "Pos": {
        "[M+H]+": -1.007825, "[M+NH4]+": -18.034374, "[M+Na]+": -22.989769,
        "[2M+H]+": -1.007825, "[2M+NH4]+": -18.034374, "[2M+Na]+": 22.989769,
        "[M-2H]2+": 2.014552, "[M+2NH4]2+": -36.067646
    },
    "Neg": {
        "[M-H]-": 1.006728, "[M+HCOO]-": -44.998752, "[2M-H]-": 1.006728,
        "[2M+HCOO]-": -44.998752, "[M-2H]2-": 2.014552, "[M+2HCOO]2-": -89.996402
    }
}


def ppm_difference(ref_mass, test_masses):
    return np.abs((test_masses - ref_mass) / ref_mass) * 1e6


def search_local_database(file_path, output_folder, mz_tolerance_ppm=3,
                          mz_tolerance_Da=0.003, stop_flag=None):
    """
    Local DB search:
    - Keep all sanitized columns
    - Only search rows with no Annotation
    - Duplicate rows if multiple matches exist
    """
    file_path = Path(file_path)
    df_input = pd.read_csv(file_path)

    # Pick annotation column
    annotation_col = None
    for col in ["Annotation", "Name"]:
        if col in df_input.columns:
            annotation_col = col
            break
    if annotation_col is None:
        raise ValueError("No 'Annotation' or 'Name' column found in input file.")

    # Filter rows with missing annotations
    search_df = df_input[df_input[annotation_col].isna() |
                         (df_input[annotation_col].astype(str).str.strip() == "")]

    # Load local DB
    program_folder = Path(__file__).resolve().parent
    db_path = program_folder / "Appendix" / "PersonalizedSearchLibrary.xlsx"
    if not db_path.exists():
        raise FileNotFoundError(f"Local database not found at {db_path}")

    df_db = pd.read_excel(db_path)
    df_db.columns = df_db.columns.str.strip()

    if "Neutral mass" not in df_db.columns:
        raise ValueError("Expected column 'Neutral mass' not found in database file.")

    df_db["Neutral mass"] = pd.to_numeric(df_db["Neutral mass"], errors="coerce")
    exact_masses = df_db["Neutral mass"].values

    results = []
                
    # Load mapping from Appendix
    program_folder = Path(__file__).resolve().parent
    mapping_path = program_folder / "Appendix" / "Headgroup_to_class.csv"
    headgroup_map = load_headgroup_to_class(mapping_path)
                
    # Iterate over search candidates
    for idx, row in search_df.iterrows():
        if stop_flag and stop_flag():
            print("Search interrupted by user.", flush=True)
            break
        mz = row.get("m/z") or row.get("m/z meas.") or None
        polarity = row.get("Polarity", "")
        if pd.isna(mz) or polarity not in adducts:
            continue
        mz = float(mz)

        for adduct, shift in adducts[polarity].items():
            if stop_flag and stop_flag():
                break
            theoretical_mz = (
                (mz + shift) / 2 if "2M" in adduct else
                ((mz + shift) * 2 if "]2+" in adduct or "]2-" in adduct else mz + shift)
            )

            mass_differences = np.abs(exact_masses - theoretical_mz)
            ppm_differences = ppm_difference(theoretical_mz, exact_masses)

            match_indices = np.where(
                (ppm_differences <= mz_tolerance_ppm) &
                (mass_differences <= mz_tolerance_Da)
            )[0]

            for i in match_indices:
                match = df_db.iloc[i]

                # Duplicate full sanitized row
                new_row = row.copy()

                # Fill annotation fields
                new_row["Matched Mass (MS matches)"] = match["Neutral mass"]
                new_row["Matched adduct (MS matches)"] = adduct
                new_row["LIPIDMAPS ID (MS matches)"] = match.get("LIPIDMAPS ID", "")
                new_row["Annotation"] = str(match.get("Annotation", ""))
                new_row["Annotation Source"] = "Lipid Maps"
                
                # Preserve "IS", otherwise add "MS match"
                if new_row.get("Annotation Type") == "IS" or new_row.get("Annotation Type") == "MS/MS match":
                    pass  # keep IS
                else:
                    new_row["Annotation Type"] = "MS match"

                # Δm/z
                delta_da = mz - theoretical_mz
                new_row["Δm/z (mDa)"] = delta_da * 1000
                new_row["Δm/z (ppm)"] = (delta_da / theoretical_mz) * 1e6 if theoretical_mz else ""

                # Headgroup + structural parsing
                headgroup = str(new_row["Annotation"]).split(" ")[0]
                new_row["Headgroup"] = headgroup

                # Add Lipid Class information. Preserve "IS" and "MS/MS match"
                if new_row.get("Annotation Type") in ("IS", "MS/MS match"):
                    # Do nothing, keep existing Lipid Class if present
                    pass
                else:
                    # Map Headgroup → Lipid Class
                    headgroup = str(new_row.get("Headgroup", "")).strip()
                    if headgroup in headgroup_map:
                        new_row["Lipid Class"] = headgroup_map[headgroup]
                    else:
                        new_row["Lipid Class"] = "Other"

                    # If Headgroup is empty, leave Lipid Class blank
                    if headgroup == "":
                        new_row["Lipid Class"] = ""
                    
                fa_info = parse_fatty_acyls(new_row["Annotation"])
                for j in range(4):
                    new_row[f"Number of carbons in fatty acyl {j+1}"] = fa_info[j][0] if len(fa_info) > j else ""
                    new_row[f"Double bonds in fatty acyl {j+1}"] = fa_info[j][1] if len(fa_info) > j else ""

                if fa_info:
                    total_c = sum(c for c, _ in fa_info)
                    total_dbe = sum(d for _, d in fa_info)
                else:
                    total_c, total_dbe = "", ""

                new_row["Number of carbons in fatty acyls"] = total_c
                new_row["Double bond equivalents"] = total_dbe
                new_row["Chain type"] = classify_chain_type(
                    total_c,
                    fa_info[0][0] if fa_info else "",
                    fa_info[1][0] if len(fa_info) > 1 else ""
                )
                new_row["PUFA?"] = classify_pufa(total_dbe)

                mods = extract_modifications(new_row["Annotation"])
                new_row["Modifications"] = mods
                new_row["# of modifications"] = count_modifications(mods)
                ratio = (total_c / total_dbe) if total_c and total_dbe not in ("", 0) else ""
                new_row["Carbons / double bond equivalent ratio"] = ratio
                new_row["Oxidized?"] = is_oxidized(mods, new_row.get("Lipid Class", ""))

                results.append(new_row)

    # Merge results back with non-searched rows
    non_searched = df_input.drop(search_df.index)
    if results:
        df_results = pd.DataFrame(results)
        df_input = pd.concat([non_searched, df_results], ignore_index=True)
    else:
        df_input = non_searched

            # Merge results back with non-searched rows
    non_searched = df_input.drop(search_df.index)
    if results:
        df_results = pd.DataFrame(results)
        df_output = pd.concat([non_searched, df_results], ignore_index=True)
    else:
        df_output = non_searched

    # --- Preserve all baseline columns from sanitized file ---
    baseline_cols = list(df_input.columns)  # all original sanitized cols
    for col in baseline_cols:
        if col not in df_output.columns:
            df_output[col] = ""

    # --- Reorder ---
    preferred_order = [
        "UniqueID", "RT [min]", "m/z meas.", "Neutral Mass", "Adducts", "Polarity",
        "QC RSD [%]", "Samples RSD [%]", 
        "Annotation", "Headgroup", "Lipid Class", "Molecular Formula",
        "Δm/z (mDa)", "Δm/z (ppm)", "Annotation tier",
        "Annotation Type", "Annotation Source", "Metaboscape Annotation Status", "MS/MS available?", 
        "MS/MS score", "mSigma", 
        "LIPIDMAPS ID (MS matches)", "Matched adduct (MS matches)", "Matched Mass (MS matches)", 
        "Number of carbons in fatty acyls", 
        "Number of carbons in fatty acyl 1", "Number of carbons in fatty acyl 2", "Number of carbons in fatty acyl 3", "Number of carbons in fatty acyl 4", 
        "Double bonds in fatty acyl 1", "Double bonds in fatty acyl 2", "Double bonds in fatty acyl 3", "Double bonds in fatty acyl 4", 
        "Chain type", "PUFA?",
        "Plasmenyl?", 
        "Modifications", "# of modifications", "Oxidized?", "Carbons / double bond equivalent ratio",
    ]

    # Final order = preferred order + any baseline/extra columns not in preferred
    ordered_cols = [c for c in preferred_order if c in df_output.columns]
    remaining_cols = [c for c in df_output.columns if c not in ordered_cols]
    df_output = df_output[ordered_cols + remaining_cols]

    # --- Save output ---
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    out_file = output_folder / f"raw_ms_search_results.csv"
    df_output.to_csv(out_file, index=False, encoding="utf-8-sig")

    return out_file, df_output


