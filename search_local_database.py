import numpy as np
import pandas as pd
from pathlib import Path
from lipid_utils import (
    parse_fatty_acyls, classify_chain_type, classify_pufa,
    extract_modifications, count_modifications, is_oxidized
)
from load_file import load_headgroup_to_class

# --- Exact masses for adduct components (monoisotopic) ---
MASS_H      = 1.0078250
MASS_NH4    = 18.033823
MASS_NA     = 22.989769
MASS_2NAH   = 44.971714
MASS_FORMATE= 44.998201  # HCOO−
OH          = 17.002740

# Adduct models: polarity -> adduct -> (nM, additive_mass, charge)
ADDUCT_MODELS = {
    "Pos": {
        "[M+H]+":        (1, +MASS_H,             +1),
        "[M+NH4]+":      (1, +MASS_NH4,           +1),
        "[M-H2O+H]+":    (1, -OH,                 +1),
        "[M+Na]+":       (1, +MASS_NA,            +1),
        "[M+2Na-H]+":    (1, +MASS_2NAH,          +1), # uncommon but included
        "[2M+H]+":       (2, +MASS_H,             +1),
        "[3M+H]+":       (3, +MASS_H,             +1),
        "[2M+NH4]+":     (2, +MASS_NH4,           +1),
        "[3M+NH4]+":     (3, +MASS_NH4,           +1),
        "[2M+Na]+":      (2, +MASS_NA,            +1),
        "[3M+Na]+":      (2, +MASS_NA,            +1),
        "[2M+2Na-H]+":   (2, +MASS_2NAH,          +1),  # uncommon but included
        "[M+2H]2+":      (1, +2*MASS_H,           +2),  # uncommon but included
        "[M+2NH4]2+":    (1, +2*MASS_NH4,         +2),
    },
    "Neg": {
        "[M-H]-":        (1, -MASS_H,             -1),
        "[M+HCOO]-":     (1, +MASS_FORMATE,       -1),
        "[2M-H]-":       (2, -MASS_H,             -1),
        "[2M+HCOO]-":    (2, +MASS_FORMATE,       -1),
        "[3M-H]-":       (2, -MASS_H,             -1),
        "[3M+HCOO]-":    (2, +MASS_FORMATE,       -1),
        "[M-2H]2-":      (1, -2*MASS_H,           -2),
        "[M+2HCOO]2-":   (1, +2*MASS_FORMATE,     -2),
    }
}

# Correct Δm/z: compare measured m/z to the m/z predicted from the matched neutral mass + adduct
def mz_from_neutral(neutral_mass, adduct):
    # masses (use exactly the same constants everywhere to avoid mDa drift)
    H      = 1.0078250
    NH4    = 18.033823
    Na     = 22.989769
    HCOO   = 44.998752
    OH     = 17.002740
    NaH    = 44.971714
    

    # map adduct to (nM, add_mass, charge_magnitude)
    model = {
        "[M+H]+":        (1,  +H,      1),
        "[M+NH4]+":      (1,  +NH4,    1),
        "[M+Na]+":       (1,  +Na,     1),
        "[M-H2O+H]+":    (1,  -OH,     1),
        "[M+2Na-H]+":    (1,  +NaH,    1),
        "[2M+H]+":       (2,  +H,      1),
        "[2M+NH4]+":     (2,  +NH4,    1),
        "[2M+Na]+":      (2,  +Na,     1),
        "[3M+H]+":       (3,  +H,      1),
        "[3M+NH4]+":     (3,  +NH4,    1),
        "[3M+Na]+":      (3,  +Na,     1),
        "[M-2H]2+":      (1,  -2*H,    2),
        "[M+2NH4]2+":    (1,  +2*NH4,  2),

        "[M-H]-":        (1,  -H,      1),
        "[M+HCOO]-":     (1,  +HCOO,   1),
        "[2M-H]-":       (2,  -H,      1),
        "[2M+HCOO]-":    (2,  +HCOO,   1),
        "[3M-H]-":       (3,  -H,      1),
        "[3M+HCOO]-":    (3,  +HCOO,   1),
        "[M-2H]2-":      (1,  -2*H,    2),
        "[M+2HCOO]2-":   (1,  +2*HCOO, 2),
    }[adduct]

    nM, add_mass, z = model
    return (nM * float(neutral_mass) + add_mass) / z  # z is magnitude for m/z

def theo_mz_from_neutral(neutral_mass_array, adduct):
    """
    Vectorized theoretical m/z from neutral mass using an adduct string present in ADDUCT_MODELS.
    Returns a numpy array.
    """
    # adduct polarity is determined outside, we pass the (nM, add, z) triple directly
    nM, add_mass, z = adduct
    z_abs = abs(z)
    return (nM * neutral_mass_array + add_mass) / z_abs

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
    df_input = pd.read_csv(file_path, low_memory=False)

    print('\nRunning MS search against local database...\n')

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

    df_db = pd.read_excel(db_path, low_memory=False)
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
        if pd.isna(mz) or polarity not in ADDUCT_MODELS:
            continue
        mz = float(mz)

        for adduct_name, model in ADDUCT_MODELS[polarity].items():
            if stop_flag and stop_flag():
                break

            # Compute theoretical m/z for all DB masses with this adduct
            mz_theo_arr = theo_mz_from_neutral(exact_masses, model)

            # Differences relative to the measured m/z
            delta_da_arr = mz - mz_theo_arr
            ppm_arr = np.abs(delta_da_arr / mz_theo_arr) * 1e6

            # Find matches
            match_indices = np.where(
                (np.abs(delta_da_arr) <= mz_tolerance_Da) &
                (ppm_arr <= mz_tolerance_ppm)
            )[0]

            # Now loop through the matches as before
            for i in match_indices:
                match = df_db.iloc[i]

                new_row = row.copy()

                # Fill annotation fields
                new_row["Matched Mass (MS matches)"] = match["Neutral mass"]
                new_row["Matched adduct (MS matches)"] = adduct_name
                new_row["LIPIDMAPS ID (MS matches)"] = match.get("LIPIDMAPS ID", "")
                new_row["Annotation"] = str(match.get("Annotation", ""))
                new_row["Annotation Source"] = "Lipid Maps"

                if new_row.get("Annotation Type") not in ("IS", "MS/MS match"):
                    new_row["Annotation Type"] = "MS match"
                    new_row["Annotation tier"] = "Low confidence"

                # Correct Δm/z using the array values
                delta_da = float(delta_da_arr[i])
                theo_mz = mz_theo_arr[i]
                new_row["Δm/z (mDa)"] = delta_da * 1000.0
                new_row["Δm/z (ppm)"] = (delta_da / theo_mz) * 1e6 if theo_mz else ""

                # everything else unchanged …
                headgroup = str(new_row["Annotation"]).split(" ")[0]
                new_row["Headgroup"] = headgroup

                if new_row.get("Annotation Type") not in ("IS", "MS/MS match"):
                    headgroup = str(new_row.get("Headgroup", "")).strip()
                    if headgroup in headgroup_map:
                        new_row["Lipid Class"] = headgroup_map[headgroup]
                    else:
                        new_row["Lipid Class"] = "Other"
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
    # Make sure df_input and df_results have the same schema
    # Preserve all original rows (including unassigned),
    # and add matched annotations as extra rows if any exist
    df_input.columns = df_input.columns.str.strip()
    if results:
        df_results = pd.DataFrame(results)
        df_results.columns = df_results.columns.str.strip()
        # align to input schema
        df_results = df_results.reindex(columns=df_input.columns, fill_value="")
        df_output = pd.concat([df_input, df_results], ignore_index=True)
    else:
        df_output = df_input.copy()

    # --- Preserve all baseline columns from sanitized file ---
    baseline_cols = list(df_input.columns)  # all original sanitized cols
    for col in baseline_cols:
        if col not in df_output.columns:
            df_output[col] = ""

    # --- Reorder ---
    preferred_order = [
        "UniqueID", 
        "RT (min)" if "RT (min)" in df_output.columns else "RT [min]",
        "m/z" if "m/z" in df_output.columns else "m/z meas.",
        "Neutral mass" if "Neutral mass" in df_output.columns else "Neutral Mass",
        "Adducts", "Polarity",
        "QC RSD (%)" if "QC RSD (%)" in df_output.columns else "QC RSD [%]",
        "Sample RSD (%)" if "Sample RSD (%)" in df_output.columns else "Samples RSD (%)",
        "Annotation", "Headgroup", "Lipid Class", "Molecular Formula",
        "Δm/z (mDa)" if "Δm/z (mDa)" in df_output.columns else "Δm/z [mDa]",
        "Δm/z (ppm)" if "Δm/z (ppm)" in df_output.columns else "Δm/z [ppm]",
        "Annotation tier",
        "Annotation Type", "Annotation Source", "Metaboscape Annotation Status", "MS/MS available?", 
        "MS/MS score", "mSigma", 
        "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]",
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

    print("Total rows going into output:", len(df_output), flush = True)
    
    # --- Save output ---
    output_folder = Path(output_folder)
    debug_folder = output_folder / "debug"
    debug_folder.mkdir(parents=True, exist_ok=True)

    out_file = debug_folder / "MS_search_results_RAW.csv"
    df_output.to_csv(out_file, index=False, encoding="utf-8-sig")
    
    print(f'\n ----- MS search finished; moving to filtering. ----- \n', flush = True)

    return out_file, df_output


