
# -----------------------------
# MAMMALIANS
# -----------------------------

import os
import pandas as pd
from pathlib import Path

# -------------------
# Load Appendix files (Adducts_and_sensitivity_scores, RT_groups, RT_windows)
# -------------------

def load_adduct_sensitivity(path):
    # Skip the first grouping row, use second row as header
    df = pd.read_csv(path, header=[1])

    ion_score_dict = {"Pos": {}, "Neg": {}}
    sensitivity = {"Pos": {}, "Neg": {}}

    for _, row in df.iterrows():
        lipid_class = str(row["Lipid Class"]).strip()

        # Positive adduct scores
        for col in ["P0", "P1", "P2", "P3", "P4"]:
            if pd.notna(row[col]):
                ion_score_dict["Pos"][col] = row[col]

        # Negative adduct scores
        for col in ["N0", "N1", "N2", "N3", "N4"]:
            if pd.notna(row[col]):
                ion_score_dict["Neg"][col] = row[col]

        # Sensitivity by lipid class
        if pd.notna(row.get("Sensitivity Pos")):
            sensitivity["Pos"][lipid_class] = row["Sensitivity Pos"]
        if pd.notna(row.get("Sensitivity Neg")):
            sensitivity["Neg"][lipid_class] = row["Sensitivity Neg"]

    return ion_score_dict, sensitivity

def load_rt_groups(groups_path, windows_path):
    groups = pd.read_csv(groups_path)
    windows = pd.read_csv(windows_path)

    # Normalize lipid_class
    groups["lipid_class"] = groups["lipid_class"].astype(str).str.strip().str.upper()

    # Map time group ID -> (lower, upper)
    rt_windows = {
        str(int(row["time group"])): (
            float(row["Lower window (min)"]),
            float(row["Upper window (min)"])
        )
        for _, row in windows.iterrows()
    }

    # Map (lipid_class, carbon_bin) -> (lower, upper)
    lipid_class_to_window = {}
    for _, row in groups.iterrows():
        lipid_class = row["lipid_class"]
        for carbon_bin, group_id in row.items():
            if carbon_bin == "lipid_class":
                continue
            carbon_bin = str(carbon_bin).strip().upper()
            if pd.notna(group_id) and int(group_id) > 0:
                group_id = str(int(group_id))
                if group_id in rt_windows:
                    lipid_class_to_window[(lipid_class, carbon_bin)] = rt_windows[group_id]

    return lipid_class_to_window

# -------------------
# Scoring Functions
# -------------------

def calculate_mz_error_score(mass_error_ppm):
    if pd.isna(mass_error_ppm):
        return 1.0
    if mass_error_ppm <= 0.8: return 0
    elif mass_error_ppm <= 1.6: return 0.2
    elif mass_error_ppm <= 2.4: return 0.4
    elif mass_error_ppm <= 3.2: return 0.6
    elif mass_error_ppm <= 4: return 0.8
    return 1.0

def assign_chain_bin(total_carbons):
    """Assign chain length bin name based on total carbons."""
    try:
        total_carbons = int(total_carbons)
    except:
        return None

    bins = [
        (1,10,"1-10C"), 
        (11,16,"11-16C"), 
        (17,22,"17-22C"),
        (23,28,"23-28C"), 
        (29,34,"29-34C"), 
        (35,40,"35-40C"),
        (41,46,"41-46C"), 
        (47,52,"47-52C"), 
        (53,58,"53-58C"),
        (59,64,"59-64C"), 
        (65,70,"65-70C"), 
        (71,999,">70C")
    ]
    for lo, hi, name in bins:
        if lo <= total_carbons <= hi:
            return name.upper().strip()
    return None

def calculate_rt_filter(rt, lipid_class, total_carbons, lipid_class_to_window):
    """Return True if the annotation passes RT filter, False if it should be excluded."""
    chain_bin = assign_chain_bin(total_carbons)
    if not chain_bin or pd.isna(rt):
        return False

    rt_range = lipid_class_to_window.get((lipid_class.upper(), chain_bin))
    if not rt_range:
        return False

    rt_min, rt_max = rt_range
    try:
        rt = float(rt)
    except:
        return False

    return rt_min <= rt <= rt_max


def calculate_adduct_score(ion, info_dict, polarity, lipid_class):
    if not isinstance(ion, str): return 1.0
    ion = ion.replace("Formate", "[M+formate]-").replace("HCOOH-H", "[M+formate]-")
    try:
        score_temp = float(info_dict["ion_score_dict"].get(polarity, {}).get(ion, 1))
        if score_temp <= 0.2: return 0
        elif score_temp <= 0.4: return 0.25
        elif score_temp <= 0.6: return 0.5
        elif score_temp <= 0.8: return 0.75
        else: return 1
    except: return 1

def calculate_sensitivity_score(polarity, adduct, lipid_class, sensitivity_dict):
    try:
        return float(sensitivity_dict.get(polarity, {}).get(adduct, 1))
    except:
        return 1

def calculate_fa_chain(number_of_carbons, lipid_class):
    if pd.isna(number_of_carbons) or number_of_carbons == "": return 1
    try:
        if int(number_of_carbons) % 2 == 0: return 0
        else: return 1
    except: return 1

def calculate_plasmenyl_score(has_plasmenyl, lipid_class):
    if has_plasmenyl == "Yes":
        if lipid_class in ['PC', 'PE', 'PG', 'PA', 'PS', 'PI', 'LPE', 'LPC', 'LPG', 'LPA', 'LPI']:
            return 1
        return 1
    return 0

def calculate_modifications_score(n_modifications, lipid_class, modifications, name):
    if not n_modifications: return 0
    try:
        n_modifications = int(n_modifications)
    except: return 1
    score = min(1, n_modifications / 6.0)
    if any(x in str(modifications) for x in ['S','Br','Cl','As','Si','F','I','T','G']):
        score = 1
    if any(x in str(name) for x in ['medication','contaminant']):
        score = 1
    return score

def calculate_carbon_dbe_ratio_score(ratio, dbe, carbons, lipid_class):
    if not ratio or pd.isna(ratio):
        return 1
    try:
        ratio = float(ratio)
        if 8 <= ratio <= 36: return 0
        elif 3.5 <= ratio < 8: return 0.5
        else: return 1
    except: return 1

def calculate_fatty_acyl_score(fa_list, lipid_class, total_carbons):
    if not fa_list: return 1
    if lipid_class == "FA" and 14 <= total_carbons <= 20:
        return 0
    if len(fa_list) == 2 and 28 <= total_carbons <= 48:
        return 0
    if len(fa_list) == 3 and 41 <= total_carbons <= 60:
        return 0
    if len(fa_list) == 4 and 56 <= total_carbons <= 76:
        return 0
    return 0.5

def calculate_abbreviation_score(name):
    if not isinstance(name, str):
        return 0
    return 1 if "NOABBR" in name.upper() else 0

# -------------------
# Apply Scoring
# -------------------

def apply_scoring(df, output_folder, weights=None):
    if weights is None:
        weights = {
            "adduct_score_weight": 1.0,
            "rt_score_weight": 1.0,
            "mz_error_score_weight": 1.0,
            "carbon_dbe_ratio_score_weight": 1.0,
            "fa_chain_score_weight": 1.0,
            "modifications_score_weight": 1.0,
            "plasmenyl_score_weight": 6.0,
            "sensitivity_score_weight": 1.0,
            "fa_score_weight": 1.0,
            "abbreviation_score_weight": 10.0,
        }

    print('\n\n -----------  APPLYING SCORES FOR MAMMALIANS --------- \n\n')

    # Split into assigned and unassigned
    ann = df["Annotation"].astype(str).str.strip()
    unassigned = df[ann.eq("") | ann.eq("nan") | df["Annotation"].isna()].copy()
    assigned = df[~(ann.eq("") | ann.eq("nan") | df["Annotation"].isna())].copy()
    # NOTE: do not reset index here, keep original alignment with rows_to_drop

    # Initialize scoring columns
    assigned["MS Score"] = 100.0
    assigned["Penalty breakdown"] = ""
    unassigned["MS Score"] = 100.0  # keep default high score
    unassigned["Penalty breakdown"] = "unassigned"

    # Load reference data
    ion_score_dict, sensitivity_dict = load_adduct_sensitivity("Appendix/Adducts_and_sensitivity_scores.csv")
    lipid_class_to_window = load_rt_groups("Appendix/RT_groups.csv", "Appendix/RT_windows.csv")

    # ensure debug folder
    output_folder = Path(output_folder)
    debug_folder = output_folder / "debug"
    debug_folder.mkdir(parents=True, exist_ok=True)
            
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(debug_folder, "rt_debug.log")
    dropped_path = os.path.join(debug_folder, "Removed_by_rt.csv")
    
    rows_to_drop = []

    # RT filtering for assigned only
    with open(log_path, "w", encoding="utf-8") as log:
        for idx, row in assigned.iterrows():
            lipid_class = str(row.get("Lipid Class", "")).strip().upper()
            total_carbons = row.get("Number of carbons in fatty acyls", "")
            rt = row.get("RT (min)", "")
            chain_bin = assign_chain_bin(total_carbons)
            key = (lipid_class, chain_bin)
            window = lipid_class_to_window.get(key)

            log.write(
                f"{idx}: class={lipid_class}, carbons={total_carbons}, bin={chain_bin}, "
                f"rt={rt}, key={key}, window={window}\n"
            )

            if not calculate_rt_filter(rt, lipid_class, total_carbons, lipid_class_to_window):
                rows_to_drop.append(idx)

    # Save dropped rows (if any) and apply filtering
    if rows_to_drop:
        try:
            df_dropped = assigned.loc[rows_to_drop].copy()

            dropped_path = debug_folder / "Removed_by_rt.csv"
            df_dropped.to_csv(dropped_path, index=False, encoding="utf-8-sig")

            # Drop those rows
            assigned = assigned.drop(index=rows_to_drop)
            print(f"[INFO] Removed {len(rows_to_drop)} features by RT filtering.")
        except Exception as e:
            print(f"[WARNING] Could not save dropped rows: {e}", flush=True)


    # Now reset index once, after dropping
    assigned = assigned.reset_index(drop=True)

    # Compute penalties for assigned rows
    for idx, row in assigned.iterrows():
        lipid_class = str(row.get("Lipid Class", "")).strip().upper()

        penalties = []
        breakdown = []

        adduct = (
            row.get("Matched adduct (MS matches)") or
            row.get("Adducts") or
            row.get("Matched adduct")
        )

        # mz error
        p = calculate_mz_error_score(row.get("Δm/z (ppm)", "")) * weights["mz_error_score_weight"]
        penalties.append(p); breakdown.append(f"mz_error={p:.2f}")

        # adducts
        p = calculate_adduct_score(adduct, {"ion_score_dict": ion_score_dict}, row.get("Polarity", ""), lipid_class) * weights["adduct_score_weight"]
        penalties.append(p); breakdown.append(f"adduct={p:.2f}")

        # sensitivity
        p = calculate_sensitivity_score(row.get("Polarity", ""), adduct, lipid_class, sensitivity_dict) * weights["sensitivity_score_weight"]
        penalties.append(p); breakdown.append(f"sensitivity={p:.2f}")

        # fa chain odd/even
        p = calculate_fa_chain(row.get("Number of carbons in fatty acyls", ""), lipid_class) * weights["fa_chain_score_weight"]
        penalties.append(p); breakdown.append(f"fa_chain={p:.2f}")

        # plasmenyl
        p = calculate_plasmenyl_score(row.get("Plasmenyl?", ""), lipid_class) * weights["plasmenyl_score_weight"]
        penalties.append(p); breakdown.append(f"plasmenyl={p:.2f}")

        # modifications
        p = calculate_modifications_score(
            row.get("# of modifications", ""), lipid_class,
            row.get("Modifications", ""), row.get("Annotation", "")
        ) * weights["modifications_score_weight"]
        penalties.append(p); breakdown.append(f"modifications={p:.2f}")

        # carbon/dbe ratio
        p = calculate_carbon_dbe_ratio_score(
            row.get("Carbons / double bond equivalent ratio", ""),
            row.get("Double bond equivalents", ""),
            row.get("Number of carbons in fatty acyls", ""),
            lipid_class
        ) * weights["carbon_dbe_ratio_score_weight"]
        penalties.append(p); breakdown.append(f"carbon_dbe={p:.2f}")

        # fatty acyl plausibility
        fatty_acyls = [
            f"{row.get(f'Number of carbons in fatty acyl {i}', '')}:{row.get(f'Double bonds in fatty acyl {i}', '')}"
            for i in range(1, 5)
            if row.get(f'Number of carbons in fatty acyl {i}', '') not in [None, ""]
        ]
        p = calculate_fatty_acyl_score(fatty_acyls, lipid_class, row.get("Number of carbons in fatty acyls", "")) * weights["fa_score_weight"]
        penalties.append(p); breakdown.append(f"fa_score={p:.2f}")
        
        # Abbreviation penalty
        p = calculate_abbreviation_score(row.get("Annotation", "")) * weights["abbreviation_score_weight"]
        penalties.append(p); breakdown.append(f"NoAbbreviation={p:.2f}")

        total_penalty = sum(penalties)
        assigned.at[idx, "MS Score"] = max(0, assigned.at[idx, "MS Score"] - total_penalty)
        assigned.at[idx, "Penalty breakdown"] = "; ".join(breakdown)

    # Recombine assigned + unassigned
    df_scored = pd.concat([assigned, unassigned], ignore_index=True)
    return df_scored



