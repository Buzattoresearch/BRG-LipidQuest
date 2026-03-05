'''
In the scoring and filtering stage, the software assigns an MS Score to each annotated candidate and uses that score plus biological plausibility rules to decide what stays. 
It first applies a retention time filter based on class specific carbon bin windows defined in the Appendix, and it logs every decision so you can audit which window was used and why a row was dropped. 
It then computes a set of penalty terms for each annotated row and subtracts the total penalty from a starting score of 100, while keeping a human readable penalty breakdown string in the output for debugging. 
Adduct scoring is handled as a ranked whitelist per lipid class and polarity (see Appendix folder), where the expected adduct forms have low or zero penalty and less expected but still allowed adducts have higher penalty, 
and any adduct not listed for that class and polarity is treated as not allowed and is removed outright with its own debug export. 
Sensitivity scoring adds a class and polarity dependent penalty to downweight classes that are rare in nature or implausible in the current sample context. 
Additional penalties capture patterns that frequently indicate artifacts, including extreme mass error, unusual carbon to double bond ratios, implausible fatty acyl counts for a claimed class, 
excessive or suspicious modification strings, and explicit flags embedded in names such as missing abbreviations.

After scoring, the software applies a plausibility filter module that encodes organism specific chemistry rules and removes structurally or biologically implausible annotations while 
saving the removed rows and the reason for removal. High confidence MetaboScape annotations labeled as lipid species or target list calls can be exempted from plausibility filtering 
so strong vendor evidence is not discarded by conservative rule sets. The software then runs a Kendrick mass defect filter within lipid classes and removes rows whose mass defect does not match the class pattern, 
which helps catch oddball matches that slip through mass only search. It collapses duplicates so each UniqueID ends up with at most one representative annotation, choosing internal standards first, 
then MS/MS matches, then MS matches, and within those categories preferring the highest MS Score. It applies a minimum score cutoff to remove low scoring annotations while still keeping genuinely 
unassigned features in the final table so the dataset remains transparent about what could not be annotated.

'''

# -----------------------------
# FUNGI
# -----------------------------

import os, re
import pandas as pd
from pathlib import Path

# -------------------
# Load Appendix files (Adducts_and_sensitivity_scores, RT_groups, RT_windows)
# -------------------

def _normalize_adduct_string(x: str) -> str:
    """
    Normalize adduct text to match what you store in the Appendix.
    Keep this minimal and explicit.
    """
    s = str(x).strip()
    
    if s.lower() in {"nan", "na", "none", "<na>", "null"}:
        return ""

    if not s:
        return ""

    # Common variants -> canonical appendix style
    s_low = s.lower()
    if s_low in {"formate", "[m+formate]-", "[m+hcoo]-"}:
        return "[M+formate]-"
    if s_low in {"hcooh-h", "hcoo-h"}:
        return "[M+formate]-"
    if s_low in {"m+h-h2o", "m-h2o+h"}:
        return "[M+H-H2O]-"

    return s

def _primary_adduct(x) -> str:
    """
    Return the first adduct term from a MetaboScape-style adduct string.
    Handles comma or semicolon separated lists.
    """
    s = _normalize_adduct_string(x)
    if not s:
        return ""

    # Split on comma or semicolon, take the first token
    # Example: "[M+NH4]+, [2M+NH4]+" -> "[M+NH4]+"
    token = re.split(r"[;,]", s, maxsplit=1)[0].strip()
    return _normalize_adduct_string(token)

def load_adduct_sensitivity(path):
    """
    Loads:
      - ion_rank_dict[Pos/Neg][LIPID_CLASS][ADDUCT] = rank (0..4)
      - sensitivity[Pos/Neg][LIPID_CLASS] = float in [0,1] (as in your file)
    """
    df = pd.read_csv(path, header=1, low_memory=False)

    ion_rank_dict = {"Pos": {}, "Neg": {}}
    sensitivity = {"Pos": {}, "Neg": {}}

    pos_cols = ["P0", "P1", "P2", "P3", "P4"]
    neg_cols = ["N0", "N1", "N2", "N3", "N4"]

    for _, row in df.iterrows():
        lipid_class = str(row.get("Lipid Class", "")).strip().upper()
        if not lipid_class or lipid_class == "NAN":
            continue

        # Build per-class allowed adduct ranks
        ion_rank_dict["Pos"].setdefault(lipid_class, {})
        ion_rank_dict["Neg"].setdefault(lipid_class, {})

        for r, col in enumerate(pos_cols):
            v = row.get(col, "")
            if pd.notna(v) and str(v).strip() != "":
                adduct = _normalize_adduct_string(v)
                if adduct:
                    ion_rank_dict["Pos"][lipid_class][adduct] = r  # P0->0 ... P4->4

        for r, col in enumerate(neg_cols):
            v = row.get(col, "")
            if pd.notna(v) and str(v).strip() != "":
                adduct = _normalize_adduct_string(v)
                if adduct:
                    ion_rank_dict["Neg"][lipid_class][adduct] = r  # N0->0 ... N4->4

        # Sensitivity by lipid class
        if pd.notna(row.get("Sensitivity Pos")):
            sensitivity["Pos"][lipid_class] = float(row["Sensitivity Pos"])
        if pd.notna(row.get("Sensitivity Neg")):
            sensitivity["Neg"][lipid_class] = float(row["Sensitivity Neg"])

    return ion_rank_dict, sensitivity

def load_rt_groups(groups_path, windows_path):
    groups = pd.read_csv(groups_path, low_memory=False)
    windows = pd.read_csv(windows_path, low_memory=False)

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
        total_carbons = pd.to_numeric(total_carbons, errors="coerce")
        total_carbons = 0 if pd.isna(total_carbons) else int(total_carbons)
    except:
        return None

    bins = [
        (0,0,"0C"), 
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
    """
    Appendix rule:
      - If ion is P0/N0: 0.00 penalty
      - P1/N1: 0.25
      - P2/N2: 0.50
      - P3/N3: 0.75
      - P4/N4: 1.00
      - Not listed for that class+polarity: NOT ALLOWED (returns (1.0, False, None))
    Returns:
      (penalty_float, allowed_bool, rank_int_or_None)
    """
    
    # Normalize polarity to Pos/Neg
    pol = str(polarity).strip().lower()
    if pol.startswith("pos"):
        pol_key = "Pos"
    elif pol.startswith("neg"):
        pol_key = "Neg"
    else:
        return 1.0, False, None

    lipid_class = str(lipid_class).strip().upper()
    if not lipid_class or lipid_class == "NAN":
        return 1.0, False, None

    # If multiple adducts exist (comma-separated), take the first one
    ion_str = str(ion)
    if "," in ion_str:
        ion_str = ion_str.split(",")[0].strip()

    ion_norm = _normalize_adduct_string(ion_str)

    if not ion_norm:
        return 1.0, False, None

    ion_rank_dict = info_dict.get("ion_rank_dict", {})
    rank_map = ion_rank_dict.get(pol_key, {}).get(lipid_class, {})

    if ion_norm not in rank_map:
        # Not in appendix for this class+polarity -> not allowed
        return 1.0, False, None

    rank = int(rank_map[ion_norm])  # 0..4
    penalty = min(1.0, max(0.0, rank * 0.25))
    return penalty, True, rank

def calculate_sensitivity_score(polarity, adduct, lipid_class, sensitivity_dict):
    """
    Sensitivity / plausibility penalty based on lipid class and polarity.

    sensitivity_dict is built in load_adduct_sensitivity as:
        sensitivity["Pos" or "Neg"][lipid_class] -> base value in [0,1]
    """
    lipid_class = str(lipid_class).strip().upper()

    # Normalize polarity to the keys used in sensitivity_dict
    pol = str(polarity).strip().lower()
    if pol.startswith("pos"):
        pol_key = "Pos"
    elif pol.startswith("neg"):
        pol_key = "Neg"
    else:
        pol_key = None

    # Base score from the sensitivity table (per class), default 1 (max penalty)
    try:
        if pol_key is None:
            score = 1.0
        else:
            score = float(sensitivity_dict.get(pol_key, {}).get(lipid_class, 1.0))
    except Exception:
        score = 1.0

    penalty = 0.0

    # Penalty due to rarity in LMSD (lipids that have been detected in nature)
    if lipid_class in ['M(IP)2C', 'PnC', 'DGTA', 'PnE', 'PGS', 'PT', 'PS-NAc', 'NAPE',
                       'SLBPA', 'PPA', 'PE-IsoK', 'LSM', 'PGP', 'DGTS', 'GlcADG', 'GP',
                       'Glc-GP', 'DGMG', 'MGMG', 'DGCC']:  # <= 0.1% of LMSD entries
        penalty += 0.9

    elif lipid_class in ['SQDG', 'DGDG', 'SQMG', 'SPBP', 'HexSPB', 'PIP', 'SCer']:  # ~0.2%
        penalty += 0.75

    elif lipid_class in ['SHexCer', 'CerP', 'FAG', 'PE-NMe', 'NAT']:  # <= 0.05%
        penalty += 0.5

    elif lipid_class in ['CDP-DG', 'LPG', 'MGDG', 'LPI']:  # <= 0.10%
        penalty += 0.25

    elif lipid_class in ['LPIM', 'LPA', 'PI-Cer', 'NAE', 'LPS', 'MG', 'SPB']:  # <= 0.20%
        penalty += 0.10

    elif lipid_class in ['TG', 'FA', 'PK', 'ST', 'HexCer']:  # very common classes
        penalty += 0.10

    # Penalty due to uncommon lipids in these samples (bacterial context)
    if lipid_class in ['PMeOH', 'PEtOH', 'WE', 'FAL', 'FOH']:
        penalty += 0.9

    if lipid_class in ['PK', 'PR', 'SL', 'Other', 'GP', 'DGTS', 'HexSPB', 'NAT', 'SHexCer',
                       'SQDG', 'SMGDG', 'CerP', 'MIPC', 'M(IP)2C', 'SCer', 'HC', 'PnC',
                       'PnE', 'FOH', 'FAL', 'LSM', 'CDP-DG', 'GlcADG', 'DGCC',
                       'SulfateHexSPB', 'MGMG', 'DGMG', 'MGDG', 'DGDG', 'PIM',
                       'PI-Cer', 'HBMP', 'ACer', 'Acer', 'PE-Cer', 'NAE', 'PIP']:
        penalty += 0.9

    elif lipid_class in ['LPA', 'LPS', 'LPI', 'NA', 'SPB', 'ACer', 'CoA', 'SHexCer', 'CerP', 'ST']:
        penalty += 0.80

    elif lipid_class in ['PS', 'PI']:
        penalty += 0.35

    elif lipid_class in ['BMP', 'CAR', 'MG']:
        penalty += 0.20

    # Combine base and penalty, clamp to [0,1]
    score += penalty
    if score >= 1.0:
        score = 1.0
    if score < 0.0:
        score = 0.0

    return score

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
    if any(x in str(modifications) for x in ['S','Br','Cl','As','Si','F','I','T','G', 'Gly', 'Leu']):
        score = 1
    if any(x in str(name) for x in ['medication','contaminant', 'plant']):
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

def apply_scoring(df, output_folder, pol_tag, weights=None):
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

    print('\n\n -----------  APPLYING SCORES FOR FUNGI --------- \n\n')

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
    program_folder = Path(__file__).resolve().parent
    appendix = program_folder / "Appendix"

    ion_rank_dict, sensitivity_dict = load_adduct_sensitivity(appendix / "Adducts_and_sensitivity_scores.csv")
    info_dict = {"ion_rank_dict": ion_rank_dict}

    lipid_class_to_window = load_rt_groups(appendix / "RT_groups.csv", appendix / "RT_windows.csv")

    # ensure debug folder
    output_folder = Path(output_folder)
    debug_folder = output_folder / "debug"
    debug_folder.mkdir(parents=True, exist_ok=True)
            
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(debug_folder, "rt_debug.log")
    dropped_path = os.path.join(debug_folder, f"{pol_tag}Annotations_Removed_by_rt.csv")
    
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

            dropped_path = debug_folder / f"{pol_tag}Annotations_Removed_by_rt.csv"
            df_dropped.to_csv(dropped_path, index=False, encoding="utf-8-sig")

            # Drop those rows
            assigned = assigned.drop(index=rows_to_drop)
            print(f"[INFO] Removed {len(rows_to_drop)} features by RT filtering.")
        except Exception as e:
            print(f"[WARNING] Could not save dropped rows: {e}", flush=True)


    # Now reset index once, after dropping
    assigned = assigned.reset_index(drop=True)

    # Compute penalties for assigned rows
    not_allowed_idx = []
    for idx, row in assigned.iterrows():
        lipid_class = str(row.get("Lipid Class", "")).strip().upper()

        penalties = []
        breakdown = []

        adduct_raw = (
            row.get("Matched adduct (MS matches)")
            if pd.notna(row.get("Matched adduct (MS matches)")) and str(row.get("Matched adduct (MS matches)")).strip() != ""
            else row.get("Adducts")
            if pd.notna(row.get("Adducts")) and str(row.get("Adducts")).strip() != ""
            else row.get("Matched adduct")
        )

        adduct_primary = _primary_adduct(adduct_raw)
    
        # mz error
        p = calculate_mz_error_score(row.get("Δm/z (ppm)", "")) * weights["mz_error_score_weight"]
        penalties.append(p); breakdown.append(f"mz_error={p:.2f}")

        # adducts (ranked whitelist from appendix)
        p_adduct, adduct_allowed, adduct_rank = calculate_adduct_score(
            adduct_primary,
            info_dict,
            row.get("Polarity", ""),
            lipid_class
        )

    
        if not adduct_allowed:
            not_allowed_idx.append(idx)
    
        p = p_adduct * weights["adduct_score_weight"]
        penalties.append(p)

        if adduct_allowed:
            breakdown.append(f"adduct=P{adduct_rank}:{p:.2f}")
        else:
            breakdown.append(f"adduct=NOT_ALLOWED:{p:.2f}")

        # sensitivity
        p = calculate_sensitivity_score(row.get("Polarity", ""), adduct_primary, lipid_class, sensitivity_dict) * weights["sensitivity_score_weight"]
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

    print(f"[INFO] not_allowed_idx count = {len(not_allowed_idx)}", flush=True)
    
    # Hard remove: adduct not allowed by Appendix whitelist
    out_path = Path(debug_folder) / f"{pol_tag}Annotations_Removed_by_adduct_not_allowed.csv"
    df_not_allowed = assigned.loc[not_allowed_idx].copy()

    # Always write a file (even if empty) so you can confirm the code path ran
    df_not_allowed.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Wrote adduct-not-allowed debug file: {out_path} (rows={len(df_not_allowed)})", flush=True)

    if not_allowed_idx:
        assigned = assigned.drop(index=not_allowed_idx).reset_index(drop=True)
        print(f"[INFO] Removed {len(not_allowed_idx)} features: adduct not allowed by Appendix.", flush=True)
        
    # Recombine assigned + unassigned
    df_scored = pd.concat([assigned, unassigned], ignore_index=True)
    return df_scored



