# -------------------------------------------------------------------------
# Detects the most recent processed result files (LOESS, median, normalized, or filtered)
# and generates:
#   - Final_Annotated.csv
#   - Final_Unknowns.csv
# inside the main results folder.

'''
For RSD QC filtering:
    a feature with 0, 1, or 2 detected QCs is not removed by QC RSD (only filtered by sample RSD);
    a feature with 3 or more detected QCs is removed by QC RSD only if it exceeds the threshold; then the within-group RSD filter still runs after.
That same behavior applies to normalized annotated, semi-quant annotated, before-normalization annotated, and unknowns.
'''

# -------------------------------------------------------------------------

import pandas as pd
import re
from pathlib import Path
from typing import Optional

# Expected number of fatty-acyl chains by Lipid Class (edit to match your names)
_CLASS_TO_ACYL_COUNT = {
    # 1-acyl (monoacyl)
    "FA": 1, "FOH": 1, "FAG": 1, "FAL": 1, "HC": 1,
    "CAR": 1, "CoA": 1,
    "NA": 1,  "NAE": 1,  "NAT": 1,
    "MG": 1,
    "MGMG": 1, "MGDG": 1,
    "LPC": 1, "LPE": 1, "LPI": 1, "LPS": 1, "LPG": 1, "LPA": 1, "LSM": 1, "LPT": 1, 
    "CE": 1, "WE": 2,
    "ST": 1,
    "Other": 1, "PK": 1, "PR": 1, "SL":1,

    # sphingolipids: 
    "Cer": 2, "CerP": 2, "GlcCer": 2, "HexCer": 2, "Hex2Cer": 2, "SM": 2, "HexSBP": 2, 
    "PE-Cer": 2, "PI-Cer": 2, "SCer": 2, "SHexCer": 2, 
    "SPB": 1, "SPBP": 1,
    "SulfateHexSPB": 1, 
    "M(IP)2C": 2, "MIPC": 2, 

    # 2-acyl (diacyl)
    "PC": 2, "PE": 2, "PI": 2, "PS": 2, "PG": 2, "PA": 2, "BMP": 2, "Glc-GP": 2, "GP": 2, 
    "PIM": 2, "PIP": 2, "PnC": 2, "PnE": 2, "PPA": 2, "PT": 2, 
    "DG": 2, "CDP-DG": 2, "DGCC": 2, "DGDG": 2, "DGMG": 2, "DGTA": 2, "DGTS": 2, "GlcADG": 2,
    "SQDG": 2, "SQMG": 2,
    "FAHFA": 2,

    # 3-acyl (triacyl)
    "TG": 3,
    "NAPE": 3,
    "ACer": 3,

    # 4-acyl (tetraacyl)
    "CL": 4,
}

def clean_sample_name(name: str) -> str:
    """Simplify sample names by removing polarity and run identifiers."""
    if not isinstance(name, str):
        return name
    cleaned = name
    cleaned = re.sub(r"\[?POS\]?|\[?NEG\]?", "", cleaned, flags=re.IGNORECASE)  # remove [POS]/[NEG]
    cleaned = re.sub(r"^(P_|N_)", "", cleaned)  # remove leading polarity letters
    cleaned = re.split(r"_P[12]", cleaned)[0]   # truncate after _P1/_P2
    return cleaned.strip("_- ")

def _build_species_annotation(df: pd.DataFrame) -> pd.Series:
    """
    Species annotation = Headgroup + (optional space) + total carbons + ":" + DBE + [;O#] + [;Modifications]

    Rules:
      - If total carbons is missing or 0 -> copy Annotation (do not build species)
      - If Headgroup contains "O-" -> no space between Headgroup and "C:D"
      - If Headgroup contains any of: HexCer, Hex(2)Cer, Hex2Cer, Hex3Cer -> force Headgroup to "HexCer"
      - For Cer/SM/HexCer: if Annotation contains ';O' followed by a number (e.g. ';O2') keep it in Species annotation
      - If Modifications is empty -> omit it
    """
    # Need Annotation for fallback copy
    ann = df["Annotation"].astype("string").fillna("").str.strip() if "Annotation" in df.columns else pd.Series([""] * len(df), index=df.index, dtype="string")

    required = [
        "Headgroup",
        "Number of carbons in fatty acyls",
        "Double bond equivalents",
        "Modifications",
    ]
    for c in required:
        if c not in df.columns:
            # missing inputs -> best effort: copy annotation
            return ann.copy()

    hg = df["Headgroup"].astype("string").fillna("").str.strip()

    # --- Normalize HexCer variants to exactly "HexCer" ---
    hex_mask = hg.str.contains(r"(?i)(?<![A-Za-z])(?:HexCer|Hex\(\d+\)Cer|Hex\d+Cer|Hex\((?:2|3|4|5|6)\)-)",
        regex=True,
        na=False
    )
    hg = hg.where(~hex_mask, "HexCer")

    carb = pd.to_numeric(df["Number of carbons in fatty acyls"], errors="coerce")
    dbe  = pd.to_numeric(df["Double bond equivalents"], errors="coerce")
    mods = df["Modifications"].astype("string").fillna("").str.strip()
    mods = mods.where(~mods.str.lower().isin({"nan", "none", "<na>", "na", "null"}), "")

    # --- If carbons missing or 0 -> copy Annotation ---
    fallback_mask = carb.isna() | (carb <= 0)
    out = pd.Series([""] * len(df), index=df.index, dtype="string")
    out.loc[fallback_mask] = ann.loc[fallback_mask]

    # Build only when core fields exist and not fallback
    build_mask = (~fallback_mask) & (hg != "") & dbe.notna()

    carb_i = carb.round(0).astype("Int64").astype("string")
    dbe_i  = dbe.round(0).astype("Int64").astype("string")

    # joiner: space unless headgroup contains "O-"
    joiner = pd.Series(" ", index=df.index, dtype="string")
    joiner = joiner.where(~hg.str.contains("O-", na=False), "")

    core = hg + joiner + carb_i + ":" + dbe_i

    # --- Extract ';O#' from Annotation (keep ONLY for molecular-level Cer/SM/HexCer) ---
    # molecular-level = has "/" or "_" in the annotation text
    is_molecular = ann.str.contains(r"[/_]", regex=True, na=False)

    oxy = ann.str.extract(r"(;O\d+)", expand=False).fillna("")
    sph_mask = hg.str.contains(r"^(Cer|SM|HexCer|GM)$", case=False, regex=True, na=False)

    # keep ;O# only when BOTH: (Cer/SM/HexCer) AND (molecular-level annotation)
    oxy = oxy.where(sph_mask & is_molecular, "")

    # Append oxygen suffix first (already contains leading ';')
    core_plus = core + oxy

    # Append Modifications if present
    has_mods = mods.ne("")
    out.loc[build_mask & has_mods] = core_plus.loc[build_mask & has_mods] + ";" + mods.loc[build_mask & has_mods]
    out.loc[build_mask & ~has_mods] = core_plus.loc[build_mask & ~has_mods]

    # Any remaining blanks: copy annotation (safe fallback)
    out = out.where(out.ne(""), ann)

    return out

def _infer_fatty_acyl_chain_count(headgroup: str, lipid_class: str, annotation: str) -> Optional[int]:
    lc = ("" if lipid_class is None else str(lipid_class)).strip()
    hg = ("" if headgroup is None else str(headgroup)).strip()
    ann = ("" if annotation is None else str(annotation)).strip()

    # normalize keys (pick ONE strategy that matches your tables)
    lc_key = lc.strip()
    hg_key = hg.strip()

    # 1) primary: Lipid Class mapping
    if lc_key in _CLASS_TO_ACYL_COUNT:
        return _CLASS_TO_ACYL_COUNT[lc_key]

    # 2) secondary: Headgroup mapping (optional)
    if hg_key in _CLASS_TO_ACYL_COUNT:
        return _CLASS_TO_ACYL_COUNT[hg_key]

    # 3) special cases (optional)
    # LNAPE vs NAPE if you encode lyso as "LNAPE" in Headgroup
    if hg_key.upper().startswith("L") and hg_key.upper().endswith("NAPE"):
        return 2
    if hg_key.upper().endswith("NAPE"):
        return 3

    # 4) last resort fallback: separators in annotation
    sep_count = len(re.findall(r"[/_]", ann))
    if sep_count >= 3:
        return 4
    if sep_count == 2:
        return 3
    if sep_count == 1:
        return 2
    if sep_count == 0:
        return 2

    return None

def _build_annotation_level(df: pd.DataFrame) -> pd.Series:
    """
    Rules:
      - 1 fatty acyl chain: always "Molecular level"
      - 2 fatty acyl chains:
          no "/" or "_" -> "Sum composition - Species level"
          "/" or "_" -> "Molecular level"
      - 3 fatty acyl chains:
          need two symbols ("/" or "_") -> "Molecular level"
          otherwise -> "Sum composition - Species level"

    Uses df["Annotation"] as the name (falls back to Species annotation if missing).
    """
    if "Annotation" in df.columns:
        name = df["Annotation"].astype("string").fillna("").str.strip()
    elif "Species annotation" in df.columns:
        name = df["Species annotation"].astype("string").fillna("").str.strip()
    else:
        name = pd.Series([""] * len(df), index=df.index, dtype="string")

    hg = df["Headgroup"].astype("string").fillna("") if "Headgroup" in df.columns else pd.Series([""] * len(df), index=df.index, dtype="string")
    lc = df["Lipid Class"].astype("string").fillna("") if "Lipid Class" in df.columns else pd.Series([""] * len(df), index=df.index, dtype="string")

    sep_count = name.str.count(r"[/_]")

    chain_counts = []
    for i in range(len(df)):
        chain_counts.append(_infer_fatty_acyl_chain_count(hg.iat[i], lc.iat[i], name.iat[i]))
    chain_counts = pd.Series(chain_counts, index=df.index, dtype="Int64")

    out = pd.Series([""] * len(df), index=df.index, dtype="string")

    m1 = chain_counts == 1
    out.loc[m1] = "Molecular level"

    m2 = chain_counts == 2
    out.loc[m2 & (sep_count == 0)] = "Sum composition - Species level"
    out.loc[m2 & (sep_count >= 1)] = "Molecular level"

    m3 = chain_counts == 3
    out.loc[m3 & (sep_count >= 2)] = "Molecular level"
    out.loc[m3 & (sep_count < 2)] = "Sum composition - Species level"

    # 4 chains
    m4 = chain_counts == 4
    out.loc[m4 & (sep_count >= 3)] = "Molecular level"
    out.loc[m4 & (sep_count < 3)] = "Sum composition - Species level"

    # any remaining unknowns: conservative default
    out.loc[out.eq("")] = "Sum composition - Species level"

    return out

def create_final_outputs(results_folder, rsd_qc_thresh=None, max_group_rsd_thresh=None):
    print(f"Results folder: {results_folder}", flush=True)
    results_folder = Path(results_folder)
    debug_folder = results_folder / "debug"
    debug_folder.mkdir(parents=True, exist_ok=True)
    
     # -------- HELPERS ---------
        
    def _apply_qc_filter(df_in, removed_frames, label_prefix=""):
        df_out = df_in.copy()

        qc_rsd_col = None
        if "RSD QCs (%)" in df_out.columns:
            qc_rsd_col = "RSD QCs (%)"

        qc_count_col = "QC detected count" if "QC detected count" in df_out.columns else None

        if rsd_qc_thresh is None or qc_rsd_col is None:
            print(f"[FINAL] {label_prefix}QC RSD filter skipped (no threshold or suitable column).")
            return df_out

        qc_vals = _to_num(df_out[qc_rsd_col])

        # Default: keep everything
        keep_qc = pd.Series(True, index=df_out.index)

        if qc_count_col is not None:
            qc_counts = pd.to_numeric(df_out[qc_count_col], errors="coerce")

            # Only apply QC RSD filter where >= 3 QCs were detected before imputation
            apply_qc_mask = qc_counts >= 3

            keep_qc.loc[apply_qc_mask] = (
                    qc_vals.loc[apply_qc_mask].isna() |
                    (qc_vals.loc[apply_qc_mask] <= float(rsd_qc_thresh))
                )
        else:
            # Fallback for old files lacking QC detected count
            keep_qc = qc_vals.isna() | (qc_vals <= float(rsd_qc_thresh))

        qc_fail_mask = ~keep_qc
        if qc_fail_mask.any():
            removed_qc_block = df_out.loc[qc_fail_mask].copy()
            removed_qc_block["Removed reason"] = (
                f"{label_prefix}{qc_rsd_col} > {rsd_qc_thresh}% "
                f"(applied only when QC detected count >= 3)"
            )
            removed_qc_block["QC RSD (parsed)"] = qc_vals.loc[qc_fail_mask].values
            if qc_count_col is not None:
                removed_qc_block["QC detected count (parsed)"] = pd.to_numeric(
                        df_out.loc[qc_fail_mask, qc_count_col], errors="coerce"
                    ).values
            removed_frames.append(removed_qc_block)
            print(f"[FINAL] {label_prefix}Removed {int(qc_fail_mask.sum())} features by QC filtering")

        return df_out.loc[keep_qc].copy()
        
    def _get_group_rsd_cols(df_in):
        cols = []
        for c in df_in.columns:
            if not c.startswith("RSD_"):
                continue
            if c in {
                "RSD QCs (%)",
                "RSD QCs observed-only (%)",
                "RSD QCs imputed (%)",
                "RSD Samples (%)"
            }:
                continue
            if c.strip().upper() == "RSD_QC [%]":
                continue
            cols.append(c)
        return cols

    # -------- END HELPERS --------

    pol_tag = ""

    print(f"Debug folder: {debug_folder}")

    folder_str = str(debug_folder).replace("\\", "/").upper()
    if "/POS/DEBUG" in folder_str.upper():
        pol_tag = "Pos_"
    elif "/NEG/DEBUG" in folder_str.upper():
        pol_tag = "Neg_"
    else:
        pol_tag = ""
        print("****  NO POLARITY TAG ***** ", flush=True)

    print("\nGenerating final files...\n")

    # --- Priority order for annotated file detection ---
    annotated_candidates = [
        (pol_tag + "8-Final_annotated_results_loess_normalized.csv", "LOESS normalization"),
        (pol_tag + "6-Final_annotated_median_normalized.csv", "Median normalization"),
        (pol_tag + "4-Final_annotated_results_normalized.csv", "Basic normalization"),
        (pol_tag + "3-Final_annotated_results_imputed.csv", "Imputed filtered only"),
    ]

    # --- Priority order for annotated file with semi_quantification detection ---
    annotated_semi_quant_candidates = [
        (pol_tag + "8b-Final_annotated_results_loess_normalized_semi_quant.csv", "LOESS normalization"),
        (pol_tag + "6c-Final_annotated_median_normalized_semi_quant.csv", "Median normalization"),
        (pol_tag + "4b-Final_annotated_results_norm_semi-quant.csv", "Basic normalization"),
    ]

    # --- Priority order for unknowns file detection ---
    unknowns_candidates = [
        (pol_tag + "9-Final_unknowns_loess_normalized.csv", "Median normalization"),
        (pol_tag + "7-Final_unknowns_median_normalized.csv", "Median normalization"),
        (pol_tag + "5-Final_unknowns_normalized.csv", "Basic normalization"),
    ]

    annotated_file = None
    annotated_file_semi_quant = None
    unknowns_file = None
    method_used = None
    method_used_unk = None

    # Detect annotated file (supports Pos_/Neg_ prefixes)
    for name, label in annotated_candidates:
        matches = sorted(debug_folder.glob(f"*{name}"))
        if not matches:
            continue
        annotated_file = matches[0]
        method_used = label
        break  # stop at the first match, respecting priority

    # Detect annotated file with semi_quant (supports Pos_/Neg_ prefixes)
    for name_semi, label_semi in annotated_semi_quant_candidates:
        matches_semi_quant = sorted(debug_folder.glob(f"*{name_semi}"))
        if not matches_semi_quant:
            continue
        annotated_file_semi_quant = matches_semi_quant[0]
        method_used = label_semi
        break  # stop at the first match, respecting priority

    # Detect unknowns file (supports Pos_/Neg_ prefixes and matches annotated polarity)
    for name_unk, label_unk in unknowns_candidates:
        matches_unk = sorted(debug_folder.glob(f"*{name_unk}"))
        if not matches_unk:
            continue
        unknowns_file = matches_unk[0]
        method_used_unk = label_unk
        break  # stop at the first match, respecting priority

    if annotated_file is None:
        raise FileNotFoundError(
            "No annotated dataset found in results/debug. "
            "Expected one of:\n" + "\n".join(n for n, _ in annotated_candidates)
        )

    print(f"\n[FINAL] ✅ GENERATING FINAL FILES: Using annotated and normalized dataset {annotated_file.name} ({method_used})", flush=True)
    print("[FINAL] ✅ GENERATING FINAL FILES: Using non-normalized annotated dataset 3-Final_annotated_results_imputed.csv → Final_Before_Normalization.csv", flush=True)
    if annotated_file_semi_quant:
        print(f"[FINAL] ✅ GENERATING FINAL FILES: Using annotated and normalized dataset with semi_quantification {annotated_file_semi_quant.name} ({method_used})", flush=True)
    else:
        print("[FINAL]⚠️ No dedicated annotated and normalized dataset with semi_quantification found.", flush=True)
    if unknowns_file:
        print(f"[FINAL] ✅ Using unknowns dataset: {unknowns_file.name} ({method_used_unk})", flush=True)
    else:
        print("[FINAL]⚠️ No dedicated unknowns file found. Only annotated file will be processed.", flush=True)

    print(f"[FINAL] RSD thresholds → QC: {rsd_qc_thresh}, Group: {max_group_rsd_thresh}\n\n", flush=True)

    # --- Load annotated data ---
    df_ann = pd.read_csv(annotated_file, low_memory=False)
    removed_ann_frames = []  # accumulate removed rows with reasons

    # ---- Apply RSD filters (post-normalization; thresholds from GUI) ----
    def _to_num(x):
        s = pd.Series(x, copy=False).astype(str)
        s = s.str.replace("%", "", regex=False)
        s = s.str.replace(",", ".", regex=False)
        s = s.str.extract(r"([-+]?[0-9]*\.?[0-9]+)")[0]
        return pd.to_numeric(s, errors="coerce")

    # 1) QC RSD filter — record removals
    df_ann = _apply_qc_filter(df_ann, removed_ann_frames, label_prefix="")

    # 2) Within-group RSD filter — record removals
    if max_group_rsd_thresh is not None:
        group_rsd_cols = _get_group_rsd_cols(df_ann)
        if group_rsd_cols:
            df_rsd = df_ann[group_rsd_cols].apply(_to_num)
            mask_all_na = df_rsd.isna().all(axis=1)
            keep_group = (df_rsd <= float(max_group_rsd_thresh)).any(axis=1) | mask_all_na
            # Keep if any group passes OR keep if all group RSDs are NaN

            group_fail_mask = ~keep_group
            if group_fail_mask.any():
                removed_group_block = df_ann.loc[group_fail_mask].copy()
                removed_group_block["Min group RSD (parsed)"] = df_rsd.min(axis=1).loc[group_fail_mask].values
                removed_group_block["Max group RSD (parsed)"] = df_rsd.max(axis=1).loc[group_fail_mask].values
                removed_group_block["Removed reason"] = f"All groups RSD > {max_group_rsd_thresh}% (or no group ≤ threshold)"
                removed_ann_frames.append(removed_group_block)
                print(f"[FINAL] Removed {int(group_fail_mask.sum())} features by within-group RSD > {max_group_rsd_thresh}% (in all groups)")

            df_ann = df_ann.loc[keep_group].copy()
        else:
            print("[FINAL] No per-group RSD columns detected; group RSD filter skipped.")
    else:
        print("[FINAL] Group RSD filter skipped (no threshold).")

    # --- Persist annotated removals (if any) ---
    removed_ann_path = debug_folder / f"{pol_tag}Removed_by_RSD_Annotated.csv"
    if removed_ann_frames:
        removed_ann = pd.concat(removed_ann_frames, ignore_index=True)
        removed_ann.to_csv(removed_ann_path, index=False, encoding="utf-8-sig")
        print(f"[FINAL] Saved RSD-removal debug file (annotated): {removed_ann_path.name} ({len(removed_ann)} rows)", flush = True)
    else:
        print("[FINAL] No annotated rows removed by RSD filters (or filters disabled).", flush = True)

    # --- Exclude internal standards (Annotation Type == "IS") ---
    if "Annotation Type" in df_ann.columns:
        initial_count = len(df_ann)
        df_ann = df_ann[~df_ann["Annotation Type"].astype(str).str.upper().eq("IS")].copy()
        removed_count = initial_count - len(df_ann)
        print(f"[INFO] Excluded {removed_count} internal standards (Annotation Type = 'IS') from annotated results.", flush = True)
    else:
        print("[WARNING] 'Annotation Type' column not found; no IS exclusion applied.", flush = True)

    # --- Add derived annotation columns ---
    df_ann["Species annotation"] = _build_species_annotation(df_ann)
    df_ann["Annotation level"] = _build_annotation_level(df_ann)

    # --- Define desired annotated columns ---
    base_cols = [
        "UniqueID", "RT (min)", "m/z", "Polarity", "Adducts", "Neutral mass",
        "QC detected count", "RSD QCs (%)", "RSD Samples (%)",
        "Annotation", "Annotation level", "Species annotation", 
        "Annotation Type", "Annotation Source",
        "Headgroup", "Lipid Class",
        "Δm/z (mDa)", "Δm/z (ppm)", "MS/MS score", "Annotation tier", "mSigma",
        "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]",
        "Molecular Formula", "Plasmenyl?",
        "Number of carbons in fatty acyls", "Double bond equivalents",
        "Chain type", "PUFA?", "Modifications",  
    ]

    # Detect sample-specific RSD columns
    rsd_cols = [c for c in df_ann.columns
                if isinstance(c, str) and c.startswith(("RSD_", "QC detected")) and c not in base_cols]

    # Detect all sample columns (intensities)
    sample_cols = [
        c for c in df_ann.columns
        if c not in base_cols and c not in rsd_cols and (c.startswith("P_") or c.startswith("N_"))
    ]
    sample_cols = [c for c in sample_cols if c != "QC detected count"]

    # --- Clean up sample column names (remove polarity, replicate suffixes, etc.) ---
    rename_map = {col: clean_sample_name(col) for col in sample_cols}
    df_ann.rename(columns=rename_map, inplace=True)
    sample_cols = list(rename_map.values())

    # Build final ordered column list (keep core + RSD + sample columns)
    final_cols = []
    for col in base_cols:
        if col in df_ann.columns:
            final_cols.append(col)
        if col == "RSD Samples (%)":
            final_cols.extend([c for c in rsd_cols if c not in final_cols])

    # Add sample columns at the end
    final_cols.extend([c for c in sample_cols if c not in final_cols])

    # Keep only columns that exist
    final_cols = [c for c in final_cols if c in df_ann.columns]

    # Apply final column filtering
    df_ann = df_ann[final_cols]

    # --- Save final outputs ---
    results_folder.mkdir(parents=True, exist_ok=True)
    annotated_path = results_folder / f"{pol_tag}Final_Annotated.csv"
    unknowns_path = results_folder / f"{pol_tag}Final_Unknowns.csv"

    df_ann.to_csv(annotated_path, index=False, encoding="utf-8-sig")

    # --- Load annotated data with semi_quantification ---
    annotated_path_semi = None
    if annotated_file_semi_quant is not None:
        df_ann_semi = pd.read_csv(annotated_file_semi_quant, low_memory=False)
        removed_ann_frames_semi = []

        # 1) QC RSD filter — record removals (semi-quant)
        df_ann_semi = _apply_qc_filter(df_ann_semi, removed_ann_frames_semi, label_prefix="[SEMI] ")

        # 2) Within-group RSD filter — record removals
        if max_group_rsd_thresh is not None:
            group_rsd_cols_semi = _get_group_rsd_cols(df_ann_semi)
            if group_rsd_cols_semi:
                df_rsd_semi = df_ann_semi[group_rsd_cols_semi].apply(_to_num)
                mask_all_na_semi = df_rsd_semi.isna().all(axis=1)
                keep_group_semi = (df_rsd_semi <= float(max_group_rsd_thresh)).any(axis=1) | mask_all_na_semi

                group_fail_mask_semi = ~keep_group_semi
                if group_fail_mask_semi.any():
                    removed_group_block_semi = df_ann_semi.loc[group_fail_mask_semi].copy()
                    removed_group_block_semi["Min group RSD (parsed)"] = df_rsd_semi.min(axis=1).loc[group_fail_mask_semi].values
                    removed_group_block_semi["Max group RSD (parsed)"] = df_rsd_semi.max(axis=1).loc[group_fail_mask_semi].values
                    removed_group_block_semi["Removed reason"] = f"All groups RSD > {max_group_rsd_thresh}% (or no group ≤ threshold)"
                    removed_ann_frames_semi.append(removed_group_block_semi)
                    print(f"[FINAL] Removed {int(group_fail_mask_semi.sum())} features by within-group RSD > {max_group_rsd_thresh}% (in all groups)")

                df_ann_semi = df_ann_semi.loc[keep_group_semi].copy()
            else:
                print("[FINAL] No per-group RSD columns detected; group RSD filter skipped.")
        else:
            print("[FINAL] Group RSD filter skipped (no threshold).")

        # --- Persist annotated removals (if any) ---
        removed_ann_path_semi = debug_folder / f"{pol_tag}Removed_by_RSD_Annotated_semi_quant.csv"
        if removed_ann_frames_semi:
            removed_ann_semi = pd.concat(removed_ann_frames_semi, ignore_index=True)
            removed_ann_semi.to_csv(removed_ann_path_semi, index=False, encoding="utf-8-sig")
            print(f"[FINAL] Saved RSD-removal debug file (annotated with semi_quantification): {removed_ann_path_semi.name} ({len(removed_ann_semi)} rows)")
        else:
            print("[FINAL] No annotated rows removed by RSD filters for semi_quantified file (or filters disabled).")

        # --- Exclude internal standards (Annotation Type == "IS") ---
        if "Annotation Type" in df_ann_semi.columns:
            initial_count_semi = len(df_ann_semi)
            df_ann_semi = df_ann_semi[~df_ann_semi["Annotation Type"].astype(str).str.upper().eq("IS")].copy()
            removed_count_semi = initial_count_semi - len(df_ann_semi)
            print(f"[INFO] Excluded {removed_count_semi} internal standards (Annotation Type = 'IS') from annotated results with semi_quantification.")
        else:
            print("[WARNING] 'Annotation Type' column not found in semi_quantified file; no IS exclusion applied.")

        # --- Add Species annotation column (requested) ---
        df_ann_semi["Species annotation"] = _build_species_annotation(df_ann_semi)
        df_ann_semi["Annotation level"] = _build_annotation_level(df_ann_semi)

        # Detect all sample columns
        rsd_cols_semi = [c for c in df_ann.columns
                        if isinstance(c, str) and c.startswith(("RSD_", "QC detected")) and c not in base_cols]
        sample_cols_semi = [
            c for c in df_ann_semi.columns
            if c not in base_cols and c not in rsd_cols_semi and (c.startswith("P_") or c.startswith("N_"))
        ]
        sample_cols_semi = [c for c in sample_cols_semi if c != "QC detected count"]

        # --- Clean up sample column names (remove polarity, replicate suffixes, etc.) ---
        rename_map_semi = {col_semi: clean_sample_name(col_semi) for col_semi in sample_cols_semi}
        df_ann_semi.rename(columns=rename_map_semi, inplace=True)
        sample_cols_semi = list(rename_map_semi.values())

        # Build final ordered column list (keep core + RSD + sample columns)
        final_cols_semi = []
        for col in base_cols:
            if col in df_ann_semi.columns:
                final_cols_semi.append(col)
            if col == "RSD Samples (%)":
                final_cols_semi.extend([c for c in rsd_cols_semi if c not in final_cols_semi])

        # Add sample columns at the end
        final_cols_semi.extend([c for c in sample_cols_semi if c not in final_cols_semi])

        # Keep only columns that exist
        final_cols_semi = [c for c in final_cols_semi if c in df_ann_semi.columns]

        # Apply final column filtering
        df_ann_semi = df_ann_semi[final_cols_semi]

        # --- Save final outputs ---
        annotated_path_semi = results_folder / f"{pol_tag}Final_Annotated_semi_quant.csv"
        df_ann_semi.to_csv(annotated_path_semi, index=False, encoding="utf-8-sig")

    # --- Also generate a BEFORE-NORMALIZATION final file ---
    before_norm_src = None
    if pol_tag:
        candidates_before = sorted(debug_folder.glob(f"{pol_tag}3-Final_annotated_results_imputed.csv"))
    else:
        candidates_before = []
    if not candidates_before:
        candidates_before = sorted(debug_folder.glob("3-Final_annotated_results_imputed.csv"))
    if candidates_before:
        before_norm_src = candidates_before[0]

    before_norm_path = results_folder / f"{pol_tag}Final_Annotated_Before_Normalization.csv"

    if before_norm_src and before_norm_src.exists():
        print(f"[FINAL] Building before-normalization table from: {before_norm_src.name}")
        df_before = pd.read_csv(before_norm_src, low_memory=False)
        removed_before_frames = []
        
        # Apply the SAME two RSD filters (QC and within-group) using the GUI thresholds (before)
        df_before = _apply_qc_filter(df_before, removed_before_frames, label_prefix="[BEFORE] ")

        if max_group_rsd_thresh is not None:
            group_rsd_cols_b = _get_group_rsd_cols(df_before)
            if group_rsd_cols_b:
                df_rsd_b = df_before[group_rsd_cols_b].apply(_to_num)
                mask_all_na_b = df_rsd_b.isna().all(axis=1)
                keep_group_b = (df_rsd_b <= float(max_group_rsd_thresh)).any(axis=1) | mask_all_na_b

                group_fail_b = ~keep_group_b
                if group_fail_b.any():
                    removed_group_b = df_before.loc[group_fail_b].copy()
                    removed_group_b["Min group RSD (parsed)"] = df_rsd_b.min(axis=1).loc[group_fail_b].values
                    removed_group_b["Max group RSD (parsed)"] = df_rsd_b.max(axis=1).loc[group_fail_b].values
                    removed_group_b["Removed reason"] = f"[BEFORE] All groups RSD > {max_group_rsd_thresh}% (or no group ≤ threshold)"
                    removed_before_frames.append(removed_group_b)
                    print(f"[FINAL] [BEFORE] Removed {int(group_fail_b.sum())} by within-group RSD > {max_group_rsd_thresh}% (in all groups)")

                df_before = df_before.loc[keep_group_b].copy()
            else:
                print("[FINAL] [BEFORE] No per-group RSD columns; group RSD filter skipped.")
        else:
            print("[FINAL] [BEFORE] Group RSD filter skipped (no threshold).")

        # Exclude IS (same as the main final file)
        if "Annotation Type" in df_before.columns:
            init_b = len(df_before)
            df_before = df_before[~df_before["Annotation Type"].astype(str).str.upper().eq("IS")].copy()
            removed_b = init_b - len(df_before)
            print(f"[INFO] [BEFORE] Excluded {removed_b} internal standards from annotated results.")
        else:
            print("[WARNING] [BEFORE] 'Annotation Type' not found; no IS exclusion applied.")

        # --- Add derived annotation columns ---
        df_before["Species annotation"] = _build_species_annotation(df_before)
        df_before["Annotation level"] = _build_annotation_level(df_before)

        base_cols_b = [
            "UniqueID", "RT (min)", "m/z", "Polarity", "Adducts", "Neutral mass",
            "QC detected count", "RSD QCs (%)", "RSD Samples (%)",
            "Annotation", "Annotation level", "Species annotation", 
            "Annotation Type", "Annotation Source",
            "Headgroup", "Lipid Class",
            "Δm/z (mDa)", "Δm/z (ppm)", "MS/MS score", "Annotation tier", "mSigma",
            "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]",
            "Molecular Formula", "Plasmenyl?",
            "Number of carbons in fatty acyls", "Double bond equivalents",
            "Chain type", "PUFA?", "Modifications",

        ]
        rsd_cols_b = [c for c in df_ann.columns
                        if isinstance(c, str) and c.startswith(("RSD_", "QC detected")) and c not in base_cols_b]
        sample_cols_b = [c for c in df_before.columns 
                         if c not in base_cols_b and c not in rsd_cols and (c.startswith("P_") or c.startswith("N_"))]
        sample_cols_b = [c for c in sample_cols_b if c != "QC detected count"]

        rename_map_b = {col: clean_sample_name(col) for col in sample_cols_b}
        df_before.rename(columns=rename_map_b, inplace=True)
        sample_cols_b = list(rename_map_b.values())

        final_cols_b = []
        for col in base_cols_b:
            if col in df_before.columns:
                final_cols_b.append(col)
            if col == "RSD Samples (%)":
                final_cols_b.extend([c for c in rsd_cols_b if c not in final_cols_b])
        final_cols_b.extend([c for c in sample_cols_b if c not in final_cols_b])
        final_cols_b = [c for c in final_cols_b if c in df_before.columns]

        df_before = df_before[final_cols_b]

        removed_before_path = debug_folder / f"{pol_tag}Removed_by_RSD_Annotated_BEFORE.csv"
        if removed_before_frames:
            removed_before = pd.concat(removed_before_frames, ignore_index=True)
            removed_before.to_csv(removed_before_path, index=False, encoding="utf-8-sig")
            print(f"[FINAL] Saved RSD-removal debug file (BEFORE): {removed_before_path.name} ({len(removed_before)} rows)")

        df_before.to_csv(before_norm_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {before_norm_path.name} ({len(df_before)} rows)")
    else:
        print("[FINAL] BEFORE file not found — skipping Final_Before_Normalization.csv")

    # --- Unknowns processing (unchanged) ---
    if unknowns_file:
        df_unk = pd.read_csv(unknowns_file, low_memory=False)
        removed_unk_frames = []

        df_unk = _apply_qc_filter(df_unk, removed_unk_frames, label_prefix="[UNK] ")

        if max_group_rsd_thresh is not None:
            group_rsd_cols_u = _get_group_rsd_cols(df_unk)
            if group_rsd_cols_u:
                df_rsd_u = df_unk[group_rsd_cols_u].apply(_to_num)
                mask_all_na_u = df_rsd_u.isna().all(axis=1)
                keep_group_u = (df_rsd_u <= float(max_group_rsd_thresh)).any(axis=1) | mask_all_na_u

                group_fail_u = ~keep_group_u
                if group_fail_u.any():
                    removed_group_u = df_unk.loc[group_fail_u].copy()
                    removed_group_u["Min group RSD (parsed)"] = df_rsd_u.min(axis=1).loc[group_fail_u].values
                    removed_group_u["Max group RSD (parsed)"] = df_rsd_u.max(axis=1).loc[group_fail_u].values
                    removed_group_u["Removed reason"] = f"[UNK] All groups RSD > {max_group_rsd_thresh}% (or no group ≤ threshold)"
                    removed_unk_frames.append(removed_group_u)
                    print(f"[FINAL] [UNK] Removed {int(group_fail_u.sum())} features by within-group RSD > {max_group_rsd_thresh}% (in all groups)")

                df_unk = df_unk.loc[keep_group_u].copy()
            else:
                print("[FINAL] [UNK] No per-group RSD columns detected; group RSD filter skipped.")
        else:
            print("[FINAL] [UNK] Group RSD filter skipped (no threshold).")

        base_cols_unk = [
            "UniqueID", "RT (min)", "m/z", "Polarity",
            "QC detected count", "RSD QCs (%)", "RSD Samples (%)",
        ]

        rsd_cols_unk = [c for c in df_ann.columns
                        if isinstance(c, str) and c.startswith(("RSD_", "QC detected")) and c not in base_cols_unk]
        sample_cols_unk = [
            c for c in df_unk.columns
            if c not in base_cols_unk and c not in rsd_cols_unk and (c.startswith("P_") or c.startswith("N_"))
        ]
        sample_cols_unk = [c for c in sample_cols_unk if c != "QC detected count"]

        rename_map_unk = {col: clean_sample_name(col) for col in sample_cols_unk}
        df_unk.rename(columns=rename_map_unk, inplace=True)
        sample_cols_unk = list(rename_map_unk.values())

        final_cols_unk = []
        for col in base_cols_unk:
            if col in df_unk.columns:
                final_cols_unk.append(col)
            if col == "RSD Samples (%)":
                final_cols_unk.extend([c for c in rsd_cols_unk if c not in final_cols_unk])

        final_cols_unk.extend([c for c in sample_cols_unk if c not in final_cols_unk])
        final_cols_unk = [c for c in final_cols_unk if c in df_unk.columns]
        df_unk = df_unk[final_cols_unk]

        removed_unk_path = debug_folder / f"{pol_tag}Removed_by_RSD_Unknowns.csv"
        if removed_unk_frames:
            removed_unk = pd.concat(removed_unk_frames, ignore_index=True)
            removed_unk.to_csv(removed_unk_path, index=False, encoding="utf-8-sig")
            print(f"[FINAL] Saved RSD-removal debug file (unknowns): {removed_unk_path.name} ({len(removed_unk)} rows)")

        df_unk.to_csv(unknowns_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {annotated_path.name} ({len(df_ann)} rows)")
    if unknowns_file:
        print(f"Saved: {unknowns_path.name} ({len(df_unk)} rows)")
    else:
        print("No unknowns file was saved (not found in debug folder).")

    return annotated_path, annotated_path_semi, unknowns_path if unknowns_file else None, method_used


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "results"
    create_final_outputs(folder)
