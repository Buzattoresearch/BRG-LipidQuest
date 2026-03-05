# -------------------------------------------------------------------------
# Polarity merging utilities for LipidQuest / MetaboScape pipeline
#
# Provides two modes:
#   1) simple:   row-wise concatenation of POS + NEG final files
#   2) best:     keep only the "best" polarity per lipid annotation,
#                based on MS/MS score, mSigma, Δm/z (mDa), and RSD QCs (%),
#                after identifying candidate POS–NEG pairs by RT and mass.
#                Unknowns are always concatenated.
#
# Input (per root results folder):
#   POS/Pos_Final_Annotated.csv
#   POS/Pos_Final_Unknowns.csv
#   NEG/Neg_Final_Annotated.csv
#   NEG/Neg_Final_Unknowns.csv
#
# Output (in <root>):
#   Final_Annotated_simple_combination.csv
#   Final_Unknowns_simple_combination.csv
#   Final_Annotated.csv
#   Final_Unknowns.csv
#
# Debug outputs (in <root>/debug_merging, best mode only, when both polarities exist):
#   candidate_pairs.csv   – all RT+mass+annotation-compatible POS–NEG pairs (with iteration)
#   scores_detailed.csv   – raw + normalized metrics, boosted scores, winners (with iteration)
#   removed_features.csv  – all features discarded by polarity competition (with iteration)
#
# Usage:
#   python merging.py /path/to/output_root --mode simple
#   python merging.py /path/to/output_root --mode best
#   python merging.py /path/to/output_root --mode both
#
# -------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Constants: column heuristics
# -------------------------------------------------------------------------

# Core annotated columns as produced by generate_final_file.create_final_outputs
BASE_ANNOTATED_COLS = [
    "UniqueID", "RT (min)", "m/z", "Polarity", "Adducts", "Neutral mass",
    "Annotation", "Annotation level", "Species annotation", 
    "Annotation Type", "Annotation Source",
    "Headgroup", "Lipid Class",
    "Δm/z (mDa)", "Δm/z (ppm)", "MS/MS score", "Annotation tier", "mSigma",
    "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]",
    "Molecular Formula", "Plasmenyl?",
    "Number of carbons in fatty acyls", "Double bond equivalents",
    "Chain type", "PUFA?", "Modifications",
    "RSD QCs (%)", "RSD Samples (%)",
]

# Core unknown columns
BASE_UNKNOWN_COLS = [
    "UniqueID", "RT (min)", "m/z", "Polarity",
    "RSD QCs (%)", "RSD Samples (%)",
]


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def _parse_percent(value: pd.Series) -> pd.Series:
    """
    Parse RSD-like columns ("xx %", comma decimal, etc.) to float.
    Returns a Series of floats with NaN for unparsable entries.
    """
    s = value.astype(str)
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.extract(r"([-+]?[0-9]*\.?[0-9]+)")[0]
    return pd.to_numeric(s, errors="coerce")


def _detect_sample_columns(df: pd.DataFrame, base_cols: List[str]) -> List[str]:
    """
    Detect sample intensity columns in a final table:
    - not in base_cols
    - not starting with 'RSD_' (RSD per group)
    """
    sample_cols: List[str] = []
    for col in df.columns:
        if col in base_cols:
            continue
        if col.startswith("RSD_"):
            continue
        sample_cols.append(col)
    return sample_cols


def _load_final_files(root: Path) -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame],
    Optional[pd.DataFrame], Optional[pd.DataFrame]
]:
    """
    Load final annotated/unknowns for POS and NEG.
    Returns (df_pos_ann, df_neg_ann, df_pos_unk, df_neg_unk), where
    each can be None if the file is missing.
    """
    root = root.resolve()

    pos_folder = root / "POS"
    neg_folder = root / "NEG"

    pos_ann_path = pos_folder / "Pos_Final_Annotated.csv"
    neg_ann_path = neg_folder / "Neg_Final_Annotated.csv"
    pos_unk_path = pos_folder / "Pos_Final_Unknowns.csv"
    neg_unk_path = neg_folder / "Neg_Final_Unknowns.csv"

    df_pos_ann = pd.read_csv(pos_ann_path) if pos_ann_path.exists() else None
    df_neg_ann = pd.read_csv(neg_ann_path) if neg_ann_path.exists() else None
    df_pos_unk = pd.read_csv(pos_unk_path) if pos_unk_path.exists() else None
    df_neg_unk = pd.read_csv(neg_unk_path) if neg_unk_path.exists() else None

    if df_pos_ann is None and df_neg_ann is None:
        raise FileNotFoundError(
            f"No final annotated files found under {root}. "
            f"Expected at least one of:\n  {pos_ann_path}\n  {neg_ann_path}"
        )

    return df_pos_ann, df_neg_ann, df_pos_unk, df_neg_unk


def _mass_difference_ok(mass_pos: float, mass_neg: float) -> bool:
    """
    Return True if pos/neg neutral masses match within ±5 ppm OR ±5 mDa.
    """
    if pd.isna(mass_pos) or pd.isna(mass_neg):
        return False

    diff = abs(mass_pos - mass_neg)
    ppm_tol = mass_pos * 5e-6  # 5 ppm

    return (diff <= 0.005) or (diff <= ppm_tol)


def _find_candidate_pairs(df_pos: pd.DataFrame, df_neg: pd.DataFrame) -> pd.DataFrame:
    """
    Create a table of all POS–NEG candidate peak pairs matching:
        RT within 12 sec (0.2 min)
        Neutral mass within 5 ppm or 5 mDa
    Returns a DataFrame listing all candidate pairings.
    """
    rows = []
    rt_tol = 12 / 60.0  # 12 sec in minutes

    # Pre-select required columns to avoid KeyErrors later
    for col in ["RT (min)", "Neutral mass"]:
        if col not in df_pos.columns or col not in df_neg.columns:
            return pd.DataFrame()

    for idx_p, row_p in df_pos.iterrows():
        rt_p = row_p["RT (min)"]
        mass_p = row_p["Neutral mass"]

        if pd.isna(rt_p) or pd.isna(mass_p):
            continue

        # restrict NEG rows by RT first
        df_rt = df_neg[(df_neg["RT (min)"] - rt_p).abs() <= rt_tol]
        if df_rt.empty:
            continue

        for idx_n, row_n in df_rt.iterrows():
            mass_n = row_n["Neutral mass"]

            if _mass_difference_ok(mass_p, mass_n):
                rows.append({
                    "POS_idx": idx_p,
                    "NEG_idx": idx_n,
                    "POS_UniqueID": row_p.get("UniqueID", ""),
                    "NEG_UniqueID": row_n.get("UniqueID", ""),
                    "POS_Annotation": row_p.get("Annotation", ""),
                    "NEG_Annotation": row_n.get("Annotation", ""),
                    "POS_RT": rt_p,
                    "NEG_RT": row_n["RT (min)"],
                    "POS_mass": mass_p,
                    "NEG_mass": mass_n,
                    "RT_diff": abs(rt_p - row_n["RT (min)"]),
                    "Mass_diff_mDa": abs(mass_p - mass_n) * 1000.0,
                })
    return pd.DataFrame(rows)


def _score_row(row: pd.Series) -> Tuple[float, Dict[str, float]]:
    """
    Compute polarity score using normalized components:

        msms_norm   in [0,1]   (higher better)
        msigma_norm in [0,1]   (1 is best, 0 worst)
        dmz_norm    in [0,1]   (1 is best, 0 worst)
        rsd_norm    in [0,1]   (1 is best, 0 worst)

    Raw ranges (approx):
        MS/MS score:   0–1000
        mSigma:        0–100
        Δm/z (mDa):    0–3
        RSD QCs (%):   0–30

    Score = 0.4*msms_norm + 0.2*msigma_norm + 0.2*dmz_norm + 0.2*rsd_norm
    """
    # --- raw values ---
    msms = pd.to_numeric(row.get("MS/MS score", np.nan), errors="coerce")
    msigma = pd.to_numeric(row.get("mSigma", np.nan), errors="coerce")
    dmz = pd.to_numeric(row.get("Δm/z (mDa)", np.nan), errors="coerce")
    rsd_qc = _parse_percent(pd.Series([row.get("RSD QCs (%)", "")])).iloc[0]

    # default "bad" when missing
    if np.isnan(msms):
        msms = 0.0
    if np.isnan(msigma):
        msigma = 100.0
    if np.isnan(dmz):
        dmz = 3.0
    if np.isnan(rsd_qc):
        rsd_qc = 30.0

    # --- clamp to expected ranges ---
    msms_clamp = max(0.0, min(msms, 1000.0))
    msigma_clamp = max(0.0, min(msigma, 250.0))
    dmz_clamp = max(0.0, min(abs(dmz), 5.0))
    rsd_clamp = max(0.0, min(rsd_qc, 30.0))

    # --- normalize to [0,1]; 1 is best ---
    msms_norm = msms_clamp / 1000.0                  # higher better
    msigma_norm = 1.0 - (msigma_clamp / 250.0)       # lower is better
    dmz_norm = 1.0 - (dmz_clamp / 5.0)               # lower is better
    rsd_norm = 1.0 - (rsd_clamp / 30.0)              # lower is better

    # safety: don’t let any slip outside [0,1]
    msms_norm = float(max(0.0, min(msms_norm, 1.0)))
    msigma_norm = float(max(0.0, min(msigma_norm, 1.0)))
    dmz_norm = float(max(0.0, min(dmz_norm, 1.0)))
    rsd_norm = float(max(0.0, min(rsd_norm, 1.0)))

    # --- composite score ---
    score = (
        0.4 * msms_norm +
        0.2 * msigma_norm +
        0.2 * dmz_norm +
        0.2 * rsd_norm
    )

    components = {
        "msms_raw": float(msms),
        "msigma_raw": float(msigma),
        "dmz_raw": float(dmz),
        "rsd_qc_raw": float(rsd_qc),
        "msms_norm": msms_norm,
        "msigma_norm": msigma_norm,
        "dmz_norm": dmz_norm,
        "rsd_norm": rsd_norm,
        "Score": float(score),
    }
    return float(score), components


def _annotations_compatible(
    row_pos: pd.Series,
    row_neg: pd.Series
) -> Tuple[bool, float, float]:
    """
    Determine if POS and NEG annotations are compatible for merging.

    Returns:
        compatible (bool),
        pos_multiplier (float),
        neg_multiplier (float)

    Rules:
      1) If Annotation identical → compatible, pos_mult = neg_mult = 1.0.
      2) If same Lipid Class AND same Headgroup AND one or both are
         molecular species (contains "_" or "/") → compatible.
         Each molecular-species side gets a 10% boost:
            multiplier = 1.1 for that polarity.
      3) Otherwise → NOT compatible (multipliers ignored).
    """
    ann_pos = str(row_pos.get("Annotation", "")).strip()
    ann_neg = str(row_neg.get("Annotation", "")).strip()

    # Case 1 — identical annotation
    if ann_pos == ann_neg and ann_pos != "":
        return True, 1.0, 1.0

    # Extract class and headgroup
    class_pos = str(row_pos.get("Lipid Class", "")).strip()
    class_neg = str(row_neg.get("Lipid Class", "")).strip()
    head_pos = str(row_pos.get("Headgroup", "")).strip()
    head_neg = str(row_neg.get("Headgroup", "")).strip()

    same_class = (class_pos != "" and class_pos == class_neg)
    same_head = (head_pos != "" and head_pos == head_neg)

    # Detect molecular species (contains "_" or "/")
    def is_molecular_species(a: str) -> bool:
        return ("_" in a) or ("/" in a)

    if same_class and same_head:
        pos_ms = is_molecular_species(ann_pos)
        neg_ms = is_molecular_species(ann_neg)
        if pos_ms or neg_ms:
            pos_mult = 1.1 if pos_ms else 1.0
            neg_mult = 1.1 if neg_ms else 1.0
            return True, pos_mult, neg_mult

    # Everything else: not compatible
    return False, 1.0, 1.0


# -------------------------------------------------------------------------
# Simple concatenation
# -------------------------------------------------------------------------

def merge_simple(root: Path):
    """
    Simple merge:
    - Add P_ / N_ prefix to UniqueID ONLY
    - Do NOT prefix sample columns (they already have no polarity prefixes)
    - Align sample columns by name and fill missing with NaN
    """
    df_pos_ann, df_neg_ann, df_pos_unk, df_neg_unk = _load_final_files(root)
    debug_dir = root / "debug_merging"
    debug_dir.mkdir(exist_ok=True)

    annotated_frames: List[pd.DataFrame] = []

    if df_pos_ann is not None:
        df = df_pos_ann.copy()
        df["UniqueID"] = "P_" + df["UniqueID"].astype(str)
        annotated_frames.append(df)

    if df_neg_ann is not None:
        df = df_neg_ann.copy()
        df["UniqueID"] = "N_" + df["UniqueID"].astype(str)
        annotated_frames.append(df)

    # Align columns across POS + NEG
    if annotated_frames:
        all_cols = sorted({c for df in annotated_frames for c in df.columns})
        aligned = [df.reindex(columns=all_cols) for df in annotated_frames]
        df_ann = pd.concat(aligned, ignore_index=True)
    else:
        df_ann = pd.DataFrame(columns=BASE_ANNOTATED_COLS)

    # Reorder annotated: metadata first, then sample columns
    sample_cols_ann = sorted(_detect_sample_columns(df_ann, BASE_ANNOTATED_COLS))
    final_cols_ann = [c for c in BASE_ANNOTATED_COLS if c in df_ann.columns] + \
        [c for c in sample_cols_ann if c in df_ann.columns]
    df_ann = df_ann.reindex(columns=final_cols_ann)

    # Unknowns
    unknown_frames: List[pd.DataFrame] = []
    if df_pos_unk is not None:
        df = df_pos_unk.copy()
        df["UniqueID"] = "P_" + df["UniqueID"].astype(str)
        unknown_frames.append(df)
    if df_neg_unk is not None:
        df = df_neg_unk.copy()
        df["UniqueID"] = "N_" + df["UniqueID"].astype(str)
        unknown_frames.append(df)

    if unknown_frames:
        all_cols_unk = sorted({c for df in unknown_frames for c in df.columns})
        aligned_unk = [df.reindex(columns=all_cols_unk) for df in unknown_frames]
        df_unk = pd.concat(aligned_unk, ignore_index=True)

        # Reorder unknowns: metadata first, then sample columns
        sample_cols_unk = sorted(_detect_sample_columns(df_unk, BASE_UNKNOWN_COLS))
        final_cols_unk = [c for c in BASE_UNKNOWN_COLS if c in df_unk.columns] + \
            [c for c in sample_cols_unk if c in df_unk.columns]
        df_unk = df_unk.reindex(columns=final_cols_unk)
    else:
        df_unk = None

    # Save
    df_ann.to_csv(debug_dir / "Final_Annotated_simple_combination.csv",
                  index=False, encoding="utf-8-sig")

    if df_unk is not None:
        df_unk.to_csv(debug_dir / "Final_Unknowns_simple_combination.csv",
                      index=False, encoding="utf-8-sig")

    return df_ann, df_unk

def _strip_polarity_from_sample_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sample columns by:
      1) Removing leading polarity prefixes: P_ or N_
      2) Removing run/injection suffixes: anything after _P1- or _P2-
    Applies only to sample intensity columns (not metadata).
    """
    new_cols = {}
    for col in df.columns:

        # Skip metadata columns
        if col in BASE_ANNOTATED_COLS:
            continue
        if col.startswith("RSD_"):
            continue

        new_name = col

        # 1) Remove leading P_ or N_
        if new_name.startswith(("P_", "N_")):
            new_name = new_name[2:]

        # 2) Remove anything after _P1- or _P2-
        if "_P1-" in new_name:
            new_name = new_name.split("_P1-")[0]
        if "_P2-" in new_name:
            new_name = new_name.split("_P2-")[0]

        new_cols[col] = new_name

    return df.rename(columns=new_cols)

def _load_pre_norm_files(root: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load pre-normalization annotated files from:
      POS/debug/Pos_3-Final_annotated_results_imputed.csv
      NEG/debug/Neg_3-Final_annotated_results_imputed.csv
    """
    pos_path = root / "POS" / "debug" / "Pos_3-Final_annotated_results_imputed.csv"
    neg_path = root / "NEG" / "debug" / "Neg_3-Final_annotated_results_imputed.csv"

    df_pos = pd.read_csv(pos_path) if pos_path.exists() else None
    df_neg = pd.read_csv(neg_path) if neg_path.exists() else None

    return df_pos, df_neg

def merge_pre_norm_simple(root: Path) -> pd.DataFrame:
    """
    Simple concatenation for the pre-normalization annotated results before merging.

    Inputs:
        <root>/POS/debug/Pos_3-Final_annotated_results_imputed.csv
        <root>/NEG/debug/Neg_3-Final_annotated_results_imputed.csv

    Steps:
        1) Load pre-normalization annotated files.
        2) Filter: keep rows with RSD QCs (%) < 30.
        3) Add P_ / N_ prefix to UniqueID.
        4) Align columns POS+NEG.
        5) Save:
            Final_Annotated_Before_Normalization_simple.csv
    """
    debug_dir = root / "debug_merging"
    debug_dir.mkdir(exist_ok=True)

    df_pos, df_neg = _load_pre_norm_files(root)

    # --- remove polarity prefixes from sample columns ---
    if df_pos is not None:
        df_pos = _strip_polarity_from_sample_columns(df_pos)
    if df_neg is not None:
        df_neg = _strip_polarity_from_sample_columns(df_neg)


    if df_pos is None and df_neg is None:
        print(f'No pre-normalized annotated file found.', flush = True)
        raise FileNotFoundError("No pre-normalization annotated files found.")

    # Filter RSD QCs < 30%
    if df_pos is not None:
        df_pos["RSD_QC_float"] = _parse_percent(df_pos["RSD QCs (%)"])
        df_pos = df_pos[df_pos["RSD_QC_float"] < 30].copy()

    if df_neg is not None:
        df_neg["RSD_QC_float"] = _parse_percent(df_neg["RSD QCs (%)"])
        df_neg = df_neg[df_neg["RSD_QC_float"] < 30].copy()

    if df_pos is not None:
        df_pos = df_pos.drop(columns=["RSD_QC_float"], errors="ignore")
    if df_neg is not None:
        df_neg = df_neg.drop(columns=["RSD_QC_float"], errors="ignore")

    frames: List[pd.DataFrame] = []

    if df_pos is not None and len(df_pos) > 0:
        df = df_pos.copy()
        df["UniqueID"] = "P_" + df["UniqueID"].astype(str)
        frames.append(df)

    if df_neg is not None and len(df_neg) > 0:
        df = df_neg.copy()
        df["UniqueID"] = "N_" + df["UniqueID"].astype(str)
        frames.append(df)

    if not frames:
        # After RSD filtering nothing left
        empty_df = pd.DataFrame(columns=BASE_ANNOTATED_COLS)
        empty_df.to_csv(root / "Final_Annotated_Before_Normalization_simple.csv",
                        index=False, encoding="utf-8-sig")
        return empty_df

    # Align column sets
    all_cols = sorted({c for df in frames for c in df.columns})
    aligned_frames = [df.reindex(columns=all_cols) for df in frames]

    df_out = pd.concat(aligned_frames, ignore_index=True)

    sample_cols = sorted(_detect_sample_columns(df_out, BASE_ANNOTATED_COLS))
    final_cols = [c for c in BASE_ANNOTATED_COLS if c in df_out.columns] + sample_cols
    df_out = df_out.reindex(columns=final_cols)

        # Drop unwanted columns before saving
    cols_to_drop = [
        "# of modifications",
        "Average Intensity (all samples)",
        "Carbons / double bond equivalent ratio",
        "Double bonds in fatty acyl 1",
        "Double bonds in fatty acyl 2",
        "Double bonds in fatty acyl 3",
        "Double bonds in fatty acyl 4",
        "Internal Standard",
        "MS/MS available?",
        "Maximum Intensity (all samples)",
        "Metaboscape Annotation Status",
        "Minimum Intensity (all samples)",
        "Number of carbons in fatty acyl 1",
        "Number of carbons in fatty acyl 2",
        "Number of carbons in fatty acyl 3",
        "Number of carbons in fatty acyl 4",
        "Oxidized?",
    ]
    df_out = df_out.drop(columns=[c for c in cols_to_drop if c in df_out.columns], errors="ignore")

    # NEW: drop rows with missing Annotation
    df_out = df_out[df_out["Annotation"].notna() & (df_out["Annotation"].astype(str).str.strip() != "")]

    # Save output
    out_path = debug_dir / "Final_Annotated_Before_Normalization_simple.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f'Pre-normalized concatenated annotated file: saved to {out_path}.', flush = True)

    return df_out

# -------------------------------------------------------------------------
# Best-polarity selection for annotated lipids (iterative)
# -------------------------------------------------------------------------

def merge_best_polarity(root: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Best-polarity merge for annotated lipids, with iterative refinement (B1).

    Steps:
      1) Load Pos_Final_Annotated and Neg_Final_Annotated.
      2) Iteratively:
           a) Identify candidate POS–NEG pairs among *currently alive* features:
                RT difference ≤ 12 seconds (0.2 min)
                Neutral mass difference ≤ 5 ppm OR ≤ 5 mDa
              Then filter by annotation compatibility:
                - identical Annotation → allowed.
                - same Lipid Class + Headgroup AND one/both molecular species
                  (contains "_" or "/") → allowed; molecular-species sides get
                  10% score boost.
                - otherwise → pair discarded.
              All such pairs (per iteration) go to candidate_pairs.csv.
           b) For each pair, compute scores (raw + normalized + boosted) and
              choose a winner (POS or NEG). All scores across all iterations
              go to scores_detailed.csv.
           c) For each feature:
                - If it never appears in any pair in this iteration → kept.
                - If it appears and *never* wins → removed in this iteration.
                - If it appears and wins at least once → kept.
              Removed rows (with iteration) go to removed_features.csv.
           d) Repeat steps (a–c) with the reduced set of alive features until
              a full iteration removes no additional features (stable).
      3) Prefix remaining UniqueIDs with P_ / N_ and concatenate POS + NEG.
      4) Unknowns are merged by simple concatenation with P_ / N_ prefixes.

    Returns:
        df_ann_best  – merged annotated table
        df_unk_best  – merged unknowns (or None if no unknown files found)
    """
    df_pos_ann, df_neg_ann, df_pos_unk, df_neg_unk = _load_final_files(root)
    debug_dir = root / "debug_merging"
    debug_dir.mkdir(exist_ok=True)

    print(f'Annotated file loaded for merging.', flush = True)

    # Case 1: only one polarity exists -> degenerate to "simple" behavior for annotated
    if df_pos_ann is None or df_neg_ann is None:
        print(f'One polarity is missing. Merging cannot be completed.', flush = True)
        annotated_frames: List[pd.DataFrame] = []
        if df_pos_ann is not None:
            df = df_pos_ann.copy()
            df["UniqueID"] = "P_" + df["UniqueID"].astype(str)
            annotated_frames.append(df)
            print(f'Positive is present.', flush = True)
        if df_neg_ann is not None:
            df = df_neg_ann.copy()
            df["UniqueID"] = "N_" + df["UniqueID"].astype(str)
            annotated_frames.append(df)
            print(f'Negative is present.', flush = True)

        if annotated_frames:
            all_cols = sorted({c for df in annotated_frames for c in df.columns})
            aligned = [df.reindex(columns=all_cols) for df in annotated_frames]
            df_out = pd.concat(aligned, ignore_index=True)
        else:
            df_out = pd.DataFrame(columns=BASE_ANNOTATED_COLS)

        sample_cols_ann = sorted(_detect_sample_columns(df_out, BASE_ANNOTATED_COLS))
        final_cols_ann = [c for c in BASE_ANNOTATED_COLS if c in df_out.columns] + \
            [c for c in sample_cols_ann if c in df_out.columns]
        df_ann_best = df_out.reindex(columns=final_cols_ann)

    else:
        # Case 2: both polarities exist -> full iterative pairing + scoring
        print(f'Both polarities are present for merging.', flush = True)
        pos_alive_idx = set(df_pos_ann.index)
        neg_alive_idx = set(df_neg_ann.index)

        all_pairs_debug: List[pd.DataFrame] = []
        all_scores_debug: List[pd.DataFrame] = []
        removed_rows: List[pd.Series] = []

        iteration = 0
        while True:
            iteration += 1
            # Current alive features
            if not pos_alive_idx and not neg_alive_idx:
                break

            df_pos_alive = df_pos_ann.loc[sorted(pos_alive_idx)] if pos_alive_idx else df_pos_ann.iloc[0:0]
            df_neg_alive = df_neg_ann.loc[sorted(neg_alive_idx)] if neg_alive_idx else df_neg_ann.iloc[0:0]

            # 1) Find candidate pairs (RT + mass only) among alive
            pairs = _find_candidate_pairs(df_pos_alive, df_neg_alive)

            if not pairs.empty:
                # Apply annotation compatibility and attach multipliers
                filtered_rows = []
                for _, p in pairs.iterrows():
                    pos = df_pos_ann.loc[p["POS_idx"]]
                    neg = df_neg_ann.loc[p["NEG_idx"]]

                    compatible, pos_mult, neg_mult = _annotations_compatible(pos, neg)
                    if compatible:
                        r = p.to_dict()
                        r["POS_multiplier"] = pos_mult
                        r["NEG_multiplier"] = neg_mult
                        r["Iteration"] = iteration
                        filtered_rows.append(r)

                pairs = pd.DataFrame(filtered_rows)

            # Nothing compatible in this iteration: stop refinement
            if pairs.empty:
                # no more pairs to compete -> all remaining alive are kept
                break

            all_pairs_debug.append(pairs.copy())

            # 2) Score each candidate pair with detailed score components
            score_records: List[Dict] = []
            for _, pair in pairs.iterrows():
                pos_idx = int(pair["POS_idx"])
                neg_idx = int(pair["NEG_idx"])

                pos = df_pos_ann.loc[pos_idx]
                neg = df_neg_ann.loc[neg_idx]

                pos_mult = float(pair.get("POS_multiplier", 1.0))
                neg_mult = float(pair.get("NEG_multiplier", 1.0))

                score_pos_raw, comp_pos = _score_row(pos)
                score_neg_raw, comp_neg = _score_row(neg)

                score_pos_boost = score_pos_raw * pos_mult
                score_neg_boost = score_neg_raw * neg_mult

                winner = "POS" if score_pos_boost >= score_neg_boost else "NEG"

                rec = pair.to_dict()
                rec.update({
                    "Iteration": iteration,

                    # POS raw metrics
                    "POS_MS/MS_raw": comp_pos["msms_raw"],
                    "POS_mSigma_raw": comp_pos["msigma_raw"],
                    "POS_dMz_raw(mDa)": comp_pos["dmz_raw"],
                    "POS_RSD_QCs_raw(%)": comp_pos["rsd_qc_raw"],
                    # POS normalized
                    "POS_MS/MS_norm": comp_pos["msms_norm"],
                    "POS_mSigma_norm": comp_pos["msigma_norm"],
                    "POS_dMz_norm": comp_pos["dmz_norm"],
                    "POS_RSD_QCs_norm": comp_pos["rsd_norm"],
                    # POS scores
                    "POS_Score_raw": score_pos_raw,
                    "POS_Score_multiplier": pos_mult,
                    "POS_Score_boosted": score_pos_boost,

                    # NEG raw metrics
                    "NEG_MS/MS_raw": comp_neg["msms_raw"],
                    "NEG_mSigma_raw": comp_neg["msigma_raw"],
                    "NEG_dMz_raw(mDa)": comp_neg["dmz_raw"],
                    "NEG_RSD_QCs_raw(%)": comp_neg["rsd_qc_raw"],
                    # NEG normalized
                    "NEG_MS/MS_norm": comp_neg["msms_norm"],
                    "NEG_mSigma_norm": comp_neg["msigma_norm"],
                    "NEG_dMz_norm": comp_neg["dmz_norm"],
                    "NEG_RSD_QCs_norm": comp_neg["rsd_norm"],
                    # NEG scores
                    "NEG_Score_raw": score_neg_raw,
                    "NEG_Score_multiplier": neg_mult,
                    "NEG_Score_boosted": score_neg_boost,

                    # Decision
                    "Winner": winner,
                })

                score_records.append(rec)

            scores_df = pd.DataFrame(score_records)
            all_scores_debug.append(scores_df.copy())

            # 3) Decide keep/remove based on this iteration’s pairs
            pos_votes: Dict[int, List[bool]] = {}
            neg_votes: Dict[int, List[bool]] = {}

            for _, row in scores_df.iterrows():
                pos_idx = int(row["POS_idx"])
                neg_idx = int(row["NEG_idx"])
                winner = row["Winner"]

                pos_votes.setdefault(pos_idx, []).append(winner == "POS")
                neg_votes.setdefault(neg_idx, []).append(winner == "NEG")

            pos_alive_before = pos_alive_idx.copy()
            neg_alive_before = neg_alive_idx.copy()

            pos_keep_iter: set[int] = set()
            neg_keep_iter: set[int] = set()
            pos_remove_iter: set[int] = set()
            neg_remove_iter: set[int] = set()

            # POS: within currently alive
            for idx in pos_alive_before:
                if idx not in pos_votes:
                    # did not appear in any pair this iteration -> keep
                    pos_keep_iter.add(idx)
                else:
                    if any(pos_votes[idx]):
                        pos_keep_iter.add(idx)
                    else:
                        pos_remove_iter.add(idx)

            # NEG: within currently alive
            for idx in neg_alive_before:
                if idx not in neg_votes:
                    neg_keep_iter.add(idx)
                else:
                    if any(neg_votes[idx]):
                        neg_keep_iter.add(idx)
                    else:
                        neg_remove_iter.add(idx)

            # Update alive sets
            pos_alive_idx = pos_keep_iter
            neg_alive_idx = neg_keep_iter

            # Record removed rows for debug
            for idx in sorted(pos_remove_iter):
                r = df_pos_ann.loc[idx].copy()
                r["PolarityRemoved"] = "POS"
                r["IterationRemoved"] = iteration
                removed_rows.append(r)
            for idx in sorted(neg_remove_iter):
                r = df_neg_ann.loc[idx].copy()
                r["PolarityRemoved"] = "NEG"
                r["IterationRemoved"] = iteration
                removed_rows.append(r)

            # Check convergence: if nothing removed this iteration, stop
            if not pos_remove_iter and not neg_remove_iter:
                break

        # After iterative refinement, keep all remaining alive features
        df_pos_final = df_pos_ann.loc[sorted(pos_alive_idx)].copy() if pos_alive_idx else df_pos_ann.iloc[0:0].copy()
        df_neg_final = df_neg_ann.loc[sorted(neg_alive_idx)].copy() if neg_alive_idx else df_neg_ann.iloc[0:0].copy()

        # Write debug files
        if all_pairs_debug:
            pairs_all = pd.concat(all_pairs_debug, ignore_index=True)
            pairs_all.to_csv(debug_dir / "candidate_pairs.csv",
                             index=False, encoding="utf-8-sig")

        if all_scores_debug:
            scores_all = pd.concat(all_scores_debug, ignore_index=True)
            scores_all.to_csv(debug_dir / "scores_detailed.csv",
                              index=False, encoding="utf-8-sig")

        if removed_rows:
            df_removed = pd.DataFrame(removed_rows)
            df_removed.to_csv(debug_dir / "removed_features.csv",
                              index=False, encoding="utf-8-sig")

        # 4) Prefix UniqueIDs and concatenate
        df_pos_final["UniqueID"] = "P_" + df_pos_final["UniqueID"].astype(str)
        df_neg_final["UniqueID"] = "N_" + df_neg_final["UniqueID"].astype(str)
        df_out = pd.concat([df_pos_final, df_neg_final], ignore_index=True)

        # Final reordered annotated table
        sample_cols_ann = sorted(_detect_sample_columns(df_out, BASE_ANNOTATED_COLS))
        final_cols_ann = [c for c in BASE_ANNOTATED_COLS if c in df_out.columns] + \
            [c for c in sample_cols_ann if c in df_out.columns]
        df_ann_best = df_out.reindex(columns=final_cols_ann)

    # --- Unknowns: always simple concatenation with P_/N_ prefixes ---

    unknown_frames: List[pd.DataFrame] = []
    if df_pos_unk is not None:
        df = df_pos_unk.copy()
        df["UniqueID"] = "P_" + df["UniqueID"].astype(str)
        unknown_frames.append(df)

    if df_neg_unk is not None:
        df = df_neg_unk.copy()
        df["UniqueID"] = "N_" + df["UniqueID"].astype(str)
        unknown_frames.append(df)

    if unknown_frames:
        all_cols_unk = sorted({c for df in unknown_frames for c in df.columns})
        df_unk_best = pd.concat(
            [df.reindex(columns=all_cols_unk) for df in unknown_frames],
            ignore_index=True
        )
        sample_cols_unk = sorted(_detect_sample_columns(df_unk_best, BASE_UNKNOWN_COLS))
        final_cols_unk = [c for c in BASE_UNKNOWN_COLS if c in df_unk_best.columns] + \
            [c for c in sample_cols_unk if c in df_unk_best.columns]
        df_unk_best = df_unk_best.reindex(columns=final_cols_unk)
    else:
        df_unk_best = None

    # --- Save ---
    out_ann = root / "Final_Annotated.csv"
    df_ann_best.to_csv(out_ann, index=False, encoding="utf-8-sig")

    if df_unk_best is not None:
        out_unk = root / "Final_Unknowns.csv"
        df_unk_best.to_csv(out_unk, index=False, encoding="utf-8-sig")

    return df_ann_best, df_unk_best

def _merge_best_from_tables(df_pos_ann, df_neg_ann):
    """
    Internal helper: run the same iterative best-polarity algorithm
    used in merge_best_polarity(), but operate entirely in memory
    on the supplied POS and NEG dataframes.

    Returns:
        df_ann_best  – merged annotated table
    """

    # Reuse your existing iterative logic — copy/paste from merge_best_polarity,
    # but replace all calls to _load_final_files() with these two local tables.

    pos_alive_idx = set(df_pos_ann.index)
    neg_alive_idx = set(df_neg_ann.index)

    iteration = 0
    while True:
        iteration += 1

        df_pos_alive = df_pos_ann.loc[sorted(pos_alive_idx)] if pos_alive_idx else df_pos_ann.iloc[0:0]
        df_neg_alive = df_neg_ann.loc[sorted(neg_alive_idx)] if neg_alive_idx else df_neg_ann.iloc[0:0]

        pairs = _find_candidate_pairs(df_pos_alive, df_neg_alive)

        # annotation compatibility
        filtered = []
        for _, r in pairs.iterrows():
            pos = df_pos_ann.loc[r["POS_idx"]]
            neg = df_neg_ann.loc[r["NEG_idx"]]
            ok, pos_mult, neg_mult = _annotations_compatible(pos, neg)
            if ok:
                d = r.to_dict()
                d["POS_multiplier"] = pos_mult
                d["NEG_multiplier"] = neg_mult
                filtered.append(d)
        pairs = pd.DataFrame(filtered)

        if pairs.empty:
            break

        # scoring
        pos_votes = {}
        neg_votes = {}

        for _, r in pairs.iterrows():
            p = df_pos_ann.loc[int(r["POS_idx"])]
            n = df_neg_ann.loc[int(r["NEG_idx"])]

            s_pos_raw, comp_pos = _score_row(p)
            s_neg_raw, comp_neg = _score_row(n)

            s_pos = s_pos_raw * float(r["POS_multiplier"])
            s_neg = s_neg_raw * float(r["NEG_multiplier"])

            winner = "POS" if s_pos >= s_neg else "NEG"

            pos_votes.setdefault(int(r["POS_idx"]), []).append(winner == "POS")
            neg_votes.setdefault(int(r["NEG_idx"]), []).append(winner == "NEG")

        # elimination
        pos_keep = set()
        neg_keep = set()

        for idx in pos_alive_idx:
            if idx not in pos_votes:
                pos_keep.add(idx)
            elif any(pos_votes[idx]):
                pos_keep.add(idx)

        for idx in neg_alive_idx:
            if idx not in neg_votes:
                neg_keep.add(idx)
            elif any(neg_votes[idx]):
                neg_keep.add(idx)

        if len(pos_keep) == len(pos_alive_idx) and len(neg_keep) == len(neg_alive_idx):
            break

        pos_alive_idx = pos_keep
        neg_alive_idx = neg_keep

    # final output assembly
    df_pos_final = df_pos_ann.loc[sorted(pos_alive_idx)].copy()
    df_neg_final = df_neg_ann.loc[sorted(neg_alive_idx)].copy()

    df_pos_final["UniqueID"] = "P_" + df_pos_final["UniqueID"].astype(str)
    df_neg_final["UniqueID"] = "N_" + df_neg_final["UniqueID"].astype(str)

    df_out = pd.concat([df_pos_final, df_neg_final], ignore_index=True)

    sample_cols = sorted(_detect_sample_columns(df_out, BASE_ANNOTATED_COLS))
    final_cols = [c for c in BASE_ANNOTATED_COLS if c in df_out.columns] + sample_cols
    return df_out.reindex(columns=final_cols)

def merge_pre_norm_best_polarity(root: Path) -> pd.DataFrame:
    df_pos, df_neg = _load_pre_norm_files(root)

    # --- remove polarity prefixes from sample columns ---
    if df_pos is not None:
        df_pos = _strip_polarity_from_sample_columns(df_pos)
    if df_neg is not None:
        df_neg = _strip_polarity_from_sample_columns(df_neg)


    if df_pos is None and df_neg is None:
        raise FileNotFoundError("No pre-normalization annotated files found.")

    # RSD QC filtering
    if df_pos is not None:
        rsd_col = [c for c in df_pos.columns if "RSD QC" in c][0]
        df_pos["__rsd__"] = _parse_percent(df_pos[rsd_col])
        df_pos = df_pos[df_pos["__rsd__"] < 30].drop(columns="__rsd__", errors="ignore")

    if df_neg is not None:
        rsd_col = [c for c in df_neg.columns if "RSD QC" in c][0]
        df_neg["__rsd__"] = _parse_percent(df_neg[rsd_col])
        df_neg = df_neg[df_neg["__rsd__"] < 30].drop(columns="__rsd__", errors="ignore")

    # handle missing polarity
    if df_pos is None or df_pos.empty:
        df_neg = df_neg.copy()
        df_neg["UniqueID"] = "N_" + df_neg["UniqueID"].astype(str)
        df_neg.to_csv(root / "Final_Annotated_Before_Normalization.csv",
                      index=False, encoding="utf-8-sig")
        return df_neg

    if df_neg is None or df_neg.empty:
        df_pos = df_pos.copy()
        df_pos["UniqueID"] = "P_" + df_pos["UniqueID"].astype(str)
        df_pos.to_csv(root / "Final_Annotated_Before_Normalization.csv",
                      index=False, encoding="utf-8-sig")
        return df_pos

    # --- save ---
    df_pre = _merge_best_from_tables(df_pos.copy(), df_neg.copy())

    # Drop unwanted columns before saving
    cols_to_drop = [
        "# of modifications",
        "Average Intensity (all samples)",
        "Carbons / double bond equivalent ratio",
        "Double bonds in fatty acyl 1",
        "Double bonds in fatty acyl 2",
        "Double bonds in fatty acyl 3",
        "Double bonds in fatty acyl 4",
        "Internal Standard",
        "MS/MS available?",
        "Maximum Intensity (all samples)",
        "Metaboscape Annotation Status",
        "Minimum Intensity (all samples)",
        "Number of carbons in fatty acyl 1",
        "Number of carbons in fatty acyl 2",
        "Number of carbons in fatty acyl 3",
        "Number of carbons in fatty acyl 4",
        "Oxidized?",
    ]
    df_pre = df_pre.drop(columns=[c for c in cols_to_drop if c in df_pre.columns], errors="ignore")

    # NEW: drop rows with missing Annotation
    df_pre = df_pre[df_pre["Annotation"].notna() & (df_pre["Annotation"].astype(str).str.strip() != "")]

    out_path = root / "Final_Annotated_Before_Normalization.csv"
    df_pre.to_csv(out_path, index=False, encoding="utf-8-sig")


    print(f"\n[MERGE] Saved pre-normalization best-polarity to {out_path}", flush = True)

    return df_pre


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Polarity merging for normalized and pre-normalized results."
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root results folder containing 'POS' and 'NEG' subfolders."
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "best", "both"],
        default="both",
        help=(
            "simple = merge normalized + pre-norm by simple concatenation; "
            "best = best-polarity merge for pre-norm only; "
            "both = run simple (normalized + pre-norm) + best (pre-norm)."
        ),
    )

    args = parser.parse_args()
    root = Path(args.root)

    # --------------------------------------------------
    # SIMPLE MERGE: normalized + pre-normalized
    # --------------------------------------------------
    if args.mode in ("simple", "both"):
        print("[MERGE] Simple merge: normalized annotated/unknowns", flush=True)
        df_ann_simple, df_unk_simple = merge_simple(root)
        print(f"[MERGE] Normalized simple annotated: {df_ann_simple.shape}", flush=True)
        if df_unk_simple is not None:
            print(f"[MERGE] Normalized simple unknowns: {df_unk_simple.shape}", flush=True)
        else:
            print("[MERGE] Normalized simple unknowns: none", flush=True)

        print("[MERGE] Simple merge: pre-normalization annotated", flush=True)
        df_pre_simple = merge_pre_norm_simple(root)
        print(f"[MERGE] Pre-normalization simple annotated: {df_pre_simple.shape}", flush=True)

    # --------------------------------------------------
    # BEST-POLARITY: **pre-normalization only**
    # --------------------------------------------------
    if args.mode in ("best", "both"):
        print("[MERGE] Best-polarity merge: pre-normalization annotated (and unknowns if present)", flush=True)
        df_pre_best = merge_pre_norm_best_polarity(root)
        print(f"[MERGE] Pre-normalization best annotated: {df_pre_best.shape}\n", flush=True)


if __name__ == "__main__":
    main()
