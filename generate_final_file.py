# -------------------------------------------------------------------------
# Detects the most recent processed result files (LOESS, median, normalized, or filtered)
# and generates:
#   - Final_Annotated.csv
#   - Final_Unknowns.csv
# inside the main results folder.
# -------------------------------------------------------------------------

import pandas as pd
import re
from pathlib import Path

def clean_sample_name(name: str) -> str:
    """Simplify sample names by removing polarity and run identifiers."""
    if not isinstance(name, str):
        return name
    cleaned = name
    cleaned = re.sub(r"\[?POS\]?|\[?NEG\]?", "", cleaned, flags=re.IGNORECASE)  # remove [POS]/[NEG]
    cleaned = re.sub(r"^(P_|N_)", "", cleaned)  # remove leading polarity letters
    cleaned = re.split(r"_P[12]", cleaned)[0]   # truncate after _P1/_P2
    # cleaned = re.sub(r"_[0-9]+(_[0-9]+)*$", "", cleaned)  # remove trailing run IDs like _1_490
    # cleaned = re.sub(r"[_\-]+$", "", cleaned)
    # cleaned = re.sub(r"[_\-]{2,}", "_", cleaned)
    return cleaned.strip("_- ")

def create_final_outputs(results_folder):
    results_folder = Path(results_folder)
    debug_folder = results_folder / "debug"

    print(f'\nGenerating final files...\n')

    # --- Priority order for annotated file detection ---
    annotated_candidates = [
        ("9-Final_annotated_results_loess_normalized.csv", "LOESS normalization"),
        ("7-Final_annotated_median_normalized.csv", "Median normalization"),
        ("5-Final_annotated_results_normalized.csv", "Basic normalization"),
        ("4-Final_annotated_results_imputed_filtered.csv", "Imputed filtered only")
    ]

    # --- Priority order for unknowns file detection ---
    unknowns_candidates = [
        "8-Final_unknowns_median_normalized.csv",
        "7-Final_unknowns.csv"
    ]

    annotated_file = None
    unknowns_file = None
    method_used = None

    # Detect annotated file
    for name, label in annotated_candidates:
        path = debug_folder / name
        if path.exists():
            annotated_file = path
            method_used = label
            break

    # Detect unknowns file
    for name in unknowns_candidates:
        path = debug_folder / name
        if path.exists():
            unknowns_file = path
            break

    if annotated_file is None:
        raise FileNotFoundError(
            "No annotated dataset found in results/debug. "
            "Expected one of:\n" + "\n".join(n for n, _ in annotated_candidates)
        )

    print(f"✅ Using annotated dataset: {annotated_file.name} ({method_used})", flush=True)
    if unknowns_file:
        print(f"✅ Using unknowns dataset: {unknowns_file.name}", flush=True)
    else:
        print("⚠️ No dedicated unknowns file found. Only annotated file will be processed.", flush=True)

    # --- Load annotated data ---
    df_ann = pd.read_csv(annotated_file, low_memory=False)

    # --- Exclude internal standards (Annotation Type == "IS") ---
    if "Annotation Type" in df_ann.columns:
        initial_count = len(df_ann)
        df_ann = df_ann[~df_ann["Annotation Type"].astype(str).str.upper().eq("IS")].copy()
        removed_count = initial_count - len(df_ann)
        print(f"[INFO] Excluded {removed_count} internal standards (Annotation Type = 'IS') from annotated results.")
    else:
        print("[WARNING] 'Annotation Type' column not found; no IS exclusion applied.")

    # --- Define desired annotated columns ---
    base_cols = [
        "UniqueID", "RT (min)", "m/z", "Polarity",
        "Annotation", "Annotation Type", "Annotation Source",
        "Headgroup", "Lipid Class",
        "Δm/z (mDa)", "Δm/z (ppm)", "MS/MS score", "Annotation tier", "mSigma",
        "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]",
        "Molecular Formula", "Plasmenyl?",
        "Number of carbons in fatty acyls", "Double bond equivalents",
        "Chain type", "PUFA?", "Modifications", "# of modifications",
        "Oxidized?", "Carbons / double bond equivalent ratio",
        "RSD QCs (%)", "RSD Samples (%)",
    ]

    # Detect sample-specific RSD columns
    rsd_cols = [c for c in df_ann.columns if c.startswith("RSD_")]
    rsd_cols = [c for c in rsd_cols if c not in base_cols]

    # Detect all sample columns (intensities)
    sample_cols = [
        c for c in df_ann.columns
        if c.startswith("[POS") or c.startswith("[NEG") or c.startswith("P_") or c.startswith("N_")
    ]
    
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
            # Insert all sample-group RSDs right after this point
            final_cols.extend([c for c in rsd_cols if c not in final_cols])

    # Add sample columns at the end
    final_cols.extend([c for c in sample_cols if c not in final_cols])

    # Keep only columns that exist
    final_cols = [c for c in final_cols if c in df_ann.columns]

    # Apply final column filtering
    df_ann = df_ann[final_cols]

    # --- Save final outputs ---
    results_folder.mkdir(parents=True, exist_ok=True)
    annotated_path = results_folder / "Final_Annotated.csv"
    unknowns_path = results_folder / "Final_Unknowns.csv"

    df_ann.to_csv(annotated_path, index=False, encoding="utf-8-sig")

    if unknowns_file:
        df_unk = pd.read_csv(unknowns_file, low_memory=False)

        # --- Define desired columns for unknowns ---
        base_cols_unk = [
            "UniqueID", "RT (min)", "m/z", "Polarity",
            "RSD QCs (%)", "RSD Samples (%)",
        ]

        # Detect sample-specific RSD columns
        rsd_cols_unk = [c for c in df_unk.columns if c.startswith("RSD_")]
        rsd_cols_unk = [c for c in rsd_cols_unk if c not in base_cols_unk]

        # Detect all sample columns (intensities)
        sample_cols_unk = [
            c for c in df_unk.columns
            if c.startswith("[POS") or c.startswith("[NEG") or c.startswith("P_") or c.startswith("N_")
        ]
        
        # --- Clean up sample column names (remove polarity, replicate suffixes, etc.) ---
        rename_map_unk = {col: clean_sample_name(col) for col in sample_cols_unk}
        df_unk.rename(columns=rename_map_unk, inplace=True)
        sample_cols_unk = list(rename_map_unk.values())

        # Build final ordered column list (keep core + RSD + sample columns)
        final_cols_unk = []
        for col in base_cols_unk:
            if col in df_unk.columns:
                final_cols_unk.append(col)
            if col == "RSD Samples (%)":
                # Insert all sample-group RSDs right after this point
                final_cols_unk.extend([c for c in rsd_cols_unk if c not in final_cols_unk])

        # Add sample columns at the end
        final_cols_unk.extend([c for c in sample_cols_unk if c not in final_cols_unk])

        # Keep only columns that exist
        final_cols_unk = [c for c in final_cols_unk if c in df_unk.columns]

        # Apply column selection
        df_unk = df_unk[final_cols_unk]

        # Save file
        df_unk.to_csv(unknowns_path, index=False, encoding="utf-8-sig")


    print(f"Saved: {annotated_path.name} ({len(df_ann)} rows)")
    if unknowns_file:
        print(f"Saved: {unknowns_path.name} ({len(df_unk)} rows)")
    else:
        print("No unknowns file was saved (not found in debug folder).")

    return annotated_path, unknowns_path if unknowns_file else None, method_used


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "results"
    create_final_outputs(folder)
