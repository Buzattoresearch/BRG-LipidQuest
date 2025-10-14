# ---------------------------------------------------------------------
# median_normalization.py
# Within-class median normalization (annotated) and
# global median normalization (unknowns), each evaluated separately.
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")

from normalization import evaluate_normalization_performance


def median_normalization(annotated_csv, unknowns_csv, sample_groups_csv, output_folder="results"):
    """
    Perform within-class median normalization for annotated features
    and global median normalization for unknowns.
    Evaluate each output separately.
    """
    print("\n[STEP] Starting median-based normalization...", flush=True)
    annotated_csv = Path(annotated_csv)
    unknowns_csv = Path(unknowns_csv)
    sample_groups_csv = Path(sample_groups_csv)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Load annotated and unknown datasets
    # --------------------------------------------------------------
    if not annotated_csv.exists():
        raise FileNotFoundError(f"Annotated file not found: {annotated_csv}", flush=True)
    df_ann = pd.read_csv(annotated_csv)
    print(f"[INFO] Loaded annotated dataset: {len(df_ann)} rows", flush=True)

    if not unknowns_csv.exists():
        print(f"[WARNING] Unknowns file not found: {unknowns_csv}", flush=True)
        df_unk = pd.DataFrame()
    else:
        df_unk = pd.read_csv(unknowns_csv)
        print(f"[INFO] Loaded unknown dataset: {len(df_unk)} rows", flush=True)

    # Identify sample columns
    sample_cols = [c for c in df_ann.columns if c.startswith("[POS") or c.startswith("[NEG]")]
    if not sample_cols:
        raise ValueError("No sample columns found in annotated file.", flush=True)

    # --------------------------------------------------------------
    # Step 1: Within-class median normalization (annotated)
    # --------------------------------------------------------------
    print("[STEP] Performing within-class median normalization for annotated features...", flush=True)
    if "Lipid Class" not in df_ann.columns:
        raise ValueError("Missing 'Lipid Class' column in annotated dataset.", flush=True)

    class_medians = (
        df_ann.groupby("Lipid Class")[sample_cols]
        .median()
        .replace(0, np.nan)
    )

    norm_ann = df_ann.copy()
    for idx, row in norm_ann.iterrows():
        lipid_class = row["Lipid Class"]
        if lipid_class not in class_medians.index:
            continue
        for col in sample_cols:
            denom = class_medians.loc[lipid_class, col]
            val = row[col]
            norm_ann.at[idx, col] = val / denom if pd.notna(val) and pd.notna(denom) and denom != 0 else np.nan

    # --------------------------------------------------------------
    # Step 1b: Recalculate RSDs using sample_groups.csv
    # --------------------------------------------------------------
    print("[STEP] Recalculating RSDs for annotated dataset...", flush=True)
    group_df = pd.read_csv(sample_groups_csv)
    group_map = {
        g.strip(): [s for s in group_df.loc[group_df["Group"] == g, "Sample"].tolist()]
        for g in group_df["Group"].dropna().unique()
    }
    qc_groups = [g for g in group_map if g.upper() == "QC"]
    qc_samples = [s for g in qc_groups for s in group_map[g]]
    non_qc_groups = [g for g in group_map if g.upper() != "QC"]

    qc_cols = [c for c in sample_cols if any(s == c or s in c for s in qc_samples)]
    non_qc_cols = [c for c in sample_cols if c not in qc_cols]

    rsd_qc_vals, rsd_sample_vals = [], []
    for _, row in norm_ann.iterrows():
        qc_vals = pd.to_numeric(row[qc_cols], errors="coerce").replace(0, np.nan).dropna()
        rsd_qc = (qc_vals.std(ddof=1) / qc_vals.mean() * 100) if len(qc_vals) > 1 else np.nan
        sample_vals = pd.to_numeric(row[non_qc_cols], errors="coerce").replace(0, np.nan).dropna()
        rsd_samples = (sample_vals.std(ddof=1) / sample_vals.mean() * 100) if len(sample_vals) > 1 else np.nan
        rsd_qc_vals.append(rsd_qc)
        rsd_sample_vals.append(rsd_samples)
    norm_ann["RSD QCs (%)"] = rsd_qc_vals
    norm_ann["RSD Samples (%)"] = rsd_sample_vals

    for g in non_qc_groups:
        cols = [c for c in sample_cols if any(s == c or s in c for s in group_map[g])]
        rsd_vals = []
        for _, row in norm_ann.iterrows():
            vals = pd.to_numeric(row[cols], errors="coerce").replace(0, np.nan).dropna()
            rsd = (vals.std(ddof=1) / vals.mean() * 100) if len(vals) > 1 else np.nan
            rsd_vals.append(rsd)
        norm_ann[f"RSD_{g} [%]"] = rsd_vals

    # --------------------------------------------------------------
    # Step 1c: Reorder columns before saving
    # --------------------------------------------------------------
    core_cols = [
        "UniqueID", "RT (min)", "m/z", "Neutral mass", "Adducts", "Polarity", "Internal Standard",
        "RSD QCs (%)", "RSD Samples (%)", "RSD_12x [%]", "RSD_15x [%]", "RSD_5x [%]",
        "RSD_8x [%]", "RSD_QC [%]", "MS/MS available?", "Annotation", "Annotation Type",
        "Metaboscape Annotation Status", "Annotation Source", "Headgroup", "Lipid Class",
        "Δm/z (mDa)", "Δm/z (ppm)", "MS/MS score", "Annotation tier", "mSigma", "Molecular Formula",
        "Plasmenyl?", "Number of carbons in fatty acyls", "Double bond equivalents",
        "Number of carbons in fatty acyl 1", "Double bonds in fatty acyl 1",
        "Number of carbons in fatty acyl 2", "Double bonds in fatty acyl 2",
        "Number of carbons in fatty acyl 3", "Double bonds in fatty acyl 3",
        "Number of carbons in fatty acyl 4", "Double bonds in fatty acyl 4",
        "Chain type", "PUFA?", "Modifications", "# of modifications", "Oxidized?",
        "Carbons / double bond equivalent ratio", "Average Intensity (all samples)",
        "Minimum Intensity (all samples)", "Maximum Intensity (all samples)",
        "Matched IS", "Matched IS Reason", "Polarity_norm"
    ]
    ordered_cols = [c for c in core_cols if c in norm_ann.columns] + sample_cols
    norm_ann = norm_ann[[c for c in ordered_cols if c in norm_ann.columns]]

    out_ann = output_folder / "5-Final_search_results_median_normalized.csv"
    norm_ann.to_csv(out_ann, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved annotated median-normalized file: {out_ann}", flush=True)

    # --------------------------------------------------------------
    # Step 2: Global median normalization (unknowns)
    # --------------------------------------------------------------
    if not df_unk.empty:
        print("[STEP] Performing global median normalization for unknown features...", flush=True)
        global_medians = df_unk[sample_cols].median().replace(0, np.nan)
        norm_unk = df_unk.copy()
        for col in sample_cols:
            denom = global_medians[col]
            norm_unk[col] = norm_unk[col] / denom if pd.notna(denom) and denom != 0 else np.nan

        out_unk = output_folder / "5-Final_unknowns_median_normalized.csv"
        norm_unk.to_csv(out_unk, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved unknowns median-normalized file: {out_unk}", flush=True)
    else:
        print("[INFO] Skipping unknown normalization — no unknowns present.", flush=True)
        out_unk = None

    # --------------------------------------------------------------
    # Step 3: Evaluate normalization performance
    # --------------------------------------------------------------
    eval_base = output_folder / "median_normalization"
    eval_base.mkdir(parents=True, exist_ok=True)

    try:
        print("\n[STEP] Evaluating normalization performance (annotated)...", flush=True)
        evaluate_normalization_performance(
            annotated_csv,
            out_ann,
            sample_groups_csv,
            eval_base / "annotated"
        )
        print("[DONE] Annotated median normalization evaluation complete.", flush=True)
    except Exception as e:
        import traceback
        print(f"[WARNING] Annotated evaluation failed: {e}", flush=True)
        print(traceback.format_exc(), flush=True)

    if out_unk:
        try:
            print("\n[STEP] Evaluating normalization performance (unknowns)...", flush=True)
            evaluate_normalization_performance(
                unknowns_csv,
                out_unk,
                sample_groups_csv,
                eval_base / "unknowns"
            )
            print("[DONE] Unknowns median normalization evaluation complete.\n", flush=True)
        except Exception as e:
            import traceback
            print(f"[WARNING] Unknowns evaluation failed: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

    print("[ALL DONE] Median normalization (separate annotated & unknowns) completed.\n", flush=True)
    return out_ann, out_unk


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Within-class and global median normalization (separate outputs).")
    parser.add_argument("--annotated", required=True, help="Path to Final_search_results_normalized.csv")
    parser.add_argument("--unknowns", required=True, help="Path to Final_unknowns.csv")
    parser.add_argument("--groups", required=True, help="Path to sample_groups.csv")
    parser.add_argument("--out", default="results", help="Output folder")
    args = parser.parse_args()

    median_normalization(args.annotated, args.unknowns, args.groups, args.out)
