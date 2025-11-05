# ---------------------------------------------------------------------
# Within-class median normalization (annotated) and
# global median normalization (unknowns), each evaluated separately.
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")

from normalization import evaluate_normalization_performance

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*is_sparse is deprecated.*",
    category=FutureWarning
)


def median_normalization(annotated_csv, unknowns_csv, sample_groups_csv, output_folder="results", min_n_per_class: int = 5):
    """
    HYBRID normalization with column reordering preserved.
      A) Global per-sample median (annotated + unknowns)
      B) Within-class median applied to the globally normalized annotated table
    """
    print("\nStarting HYBRID median-based normalization...\n", flush=True)
    annotated_csv = Path(annotated_csv)
    unknowns_csv = Path(unknowns_csv)
    sample_groups_csv = Path(sample_groups_csv)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # ---------- Load ----------
    if not annotated_csv.exists():
        raise FileNotFoundError(f"Annotated file not found: {annotated_csv}")
    df_ann = pd.read_csv(annotated_csv, low_memory=False)
    print(f"[INFO] Loaded annotated dataset: {len(df_ann)} rows", flush=True)

    if unknowns_csv.exists():
        df_unk = pd.read_csv(unknowns_csv, low_memory=False)
        print(f"[INFO] Loaded unknown dataset: {len(df_unk)} rows", flush=True)
    else:
        df_unk = pd.DataFrame()
        print(f"[WARNING] Unknowns file not found: {unknowns_csv}", flush=True)

    # ---------- Identify sample columns FIRST ----------
    sample_cols = [c for c in df_ann.columns if c.startswith("[POS") or c.startswith("[NEG")]
    if not sample_cols:
        raise ValueError("No sample columns found in annotated file.")

    # ---------- Coerce numeric (zeros -> NaN for medians/ratios) ----------
    def _coerce_numeric(df, cols):
        df = df.copy()
        df.loc[:, cols] = df[cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
        return df

    df_ann = _coerce_numeric(df_ann, sample_cols)
    if not df_unk.empty:
        ucols = [c for c in df_unk.columns if c in sample_cols]
        df_unk = _coerce_numeric(df_unk, ucols)

    # ---------- Column reordering helper ----------
    core_cols = [
        "UniqueID","RT (min)","m/z","Neutral mass","Adducts","Polarity","Internal Standard",
        "RSD QCs (%)","RSD Samples (%)","RSD_12x [%]","RSD_15x [%]","RSD_5x [%]","RSD_8x [%]","RSD_QC [%]",
        "MS/MS available?","Annotation","Annotation Type","Metaboscape Annotation Status","Annotation Source",
        "Headgroup","Lipid Class","Δm/z (mDa)","Δm/z (ppm)","MS/MS score","Annotation tier","mSigma",
        "CCS (Å²)","Mob. 1/K0","ΔCCS [%]","Molecular Formula","Plasmenyl?","Number of carbons in fatty acyls",
        "Double bond equivalents","Number of carbons in fatty acyl 1","Double bonds in fatty acyl 1",
        "Number of carbons in fatty acyl 2","Double bonds in fatty acyl 2","Number of carbons in fatty acyl 3",
        "Double bonds in fatty acyl 3","Number of carbons in fatty acyl 4","Double bonds in fatty acyl 4",
        "Chain type","PUFA?","Modifications","# of modifications","Oxidized?",
        "Carbons / double bond equivalent ratio","Average Intensity (all samples)",
        "Minimum Intensity (all samples)","Maximum Intensity (all samples)",
        "Matched IS","Matched IS Reason","Polarity_norm"
    ]
    def _reorder(df):
        ordered = [c for c in core_cols if c in df.columns] + [c for c in sample_cols if c in df.columns]
        return df[[c for c in ordered if c in df.columns]]

    # ---------- A) Global per-sample median ----------
    print("[STEP A] Global per-sample median normalization...", flush=True)
    per_sample_median_ann = df_ann[sample_cols].median(axis=0, skipna=True).replace(0, np.nan)
    norm_ann_global = df_ann.copy()
    norm_ann_global.loc[:, sample_cols] = norm_ann_global[sample_cols].div(per_sample_median_ann, axis=1)

    if not df_unk.empty:
        shared = [c for c in sample_cols if c in df_unk.columns]
        per_sample_median_unk = df_unk[shared].median(axis=0, skipna=True).replace(0, np.nan)
        norm_unk_global = df_unk.copy()
        norm_unk_global.loc[:, shared] = norm_unk_global[shared].div(per_sample_median_unk, axis=1)
    else:
        norm_unk_global = None

    # ---------- B) Within-class on top of global (annotated only, gated by min_n_per_class) ----------
    print(f"[STEP B] Within-class median normalization on globally normalized annotated "
      f"(only for classes with n_features >= {min_n_per_class})...", flush=True)

    if "Lipid Class" not in norm_ann_global.columns:
        raise ValueError("Missing 'Lipid Class' column in annotated dataset.")

    # Count features per class (rows per class in the annotated table)
    class_sizes = norm_ann_global.groupby("Lipid Class").size()
    eligible_classes = set(class_sizes[class_sizes >= min_n_per_class].index)

    # Per-sample medians for each class (computed on the *globally* normalized table)
    class_medians = (
        norm_ann_global.groupby("Lipid Class")[sample_cols]
        .median()
        .replace(0, np.nan)
    )

    # Build denominators matrix aligned to rows; default = 1.0 (no change) for ineligible classes
    norm_ann_class = norm_ann_global.copy()
    # Map per-row class medians
    denom = class_medians.reindex(norm_ann_class["Lipid Class"]).to_numpy()  # shape (n_feat, n_samples)

    # Mask rows whose class is NOT eligible -> set denom to 1.0 so division does nothing
    row_classes = norm_ann_class["Lipid Class"].astype(str).values
    ineligible_rows = ~np.isin(row_classes, list(eligible_classes))
    if ineligible_rows.any():
        denom[ineligible_rows, :] = 1.0

    # Any non-finite denominators (NaN/inf) -> set to 1.0 (no change)
    denom = np.where(np.isfinite(denom) & (denom != 0), denom, 1.0)

    # Apply normalization (only rows with denom != 1 actually change)
    vals = norm_ann_class[sample_cols].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_ann_class.loc[:, sample_cols] = vals / denom

    # Save a summary of which classes were normalized
    summary = (
        class_sizes.rename("n_features")
        .to_frame()
        .assign(eligible=lambda s: s.index.to_series().isin(eligible_classes))
    )
    (output_folder / "debug").mkdir(parents=True, exist_ok=True)
    summary_path = output_folder / "debug" / "within_class_normalization_summary.csv"
    summary.to_csv(summary_path, index=True, encoding="utf-8-sig")
    print(f"[INFO] Within-class normalization summary → {summary_path}", flush=True)

    # ---------- RSDs (QC and groups) ----------
    group_df = pd.read_csv(sample_groups_csv, low_memory=False)
    group_map = {
        g: [s for s in group_df.loc[group_df["Group"] == g, "Sample"].tolist()]
        for g in group_df["Group"].dropna().unique()
    }
    qc_samples = set(group_map.get("QC", [])) | set(group_map.get("qc", [])) | set(group_map.get("Qc", []))
    qc_cols = [c for c in sample_cols if c in qc_samples]
    non_qc_cols = [c for c in sample_cols if c not in qc_cols]

    def _add_rsd_columns(df_in):
        df_out = df_in.copy()
        def rsd(a):
            a = pd.to_numeric(a, errors="coerce")
            a = a[~a.isna()]
            return float(a.std(ddof=1) / a.mean() * 100) if len(a) > 1 and a.mean() not in (0, np.nan) else np.nan
        df_out["RSD QCs (%)"]     = df_out[qc_cols].apply(rsd, axis=1) if qc_cols else np.nan
        df_out["RSD Samples (%)"] = df_out[non_qc_cols].apply(rsd, axis=1) if non_qc_cols else np.nan
        for g, slist in group_map.items():
            cols = [c for c in slist if c in sample_cols]
            if cols:
                df_out[f"RSD_{g} [%]"] = df_out[cols].apply(rsd, axis=1)
        return df_out

    norm_ann_global = _add_rsd_columns(norm_ann_global)
    norm_ann_class  = _add_rsd_columns(norm_ann_class)
    if norm_unk_global is not None:
        # Unknowns don’t have Lipid Class, just keep global normalization; no RSDs by group unless needed
        pass

    # ---------- Reorder columns & save ----------
    out_ann_global = output_folder / "debug" / "7a-Final_annotated_global_median_normalized.csv"
    out_ann_class  = output_folder / "debug" / "7-Final_annotated_median_normalized.csv"
    norm_ann_global = _reorder(norm_ann_global)
    norm_ann_class  = _reorder(norm_ann_class)
    norm_ann_global.to_csv(out_ann_global, index=False, encoding="utf-8-sig")
    norm_ann_class.to_csv(out_ann_class,   index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {out_ann_global}", flush=True)
    print(f"[INFO] Saved: {out_ann_class}",  flush=True)

    if norm_unk_global is not None:
        out_unk_global = output_folder / "debug" / "8-Final_unknowns_median_normalized.csv"
        # reorder unknowns too (core cols subset + sample cols)
        cols_u = [c for c in core_cols if c in norm_unk_global.columns] + [c for c in sample_cols if c in norm_unk_global.columns]
        norm_unk_global[cols_u].to_csv(out_unk_global, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved: {out_unk_global}", flush=True)
    else:
        out_unk_global = None

    # ---------- Evaluate (both annotated, plus unknowns if present) ----------
    eval_base = output_folder / "debug" / "median_normalization"
    eval_base.mkdir(parents=True, exist_ok=True)

    try:
        print("\n[Eval] Global median (annotated)...", flush=True)
        evaluate_normalization_performance(annotated_csv, out_ann_global, sample_groups_csv, eval_base / "annotated_global")
    except Exception as e:
        print(f"[WARNING] Global annotated evaluation failed: {e}", flush=True)

    try:
        print("\n[Eval] Within-class after global (annotated)...", flush=True)
        evaluate_normalization_performance(annotated_csv, out_ann_class, sample_groups_csv, eval_base / "annotated_within_class")
    except Exception as e:
        print(f"[WARNING] Within-class annotated evaluation failed: {e}", flush=True)

    if out_unk_global:
        try:
            print("\n[Eval] Global median (unknowns)...", flush=True)
            evaluate_normalization_performance(unknowns_csv, out_unk_global, sample_groups_csv, eval_base / "unknowns_global")
        except Exception as e:
            print(f"[WARNING] Unknowns evaluation failed: {e}", flush=True)

    print("\n[ALL DONE] Hybrid normalization complete.\n", flush=True)
    return out_ann_class, out_unk_global



# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Within-class and global median normalization (separate outputs).")
    parser.add_argument("--annotated", required=True, help="Path to Final_annotated_results_normalized.csv")
    parser.add_argument("--unknowns", required=True, help="Path to Final_unknowns.csv")
    parser.add_argument("--groups", required=True, help="Path to sample_groups.csv")
    parser.add_argument("--out", default="results", help="Output folder")
    args = parser.parse_args()

    median_normalization(args.annotated, args.unknowns, args.groups, args.out)
