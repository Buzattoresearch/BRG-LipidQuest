# ---------------------------------------------------------------------
# LOESS-based drift correction for LC-MS lipidomics datasets
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from normalization import evaluate_normalization_performance

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*is_sparse is deprecated.*",
    category=FutureWarning
)

def loess_normalization(annotated_csv, unknowns_csv, sample_groups_csv, output_folder="results", frac=0.3):
    """
    Apply LOESS drift correction using QC samples over injection order.

    Parameters
    ----------
    annotated_csv: str or Path
        Path to dataset (e.g., Final_annotated_results_imputed_filtered.csv or normalized file)
    sample_groups_csv : str or Path
        Path to sample_groups.csv with 'Sample', 'Group', and 'Order' columns
    output_folder : str or Path
        Output directory
    frac : float
        Fraction of points used for local regression (controls smoothing)

    Returns
    -------
    out_path : Path
        Path to LOESS-corrected output file
    """
    print("\nStarting LOESS drift correction...\n", flush=True)
    annotated_csv = Path(annotated_csv)
    sample_groups_csv = Path(sample_groups_csv)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(annotated_csv, low_memory=False)
    group_df = pd.read_csv(sample_groups_csv, low_memory=False)

    # Verify required columns
    if "Order" not in group_df.columns:
        raise ValueError("sample_groups.csv must include an 'Order' column for LOESS correction.")

    # Identify sample columns
    sample_cols = [c for c in df.columns if c.startswith("[POS") or c.startswith("[NEG]")]
    if not sample_cols:
        raise ValueError("No sample columns found starting with [POS or [NEG].")

    # Merge order info
    sample_meta = group_df.set_index("Sample").reindex(sample_cols)
    if sample_meta["Order"].isna().any():
        raise ValueError("Some samples are missing injection order in sample_groups.csv.")

    # Ensure numeric order
    sample_meta["Order"] = pd.to_numeric(sample_meta["Order"], errors="coerce")
    order = sample_meta["Order"].values
    qc_mask = sample_meta["Group"].str.upper().str.strip() == "QC"
    qc_samples = sample_meta.index[qc_mask].tolist()
    print(f"[INFO] Found {len(qc_samples)} QC samples for LOESS fitting.", flush=True)

    # -----------------------------------------------------------------
    # LOESS correction per feature
    # -----------------------------------------------------------------
    corrected_df = df.copy()
    drift_plots_folder = output_folder / "debug" / "loess_normalization"
    drift_plots_folder.mkdir(exist_ok=True)

    for idx, row in df.iterrows():
        intensities = row[sample_cols].astype(float).values

        # Skip empty features
        if np.isnan(intensities).all():
            continue

                # --- Use only QC samples; operate in log10 space; LOOCV at QC points ---
        qc_idx = np.where(qc_mask.values)[0]
        qc_y = intensities[qc_idx]
        qc_x = order[qc_idx]

        # require >=3 non-NaN QCs
        valid = ~np.isnan(qc_y)
        if valid.sum() < 3:
            continue

        x_qc = qc_x[valid].astype(float)
        y_qc = qc_y[valid].astype(float)
        y_qc_log = np.log10(y_qc)

        # helper: predict trend (log-space). Use LOWESS if >=5 points, else linear in log-space.
        def loess_predict(x_train, y_train_log, x_grid, frac_use):
            x_train = np.asarray(x_train, dtype=float)
            y_train_log = np.asarray(y_train_log, dtype=float)
            x_grid = np.asarray(x_grid, dtype=float)

            if len(x_train) >= 5:
                fitted_train = lowess(y_train_log, x_train, frac=frac_use, it=0, return_sorted=False)
                order_idx = np.argsort(x_train)
                return np.interp(x_grid, x_train[order_idx], fitted_train[order_idx])
            elif len(x_train) >= 2:
                coeffs = np.polyfit(x_train, y_train_log, 1)
                return np.polyval(coeffs, x_grid)
            else:
                return np.full_like(x_grid, np.nan, dtype=float)

        # choose smoothing (more smoothing when few QCs)
        frac_use = 1.0 if len(x_qc) <= 6 else frac

        # ---- LOOCV trend at the QC orders (prevents collapse to a constant) ----
        trend_qc_log = np.empty_like(y_qc_log)
        for j in range(len(x_qc)):
            mask_j = np.ones(len(x_qc), dtype=bool)
            mask_j[j] = False
            pred = loess_predict(x_qc[mask_j], y_qc_log[mask_j], np.array([x_qc[j]]), frac_use)
            trend_qc_log[j] = pred[0]

        # ---- Single fit for ALL orders (used for non-QC samples) ----
        trend_all_log = loess_predict(x_qc, y_qc_log, order.astype(float), frac_use)

        # build per-sample trend in log space with LOOCV at QC positions
        trend_log = trend_all_log.copy()
        valid_global_idx = qc_idx[valid]            # positions (in all samples) of valid QC injections
        trend_log[valid_global_idx] = trend_qc_log  # overwrite LOOCV predictions at those QC injections

        # guard: replace any non-finite trends with median of the trend
        if not np.isfinite(trend_log).all():
            med_trend_log = np.nanmedian(trend_log)
            trend_log = np.where(np.isfinite(trend_log), trend_log, med_trend_log)

        # anchor to the median fitted QC trend (not raw QC median)
        qc_trend_med_log = np.nanmedian(trend_qc_log)
        trend = np.power(10.0, trend_log)
        anchor = np.power(10.0, qc_trend_med_log)

        corrected_vals = (intensities / trend) * anchor
        corrected_df.loc[idx, sample_cols] = corrected_vals

        # Optional: quick diagnostic plot for a tiny random subset
        if np.random.rand() < 0.002:
            plt.figure(figsize=(5, 4))
            # raw QC points
            plt.scatter(x_qc, y_qc, label="QC raw", s=20)
            # fitted trend over all injections
            plt.plot(order, np.power(10.0, trend_all_log), label="Trend (all fit)", lw=2)
            # LOOCV trend only at QC positions
            plt.scatter(x_qc, np.power(10.0, trend_qc_log), label="LOOCV @ QC", s=20)
            plt.xlabel("Injection order")
            plt.ylabel("Intensity")
            plt.title(f"Feature {idx} drift")
            plt.legend()
            plt.tight_layout()
            plt.savefig(drift_plots_folder / f"feature_{idx}_drift.png", dpi=200)
            plt.close()

    # -----------------------------------------------------------------
    # Recalculate RSDs using sample_groups.csv (QC defined by Group column)
    # -----------------------------------------------------------------
    print("[STEP] Recalculating RSDs after LOESS correction...", flush=True)
    group_df = pd.read_csv(sample_groups_csv, low_memory=False)

    # Identify sample columns (intensity data)
    sample_cols = [c for c in corrected_df.columns if c.startswith("[POS") or c.startswith("[NEG]")]
    if not sample_cols:
        raise ValueError("No sample columns found. Expected columns starting with [POS or [NEG].")

    # Build a mapping: group → list of sample names
    group_map = {
        g.strip(): [s for s in group_df.loc[group_df["Group"] == g, "Sample"].tolist()]
        for g in group_df["Group"].dropna().unique()
    }

    # Identify QC and non-QC groups directly from sample_groups.csv
    qc_groups = [g for g in group_map if g.upper() == "QC"]
    qc_samples = [s for g in qc_groups for s in group_map[g]]
    non_qc_groups = [g for g in group_map if g.upper() != "QC"]

    # Match to dataframe column names
    qc_cols = [c for c in sample_cols if any(s == c or s in c for s in qc_samples)]
    non_qc_cols = [c for c in sample_cols if c not in qc_cols]

    rsd_qc_vals, rsd_sample_vals = [], []

    # Compute global QC and sample RSDs
    for _, row in corrected_df.iterrows():
        # QC RSD
        qc_vals = pd.to_numeric(row[qc_cols], errors="coerce").replace(0, np.nan).dropna()
        rsd_qc = (qc_vals.std(ddof=1) / qc_vals.mean() * 100) if len(qc_vals) > 1 else np.nan

        # Non-QC RSD (all other groups combined)
        sample_vals = pd.to_numeric(row[non_qc_cols], errors="coerce").replace(0, np.nan).dropna()
        rsd_samples = (sample_vals.std(ddof=1) / sample_vals.mean() * 100) if len(sample_vals) > 1 else np.nan

        rsd_qc_vals.append(rsd_qc)
        rsd_sample_vals.append(rsd_samples)

    corrected_df["RSD QCs (%)"] = rsd_qc_vals
    corrected_df["RSD Samples (%)"] = rsd_sample_vals

    # Compute per-group RSD columns based on sample_groups.csv
    for g in non_qc_groups:
        cols = [c for c in sample_cols if any(s == c or s in c for s in group_map[g])]
        rsd_vals = []
        for _, row in corrected_df.iterrows():
            vals = pd.to_numeric(row[cols], errors="coerce").replace(0, np.nan).dropna()
            rsd = (vals.std(ddof=1) / vals.mean() * 100) if len(vals) > 1 else np.nan
            rsd_vals.append(rsd)
        corrected_df[f"RSD_{g} [%]"] = rsd_vals

    print(f"[INFO] RSD recalculation complete. Added columns: "
          f"RSD QCs (%), RSD Samples (%), and {len(non_qc_groups)} group-specific RSDs.", flush=True)

    # -----------------------------------------------------------------
    # Reorder columns before saving (no metadata after sample columns)
    # -----------------------------------------------------------------
    print("[STEP] Reordering columns before saving...", flush=True)
    core_cols = [
        "UniqueID", "RT (min)", "m/z", "Neutral mass", "Adducts", "Polarity", "Internal Standard",
        "RSD QCs (%)", "RSD Samples (%)", "RSD_12x [%]", "RSD_15x [%]", "RSD_5x [%]",
        "RSD_8x [%]", "RSD_QC [%]", "MS/MS available?", "Annotation", "Annotation Type",
        "Metaboscape Annotation Status", "Annotation Source", "Headgroup", "Lipid Class",
        "Δm/z (mDa)", "Δm/z (ppm)", "MS/MS score", "Annotation tier", "mSigma", 
        "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]",
        "Molecular Formula",
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

    sample_cols = [c for c in corrected_df.columns if c.startswith("[POS") or c.startswith("[NEG]")]
    ordered_cols = [c for c in core_cols if c in corrected_df.columns] + sample_cols
    corrected_df = corrected_df[[c for c in ordered_cols if c in corrected_df.columns]]

    # -----------------------------------------------------------------
    # Save corrected data
    # -----------------------------------------------------------------
    out_path = output_folder / "debug" /"9-Final_annotated_results_loess_normalized.csv"
    corrected_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved LOESS-corrected file: {out_path}", flush=True)

    # -----------------------------------------------------------------
    # Evaluate performance
    # -----------------------------------------------------------------
    eval_folder = output_folder / "debug" / "loess_normalization"
    eval_folder.mkdir(exist_ok=True)

    try:
        print("\n[STEP] Evaluating LOESS normalization performance...", flush=True)
        evaluate_normalization_performance(
            annotated_csv,
            out_path,
            sample_groups_csv,
            eval_folder
        )
        print("[DONE] LOESS normalization performance evaluation complete.\n", flush=True)
    except Exception as e:
        import traceback
        print(f"[WARNING] LOESS evaluation failed: {e}", flush=True)
        print(traceback.format_exc(), flush=True)

    return out_path


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply LOESS drift correction using QC samples.")
    parser.add_argument("--features", required=True, help="Path to the input CSV (e.g., Final_annotated_results_imputed_filtered.csv).")
    parser.add_argument("--groups", required=True, help="Path to sample_groups.csv (must include 'Order').")
    parser.add_argument("--out", default="results", help="Output folder.")
    parser.add_argument("--frac", type=float, default=0.3, help="LOESS smoothing fraction (default=0.3).")
    args = parser.parse_args()

    loess_normalization(args.features, args.groups, args.out, args.frac)
