# ---------------------------------------------------------------------
# Missing value imputation and RSD recomputation for lipidomics datasets
# Integrated with QC RSD filtering consistent with data_cleansing.py
# ---------------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*is_sparse is deprecated.*",
    category=FutureWarning
)

def load_qc_threshold():
    """Try to read qc_rsd_threshold from data_cleansing.py, else default to 30."""
    try:
        spec = importlib.util.spec_from_file_location("data_cleansing", Path(__file__).parent / "data_cleansing.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "qc_rsd_threshold"):
            return float(module.qc_rsd_threshold)
    except Exception:
        pass
    return 30.0


def impute_missing_values(final_csv, group_csv, output_folder="results",
                          detection_thresholds=(0.8, 0.5), qc_rsd_threshold=None):
    """
    Replace missing/zero values in sample intensity columns using per-feature (row-wise) LOD
    estimated as the 1st percentile of positive values within each group, with fallbacks:
    row-in-group → row-global → group-global → dataset-global. Recompute RSDs and apply QC RSD filter.
    """
    final_csv = Path(final_csv)
    group_csv = Path(group_csv)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(final_csv, low_memory=False)
    group_df = pd.read_csv(group_csv, low_memory=False)

    # --- Identify sample columns ---
    sample_cols = [c for c in df.columns if str(c).strip().startswith(("[POS", "[NEG"))]
    if not sample_cols:
        raise ValueError("No sample columns found. Expected columns starting with [POS] or [NEG].")

    # --- Map samples to groups (warn on missing) ---
    sample_to_group = dict(zip(group_df["Sample"], group_df["Group"]))
    missing_in_map = [s for s in sample_cols if s not in sample_to_group]
    if missing_in_map:
        print(f"Warning: {len(missing_in_map)} samples not found in group map: {missing_in_map}", flush=True)
    groups = group_df["Group"].unique()
    high_thresh, mid_thresh = detection_thresholds

    # =======================
    # 1) Minima from POSITIVES (BEFORE zero→NaN)
    # =======================
    vals_raw = pd.to_numeric(df[sample_cols].stack(), errors="coerce").values
    pos_vals = vals_raw[(~np.isnan(vals_raw)) & (vals_raw > 0)]
    global_min = float(np.percentile(pos_vals, 1)) if pos_vals.size else 1.0
    print(f"Global positive 1st percentile: {global_min:.6g}", flush=True)

    group_min = {}
    for g in groups:
        g_samples = [s for s in group_df.loc[group_df["Group"] == g, "Sample"] if s in sample_cols]
        if not g_samples:
            group_min[g] = global_min
            continue
        gv_raw = pd.to_numeric(df[g_samples].stack(), errors="coerce").values
        gv_pos = gv_raw[(~np.isnan(gv_raw)) & (gv_raw > 0)]
        group_min[g] = float(np.percentile(gv_pos, 1)) if gv_pos.size else global_min

    print("Per-group positive 1st percentiles:")
    for k, v in group_min.items():
        print(f"  {k}: {v:.6g}", flush=True)

    # =======================
    # 2) Normalize missings (AFTER minima computed)
    # =======================
    df[sample_cols] = (
        df[sample_cols]
        .replace(["NA", "NaN", "nan", ""], np.nan)
        .astype(float)
    )
    # zeros treated as missing per your convention
    df[sample_cols] = df[sample_cols].mask(df[sample_cols] == 0, np.nan)

    # =======================
    # 3) Vectorized per-feature/per-group imputation
    # =======================
    X = df[sample_cols].to_numpy(copy=True)  # shape (n_feat, n_samp)

    # Detect sample indices per group
    col_index = {c: i for i, c in enumerate(sample_cols)}
    group_to_idxs = {
        g: [col_index[c] for c in group_df.loc[group_df["Group"] == g, "Sample"] if c in col_index]
        for g in groups
    }
    group_to_idxs = {g: idxs for g, idxs in group_to_idxs.items() if idxs}

    # Row-global positive 1st percentile fallback
    Xp = X.copy()
    Xp[~(Xp > 0)] = np.nan
    row_global_min = np.nanpercentile(Xp, 1, axis=1)  # per row; NaN if no positives

    for g, idxs in group_to_idxs.items():
        G = X[:, idxs]              # submatrix for group g
        G_pos = G.copy()
        G_pos[~(G_pos > 0)] = np.nan

        # row-wise LOD in this group = 1st pct of positives across the group's samples
        row_group_lod = np.nanpercentile(G_pos, 1, axis=1)

        # detection rate per row in this group (positives count)
        det_rate = np.sum(G > 0, axis=1) / float(len(idxs))

        # fallback chain for LOD
        lod = row_group_lod.copy()
        nan_mask = ~np.isfinite(lod)
        if nan_mask.any():
            lod[nan_mask] = row_global_min[nan_mask]
        nan_mask = ~np.isfinite(lod) | (lod <= 0)
        if nan_mask.any():
            lod[nan_mask] = group_min[g]
        nan_mask = ~np.isfinite(lod) | (lod <= 0)
        if nan_mask.any():
            lod[nan_mask] = global_min

        # replacement magnitude by detection rate
        repl = np.where(det_rate >= high_thresh, lod / 3.0,
                np.where(det_rate >= mid_thresh, np.maximum(lod / 3.0, global_min / 5.0),
                         lod / 5.0))

        # impute NaNs in this group with per-row repl
        nan_cells = np.isnan(G)
        if nan_cells.any():
            for j, col_idx in enumerate(idxs):
                mask = nan_cells[:, j]
                X[mask, col_idx] = repl[mask]

    df_imputed = df.copy()
    df_imputed.loc[:, sample_cols] = X

    # =======================
    # 4) Recompute summaries and RSDs
    # =======================
    def rsd(series):
        m = np.nanmean(series)
        s = np.nanstd(series)
        return 100 * s / m if m and not np.isnan(m) else np.nan

    for col in ["Average Intensity (all samples)", "Minimum Intensity (all samples)", "Maximum Intensity (all samples)"]:
        if col in df_imputed.columns:
            df_imputed.drop(columns=[col], inplace=True)

    df_imputed["Mean Intensity (All)"] = df_imputed[sample_cols].mean(axis=1)
    df_imputed["Median Intensity (All)"] = df_imputed[sample_cols].median(axis=1)
    df_imputed["Min Intensity (All)"] = df_imputed[sample_cols].min(axis=1)
    df_imputed["Max Intensity (All)"] = df_imputed[sample_cols].max(axis=1)
    df_imputed["RSD (%) (All)"] = df_imputed[sample_cols].apply(rsd, axis=1)

    df_imputed.rename(columns={
        "Mean Intensity (All)": "Average Intensity (all samples)",
        "Min Intensity (All)": "Minimum Intensity (all samples)",
        "Max Intensity (All)": "Maximum Intensity (all samples)"
    }, inplace=True)

    if "Carbons / double bond equivalent ratio" in df_imputed.columns:
        summary_cols = [
            "Average Intensity (all samples)",
            "Minimum Intensity (all samples)",
            "Maximum Intensity (all samples)"
        ]
        existing = [c for c in summary_cols if c in df_imputed.columns]
        cols = list(df_imputed.columns)
        anchor = cols.index("Carbons / double bond equivalent ratio") + 1
        for c in existing:
            cols.remove(c)
        for offset, c in enumerate(existing):
            cols.insert(anchor + offset, c)
        df_imputed = df_imputed[cols]

    qc_samples = [s for s in group_df.loc[group_df["Group"].str.lower() == "qc", "Sample"] if s in sample_cols]
    non_qc_samples = [s for s in sample_cols if s not in qc_samples]

    df_imputed["RSD QCs (%)"] = df_imputed[qc_samples].apply(rsd, axis=1) if qc_samples else np.nan
    if not qc_samples:
        print("Warning: No QC samples found; RSD QCs (%) set to NaN.", flush=True)

    df_imputed["RSD Samples (%)"] = df_imputed[non_qc_samples].apply(rsd, axis=1) if non_qc_samples else np.nan
    if not non_qc_samples:
        print("Warning: No non-QC samples found; RSD Samples (%) set to NaN.", flush=True)

    for g in groups:
        g_samples = [s for s in group_df.loc[group_df["Group"] == g, "Sample"] if s in sample_cols]
        if not g_samples:
            continue
        df_imputed[f"Mean Intensity ({g})"] = df_imputed[g_samples].mean(axis=1)
        df_imputed[f"Median Intensity ({g})"] = df_imputed[g_samples].median(axis=1)
        df_imputed[f"Min Intensity ({g})"] = df_imputed[g_samples].min(axis=1)
        df_imputed[f"Max Intensity ({g})"] = df_imputed[g_samples].max(axis=1)
        df_imputed[f"RSD (%) ({g})"] = df_imputed[g_samples].apply(rsd, axis=1)

    # =======================
    # 5) Apply QC RSD filter and save
    # =======================
    qc_threshold = load_qc_threshold() if qc_rsd_threshold is None else float(qc_rsd_threshold)
    print(f"Applying QC RSD filter: remove features with QC RSD > {qc_threshold:.1f}%.", flush=True)

    filtered_df = df_imputed[df_imputed["RSD QCs (%)"] <= qc_threshold].copy()
    removed_df  = df_imputed[df_imputed["RSD QCs (%)"] >  qc_threshold].copy()

    # Drop verbose per-group stat columns from disk outputs if you want lighter files
    keep_rsd_cols = {
        "RSD QCs (%)", "RSD Samples (%)",
        "RSD_12x [%]", "RSD_15x [%]", "RSD_5x [%]", "RSD_8x [%]", "RSD_QC [%]",
        "Average Intensity (all samples)", "Minimum Intensity (all samples)", "Maximum Intensity (all samples)"
    }
    drop_patterns = ("Mean Intensity (", "Median Intensity (", "Min Intensity (", "Max Intensity (", "RSD (%) (")

    def _drop_verbose(df_):
        cols_to_drop = [c for c in df_.columns if c.startswith(drop_patterns) and c not in keep_rsd_cols]
        if cols_to_drop:
            df_.drop(columns=cols_to_drop, inplace=True)

    _drop_verbose(df_imputed.copy())  # optional no-op; keep full in "imputed_before_filtering"
    _drop_verbose(filtered_df)
    _drop_verbose(removed_df)

    out_full     = output_folder / "debug" / "3-Final_annotated_results_imputed_before_filtering.csv"
    out_filtered = output_folder / "debug" / "4-Final_annotated_results_imputed_filtered.csv"
    out_removed  = output_folder / "debug" / "Removed_high_QC_RSD.csv"
    out_full.parent.mkdir(parents=True, exist_ok=True)

    df_imputed.to_csv(out_full, index=False, encoding="utf-8-sig")
    filtered_df.to_csv(out_filtered, index=False, encoding="utf-8-sig")
    removed_df.to_csv(out_removed, index=False, encoding="utf-8-sig")

    print("Imputation complete. Overwrote QC/Sample RSDs.", flush=True)
    print(f"Saved:\n - Imputed: {out_full}\n - Filtered: {out_filtered}\n - Removed (QC>thr): {out_removed}\n", flush=True)

    return out_filtered

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Impute missing values, recompute stats, and apply QC RSD filtering.")
    parser.add_argument("--final", required=True, help="Path to Final_MS_results.csv")
    parser.add_argument("--groups", required=True, help="Path to sample_groups.csv")
    parser.add_argument("--out", default="results", help="Output folder")
    args = parser.parse_args()

    impute_missing_values(args.final, args.groups, args.out)
