# ---------------------------------------------------------------------
# Missing value imputation and RSD recomputation for lipidomics datasets
# Integrated with QC RSD filtering consistent with data_cleansing.py
# ---------------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util

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
    Replace missing or zero values in sample intensity columns based on group-level detection rates,
    recompute RSDs, overwrite QC/sample RSD columns, and filter based on QC RSD threshold.
    """

    final_csv = Path(final_csv)
    group_csv = Path(group_csv)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(final_csv)
    group_df = pd.read_csv(group_csv)

    # --- Identify sample columns ---
    sample_cols = [c for c in df.columns if c.startswith("[POS") or c.startswith("[NEG")]
    if not sample_cols:
        raise ValueError("No sample columns found. Expected columns starting with [POS or [NEG].")

    # --- Normalize missing values ---
    df[sample_cols] = df[sample_cols].replace(["NA", "NaN", "nan", "", 0], np.nan).astype(float)

    # --- Map samples to groups ---
    sample_to_group = dict(zip(group_df["Sample"], group_df["Group"]))
    missing_in_map = [s for s in sample_cols if s not in sample_to_group]
    if missing_in_map:
        print(f"Warning: {len(missing_in_map)} samples not found in group map: {missing_in_map}")

    groups = group_df["Group"].unique()
    high_thresh, mid_thresh = detection_thresholds

    # --- Compute global minimum ---
    all_values = df[sample_cols].to_numpy().flatten()
    all_values = all_values[~np.isnan(all_values)]
    global_min = np.nanmin(all_values) if len(all_values) > 0 else 1.0
    print(f"Global minimum intensity: {global_min:.6g}")

    # --- Compute per-group minimums ---
    group_min = {}
    for g in groups:
        g_samples = [s for s in group_df.loc[group_df["Group"] == g, "Sample"] if s in sample_cols]
        vals = df[g_samples].to_numpy().flatten()
        vals = vals[~np.isnan(vals)]
        group_min[g] = np.nanmin(vals) if len(vals) > 0 else global_min

    # --- Imputation ---
    df_imputed = df.copy()

    for idx, row in df.iterrows():
        for g in groups:
            g_samples = [s for s in group_df.loc[group_df["Group"] == g, "Sample"] if s in sample_cols]
            if not g_samples:
                continue
            vals = row[g_samples].astype(float).values
            det_rate = np.sum(~np.isnan(vals)) / len(g_samples)
            if det_rate >= high_thresh:
                repl = group_min[g] / 3
            elif det_rate >= mid_thresh:
                repl = global_min / 3
            else:
                repl = global_min / 5
            vals[np.isnan(vals)] = repl
            df_imputed.loc[idx, g_samples] = vals

    # --- RSD helper ---
    def rsd(series):
        m = np.nanmean(series)
        s = np.nanstd(series)
        return 100 * s / m if m and not np.isnan(m) else np.nan

    # --- Remove any old summary columns before recalculating ---
    for col in ["Average Intensity (all samples)", "Minimum Intensity (all samples)", "Maximum Intensity (all samples)"]:
        if col in df_imputed.columns:
            df_imputed.drop(columns=[col], inplace=True)

    # --- Global stats ---
    df_imputed["Mean Intensity (All)"] = df_imputed[sample_cols].mean(axis=1)
    df_imputed["Median Intensity (All)"] = df_imputed[sample_cols].median(axis=1)
    df_imputed["Min Intensity (All)"] = df_imputed[sample_cols].min(axis=1)
    df_imputed["Max Intensity (All)"] = df_imputed[sample_cols].max(axis=1)
    df_imputed["RSD (%) (All)"] = df_imputed[sample_cols].apply(rsd, axis=1)

    # --- Rename global columns to match final format ---
    df_imputed.rename(columns={
        "Mean Intensity (All)": "Average Intensity (all samples)",
        "Min Intensity (All)": "Minimum Intensity (all samples)",
        "Max Intensity (All)": "Maximum Intensity (all samples)"
    }, inplace=True)

    # --- Reorder intensity summary columns to appear after 'Carbons / double bond equivalent ratio' ---
    summary_cols = [
        "Average Intensity (all samples)",
        "Minimum Intensity (all samples)",
        "Maximum Intensity (all samples)"
    ]

    existing_cols = [c for c in summary_cols if c in df_imputed.columns]
    all_cols = list(df_imputed.columns)

    if "Carbons / double bond equivalent ratio" in all_cols:
        idx = all_cols.index("Carbons / double bond equivalent ratio") + 1
        # Remove summary columns if they already exist elsewhere
        for c in existing_cols:
            if c in all_cols:
                all_cols.remove(c)
        # Insert summary columns right after the ratio column
        for offset, c in enumerate(existing_cols):
            all_cols.insert(idx + offset, c)
        df_imputed = df_imputed[all_cols]

    # --- Compute RSDs for QC and non-QC samples separately ---
    qc_samples = [s for s in group_df.loc[group_df["Group"].str.lower() == "qc", "Sample"] if s in sample_cols]
    non_qc_samples = [s for s in sample_cols if s not in qc_samples]

    if qc_samples:
        df_imputed["RSD QCs (%)"] = df_imputed[qc_samples].apply(rsd, axis=1)
    else:
        df_imputed["RSD QCs (%)"] = np.nan
        print("Warning: No QC samples found in group assignment; RSD QCs (%) set to NaN.")

    if non_qc_samples:
        df_imputed["RSD Samples (%)"] = df_imputed[non_qc_samples].apply(rsd, axis=1)
    else:
        df_imputed["RSD Samples (%)"] = np.nan
        print("Warning: No non-QC samples found in group assignment; RSD Samples (%) set to NaN.")

    # --- Per-group stats ---
    for g in groups:
        g_samples = [s for s in group_df.loc[group_df["Group"] == g, "Sample"] if s in sample_cols]
        if not g_samples:
            continue
        df_imputed[f"Mean Intensity ({g})"] = df_imputed[g_samples].mean(axis=1)
        df_imputed[f"Median Intensity ({g})"] = df_imputed[g_samples].median(axis=1)
        df_imputed[f"Min Intensity ({g})"] = df_imputed[g_samples].min(axis=1)
        df_imputed[f"Max Intensity ({g})"] = df_imputed[g_samples].max(axis=1)
        df_imputed[f"RSD (%) ({g})"] = df_imputed[g_samples].apply(rsd, axis=1)

    # --- QC filtering ---
    # Determine QC threshold
    if qc_rsd_threshold is None:
        qc_threshold = load_qc_threshold()
    else:
        qc_threshold = float(qc_rsd_threshold)
    print(f"Applying QC RSD filter: features with QC RSD > {qc_threshold:.1f}% will be removed.", flush = True)
    filtered_df = df_imputed[df_imputed["RSD QCs (%)"] <= qc_threshold].copy()
    removed_df = df_imputed[df_imputed["RSD QCs (%)"] > qc_threshold].copy()

    # --- Column cleanup: keep only relevant updated columns ---
    keep_rsd_cols = [
        "RSD QCs (%)", "RSD Samples (%)",
        "RSD_12x [%]", "RSD_15x [%]", "RSD_5x [%]", "RSD_8x [%]", "RSD_QC [%]",
        "Average Intensity (all samples)", "Minimum Intensity (all samples)", "Maximum Intensity (all samples)"
    ]

    # Find existing matching columns
    existing_rsd_cols = [c for c in df_imputed.columns if c in keep_rsd_cols]

    # Drop other RSD/stat columns that start with any of these prefixes
    drop_patterns = ("Mean Intensity (", "Median Intensity (", "Min Intensity (", "Max Intensity (", "RSD (%) (")
    cols_to_drop = [c for c in df_imputed.columns if c.startswith(drop_patterns) and c not in existing_rsd_cols]
    if cols_to_drop:
        df_imputed.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped {len(cols_to_drop)} redundant intensity/RSD columns.")
    cols_to_drop = [c for c in filtered_df.columns if c.startswith(drop_patterns) and c not in existing_rsd_cols]
    if cols_to_drop:
        filtered_df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped {len(cols_to_drop)} redundant intensity/RSD columns.")

    # --- Save outputs ---
    output_path_full = output_folder / "debug" / "Final_search_results_imputed_before_filtering.csv"
    output_path_filtered = output_folder / "3-Final_search_results_imputed_filtered.csv"
    output_path_removed = output_folder / "debug" / "Removed_high_QC_RSD.csv"

    df_imputed.to_csv(output_path_full, index=False, encoding="utf-8-sig")
    filtered_df.to_csv(output_path_filtered, index=False, encoding="utf-8-sig")
    removed_df.to_csv(output_path_removed, index=False, encoding="utf-8-sig")

    print(f"Imputation complete. Overwritten QC/Sample RSDs.")
    print(f"Saved all results:\n - Imputed: {output_path_full}\n - Filtered: {output_path_filtered}\n - Removed QC>threshold: {output_path_removed}")

    return output_path_filtered


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Impute missing values, recompute stats, and apply QC RSD filtering.")
    parser.add_argument("--final", required=True, help="Path to Final_search_results.csv")
    parser.add_argument("--groups", required=True, help="Path to sample_groups.csv")
    parser.add_argument("--out", default="results", help="Output folder")
    args = parser.parse_args()

    impute_missing_values(args.final, args.groups, args.out)
