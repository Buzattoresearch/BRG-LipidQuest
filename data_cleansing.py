#TODO: Check filtering performance.

import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend (no GUI required)
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def _normalize_polarity(x):
    s = str(x).strip().lower()
    if s in {"pos", "positive", "+", "1"}:
        return "pos"
    if s in {"neg", "negative", "-", "-1"}:
        return "neg"
    return None


def _load_contaminants_by_polarity(contaminant_csv_path):
    """Return {'pos':[mz...], 'neg':[mz...]} from Appendix/Contaminants.csv"""
    cdf = pd.read_csv(contaminant_csv_path)
    if "polarity" not in cdf.columns or "m/z" not in cdf.columns:
        raise ValueError("Contaminants.csv must have columns: 'm/z', 'polarity'")
    cdf["polarity"] = cdf["polarity"].map(_normalize_polarity)
    cdf = cdf.dropna(subset=["polarity", "m/z"])
    return {
        "pos": cdf.loc[cdf["polarity"] == "pos", "m/z"].astype(float).tolist(),
        "neg": cdf.loc[cdf["polarity"] == "neg", "m/z"].astype(float).tolist(),
    }


def _auto_mz_col(df):
    for cand in ["m/z meas.", "m/z", "m/z measured", "mz"]:
        if cand in df.columns:
            return cand
    raise ValueError("Could not find an m/z column.")


# -------------------------------------------------------------------------
# Baseline contaminant detection
# -------------------------------------------------------------------------
def _row_is_internal_standard(row, is_columns=("IS","Type","Sample Type")):
    """Return True if any column suggests an internal standard."""
    for col in is_columns:
        if col in row.index:
            val = str(row[col]).strip().lower()
            if val in {"is", "internal standard", "std", "standard"}:
                return True
    for col in row.index:
        if isinstance(row[col], str):
            txt = row[col].lower()
            if "internal standard" in txt or "avanti splash" in txt:
                return True
    return False


def detect_flat_features(
    df: pd.DataFrame,
    prefix="[POS",
    rel_std_thresh=0.075,
    intensity_quantile_low=0.90,
    intensity_quantile_high=0.999,
    debug_folder=None,
    plot=True,
    exclude_is=True,
    is_columns=("IS","Type","Sample Type")
):
    """
    Detect flat-intensity features to remove baseline contaminants (low intensity)
    and saturated peaks (high intensity) using RSD thresholds only.

    Parameters
    ----------
    rel_std_thresh : float
        RSD cutoff below which features are considered "flat".
    intensity_quantile_low : float
        Lower intensity quantile (baseline cutoff).
    intensity_quantile_high : float
        Upper intensity quantile (saturation cutoff).
    exclude_is : bool
        Skip internal standards (IS) from removal.
    """

    sample_cols = [c for c in df.columns if str(c).startswith(prefix)]
    if not sample_cols:
        print(f"No sample columns found starting with '{prefix}'. Skipping baseline detection.")
        return [], pd.DataFrame(columns=df.columns)

    # Compute per-feature mean intensity and RSD
    sample_data = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    mean_intensity = sample_data.mean(axis=1)
    rel_std = sample_data.std(axis=1) / mean_intensity.replace(0, np.nan)

    # Quantile cutoffs
    q_low = np.nanquantile(mean_intensity, intensity_quantile_low)
    q_high = np.nanquantile(mean_intensity, intensity_quantile_high)

    # Optional IS exclusion
    if exclude_is:
        is_mask = df.apply(_row_is_internal_standard, axis=1)
    else:
        is_mask = pd.Series(False, index=df.index)

    # Category masks
    baseline_mask = (rel_std < rel_std_thresh) & (mean_intensity <= q_low) & (~is_mask)
    saturated_mask = (rel_std < rel_std_thresh) & (mean_intensity >= q_high) & (~is_mask)

    # Combine and label
    combined_mask = baseline_mask | saturated_mask
    baseline_df = df.loc[combined_mask].copy()
    baseline_df["data_cleanup_reason"] = np.where(
        baseline_mask.loc[combined_mask],
        "Baseline contaminant (flat low intensity)",
        "Saturated feature (flat high intensity)"
    )

    # Annotate for plotting
    df["_MeanIntensity"] = mean_intensity
    df["_RelStd"] = rel_std
    df["_Flag"] = combined_mask.astype(int)
    df["_FlagType"] = np.where(baseline_mask, "baseline", np.where(saturated_mask, "saturated", "none"))

    # Counts
    n_total = len(df)
    n_baseline = int(baseline_mask.sum())
    n_saturated = int(saturated_mask.sum())
    n_flagged = int(combined_mask.sum())

    print(f"Baseline/saturation detection ({prefix}):", flush=True)
    print(f"  RSD threshold     : {rel_std_thresh}", flush=True)
    print(f"  Low-intensity q{intensity_quantile_low*100:.0f} cutoff : {q_low:.2f}", flush=True)
    print(f"  High-intensity q{intensity_quantile_high*100:.0f} cutoff: {q_high:.2f}", flush=True)
    print(f"  Baseline features : {n_baseline}", flush=True)
    print(f"  Saturated features: {n_saturated}", flush=True)
    print(f"  Total flagged     : {n_flagged} ({n_flagged/n_total*100:.2f} % of total)", flush=True)

    # Save summary log
    if debug_folder:
        debug_folder = Path(debug_folder)
        debug_folder.mkdir(parents=True, exist_ok=True)
        with open(debug_folder / "baseline_filter_report.txt", "w", encoding="utf-8") as f:
            f.write(f"Baseline/saturation detection ({prefix})\n")
            f.write(f"rel_std_thresh = {rel_std_thresh}\n")
            f.write(f"intensity_quantile_low  = {intensity_quantile_low}\n")
            f.write(f"intensity_quantile_high = {intensity_quantile_high}\n")
            f.write(f"Baseline (flat low)     = {n_baseline}\n")
            f.write(f"Saturated (flat high)   = {n_saturated}\n")
            f.write(f"Total flagged           = {n_flagged}\n")

    # Plot diagnostics
    if plot and debug_folder is not None:
        try:
            plt.figure(figsize=(7, 6))
            plt.scatter(df["_MeanIntensity"], df["_RelStd"], s=20, alpha=0.3, color="gray", label="All features")
            if n_baseline:
                plt.scatter(
                    df.loc[baseline_mask, "_MeanIntensity"],
                    df.loc[baseline_mask, "_RelStd"],
                    s=25, color="red", label=f"Baseline (n={n_baseline})"
                )
            if n_saturated:
                plt.scatter(
                    df.loc[saturated_mask, "_MeanIntensity"],
                    df.loc[saturated_mask, "_RelStd"],
                    s=25, color="orange", label=f"Saturated (n={n_saturated})"
                )
            plt.axhline(rel_std_thresh, color="blue", ls="--", lw=1, label=f"RSD<{rel_std_thresh}")
            plt.axvline(q_low, color="purple", ls=":", lw=1, label=f"Low q{intensity_quantile_low*100:.0f}")
            plt.axvline(q_high, color="green", ls=":", lw=1, label=f"High q{intensity_quantile_high*100:.0f}")
            plt.xscale("log"); plt.yscale("log")
            plt.xlabel("Mean Intensity (log scale)")
            plt.ylabel("Relative Standard Deviation (std/mean, log scale)")
            plt.title(f"Baseline and Saturation Detection ({prefix})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(debug_folder) / "Baseline_flag_vs_meanIntensity.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"[WARNING] Plot failed: {e}")

    return list(df.index[combined_mask]), baseline_df



# -------------------------------------------------------------------------
# Main data cleansing
# -------------------------------------------------------------------------
def apply_data_cleansing(
    df: pd.DataFrame,
    output_folder,
    contaminant_file="Appendix/Contaminants.csv",
    ppm_tolerance=5,
    min_int=None,
    prefix="[POS",
    rsd_thresh=None,                
    rsd_qc_thresh=30.0,
    min_detect_in_group=80.0,
    max_group_rsd_thresh=50.0
):

    """
    Cleansing steps:
      1. Remove known contaminants
      2. Remove baseline/saturated features
      3. Remove features below intensity threshold
      4. Apply statistical QC filters:
         - High QC RSD
         - Low within-group detection
         - High group RSD

    Outputs (under output_folder/debug):
        Removed_contaminants.csv
        Removed_baseline.csv
        Cleaned_data.csv
    """
    output_folder = Path(output_folder)
    debug_folder = output_folder / "debug"
    debug_folder.mkdir(parents=True, exist_ok=True)

    if rsd_thresh is None:
        rsd_thresh = 0.075
    # --- 1. Remove known m/z contaminants ---
    if "Polarity" not in df.columns:
        raise ValueError("Input DataFrame must have a 'Polarity' column.")
    mz_col = _auto_mz_col(df)
    cont = _load_contaminants_by_polarity(contaminant_file)
    pol = df["Polarity"].map(_normalize_polarity)
    mz_vals = pd.to_numeric(df[mz_col], errors="coerce").values

    def match_any(mz_array, targets, ppm):
        if not targets:
            return np.zeros_like(mz_array, dtype=bool), np.full(len(mz_array), np.nan)
        overall = np.zeros_like(mz_array, dtype=bool)
        which = np.full(len(mz_array), np.nan)
        for t in targets:
            lower, upper = t * (1 - ppm / 1e6), t * (1 + ppm / 1e6)
            mask = (mz_array >= lower) & (mz_array <= upper)
            newly = mask & (~overall)
            which[newly] = t
            overall |= mask
        return overall, which

    pos_mask = pol == "pos"
    neg_mask = pol == "neg"
    pos_hits, pos_target = match_any(mz_vals, cont["pos"], ppm_tolerance)
    neg_hits, neg_target = match_any(mz_vals, cont["neg"], ppm_tolerance)

    remove_flag = (pos_mask.values & pos_hits) | (neg_mask.values & neg_hits)
    reasons = np.full(len(df), "", dtype=object)
    reasons[pos_mask.values & pos_hits] = [
        f"contaminant (pos) @ {t:.4f} ppm±{ppm_tolerance}" if not np.isnan(t) else "contaminant (pos)"
        for t in pos_target[pos_mask.values & pos_hits]
    ]
    reasons[neg_mask.values & neg_hits] = [
        f"contaminant (neg) @ {t:.4f} ppm±{ppm_tolerance}" if not np.isnan(t) else "contaminant (neg)"
        for t in neg_target[neg_mask.values & neg_hits]
    ]

    removed_known = df.loc[remove_flag].copy()
    if not removed_known.empty:
        removed_known["data_cleanup_reason"] = reasons[remove_flag]
        removed_known.to_csv(debug_folder / "Removed_contaminants.csv", index=False, encoding="utf-8-sig")

    df_clean = df.loc[~remove_flag].copy()

    # --- 2. Baseline contaminants ---
    baseline_indices, baseline_df = detect_flat_features(df_clean, prefix=prefix, debug_folder=debug_folder)
    if not baseline_df.empty:
        baseline_df.to_csv(debug_folder / "Removed_baseline.csv", index=False, encoding="utf-8-sig")
        df_clean = df_clean.drop(index=baseline_indices)

    # --- 2.5 Remove repeated low-intensity m/z detections with similar intensities ---
    print("\n[STEP] Checking for repeated low-intensity m/z features with similar intensities...", flush=True)
    mz_col = _auto_mz_col(df_clean)

    # Parameters
    repetition_threshold = 3            # remove if detected ≥3 times
    low_intensity_threshold = 100000     # all mean intensities < this
    similarity_tolerance = 1.00         # ±100% relative difference allowed between intensities

    # Compute per-feature mean intensity
    sample_cols = [c for c in df_clean.columns if c.startswith("[POS") or c.startswith("[NEG]")]
    df_clean["_mean_intensity"] = df_clean[sample_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Round m/z to the third decimal for grouping
    df_clean["_mz_rounded"] = pd.to_numeric(df_clean[mz_col], errors="coerce").round(2)

    removed_dup_rows = []
    keep_indices = []

    for mz_val, group in df_clean.groupby("_mz_rounded"):
        if len(group) >= repetition_threshold and (group["_mean_intensity"] < low_intensity_threshold).all():
            # Check intensity similarity (within ±20%)
            mean_vals = group["_mean_intensity"].dropna()
            if mean_vals.empty:
                continue
            intensity_range = mean_vals.max() / max(mean_vals.min(), 1e-9)
            if intensity_range <= (1 + similarity_tolerance):
                # Rank by missing count (fewer first), then by mean intensity (higher first)
                group["_missing_count"] = group[sample_cols].isna().sum(axis=1)
                group_sorted = group.sort_values(["_missing_count", "_mean_intensity"], ascending=[True, False])
                best = group_sorted.iloc[0]
                keep_indices.append(best.name)

                to_remove = group_sorted.iloc[1:]
                removed_dup_rows.append(to_remove)
            else:
                # Intensities not similar → keep all
                keep_indices.extend(group.index.tolist())
        else:
            keep_indices.extend(group.index.tolist())

    if removed_dup_rows:
        removed_dup_df = pd.concat(removed_dup_rows, ignore_index=True)
        removed_dup_df["data_cleanup_reason"] = (
            f"Duplicate low-intensity m/z (≥{repetition_threshold} detections, <{low_intensity_threshold}, similar ±{similarity_tolerance*100:.0f}%)"
        )
        removed_dup_df.to_csv(debug_folder / "Removed_duplicate_low_intensity_mz.csv",
                              index=False, encoding="utf-8-sig")
        print(f"Removed {len(removed_dup_df)} repeated low-intensity m/z features with similar intensities.", flush=True)

        df_clean = df_clean.loc[sorted(set(keep_indices))].copy()
    else:
        print("No repeated low-intensity m/z features detected.", flush=True)


    # --- Identify Internal Standards (IS) to exclude from all statistical filtering ---
    is_mask = df_clean.apply(_row_is_internal_standard, axis=1)
    print(f"[INFO] Identified {is_mask.sum()} internal standards to skip during filtering", flush=True)

    # --- 3. Remove features below minimum intensity threshold ---
    if min_int is not None:
        if "Average Intensity (all samples)" not in df_clean.columns:
            print("[WARNING] Average Intensity column not found — skipping intensity filtering.")
        else:
            before = len(df_clean)
            df_clean = df_clean[df_clean["Average Intensity (all samples)"] >= float(min_int)].copy()
            after = len(df_clean)
            removed_intensity = before - after

            print(f"Removed {removed_intensity} features below minimum intensity {min_int}. "
                  f"Kept {after} rows.")

            # Save removed features for transparency
            removed_low_intensity = df.loc[
                df["Average Intensity (all samples)"] < float(min_int)
            ].copy()
            if not removed_low_intensity.empty:
                removed_low_intensity["data_cleanup_reason"] = f"Average intensity < {min_int}"
                removed_low_intensity.to_csv(
                    debug_folder / "Removed_low_intensity.csv",
                    index=False,
                    encoding="utf-8-sig"
                )

    # --- 4. Additional statistical QC filters (robust + diagnostic) ---
    qc_rsd_max = float(rsd_qc_thresh)
    min_detect_frac = float(min_detect_in_group) / 100.0
    group_rsd_max = float(max_group_rsd_thresh)

    print("\n=== Statistical QC Filtering Parameters ===", flush=True)
    print(f"  QC RSD threshold            : {qc_rsd_max:.1f}%")
    print(f"  Min. detection within group : {min_detect_frac*100:.1f}%")
    print(f"  Max. within-group RSD       : {group_rsd_max:.1f}%")
    print("============================================\n", flush=True)

    removed_stats = []
    debug_folder.mkdir(parents=True, exist_ok=True)

    def _to_num(series):
        return pd.to_numeric(
            series.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.extract(r"([-+]?[0-9]*\.?[0-9]+)")[0]
            .replace(["", "nan", "None"], np.nan),
            errors="coerce"
        )

    # === 4.1 Remove features with high QC RSD (ignore missing) ===
    qc_cols = [c for c in df_clean.columns if re.search(r"(?i)\brsd[_\s\-]*qc", c)]
    print(f"[DEBUG] QC RSD columns detected: {qc_cols}", flush=True)

    if qc_cols:
        qc_col = qc_cols[0]
        qc_values = _to_num(df_clean[qc_col])
        is_mask = df_clean.apply(_row_is_internal_standard, axis=1)

        mask_remove = (qc_values.notna() & (qc_values >= qc_rsd_max)) & (~is_mask)
        mask_keep = ~mask_remove
        n_removed = int(mask_remove.sum())

        if n_removed > 0:
            removed_rsd_df = df_clean.loc[mask_remove].copy()
            removed_rsd_df["data_cleanup_reason"] = f"QC RSD ≥ {qc_rsd_max}%"
            removed_rsd_df.to_csv(debug_folder / "Removed_by_RSD.csv", index=False, encoding="utf-8-sig")
            print(f"Removed {n_removed} features with QC RSD ≥ {qc_rsd_max}%")
            df_clean = df_clean.loc[mask_keep].copy()

        else:
            print(f"No features exceeded QC RSD ≥ {qc_rsd_max}%", flush=True)

        removed_stats.append(f"High QC RSD (≥{qc_rsd_max}%)")
    else:
        print("[WARNING] No QC RSD column detected for filtering", flush=True)


    # === 4.2 Remove features not detected in ≥80% of samples within any group ===
    group_file = Path(output_folder) / "sample_groups.csv"
    if group_file.exists():
        try:
            group_df = pd.read_csv(group_file)
            group_map = dict(zip(group_df["Sample"], group_df["Group"]))
            group_names = sorted(set(group_map.values()))
            print(f"[DEBUG] Groups detected for filtering: {group_names}", flush=True)

            detection_masks = []
            for group in group_names:
                sample_list = [s for s, g in group_map.items() if g == group and s in df_clean.columns]
                print(f"[DEBUG] Group {group}: {len(sample_list)} samples → {sample_list}", flush=True)
                if not sample_list:
                    continue

                vals = df_clean[sample_list].apply(pd.to_numeric, errors="coerce")
                # count non-NaN and >0 as detections
                detect_mask = vals.fillna(0) > 1e-9
                detect_frac = detect_mask.sum(axis=1) / len(sample_list)

                detection_masks.append(detect_frac >= min_detect_frac)

            if detection_masks:
                # Keep features detected in ≥80% of samples in ANY group
                keep_mask = pd.concat(detection_masks, axis=1).any(axis=1)
                remove_mask = (~keep_mask) & (~is_mask)
                n_removed = int(remove_mask.sum())

                if n_removed > 0:
                    removed_low_detect = df_clean.loc[remove_mask].copy()
                    removed_low_detect["data_cleanup_reason"] = f"<{min_detect_frac*100:.0f}% detected in all groups"
                    removed_low_detect_path = debug_folder / "Removed_low_detection.csv"
                    removed_low_detect.to_csv(removed_low_detect_path, index=False, encoding="utf-8-sig")
                    print(f"Removed {n_removed} features detected in <{min_detect_frac*100:.0f}% of samples for every group → {removed_low_detect_path}", flush=True)
                    df_clean = df_clean.loc[keep_mask].copy()
                else:
                    print(f"No features removed by <{min_detect_frac*100:.0f}% detection rule", flush=True)

                removed_stats.append(f"<{min_detect_frac*100:.0f}% detected in all groups")
            else:
                print("[WARNING] No valid group sample lists found", flush=True)

        except Exception as e:
            print(f"[WARNING] Detection filter skipped: {e}", flush=True)
    else:
        print("[WARNING] No sample_groups.csv found — skipping detection filter", flush=True)


        # === 4.3 Remove features with all group RSD ≥ 50% (ignore missing) ===
    group_rsd_cols = [
        c for c in df_clean.columns
        if re.search(r"(?i)\brsd[_\s]*", c)
        and not re.search(r"(?i)samples|qc", c)
    ]
    print(f"[DEBUG] Group RSD columns detected: {group_rsd_cols}", flush=True)

    if group_rsd_cols:
        df_rsd = df_clean[group_rsd_cols].apply(_to_num)
        print(f"[DEBUG] Group RSD overall range: min={df_rsd.min().min()}, max={df_rsd.max().max()}, median={df_rsd.median().median()}", flush=True)

        # Keep features if at least one group has RSD < threshold OR if all RSDs are missing
        mask_all_na = df_rsd.isna().all(axis=1)
        mask_keep = (df_rsd < group_rsd_max).any(axis=1) | mask_all_na
        mask_remove = (~mask_keep) & (~is_mask)

        n_removed = int(mask_remove.sum())
        if n_removed > 0:
            removed_high_rsd = df_clean.loc[mask_remove].copy()
            removed_high_rsd["data_cleanup_reason"] = f"All group RSD ≥ {group_rsd_max}%"
            removed_high_rsd_path = debug_folder / "Removed_high_group_RSD.csv"
            removed_high_rsd.to_csv(removed_high_rsd_path, index=False, encoding="utf-8-sig")
            print(f"Removed {n_removed} features with all group RSD ≥ {group_rsd_max}% → {removed_high_rsd_path}", flush=True)
            df_clean = df_clean.loc[mask_keep].copy()
        else:
            print(f"No features removed by group RSD ≥ {group_rsd_max}% rule", flush=True)

        removed_stats.append(f"All group RSD ≥ {group_rsd_max}%")
    else:
        print("[WARNING] No group RSD columns detected for filtering", flush=True)

    
    # --- 5. Plot filtering summary (accurate counts after each step) ---
    try:
        step_labels = []
        step_counts = []

        # Initial count
        step_labels.append("Initial")
        step_counts.append(len(df))

        # After contaminant removal
        df_after_contam = df.loc[~remove_flag]
        step_labels.append("Removed contaminants")
        step_counts.append(len(df_after_contam))

        # After baseline removal
        df_after_baseline = df_after_contam.drop(index=baseline_indices, errors="ignore")
        step_labels.append("Removed baseline/saturated")
        step_counts.append(len(df_after_baseline))

        # After minimum intensity filter
        if min_int is not None and "Average Intensity (all samples)" in df_after_baseline.columns:
            df_after_intensity = df_after_baseline[
                df_after_baseline["Average Intensity (all samples)"] >= float(min_int)
            ].copy()
        else:
            df_after_intensity = df_after_baseline.copy()
        step_labels.append("Min intensity filter")
        step_counts.append(len(df_after_intensity))

        # After QC RSD ≥ threshold
        df_after_qc = df_after_intensity.copy()
        qc_cols = [c for c in df_after_qc.columns if re.search(r"(?i)\brsd[_\s\-]*qc", c)]
        if qc_cols:
            qc_col = qc_cols[0]
            qc_values = pd.to_numeric(
                df_after_qc[qc_col].astype(str).str.replace("%", "", regex=False),
                errors="coerce"
            )
            df_after_qc = df_after_qc[qc_values < qc_rsd_max].copy()
        step_labels.append(f"QC RSD ≥ {qc_rsd_max}%")
        step_counts.append(len(df_after_qc))

        # After <80% detection rule
        df_after_detect = df_after_qc.copy()
        if group_file.exists():
            try:
                group_df = pd.read_csv(group_file)
                group_map = dict(zip(group_df["Sample"], group_df["Group"]))
                detection_masks = []
                for group in sorted(set(group_map.values())):
                    sample_list = [s for s, g in group_map.items() if g == group and s in df_after_detect.columns]
                    if not sample_list:
                        continue
                    vals = df_after_detect[sample_list].apply(pd.to_numeric, errors="coerce")
                    detect_mask = vals.fillna(0) > 1e-9
                    detect_frac = detect_mask.sum(axis=1) / len(sample_list)
                    detection_masks.append(detect_frac >= min_detect_frac)
                if detection_masks:
                    keep_mask = pd.concat(detection_masks, axis=1).any(axis=1)
                    df_after_detect = df_after_detect.loc[keep_mask].copy()
            except Exception as e:
                print(f"[WARNING] Detection summary step skipped: {e}", flush=True)
        step_labels.append(f"<{min_detect_frac*100:.0f}% detected in all groups")
        step_counts.append(len(df_after_detect))

        # After group RSD ≥ threshold
        df_after_group_rsd = df_after_detect.copy()
        group_rsd_cols = [
            c for c in df_after_group_rsd.columns
            if re.search(r"(?i)\brsd[_\s]*", c)
            and not re.search(r"(?i)samples|qc", c)
        ]
        if group_rsd_cols:
            df_rsd = df_after_group_rsd[group_rsd_cols].apply(
                lambda s: pd.to_numeric(
                    s.astype(str)
                    .str.replace("%", "", regex=False)
                    .str.extract(r"([-+]?[0-9]*\.?[0-9]+)")[0],
                    errors="coerce",
                )
            )
            mask_keep = (df_rsd < group_rsd_max).any(axis=1) | (df_rsd.isna().all(axis=1))
            df_after_group_rsd = df_after_group_rsd.loc[mask_keep].copy()
        step_labels.append(f"All group RSD ≥ {group_rsd_max}%")
        step_counts.append(len(df_after_group_rsd))

        # Final kept
        step_labels.append("Final (kept)")
        step_counts.append(len(df_after_group_rsd))

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(step_labels)), step_counts, color="#87CEFA", edgecolor="black")
        ax.set_xticks(range(len(step_labels)))
        ax.set_xticklabels(step_labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Features retained")
        ax.set_title("Feature Retention Across Data Cleansing Steps")

        for i, count in enumerate(step_counts):
            ax.text(i, count + max(step_counts) * 0.01, str(count), ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        plt.savefig(debug_folder / "Filtering_summary.png", dpi=200)
        plt.close(fig)

        print(f"Filtering summary plot saved → {debug_folder}/Filtering_summary.png", flush=True)

    except Exception as e:
        print(f"[WARNING] Failed to generate Filtering_summary plot: {e}", flush=True)

    # --- Final save ---
    df_clean.to_csv(debug_folder / "Cleaned_data.csv", index=False, encoding="utf-8-sig")
    
    with open(debug_folder / "thresholds_used.txt", "w", encoding="utf-8") as f:
        f.write("Data Cleansing Thresholds Used:\n")
        f.write(f"QC RSD threshold (%)             : {qc_rsd_max}\n")
        f.write(f"Minimum detection per group (%)  : {min_detect_frac*100:.1f}\n")
        f.write(f"Max within-group RSD threshold (%) : {group_rsd_max}\n")
        f.write(f"Flat peak RSD threshold          : {rsd_thresh}\n")
        f.write(f"Minimum intensity                : {min_int}\n")

    print(f"Removed {len(removed_known)} known contaminants and {len(baseline_df)} baseline features. "
          f"Kept {len(df_clean)} rows.", flush = True)

    return df_clean, removed_known, baseline_df
