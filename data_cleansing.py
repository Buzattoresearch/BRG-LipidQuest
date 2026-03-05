
'''
This script is designed to remove background noise, contaminants, and unstable or poorly detected features. Internal standards are not affected. The result is a cleaner, more statistically reliable feature table for downstream lipidomics.
This is a multi-stage filter that targets:
    
    known contaminant masses (explicit blacklist): each detected m/z is compared to a list of known background contaminants. If a feature matches a known contaminant mass (within a small tolerance of 5 ppm), it is removed. 
    Example: common plasticizers, solvent background, media components, etc.
    
    near-constant peaks across samples at the bottom or top of intensity distribution (baseline bleed and saturation artifacts). It looks for features that:
        Have almost no variation across samples
        Are consistently very low intensity (baseline noise)
        Or consistently extremely high intensity (detector saturation)
        If a peak is basically constant across all samples and sits at the very low or very high end of intensity, it is flagged and removed.
    
    repeated low-intensity m/z “clones”
        If the same mass shows up multiple times at low intensity with nearly identical signal patterns, the script keeps the best representative and removes the redundant duplicates.
        This helps clean up repeated background artifacts.
        
    unstable features in QC (RSD >70%), with a stability-based rescue
        If QC samples are defined, it calculates how much each feature varies across QCs.
        If variation is too high (>70% for raw intensities - before normalization), the feature is removed. However, there is a safeguard:
        If a feature is stable within at least one biological group (RSD sample <30%), it is rescued even if QC variability is high.
    
    features with poor detection rates within groups
        If a minimum intensity threshold is provided, features below that average intensity are removed (default: 3000).
        
    features that are rarely detected
        For each biological group, the script checks how often a feature is detected.
        If a feature is missing in too many samples across all groups (detected in less than 80% across all sample groups), it is removed. This eliminates features that are mostly absent or sporadic.                  
'''

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
    cdf = pd.read_csv(contaminant_csv_path, low_memory=False)
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
    prefix="P_",
    rel_std_thresh=0.05,
    rel_std_thresh_high=1.2,
    intensity_quantile_low=0.10,
    intensity_quantile_high=0.99,
    debug_folder=None,
    plot=True,
    exclude_is=True,
    is_columns=("IS","Type","Sample Type"),
    pol_tag: str = ""
):

    print("\nRemoving flat-intensity features (baseline contaminants or saturated features)...\n")

    # 1) pick sample columns for this polarity
    # prefix may be None (P_/N_ workflow)
    if prefix is None:
        # Do NOT select any sample columns — skip baseline detection entirely
        print("\n===== prefix=None → skipping baseline/saturation detection =====\n")
        return [], pd.DataFrame(columns=df.columns)

    # Normal behavior for prefix = "P_" or "N_" or "[POS"
    sample_cols = [c for c in df.columns if isinstance(c, str) and c.strip().startswith(prefix)]
    if not sample_cols:
        print(f"\n===== No sample columns found starting with '{prefix}'. Skipping baseline detection. =====\n")
        return [], pd.DataFrame(columns=df.columns)


    # 2) mean and RSD
    sample_data = df[sample_cols].apply(pd.to_numeric, errors="coerce")

    # Treat non-detections as missing for baseline stats
    detect_mask = sample_data > 1e-9
    n_detect = detect_mask.sum(axis=1)

    sample_data_det = sample_data.mask(~detect_mask, np.nan)

    mean_intensity = sample_data_det.mean(axis=1)
    rel_std = sample_data_det.std(axis=1, ddof=1) / mean_intensity.replace(0, np.nan)

    # Optional: make RSD undefined when too few detections
    rel_std = rel_std.where(n_detect >= 3, np.nan)

    # Store for plotting/debug
    df["_N_detect"] = n_detect

    # 3) safe quantiles (avoid empty / all-NaN)
    vals = mean_intensity.to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        # nothing to evaluate; return early
        print("All mean intensities are NaN/0; skipping baseline detection.\n")
        df["_MeanIntensity"] = mean_intensity
        df["_RelStd"] = rel_std
        df["_Flag"] = 0
        df["_FlagType"] = "none"
        return [], pd.DataFrame(columns=df.columns)

    q_low  = np.nanquantile(vals, intensity_quantile_low)
    q_high = np.nanquantile(vals, intensity_quantile_high)
    if q_low > q_high:
        q_low, q_high = q_high, q_low

    # 4) internal standards mask
    is_mask = df.apply(_row_is_internal_standard, axis=1) if exclude_is else pd.Series(False, index=df.index)

    # 5) classify
    baseline_mask       = (rel_std < rel_std_thresh)       & (mean_intensity <= q_low)  & (~is_mask)
    baseline_mask_high  = (rel_std > rel_std_thresh_high)  & (mean_intensity <= q_low)  & (~is_mask)
    saturated_mask      = (rel_std < rel_std_thresh)       & (mean_intensity >= q_high) & (~is_mask)
    combined_mask       = baseline_mask | baseline_mask_high | saturated_mask

    baseline_df = df.loc[combined_mask].copy()

    # explicit 3-class reason assignment, aligned to baseline_df index
    reason = pd.Series("", index=baseline_df.index, dtype=object)
    reason.loc[baseline_df.index[baseline_mask.loc[baseline_df.index]]] = "Baseline contaminant (flat low intensity)"
    reason.loc[baseline_df.index[baseline_mask_high.loc[baseline_df.index]]] = "Low-intensity unstable feature (high RSD)"
    reason.loc[baseline_df.index[saturated_mask.loc[baseline_df.index]]] = "Saturated feature (flat high intensity)"
    baseline_df["data_cleanup_reason"] = reason.values

    # annotate for plotting/logging
    df["_MeanIntensity"] = mean_intensity
    df["_RelStd"] = rel_std
    df["_Flag"] = combined_mask.astype(int)

    df["_FlagType"] = "none"
    df.loc[baseline_mask, "_FlagType"] = "baseline"
    df.loc[baseline_mask_high, "_FlagType"] = "baseline_high"
    df.loc[saturated_mask, "_FlagType"] = "saturated"

    n_total     = len(df)
    n_baseline  = int(baseline_mask.sum())
    n_saturated = int(saturated_mask.sum())
    n_flagged   = int(combined_mask.sum())

    print(f"Baseline/saturation detection ({prefix}):")
    print(f"  RSD threshold                : {rel_std_thresh}")
    print(f"  Low-intensity q{intensity_quantile_low*100:.1f} cutoff : {q_low:.2f}")
    print(f"  High-intensity q{intensity_quantile_high*100:.1f} cutoff: {q_high:.2f}")
    print(f"  Baseline features            : {n_baseline}")
    print(f"  Saturated features           : {n_saturated}")
    print(f"  Total flagged                : {n_flagged} ({n_flagged/n_total*100:.2f} % of total)")

    # write report
    if debug_folder:
        d = Path(debug_folder); d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{pol_tag}baseline_filter_report.txt", "w", encoding="utf-8") as f:
            f.write(f"Baseline/saturation detection ({prefix})\n")
            f.write(f"rel_std_thresh = {rel_std_thresh}\n")
            f.write(f"intensity_quantile_low  = {intensity_quantile_low}\n")
            f.write(f"intensity_quantile_high = {intensity_quantile_high}\n")
            f.write(f"q_low  = {q_low}\nq_high = {q_high}\n")
            f.write(f"Baseline (flat low)     = {n_baseline}\n")
            f.write(f"Saturated (flat high)   = {n_saturated}\n")
            f.write(f"Total flagged           = {n_flagged}\n")

        if plot:
            try:
                n_baseline_high = int(baseline_mask_high.sum())

                plt.figure(figsize=(7, 6))
                plt.scatter(
                    df["_MeanIntensity"], df["_RelStd"],
                    s=20, alpha=0.3, color="gray", label="All"
                )

                if n_baseline:
                    plt.scatter(
                        df.loc[baseline_mask, "_MeanIntensity"],
                        df.loc[baseline_mask, "_RelStd"],
                        s=25, color="red", label=f"Baseline (n={n_baseline})"
                    )

                if n_baseline_high:
                    plt.scatter(
                        df.loc[baseline_mask_high, "_MeanIntensity"],
                        df.loc[baseline_mask_high, "_RelStd"],
                        s=25, color="magenta", label=f"Low-int unstable (n={n_baseline_high})"
                    )

                if n_saturated:
                    plt.scatter(
                        df.loc[saturated_mask, "_MeanIntensity"],
                        df.loc[saturated_mask, "_RelStd"],
                        s=25, color="orange", label=f"Saturated (n={n_saturated})"
                    )

                plt.axhline(rel_std_thresh, color="blue", ls="--", lw=1, label=f"RSD<{rel_std_thresh}")
                plt.axhline(rel_std_thresh_high, color="magenta", ls="--", lw=1, label=f"RSD>{rel_std_thresh_high}")
                plt.axvline(q_low,  color="purple", ls=":", lw=1, label=f"Low q{intensity_quantile_low*100:.1f}")
                plt.axvline(q_high, color="green",  ls=":", lw=1, label=f"High q{intensity_quantile_high*100:.1f}")

                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("Mean Intensity (log)")
                plt.ylabel("RSD (std/mean, log)")
                plt.title(f"Baseline & Saturation ({prefix})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(Path(debug_folder) / f"{pol_tag}Baseline_flag_vs_meanIntensity.png", dpi=100)
                plt.close()
            except Exception as e:
                print(f"[WARNING] Plot failed: {e}")

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
    prefix="P_",
    rsd_thresh=None,
    rsd_qc_thresh=75.0,      # percent
    qc_min_cols=3,
    min_detect_in_group=80.0,
    **_
):

    """
    Cleansing steps:
      1. Remove known contaminants
      2. Remove baseline/saturated features
      3. Remove features below intensity threshold
      4. Apply statistical QC filters:
         - Low within-group detection
        * RSD filtering is applied after normalization

    Outputs (under output_folder/debug):
        Removed_contaminants.csv
        Removed_baseline.csv
        Cleaned_data.csv
    """
    
    output_folder = Path(output_folder)
    debug_folder = output_folder / "debug"
    debug_folder.mkdir(parents=True, exist_ok=True)

    print(f'\nApplying data cleansing... \n')

    # --- Determine polarity tag for output filenames ---
    if "Polarity" in df.columns:
        first_pol = df["Polarity"].dropna().astype(str).str.lower().iloc[0]
        if "pos" in first_pol:
            pol_tag = "Pos_"
        elif "neg" in first_pol:
            pol_tag = "Neg_"
        else:
            pol_tag = ""
    else:
        pol_tag = ""


    if rsd_thresh is None:
        rsd_thresh = 0.05
        
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
        removed_known.to_csv(debug_folder / f"{pol_tag}Removed_contaminants.csv", index=False, encoding="utf-8-sig")

    df_clean = df.loc[~remove_flag].copy()

    #    # --- 2. Baseline contaminants ---
    baseline_indices, baseline_df = detect_flat_features(
        df_clean,
        prefix=prefix,
        rel_std_thresh=rsd_thresh,
        rel_std_thresh_high=1.2,
        debug_folder=debug_folder,
        pol_tag=pol_tag
    )
    if not baseline_df.empty:
        baseline_df.to_csv(debug_folder / f"{pol_tag}Removed_baseline.csv", index=False, encoding="utf-8-sig")
        df_clean = df_clean.drop(index=baseline_indices)

    # --- 2.5 Remove repeated low-intensity m/z detections with similar intensities ---
    print("\n[STEP] Checking for repeated low-intensity m/z features with similar intensities...", flush=True)
    mz_col = _auto_mz_col(df_clean)

    # Parameters
    repetition_threshold = 3            # remove if detected ≥3 times
    low_intensity_threshold = 100000     # all mean intensities < this
    similarity_tolerance = 1.00         # ±100% relative difference allowed between intensities

    # Compute per-feature mean intensity
    sample_cols = [c for c in df_clean.columns if c.startswith("P_") or c.startswith("N_")]
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
        removed_dup_df.to_csv(debug_folder / f"{pol_tag}Removed_duplicate_low_intensity_mz.csv",
                              index=False, encoding="utf-8-sig")
        print(f"Removed {len(removed_dup_df)} repeated low-intensity m/z features with similar intensities.", flush=True)

        df_clean = df_clean.loc[sorted(set(keep_indices))].copy()
    else:
        print("No repeated low-intensity m/z features detected.", flush=True)


    # --- Identify Internal Standards (IS) to exclude from all statistical filtering ---
    is_mask = df_clean.apply(_row_is_internal_standard, axis=1)
    print(f"[INFO] Identified {is_mask.sum()} internal standards to skip during filtering", flush=True)
    
    # --- 3 Rough QC RSD filter (based on sample_groups.csv, Group == 'QC') ---
    print(f"\n[STEP] QC RSD filtering (sample_groups.csv, Group == 'QC')... (threshold: {rsd_qc_thresh})", flush=True)

    group_file = Path(output_folder).parent / "sample_groups.csv"
    if not group_file.exists():
        print(f"[WARNING] No sample_groups.csv found at {group_file} — skipping QC RSD filter.", flush=True)
    else:
        try:
            group_df = pd.read_csv(group_file, low_memory=False)
            group_map = dict(zip(group_df["Sample"], group_df["Group"]))

            # Identify sample columns (same convention used elsewhere)
            sample_cols = [c for c in df_clean.columns if isinstance(c, str) and (c.startswith("P_") or c.startswith("N_"))]

            qc_cols = [c for c in sample_cols if str(group_map.get(c, "")).strip().lower() == "qc"]

            if len(qc_cols) < int(qc_min_cols):
                print(f"[WARNING] Found {len(qc_cols)} QC columns (<{qc_min_cols}) — skipping QC RSD filter.", flush=True)
            else:
                qc_data = df_clean[qc_cols].apply(pd.to_numeric, errors="coerce")

                qc_mean = qc_data.mean(axis=1)
                qc_std = qc_data.std(axis=1, ddof=1)
                qc_rsd_pct = (qc_std / qc_mean.replace(0, np.nan)) * 100.0

                # Require at least 2 detected QC values for RSD to mean anything
                qc_detect_n = (qc_data.fillna(0) > 1e-9).sum(axis=1)

                # ---------------------------------------------------------
                # Rescue: keep features that are stable in at least one NON-QC group
                # (RSD < 30% in at least one sample group)
                # ---------------------------------------------------------
                rescue_rsd_thresh = 30.0  # percent

                # collect non-QC groups present in sample_groups.csv
                non_qc_groups = sorted({
                    str(g).strip() for g in group_map.values()
                    if str(g).strip() and str(g).strip().lower() != "qc"
                })

                # default: no rescue
                keep_due_to_group_stability = pd.Series(False, index=df_clean.index)

                # compute within-group RSDs and build rescue mask
                for g in non_qc_groups:
                    g_cols = [c for c in sample_cols if str(group_map.get(c, "")).strip() == g]
                    if len(g_cols) < 3:
                        # RSD is weak with 1–2 replicates; skip group
                        continue

                    g_data = df_clean[g_cols].apply(pd.to_numeric, errors="coerce")
                    g_mean = g_data.mean(axis=1)
                    g_std  = g_data.std(axis=1, ddof=1)
                    g_rsd_pct = (g_std / g_mean.replace(0, np.nan)) * 100.0

                    # require at least 2 detections in the group
                    g_detect_n = (g_data.fillna(0) > 1e-9).sum(axis=1)

                    stable_in_group = (g_rsd_pct < rescue_rsd_thresh) & (g_detect_n >= 2)
                    keep_due_to_group_stability |= stable_in_group.fillna(False)
                    
                remove_mask_qc = (
                    (qc_rsd_pct > float(rsd_qc_thresh)) &
                    (qc_detect_n >= 2) &
                    (~is_mask) &
                    (~keep_due_to_group_stability)
                )

                n_removed_qc = int(remove_mask_qc.sum())

                if n_removed_qc > 0:
                    
                    removed_qc = df_clean.loc[remove_mask_qc].copy()
                    removed_qc["QC RSD [%]"] = qc_rsd_pct.loc[remove_mask_qc].values
                    removed_qc["_QC_N_detect"] = qc_detect_n.loc[remove_mask_qc].values
                    removed_qc["data_cleanup_reason"] = f"QC RSD > {rsd_qc_thresh:.1f}%"

                    out_path = debug_folder / f"{pol_tag}Removed_high_QC_RSD_{rsd_qc_thresh}.csv"
                    removed_qc.to_csv(out_path, index=False, encoding="utf-8-sig")

                    df_clean = df_clean.loc[~remove_mask_qc].copy()
                    print(f"Removed {n_removed_qc} features with QC RSD > {rsd_qc_thresh:.1f}% → {out_path}", flush=True)
                    n_rescued = int(((qc_rsd_pct > float(rsd_qc_thresh)) & (qc_detect_n >= 2) & (~is_mask) & (keep_due_to_group_stability)).sum())
                    print(f"[INFO] QC RSD rescue kept {n_rescued} features (stable in ≥1 non-QC group, RSD<{rescue_rsd_thresh}%)", flush=True)
                
                else:
                    print(f"No features removed by QC RSD > {rsd_qc_thresh:.1f}%", flush=True)

        except Exception as e:
            print(f"[WARNING] QC RSD filter skipped: {e}", flush=True)

    # --- 4. Remove features below minimum intensity threshold ---
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
                    debug_folder / f"{pol_tag}Removed_below_min_average_intensity.csv",
                    index=False,
                    encoding="utf-8-sig"
                )

    # --- 5. Additional statistical QC filters (robust + diagnostic) ---
    min_detect_frac = float(min_detect_in_group) / 100.0

    print("\n=== Statistical QC Filtering Parameters ===", flush=True)
    print(f"  Min. detection within group : {min_detect_frac*100:.1f}%")
    print("============================================\n", flush=True)

    removed_stats = []
    debug_folder.mkdir(parents=True, exist_ok=True)

    # === 4.1 Remove features not detected in ≥80% of samples within any group ===
    group_file = Path(output_folder).parent / "sample_groups.csv"
    if group_file.exists():
        try:
            group_df = pd.read_csv(group_file, low_memory=False)
            group_map = dict(zip(group_df["Sample"], group_df["Group"]))
            group_names = sorted(set(group_map.values()))

            detection_masks = []
            for group in group_names:
                sample_list = [s for s, g in group_map.items() if g == group and s in df_clean.columns]
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
                    removed_low_detect_path = debug_folder / f"{pol_tag}Removed_low_detection.csv"
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
       
    
    # --- 6. Plot filtering summary (accurate counts after each step) ---
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

        # After <80% detection rule
        df_after_detect = df_after_intensity.copy()
        if group_file.exists():
            try:
                group_df = pd.read_csv(group_file, low_memory=False)
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
       
        # Final kept
        step_labels.append("Final (kept)")
        step_counts.append(len(df_after_detect))

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
        plt.savefig(debug_folder / f"{pol_tag}Filtering_summary.png", dpi=100)
        plt.close(fig)

        print(f"Filtering summary plot saved → {debug_folder}/Filtering_summary.png", flush=True)

    except Exception as e:
        print(f"[WARNING] Failed to generate Filtering_summary plot: {e}", flush=True)

    # --- Final save ---
    df_clean.to_csv(debug_folder / f"{pol_tag}Cleaned_data.csv", index=False, encoding="utf-8-sig")
    
    with open(debug_folder / f"{pol_tag}thresholds_used.txt", "w", encoding="utf-8") as f:
        f.write("Data Cleansing Thresholds Used:\n")
        f.write(f"Minimum detection per group (%)  : {min_detect_frac*100:.1f}\n")
        f.write(f"Flat peak RSD threshold          : {rsd_thresh}\n")
        f.write(f"QC RSD threshold (%)             : {rsd_qc_thresh}\n")
        f.write(f"QC min columns                   : {qc_min_cols}\n")
        f.write(f"Minimum intensity                : {min_int}\n")

    print(f"Removed {len(removed_known)} known contaminants and {len(baseline_df)} baseline features. "
          f"Kept {len(df_clean)} rows.", flush = True)
    print(f'\nData cleansing complete.\n')
    return df_clean, removed_known, baseline_df
