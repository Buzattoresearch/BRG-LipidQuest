# ---------------------------------------------------------------------
# Missing value imputation and RSD recomputation for lipidomics datasets
# Integrated with QC RSD filtering consistent with data_cleansing.py
# ---------------------------------------------------------------------

'''
This module imputes missing intensities in a “final” lipidomics table and regenerates RSD and QC diagnostics in a way that matches downstream expectations.

It loads the feature table and the sample group map, detects sample intensity columns (P_ and N_), and infers a polarity tag for filenames. 
It then generates a QC plot showing the percent of missing values per sample before imputation. “Missing” means NA-like strings, NaN, or zeros.

Before converting zeros to NaN, it estimates LOD-style minima from the distribution of positive signals. 
It computes a dataset-wide positive 1st percentile and also a per-group positive 1st percentile, using the sample group definitions. Those values serve as fallback floors when a feature lacks enough positives.

Imputation runs vectorized per feature and per group. 
For each group, it computes a row wise 1st percentile of positives within that group, plus the detection rate of that feature within the group. 
It fills missing values in that group using a replacement value derived from the feature’s group LOD scaled by detection rate. 
High detection features get a larger replacement (LOD/3), mid detection gets an intermediate replacement, and low detection gets a smaller replacement (LOD/5). 
If the group specific row LOD cannot be computed, it falls back in order to the row global LOD, then the group global LOD, then the dataset global LOD.

After imputation it recomputes per feature summary metrics across all samples (average, min, max) and RSD across all samples. 
It then recomputes QC RSD and non-QC sample RSD using the QC labels in the group file, and also writes per-group RSD columns in the format RSD_<group> [%] so later steps such as LOESS and final file generation can find them.

Before saving, it drops verbose intermediate per-group intensity stats to keep outputs smaller, and it removes a small set of known temporary columns if present. 
It saves the imputed table to debug with the expected “3-Final_annotated_results_imputed.csv” style filename.

Finally it generates QC diagnostics to check whether imputation distorted signal. 
It compares summed intensities per sample before versus after, runs paired Wilcoxon tests per group on those sums, and saves a statistics table. 
It also saves before versus after group bar plots, group boxplots with jittered points, and a before versus after intensity distribution histogram on log10 scale.
'''

import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt

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
    row-in-group → row-global → group-global → dataset-global. Recompute RSDs.
    """
    import matplotlib.pyplot as plt
    final_csv = Path(final_csv)
    group_csv = Path(group_csv)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    # QC plot directory (used by all plots in this function)
    qc_dir = output_folder / "debug" / "missing_value_imputation"
    qc_dir.mkdir(parents=True, exist_ok=True)


    print(f'\nStarting missing value imputation...', flush=True)
    df = pd.read_csv(final_csv, low_memory=False)
    group_df = pd.read_csv(group_csv, low_memory=False)

    # --- Identify sample columns ---
    sample_cols = [c for c in df.columns if str(c).strip().startswith(("P_", "N_"))]
    if not sample_cols:
        print("No sample columns found. Expected columns starting with P_ or N_.", flush = True)
        raise ValueError("No sample columns found. Expected columns starting with P_ or N_.")

    # --- Determine polarity tag for filenames ---
    pol_tag = ""
    if "Polarity" in df.columns:
        pol_series = df["Polarity"].dropna().astype(str).str.lower()
        if not pol_series.empty:
            first_pol = pol_series.iloc[0]
            if "pos" in first_pol:
                pol_tag = "Pos_"
            elif "neg" in first_pol:
                pol_tag = "Neg_"
    if not pol_tag:
        has_pos = any(str(c).startswith("P_") for c in sample_cols)
        has_neg = any(str(c).startswith("N_") for c in sample_cols)
        if has_pos and not has_neg:
            pol_tag = "Pos_"
        elif has_neg and not has_pos:
            pol_tag = "Neg_"

    # --- Map samples to groups (warn on missing) ---
    sample_to_group = dict(zip(group_df["Sample"], group_df["Group"]))
    missing_in_map = [s for s in sample_cols if s not in sample_to_group]
    if missing_in_map:
        print(f"Warning: {len(missing_in_map)} samples not found in group map: {missing_in_map}", flush=True)
    groups = group_df["Group"].unique()
    high_thresh, mid_thresh = detection_thresholds

    # ---- Capture BEFORE intensities (numeric) for histograms ----
    X_before = pd.to_numeric(df[sample_cols].stack(), errors="coerce").values

    # Summed intensities BEFORE imputation, per sample column
    summed_before = df[sample_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)

    # ------------------------------------------------------------
    # Missing-value percentage per sample (BEFORE imputation)
    # ------------------------------------------------------------
    try:
        missing_counts = {}
        missing_percent = {}

        for s in sample_cols:
            col_vals = pd.to_numeric(df[s], errors="coerce")
            n_total = len(col_vals)
            n_missing = np.sum((col_vals.isna()) | (col_vals == 0))
            missing_counts[s] = n_missing
            missing_percent[s] = 100 * n_missing / n_total

        # Order samples by group (QC last) for plot
        unique_groups = sorted({sample_to_group.get(s, "ungrouped") for s in sample_cols})

        sample_only_groups = {}   # group → list of sample names
        for g in unique_groups:
            sample_only_groups[g] = sorted([s for s in sample_cols
                                            if sample_to_group.get(s, "ungrouped") == g and
                                            sample_to_group.get(s, "").lower() != "qc"])

        qc_samples = sorted([s for s in sample_cols
                            if sample_to_group.get(s, "").lower() == "qc"])

        ordered_samples = []
        for g in unique_groups:
            ordered_samples.extend(sample_only_groups[g])
        if qc_samples:
            ordered_samples.extend(qc_samples)

        plot_groups = unique_groups.copy()
        if qc_samples:
            plot_groups.append("QC")

        # One color per group
        cmap = plt.cm.tab20(np.linspace(0,1,len(plot_groups)))
        group_to_color = dict(zip(plot_groups, cmap))

        colors = []
        for s in ordered_samples:
            g = sample_to_group.get(s, "ungrouped")
            if s in qc_samples:
                colors.append(group_to_color["QC"])
            else:
                colors.append(group_to_color[g])

        # ------------------------------------------------------------
        # BAR PLOT: Missing-value % per sample (before imputation)
        # ------------------------------------------------------------

        plt.figure(figsize=(11,5))
        yvals = [missing_percent[s] for s in ordered_samples]

        plt.bar(range(len(ordered_samples)), yvals, color=colors)
        plt.xticks(range(len(ordered_samples)), ordered_samples, rotation=60, ha="right")

        plt.ylabel("Missing values (%)")
        plt.title(f"Missing-value percentage per sample (before imputation) — {pol_tag.replace('_','')}")

        # Legend
        handles = [plt.Rectangle((0,0), 1,1, color=group_to_color[g]) for g in plot_groups]
        plt.legend(handles, plot_groups, title="Groups",
                bbox_to_anchor=(1.02,1), loc="upper left")

        plt.tight_layout()
        out_missing = qc_dir / f"{pol_tag}missing_value_percentage_per_sample.png"
        plt.savefig(out_missing, dpi=100)
        plt.close()

        print(f"Saved missing-value % plot → {out_missing}", flush=True)

    except Exception as e:
        print(f"[WARNING] Failed to generate missing% plot: {e}", flush = True)  

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

    print("Per-group positive 1st percentiles (global per-group minimum LOD estimate):")
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
    # zeros treated as missing
    df[sample_cols] = df[sample_cols].mask(df[sample_cols] == 0, np.nan)
    
    # Keep pre-imputation values for QC detectability metrics
    df_pre_impute = df.copy()

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

    # Remove old summary columns if present
    for col in ["Average Intensity (all samples)", "Minimum Intensity (all samples)", "Maximum Intensity (all samples)"]:
        if col in df_imputed.columns:
            df_imputed.drop(columns=[col], inplace=True)

    # New global summaries
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

    # Insert summary columns right after "Carbons / double bond equivalent ratio", if present
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

    # QC vs non-QC RSDs
    # QC vs non-QC RSDs
    qc_samples = [s for s in group_df.loc[group_df["Group"].str.lower() == "qc", "Sample"] if s in sample_cols]
    non_qc_samples = [s for s in sample_cols if s not in qc_samples]

    if qc_samples:
        # Count QCs detected BEFORE imputation
        df_imputed["QC detected count"] = df_pre_impute[qc_samples].notna().sum(axis=1)

        # QC RSD based only on observed QC values BEFORE imputation
        df_imputed["RSD QCs observed-only (%)"] = df_pre_impute[qc_samples].apply(rsd, axis=1)

        # Keep backward-compatible column name for downstream use if needed
        df_imputed["RSD QCs (%)"] = df_imputed["RSD QCs observed-only (%)"]
    else:
        df_imputed["QC detected count"] = np.nan
        df_imputed["RSD QCs observed-only (%)"] = np.nan
        df_imputed["RSD QCs (%)"] = np.nan
        print("Warning: No QC samples found; QC metrics set to NaN.", flush=True)

    df_imputed["RSD Samples (%)"] = df_imputed[non_qc_samples].apply(rsd, axis=1) if non_qc_samples else np.nan
    if not non_qc_samples:
        print("Warning: No non-QC samples found; RSD Samples (%) set to NaN.", flush=True)

    # Per-group stats + RSD_<group> [%] (to align with LOESS / median / generate_final_file)
    for g in groups:
        g_samples = [s for s in group_df.loc[group_df["Group"] == g, "Sample"] if s in sample_cols]
        if not g_samples:
            continue
        df_imputed[f"Mean Intensity ({g})"] = df_imputed[g_samples].mean(axis=1)
        df_imputed[f"Median Intensity ({g})"] = df_imputed[g_samples].median(axis=1)
        df_imputed[f"Min Intensity ({g})"] = df_imputed[g_samples].min(axis=1)
        df_imputed[f"Max Intensity ({g})"] = df_imputed[g_samples].max(axis=1)
        # New naming convention, consistent with LOESS / median
        df_imputed[f"RSD_{g} [%]"] = df_imputed[g_samples].apply(rsd, axis=1)
        
    # Reorder new QC columns to appear immediately after the QC group RSD column
    cols = list(df_imputed.columns)
    new_qc_cols = ["QC detected count", "RSD QCs observed-only (%)"]
    existing_new_qc_cols = [c for c in new_qc_cols if c in cols]

    qc_group_names = [str(g) for g in groups if str(g).strip().lower() == "qc"]
    anchor_candidates = [f"RSD_{g} [%]" for g in qc_group_names]
    anchor_col = next((c for c in anchor_candidates if c in cols), None)

    if anchor_col is not None and existing_new_qc_cols:
        for c in existing_new_qc_cols:
            cols.remove(c)
        anchor_idx = cols.index(anchor_col) + 1
        for offset, c in enumerate(existing_new_qc_cols):
            cols.insert(anchor_idx + offset, c)
        df_imputed = df_imputed[cols]

    # =======================
    # 5) Save
    # =======================

    # Drop verbose per-group stat columns from disk outputs if you want lighter files
    keep_rsd_cols = {
        "RSD QCs (%)", "RSD Samples (%)",
        "RSD_12x [%]", "RSD_15x [%]", "RSD_5x [%]", "RSD_8x [%]", "RSD_QC [%]",
        "Average Intensity (all samples)", "Minimum Intensity (all samples)", "Maximum Intensity (all samples)"
    }
    drop_patterns = ("Mean Intensity (", "Median Intensity (", "Min Intensity (", "Max Intensity (", "RSD (%) (")

    def _drop_verbose(df_):
        cols_to_drop = [
            c for c in df_.columns
            if c.startswith(drop_patterns) and c not in keep_rsd_cols
        ]
        if cols_to_drop:
            df_.drop(columns=cols_to_drop, inplace=True)

    # Apply verbose drop to the output table (keep the compact set required downstream)
    _drop_verbose(df_imputed)

    # --------------------------------------------------
    # DROP COLUMNS BEFORE SAVING
    # --------------------------------------------------
    columns_to_remove = [
        # example columns:
        "Annotation Type norm",
        "Annotation_norm",
        "type_priority",
        "RT_seconds",
        "RT_seconds",
        "missing_count",
        "mean_intensity"
    ]

    # Only drop if they exist
    df_imputed.drop(
        columns=[c for c in columns_to_remove if c in df_imputed.columns],
        inplace=True
    )

    out_full = output_folder / "debug" / f"{pol_tag}3-Final_annotated_results_imputed.csv"
    out_full.parent.mkdir(parents=True, exist_ok=True)

    df_imputed.to_csv(out_full, index=False, encoding="utf-8-sig")

    print("Imputation complete. Overwrote QC/Sample RSDs.", flush=True)
    print(f"Saved:\n - Imputed: {out_full}\n", flush=True)

    # -------------------------------------------
    # PLOTS FOR QUALITY CONTROL
    # -------------------------------------------

    # After imputation
    X_after = df_imputed[sample_cols].to_numpy(dtype=float).reshape(-1)

    # Summed intensities AFTER imputation, per sample column
    summed_after = df_imputed[sample_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)

    # ------------------------------------------------------------
    # STATISTICAL TEST: per-group before/after summed intensities
    # ------------------------------------------------------------
    try:
        from scipy.stats import wilcoxon

        group_stats = []
        unique_groups = sorted({sample_to_group.get(s, "ungrouped") for s in sample_cols})

        for g in unique_groups:
            members = [s for s in sample_cols if sample_to_group.get(s, "ungrouped") == g]

            if len(members) < 3:
                continue  # skip groups with too few samples

            before_vals = np.array([summed_before[s] for s in members], dtype=float)
            after_vals  = np.array([summed_after[s]  for s in members], dtype=float)

            # Paired Wilcoxon signed-rank test
            try:
                stat, p = wilcoxon(before_vals, after_vals, zero_method="wilcox")
            except Exception:
                stat, p = np.nan, np.nan

            group_stats.append((g, len(members), before_vals.mean(), after_vals.mean(), p))

        # ------------------------------------------------------------
        # Build statistics table
        # ------------------------------------------------------------
        stats_df = pd.DataFrame(
            group_stats,
            columns=["Group", "N samples", "Mean Before", "Mean After", "Wilcoxon p-value"]
        )

        # Save
        stats_path = qc_dir / f"{pol_tag}summed_intensity_before_after_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved before/after significance stats → {stats_path}", flush=True)


        stats_path = qc_dir / f"{pol_tag}summed_intensity_before_after_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved before/after significance stats → {stats_path}", flush=True)

    except Exception as e:
        print(f"[WARNING] Failed to generate statistical significance: {e}", flush = True) 
    # ------------------------------------------------------------
    # Build per-group total-intensity vectors (BEFORE vs AFTER)
    # ------------------------------------------------------------

    group_sums_before = {}
    group_sums_after  = {}

    # non-QC groups in alphabetical order
    non_qc_groups = sorted({sample_to_group.get(s, "ungrouped") for s in sample_cols if sample_to_group.get(s,"").lower() != "qc"})

    for g in non_qc_groups:
        members = [s for s in sample_cols if sample_to_group.get(s,"ungrouped") == g]
        if members:
            group_sums_before[g] = [summed_before[s] for s in members]
            group_sums_after[g]  = [summed_after[s]  for s in members]

    # QC last
    qc_members = [s for s in sample_cols if sample_to_group.get(s,"").lower() == "qc"]
    if qc_members:
        group_sums_before["QC"] = [summed_before[s] for s in qc_members]
        group_sums_after["QC"]  = [summed_after[s]  for s in qc_members]

    labels = list(group_sums_before.keys())

    # ------------------------------------------------------------
    # BARPLOT: total intensity per group (BEFORE vs AFTER)
    # ------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8,5))

    means_before = [np.mean(group_sums_before[g]) for g in labels]
    means_after  = [np.mean(group_sums_after[g])  for g in labels]

    x = np.arange(len(labels))
    width = 0.38

    ax.bar(x - width/2, means_before, width, label="Before", alpha=0.75)
    ax.bar(x + width/2, means_after,  width, label="After",  alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Total intensity (summed signals)")
    ax.set_title(f"Group total intensity — before vs after missing-value substitution ({pol_tag.replace('_','')})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_folder / "debug" / "missing_value_imputation" / f"{pol_tag}total_intensity_before_after_BAR.png", dpi=100)
    plt.close()

    # ------------------------------------------------------------
    # BOXPLOT: total intensity per group (BEFORE vs AFTER)
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9,5))

    # Stack data in "long" style: each group contributes 2 boxes
    data_box = []
    labels_box = []

    for g in labels:
        data_box.append(group_sums_before[g])
        labels_box.append(f"{g} (before)")
        data_box.append(group_sums_after[g])
        labels_box.append(f"{g} (after)")

    # consistent colors: same color for a group's before/after pair
    cmap = plt.cm.tab20(np.linspace(0,1,len(labels)))
    group_colors = dict(zip(labels, cmap))

    box_colors = []
    for g in labels:
        box_colors.extend([group_colors[g], group_colors[g]])  # before, after

    bp = ax.boxplot(
        data_box,
        labels=labels_box,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    # apply colors
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")

    # scatter all points ON TOP
    x_positions = np.arange(1, len(data_box)+1)

    idx = 0
    for g in labels:
        # BEFORE
        vals_b = group_sums_before[g]
        xjit_b = np.random.normal(x_positions[idx], 0.05, size=len(vals_b))
        ax.scatter(xjit_b, vals_b, s=30, color="black", alpha=0.7, zorder=4)
        idx += 1

        # AFTER
        vals_a = group_sums_after[g]
        xjit_a = np.random.normal(x_positions[idx], 0.05, size=len(vals_a))
        ax.scatter(xjit_a, vals_a, s=30, color="black", alpha=0.7, zorder=4)
        idx += 1

    ax.set_ylabel("Total intensity (summed signals)")
    ax.set_title(f"Total intensity per group — before vs after ({pol_tag.replace('_','')})")

    plt.xticks(rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(output_folder / "debug" / "missing_value_imputation" / f"{pol_tag}total_intensity_before_after_BOX.png", dpi=100)
    plt.close()


    # -------------------------------------------
    # HISTOGRAMS BEFORE/AFTER IMPUTATION
    # -------------------------------------------

    plt.figure(figsize=(7,4))
    plt.hist(np.log10(X_before + 1e-10), bins=100, alpha=0.5, label="before")
    plt.hist(np.log10(X_after  + 1e-10), bins=100, alpha=0.5, label="after")
    plt.legend()
    plt.xlabel("log10(intensity)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(qc_dir / f"{pol_tag}hist_before_after.png", dpi=100)
    plt.close()

    return out_full

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Impute missing values, recompute stats, and apply QC RSD filtering.")
    parser.add_argument("--final", required=True, help="Path to Final_MS_results.csv")
    parser.add_argument("--groups", required=True, help="Path to sample_groups.csv")
    parser.add_argument("--out", default="results", help="Output folder")
    args = parser.parse_args()

    impute_missing_values(args.final, args.groups, args.out)
