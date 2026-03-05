'''
This script runs the annotation filtering stage that turns raw candidate matches into a single curated lipid list. 
It loads the raw search results table and determines ion mode from the sample columns so all outputs are tagged as positive or negative. 
It applies a scoring module to assign an MS Score to each candidate match and to apply any initial retention time constraints defined in the scoring rules. 
Internal standards are then separated from the rest of the dataset so they are never removed by plausibility filters or score cutoffs.

The software filters the non internal standard annotations for biological plausibility using a configurable plausibility module. 
High confidence MetaboScape annotations labeled as Lipid Species or Target List are exempted from this plausibility filter and are carried through unchanged. 
A Kendrick mass defect filter is then applied within lipid classes to remove candidates whose mass defect is inconsistent with the class pattern, and the removed features are logged. 
After filtering, the software collapses duplicate rows so that each feature UniqueID is represented at most once, prioritizing internal standards first, then MS/MS matches, then MS matches, and finally choosing the highest scoring option.

The software applies a minimum MS Score cutoff to keep low quality annotations out of the final table, while still retaining truly unassigned features for transparency. 
Internal standards are then recombined with the filtered lipid list, and a CCS sanity check flags features whose CCS is inconsistent with the expected CCS versus carbon trend for their class and adduct. 
If an internal CCS reference library is available, it is incorporated into the CCS fit and plotted alongside project data. 
Flagged CCS outliers are removed, and all CCS flags and reasons are recorded.

Final outputs are saved in a standardized column order, including a scored intermediate table and the final filtered table. 
Internal standards are exported to a dedicated file and plotted to provide QC visibility. 
The pipeline then resolves redundant adduct representations for the same feature by merging adducts within a tight retention time window, writes kept and removed adduct tables, 
and generates debug plots showing which adducts were retained or discarded for each headgroup. 
Finally, the script produces summary plots of the retained annotations and separate plots for annotations removed by plausibility, Kendrick mass defect, and retention time filters, 
including a plot that counts removals by plausibility reason when those labels are available.
'''


'''
    Lipid filtering pipeline:
    1. Load raw search results
    2. Apply scoring
    3. Filter by biological plausibility (inline here)
    4. Collapse duplicates
    5. Apply minimum score cutoff
    6. Save outputs
    7. Plot results
    '''
    
import pandas as pd
import numpy as np
import copy
import glob
from pathlib import Path
import importlib
import re
import matplotlib.pyplot as plt
from typing import Optional
from internal_standard_plots import plot_internal_standards
from handle_adducts import handle_adducts
from generate_annotation_plots import plot_results, plot_kendrick_mass_vs_defect

# -----------------------------------------------------------------

#                      HELPER FUNCTIONS

# -----------------------------------------------------------------

def collapse_duplicates(df):
    """
    Enforce true uniqueness of UniqueID.
    For each UniqueID:
      1) If there are annotated rows, keep exactly one annotated row
         selected by highest priority then highest MS Score.
      2) If there are no annotated rows, keep exactly one unassigned row
         selected by highest MS Score.
    Rows without a UniqueID are preserved.
    """
    # Preconditions
    if "UniqueID" not in df.columns or "Annotation" not in df.columns:
        return df.copy()

    # Make a working copy
    dfx = df.copy()

    # Normalize helper columns
    ann = dfx["Annotation"].astype(str).str.strip()
    is_unassigned = ann.eq("") | ann.eq("nan") | dfx["Annotation"].isna()

    # Priority for Annotation Type
    # Lower number means higher priority
    
    def anno_type_priority(s):
        s = str(s).strip().upper()
        if s == "IS":
            return 0
        if s == "MS/MS MATCH":
            return 1
        if s == "MS MATCH":
            return 2
        return 3

    # Build a selection key
    # 1) has_annotation: 1 if annotated, 0 if unassigned
    # 2) type priority as above
    # 3) MS Score descending
    has_ann = (~is_unassigned).astype(int)
    type_pri = dfx.get("Annotation Type", "").apply(anno_type_priority)
    ms_score = pd.to_numeric(dfx.get("MS Score", 0), errors="coerce").fillna(0)

    dfx["_has_ann"] = has_ann
    dfx["_type_pri"] = type_pri
    dfx["_ms_score"] = ms_score

    # Split rows with and without UniqueID
    with_uid = dfx[dfx["UniqueID"].notna()].copy()
    without_uid = dfx[dfx["UniqueID"].isna()].copy()

    # For each UniqueID, pick the single best row
    # Sort so the first row per group is the keeper
    with_uid_sorted = (
        with_uid
        .sort_values(
            by=["_has_ann", "_type_pri", "_ms_score"],
            ascending=[False, True, False]
        )
        .drop_duplicates(subset=["UniqueID"], keep="first")
    )

    # Clean helper cols
    with_uid_sorted = with_uid_sorted.drop(columns=["_has_ann", "_type_pri", "_ms_score"], errors="ignore")
    without_uid = without_uid.drop(columns=["_has_ann", "_type_pri", "_ms_score"], errors="ignore")

    # Recombine
    out = pd.concat([with_uid_sorted, without_uid], ignore_index=True)

    # Guarantee that each UniqueID appears at most once
    # Defensive assertion in case of unexpected input
    # If you prefer silent behavior, comment this out
    # dup_check = out["UniqueID"].dropna()
    # assert not dup_check.duplicated().any(), "Duplicate UniqueID after collapsing"

    return out

def count_unassigned(df):
    if "Annotation" not in df.columns:
        return 0
    ann = df["Annotation"].astype(str).str.strip()
    return int(
        (ann.eq("") | ann.eq("nan") | ann.eq("Unassigned") | df["Annotation"].isna()).sum()
    )

def plot_adducts_kept_vs_removed_by_headgroup(
    kept_csv,
    removed_csv,
    output_folder,
    pol_tag="",
    headgroup_col="Headgroup",
    adduct_col="Adducts",
    min_total_per_headgroup=5,
    max_adducts_per_headgroup=12
):
    """
    Debug plots showing which adducts were KEPT vs REMOVED per headgroup.

    Reads the post-handle_adducts kept/removed CSVs.
    Saves plots under: output_folder/debug/adducts_by_headgroup/
    Hard-safe: never raises if columns/files missing.
    """
    try:
        kept_csv = Path(kept_csv)
        removed_csv = Path(removed_csv) if removed_csv else None
        output_folder = Path(output_folder)

        if not kept_csv.exists():
            print(f"[INFO] Adduct debug plots skipped (kept file missing): {kept_csv}", flush=True)
            return

        kept = pd.read_csv(kept_csv, low_memory=False)
        removed = pd.DataFrame()
        if removed_csv is not None and removed_csv.exists():
            removed = pd.read_csv(removed_csv, low_memory=False)

        # Column checks
        for df_name, df_ in [("kept", kept), ("removed", removed)]:
            if df_.empty:
                continue
            if headgroup_col not in df_.columns:
                # fallback: try Lipid Class as headgroup proxy
                if "Lipid Class" in df_.columns:
                    df_[headgroup_col] = df_["Lipid Class"]
                else:
                    df_[headgroup_col] = ""
            if adduct_col not in df_.columns:
                df_[adduct_col] = ""

        # Normalize adduct: first token only
        def _norm_adduct_series(s):
            s = s.astype(str).fillna("").str.strip()
            s = s.str.split(",").str[0].str.strip()
            s = s.replace({"nan": "", "NaN": "", "None": ""})
            return s

        def _norm_hg_series(s):
            s = s.astype(str).fillna("").str.strip()
            s_low = s.str.lower()
            bad = s.eq("") | s_low.isin({"nan", "none", "<na>", "na", "null"})
            s = s.mask(bad, "")
            return s

        kept["_hg"] = _norm_hg_series(kept[headgroup_col])
        kept["_adduct"] = _norm_adduct_series(kept[adduct_col])

        if not removed.empty:
            removed["_hg"] = _norm_hg_series(removed[headgroup_col])
            removed["_adduct"] = _norm_adduct_series(removed[adduct_col])
        else:
            removed = pd.DataFrame(columns=["_hg", "_adduct"])

        # Drop blanks
        kept = kept[(kept["_hg"] != "") & (kept["_adduct"] != "")]
        removed = removed[(removed["_hg"] != "") & (removed["_adduct"] != "")]

        if kept.empty and removed.empty:
            print("[INFO] Adduct debug plots skipped (no headgroup+adduct rows).", flush=True)
            return

        plot_dir = output_folder / "debug" / "adducts_by_headgroup"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # ---- Global heatmaps (kept and removed separately) ----
        # Keep only adducts seen at least once overall (kept+removed)
        all_df = pd.concat(
            [kept[["_hg", "_adduct"]], removed[["_hg", "_adduct"]]],
            ignore_index=True
        )
        if not all_df.empty:
            hg_order = (
                all_df["_hg"].value_counts().index.tolist()
            )
            adduct_order = (
                all_df["_adduct"].value_counts().index.tolist()
            )

            def _heatmap(df_in, title, out_name):
                if df_in.empty:
                    return
                tab = pd.crosstab(df_in["_hg"], df_in["_adduct"])
                tab = tab.reindex(index=hg_order, columns=adduct_order, fill_value=0)

                # drop very sparse headgroups
                tab = tab.loc[tab.sum(axis=1) >= int(min_total_per_headgroup)]
                if tab.empty:
                    return

                plt.figure(figsize=(max(8, 0.6 * tab.shape[1]), max(4, 0.35 * tab.shape[0])))
                plt.imshow(tab.to_numpy(), aspect="auto", interpolation="nearest")
                plt.yticks(range(tab.shape[0]), tab.index.tolist())
                plt.xticks(range(tab.shape[1]), tab.columns.tolist(), rotation=60, ha="right")
                plt.colorbar(label="Count")
                plt.title(title)
                plt.tight_layout()
                plt.savefig(plot_dir / out_name, dpi=100)
                plt.close()

            _heatmap(
                kept,
                f"{pol_tag.replace('_','')} Adduct counts by headgroup (KEPT)",
                f"{pol_tag}Adducts_by_headgroup_KEPT_heatmap.png"
            )
            _heatmap(
                removed,
                f"{pol_tag.replace('_','')} Adduct counts by headgroup (REMOVED)",
                f"{pol_tag}Adducts_by_headgroup_REMOVED_heatmap.png"
            )

        # ---- Per-headgroup bar charts: kept vs removed (side-by-side) ----
        hg_list = sorted(set(kept["_hg"].tolist() + removed["_hg"].tolist()))
        for hg in hg_list:
            k = kept.loc[kept["_hg"] == hg, "_adduct"].value_counts()
            r = removed.loc[removed["_hg"] == hg, "_adduct"].value_counts()

            total = int(k.sum()) + int(r.sum())
            if total < int(min_total_per_headgroup):
                continue

            # focus on the most common adducts for readability
            top_adducts = (k.add(r, fill_value=0)).sort_values(ascending=False).head(int(max_adducts_per_headgroup)).index.tolist()

            k2 = k.reindex(top_adducts, fill_value=0)
            r2 = r.reindex(top_adducts, fill_value=0)

            y = np.arange(len(top_adducts))
            h = 0.40

            plt.figure(figsize=(8.5, max(3.2, 0.45 * len(top_adducts))))
            plt.barh(y - h/2, k2.values, height=h, label="Kept", alpha=0.8)
            plt.barh(y + h/2, r2.values, height=h, label="Removed", alpha=0.8)

            plt.yticks(y, top_adducts)
            plt.xlabel("Count")
            plt.title(f"{pol_tag.replace('_','')} {hg}: adducts kept vs removed by adduct merging")
            plt.legend()
            plt.tight_layout()

            safe_hg = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(hg))[:60]
            plt.savefig(plot_dir / f"{pol_tag}Adducts_{safe_hg}_kept_vs_removed.png", dpi=100)
            plt.close()

        # Optional: write tables for quick grepping
        try:
            kept_tab = pd.crosstab(kept["_hg"], kept["_adduct"])
            removed_tab = pd.crosstab(removed["_hg"], removed["_adduct"]) if not removed.empty else pd.DataFrame()
            kept_tab.to_csv(plot_dir / f"{pol_tag}Adducts_by_headgroup_KEPT_table.csv", encoding="utf-8-sig")
            if not removed_tab.empty:
                removed_tab.to_csv(plot_dir / f"{pol_tag}Adducts_by_headgroup_REMOVED_table.csv", encoding="utf-8-sig")
        except Exception:
            pass

        print(f"[INFO] Saved adduct debug plots to: {plot_dir}", flush=True)

    except Exception as e:
        print(f"[WARNING] Adduct debug plots skipped (error: {e})", flush=True)
        try:
            plt.close()
        except Exception:
            pass
        
# === Helper to reorder columns consistently ===
def reorder_columns(df):
    sample_cols = [c for c in df.columns if str(c).startswith(("P_", "N_"))]
    rsd_cols = [c for c in df.columns if re.match(r"RSD.*\[%\]", c)]
    preferred_order = [
        "UniqueID", "RT (min)", "m/z", "Neutral mass", "Adducts", "Polarity",
        "Internal Standard", "RSD QCs (%)", "RSD Samples (%)"
    ]
    group_rsd_cols = sorted([c for c in rsd_cols if c not in ("RSD QCs (%)", "RSD Samples (%)")])
    metadata_following = [
            "MS/MS available?", "Annotation", "Annotation Type",
            "Metaboscape Annotation Status", "Annotation Source", "Headgroup", "Lipid Class",
            "Δm/z (mDa)", "Δm/z (ppm)", "MS/MS score", "Annotation tier", "mSigma",
            "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]", "CCS_outlier?", "CCS_outlier_score", "CCS_outlier_reason",
            "Molecular Formula", "Plasmenyl?", "Number of carbons in fatty acyls",
            "Double bond equivalents", "Number of carbons in fatty acyl 1", "Double bonds in fatty acyl 1",
            "Number of carbons in fatty acyl 2", "Double bonds in fatty acyl 2",
            "Number of carbons in fatty acyl 3", "Double bonds in fatty acyl 3",
            "Number of carbons in fatty acyl 4", "Double bonds in fatty acyl 4",
            "Chain type", "PUFA?", "Modifications", "# of modifications",
            "Oxidized?", "Carbons / double bond equivalent ratio"
    ]
    intensity_cols = [
            "Average Intensity (all samples, from MetaboScape)",
            "Average Intensity (all samples)", "Minimum Intensity (all samples)",
            "Maximum Intensity (all samples)"
    ]
    flags_cols = ["Relative Stdev", "Flags", "Flag type"]

    new_order = (
            [c for c in preferred_order if c in df.columns] +
            [c for c in group_rsd_cols if c in df.columns] +
            [c for c in metadata_following if c in df.columns] +
            [c for c in intensity_cols if c in df.columns] +
            [c for c in flags_cols if c in df.columns] +
            sample_cols
    )
    new_order = [c for c in new_order if c in df.columns]
    return df[new_order]

# ---------------------------------------------------------------------
# Helper: plotting removed reasons (plausability filter)
# ---------------------------------------------------------------------

def plot_removed_reason_counts(
    input_csv,
    output_folder,
    pol_tag="",
    suffix=""
):
    """
    Plot counts of removed features by removal reason.
    Expects a column named 'removed_reason'.
    """
    df = pd.read_csv(input_csv, low_memory=False)
    print(f'[INFO] Plotting the removed reasons (plausability filter)...', flush = True)
    
    if "removed_reason" not in df.columns:
        print("\n\n[INFO] No 'removed_reason' column found; skipping reason plot.\n\n", flush=True)
        return

    counts = (
        df["removed_reason"]
        .astype(str)
        .str.strip()
        .value_counts()
        .sort_values(ascending=False)
    )

    if counts.empty:
        print("[INFO] No removal reasons to plot.", flush=True)
        return

    plt.figure(figsize=(8, max(3, 0.4 * len(counts))))
    counts.plot(kind="barh")
    plt.xlabel("Number of removed features")
    plt.ylabel("Removal reason")
    plt.title("Annotations removed by plausibility filter")
    plt.tight_layout()

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    out_path = (
        output_folder
        / f"{pol_tag}removed_plausibility_reason_counts{suffix}.png"
    )
    print(f'[INFO] Saving the removed reasons plot to: {out_path}', flush = True)
    plt.savefig(out_path, dpi=100)
    plt.close()

    print(f"[INFO] Saved removal-reason plot to {out_path}", flush=True)
    
# ---------------------------------------------------------------------
# Kendrick Mass Defect (KMD) filtering for lipidomics datasets
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Kendrick mass calculation
# ---------------------------------------------------------------------
def calculate_kendrick_mass_defect(
    mass,
    base_unit_exact=14.01565,
    base_unit_nominal=14.00000
):
    """Return Kendrick mass and Kendrick mass defect (KMD)."""
    kendrick_mass = mass * base_unit_nominal / base_unit_exact
    kendrick_nominal = np.round(kendrick_mass)
    kendrick_defect = kendrick_mass - kendrick_nominal
    return kendrick_mass, kendrick_defect

def _robust_mad(x):
    """Median absolute deviation scaled to be comparable to std."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad  # consistency constant


def flag_and_plot_ccs_outliers_by_class(
    df: pd.DataFrame,
    output_folder,
    pol_tag="",
    class_col="Lipid Class",
    headgroup_col="Headgroup",
    adduct_col="Adducts",
    carbons_col="Number of carbons in fatty acyls",
    ccs_col="CCS (Å²)",
    annotation_col="Annotation",
    uid_col="UniqueID",
    min_class_size=10,
    min_points_for_fit=6,
    plot_min_points=1,
    mad_z_thresh=10,
    max_labels_per_class=20,
    skip_ccs_classes=None,
    use_headgroup_for_classes=None,
    reference_library_path: Optional[str] = None,
    reference_label: str = "Measured CCS Library"
):
    """
    Hard-safe:
      - ALWAYS creates CCS flag columns in the returned df.
      - NEVER raises if CCS/required columns are missing.
      - Plots only when it has enough valid data.

    Flags outliers within each class using robust MAD-z of residuals from linear CCS~carbons.
    """
    out = df.copy()

    # Always reset output columns (avoid carrying flags across reruns)
    out["CCS_outlier?"] = ""
    out["CCS_outlier_score"] = np.nan
    out["CCS_outlier_reason"] = ""
    out["CCS_residual_A2"] = np.nan

    # If required columns missing → return without crashing
    required = [class_col, adduct_col, carbons_col, ccs_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        out["CCS_outlier_reason"] = (
            out["CCS_outlier_reason"].astype(str).where(out["CCS_outlier_reason"].astype(str).str.strip() != "", "")
        )
        # Put a single global reason only where reason is blank (avoid overwriting existing reasons)
        global_reason = f"CCS outlier check skipped (missing columns: {', '.join(missing)})"
        blank_reason = out["CCS_outlier_reason"].astype(str).str.strip() == ""
        out.loc[blank_reason, "CCS_outlier_reason"] = global_reason
        return out

    # Numeric conversion
    ccs_num = pd.to_numeric(out[ccs_col], errors="coerce")
    carb_num = pd.to_numeric(out[carbons_col], errors="coerce")
    
    # -----------------------------
    # Optional: load Appendix CCS library (used for fit and plotted separately)
    # -----------------------------
    ref_df = None
    if reference_library_path:
        try:
            ref_path = Path(reference_library_path)

            if ref_path.suffix.lower() in [".xlsx", ".xls"]:
                ref_df = pd.read_excel(ref_path, sheet_name=0)
            else:
                ref_df = pd.read_csv(ref_path, low_memory=False)

            # Normalize CCS column name between Å and Å variants
            if ccs_col not in ref_df.columns:
                if ccs_col == "CCS (Å²)" and "CCS (Å²)" in ref_df.columns:
                    ref_df["CCS (Å²)"] = ref_df["CCS (Å²)"]
                elif ccs_col == "CCS (Å²)" and "CCS (Å²)" in ref_df.columns:
                    ref_df["CCS (Å²)"] = ref_df["CCS (Å²)"]

            # Check required columns in reference
            req_ref = [class_col, adduct_col, carbons_col, ccs_col]
            missing_ref = [c for c in req_ref if c not in ref_df.columns]
            if missing_ref:
                print(f"[WARNING] CCS library ignored (missing columns: {', '.join(missing_ref)})", flush=True)
                ref_df = None
            else:
                ref_df = ref_df.copy()
                ref_df["_is_reference"] = True
        except Exception as e:
            print(f"[WARNING] CCS library load failed: {e}", flush=True)
            ref_df = None

    out["_is_reference"] = False

    valid_global = ccs_num.notna() & carb_num.notna()
    if int(valid_global.sum()) == 0:
        global_reason = "CCS outlier check skipped (no numeric CCS + carbons rows)"
        blank_reason = out["CCS_outlier_reason"].astype(str).str.strip() == ""
        out.loc[blank_reason, "CCS_outlier_reason"] = global_reason
        return out

    # Prepare output folders safely
    output_folder = Path(output_folder)
    debug_folder = output_folder / "debug"
    debug_folder.mkdir(parents=True, exist_ok=True)
    plot_dir = debug_folder / "ccs_by_class"
    plot_dir.mkdir(parents=True, exist_ok=True)

    flagged_rows = []

    # Iterate classes
    # -----------------------------
    # Build a combined fit table (project + optional reference)
    # Fit uses combined table, flags apply only to project rows
    # -----------------------------
    fit_df = out.copy()
    n_project = len(out)

    if ref_df is not None and len(ref_df) > 0:
        # Align columns safely
        for col in fit_df.columns:
            if col not in ref_df.columns:
                ref_df[col] = np.nan
        for col in ref_df.columns:
            if col not in fit_df.columns:
                fit_df[col] = np.nan

        # Keep project indices stable so we can write flags back to `out`
        ref_block = ref_df[fit_df.columns].copy()
        ref_block.index = range(int(out.index.max()) + 1, int(out.index.max()) + 1 + len(ref_block))
        fit_df = pd.concat([fit_df, ref_block], axis=0)

    # Numeric conversion on fit_df
    fit_df["_ccs_num_fit"] = pd.to_numeric(fit_df[ccs_col], errors="coerce")
    fit_df["_carb_num_fit"] = pd.to_numeric(fit_df[carbons_col], errors="coerce")
    valid_fit = fit_df["_ccs_num_fit"].notna() & fit_df["_carb_num_fit"].notna()

    # normalize adduct to first token
    _adduct_norm = (
        fit_df[adduct_col].astype(str).fillna("").str.strip()
        .str.split(",").str[0].str.strip()
    )

    _class_raw = fit_df[class_col].astype(str).fillna("").str.strip()
    _head_raw = (
        fit_df[headgroup_col].astype(str).fillna("").str.strip()
        if headgroup_col in fit_df.columns
        else pd.Series([""] * len(fit_df), index=fit_df.index)
    )

    skip_set = set(str(s).strip() for s in (skip_ccs_classes or []))
    hg_set = set(str(s).strip() for s in (use_headgroup_for_classes or []))

    _ccs_label = _class_raw.copy()

    _is_skipped = _class_raw.isin(skip_set)
    _ccs_label.loc[_is_skipped] = ""

    _is_hg = _class_raw.isin(hg_set) & (~_is_skipped)
    _ccs_label.loc[_is_hg] = _head_raw.loc[_is_hg].where(_head_raw.loc[_is_hg].ne(""), _class_raw.loc[_is_hg])

    fit_df["_ccs_group_key"] = _ccs_label + " | " + _adduct_norm
    fit_df.loc[_ccs_label.eq("") | _adduct_norm.eq(""), "_ccs_group_key"] = ""

    # Only plot groups that exist in the current project (not reference-only groups)
    project_df = fit_df.iloc[:n_project].copy()
    project_df["_ccs_num_fit"] = pd.to_numeric(project_df[ccs_col], errors="coerce")
    project_df["_carb_num_fit"] = pd.to_numeric(project_df[carbons_col], errors="coerce")
    valid_project = project_df["_ccs_num_fit"].notna() & project_df["_carb_num_fit"].notna()

    project_groups = set(
        project_df.loc[valid_project & project_df["_ccs_group_key"].ne(""), "_ccs_group_key"]
        .astype(str)
        .tolist()
    )

    groups = fit_df.loc[valid_fit & fit_df["_ccs_group_key"].ne(""), "_ccs_group_key"].unique()
    groups = [g for g in groups if str(g) in project_groups]

    for gkey in sorted(groups):
        sub = fit_df.loc[valid_fit & (fit_df["_ccs_group_key"] == gkey)].copy()
        if len(sub) < int(min_class_size):
            continue
        
        # for display in messages/plots
        cls, adduct = (gkey.split(" | ", 1) + [""])[:2]

        x = pd.to_numeric(sub[carbons_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[ccs_col], errors="coerce").to_numpy(dtype=float)

        sub_is_ref = sub["_is_reference"].fillna(False).astype(bool).to_numpy()

        # Split reference vs project
        x_ref = x[sub_is_ref]
        y_ref = y[sub_is_ref]
        x_proj_all = x[~sub_is_ref]
        y_proj_all = y[~sub_is_ref]

        # If the project has no points, nothing to plot
        if x_proj_all.size < int(plot_min_points):
            continue

        # Choose fit source:
        # 1) reference if enough points
        # 2) else project if enough points
        fit_source = None
        if x_ref.size >= int(min_points_for_fit):
            fit_source = "reference"
            x_fit, y_fit = x_ref, y_ref
        elif x_proj_all.size >= int(min_points_for_fit):
            fit_source = "project"
            x_fit, y_fit = x_proj_all, y_proj_all
        else:
            fit_source = None  # will plot points only, no line, no outlier flagging

        a = b = np.nan
        mad_ref = np.nan
        resid_ref_med = np.nan

        if fit_source is not None:
            X = np.column_stack([x_fit, np.ones_like(x_fit)])
            try:
                coef, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
                a, b = float(coef[0]), float(coef[1])
            except Exception:
                fit_source = None

        # If we have a fit, compute residual scale on the fit source
        if fit_source is not None and np.isfinite(a) and np.isfinite(b):
            yhat_fit = a * x_fit + b
            resid_fit = y_fit - yhat_fit
            mad_ref = _robust_mad(resid_fit)
            resid_ref_med = np.nanmedian(resid_fit)

        # Indices of project rows within this sub-table (these indices match `out`)
        proj_idx_in_sub = sub.index[~sub_is_ref]

        # Flag outliers for PROJECT points only
        is_out_proj = np.zeros(x_proj_all.shape[0], dtype=bool)
        z_proj = np.full(x_proj_all.shape[0], np.nan, dtype=float)

        if fit_source is not None and np.isfinite(mad_ref) and mad_ref > 0 and np.isfinite(a) and np.isfinite(b):
            yhat_proj = a * x_proj_all + b
            resid_proj = y_proj_all - yhat_proj

            # store residual for debugging
            try:
                out.loc[proj_idx_in_sub, "CCS_residual_A2"] = resid_proj
            except Exception:
                pass

            # robust z using MAD (with a floor to avoid tiny MAD exploding)
            mad_eff = max(float(mad_ref), 0.75)  # Å² floor
            z_proj = (resid_proj - float(resid_ref_med)) / mad_eff

            # do not flag points that sit close to the fit line in absolute CCS units
            close_abs = np.isfinite(resid_proj) & (np.abs(resid_proj) <= 2.0)  # Å² tolerance

            # flag only if not close and |z| exceeds threshold
            is_out_proj = (~close_abs) & np.isfinite(z_proj) & (np.abs(z_proj) >= float(mad_z_thresh))

        if is_out_proj.any():
            outlier_idx_project = proj_idx_in_sub[is_out_proj]

            out.loc[outlier_idx_project, "CCS_outlier?"] = "Yes"
            out.loc[outlier_idx_project, "CCS_outlier_score"] = z_proj[is_out_proj]

            reason = f"Group '{cls}' + '{adduct}': |robust_z| >= {mad_z_thresh:.1f} vs {fit_source} CCS~carbons"
            blank_reason = out.loc[outlier_idx_project, "CCS_outlier_reason"].astype(str).str.strip() == ""
            out.loc[outlier_idx_project[blank_reason.values], "CCS_outlier_reason"] = reason

            flagged_rows.append(out.loc[outlier_idx_project].copy())

        # Do not plot if the current project has zero points in this group
        sub_is_ref = sub["_is_reference"].fillna(False).astype(bool)
        if int((~sub_is_ref).sum()) == 0:
            continue

        # Plot
        try:
            
            plt.figure(figsize=(7.5, 5.5))

            # Split project vs reference for plotting
            sub_is_ref = sub["_is_reference"].fillna(False).astype(bool).to_numpy()

            # Project points
            plt.scatter(
                x[~sub_is_ref], y[~sub_is_ref],
                s=22, alpha=0.35,
                c="tab:blue",
                label="Project"
            )

            # Reference points
            if sub_is_ref.any():
                plt.scatter(
                    x[sub_is_ref], y[sub_is_ref],
                    s=26, alpha=0.65,
                    c="tab:orange",
                    label=reference_label
                )

           # Flagged project points (relative to reference trend)
            if is_out_proj.any():
                plt.scatter(
                    x_proj_all[is_out_proj], y_proj_all[is_out_proj],
                    s=40, alpha=0.95,
                    c="tab:red",
                    label="Flagged (project)"
                )

                # Label up to N strongest flagged project points
                outlier_points = sub.loc[proj_idx_in_sub[is_out_proj]].copy()
                outlier_points["_absz"] = np.abs(z_proj[is_out_proj])
                outlier_points = outlier_points.sort_values("_absz", ascending=False).head(int(max_labels_per_class))
                outlier_points = outlier_points.sort_values("_absz", ascending=False).head(int(max_labels_per_class))

                for ridx in outlier_points.index:
                    uid = outlier_points.at[ridx, uid_col] if uid_col in outlier_points.columns else ""
                    ann = outlier_points.at[ridx, annotation_col] if annotation_col in outlier_points.columns else ""
                    label = f"{uid}: {ann}".strip(": ").strip()
                    plt.text(
                        float(pd.to_numeric(outlier_points.at[ridx, carbons_col], errors="coerce")) + 0.1,
                        float(pd.to_numeric(outlier_points.at[ridx, ccs_col], errors="coerce")),
                        str(label)[:80],
                        fontsize=7
                    )

            # Fit line for context
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
            if np.isfinite(a) and np.isfinite(b):
                ys = a * xs + b
                plt.plot(xs, ys, linewidth=1.2)

            plt.xlabel("Total carbons (summed)")
            plt.ylabel("CCS (A^2)")
            plt.title(f"CCS vs carbons — {cls} {adduct} ({pol_tag.replace('_','')})")
            plt.legend()
            plt.tight_layout()

            safe_cls = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(cls))[:60]
            safe_adduct = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(adduct))[:40]
            out_path = plot_dir / f"{pol_tag}CCS_vs_carbons_{safe_cls}__{safe_adduct}.png"
            plt.savefig(out_path, dpi=100)
            plt.close()
        except Exception:
            # plotting must never block the pipeline
            try:
                plt.close()
            except Exception:
                pass

    # If we ran and did not flag anything, fill a benign reason only if blank
    if not flagged_rows:
        global_reason = "CCS outlier check ran; no outliers flagged at current thresholds"
        blank_reason = out["CCS_outlier_reason"].astype(str).str.strip() == ""
        out.loc[blank_reason, "CCS_outlier_reason"] = global_reason
    else:
        flagged_df = pd.concat(flagged_rows, ignore_index=True)
        out_csv = debug_folder / f"{pol_tag}CCS_flagged_outliers.csv"
        try:
            flagged_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        except Exception:
            pass

    # Ensure non-flagged rows have explicit "No" when the check ran on them
    # Only set "No" where empty and row had valid CCS+carbons
    empty_flag = out["CCS_outlier?"].astype(str).str.strip() == ""
    out.loc[empty_flag & valid_global, "CCS_outlier?"] = "No"
    
    out = out.drop(columns=["_ccs_group_key"], errors="ignore")

    return out

def flag_ccs_outliers_by_class(
    df: pd.DataFrame,
    ccs_col: str = "CCS (Å²)",                  # use Å not Å to avoid font warnings
    class_col: str = "Lipid Class",
    x_col: str = "Number of carbons in fatty acyls",
    annotation_col: str = "Annotation",
    id_col: str = "UniqueID",
    min_class_size: int = 12,
    mad_k: float = 10.0,                         # stricter: 5–6; looser: 7–8
    min_points_for_fit: int = 6,                # minimum valid points in class after NaN filtering
    output_folder: Optional[Path] = None,
    pol_tag: str = "",
    make_plots: bool = True,
    label_max_per_class: int = 20
):
    """
    Flags CCS outliers per Lipid Class using robust residuals from CCS ~ carbons.
    Adds columns:
      - 'CCS_outlier_flag' : True/False
      - 'CCS_outlier_reason'
      - 'CCS_fit_residual'
      - 'CCS_fit_mad'
    Does NOT crash if CCS column is missing.
    """

    out = df.copy()

    # Default columns, always present after this function
    out["CCS_outlier_flag"] = False
    out["CCS_outlier_reason"] = ""
    out["CCS_fit_residual"] = np.nan
    out["CCS_fit_mad"] = np.nan

    if ccs_col not in out.columns:
        # silently skip, no crash
        return out

    if class_col not in out.columns or x_col not in out.columns:
        return out

    # numeric coercion
    out["_ccs_num"] = pd.to_numeric(out[ccs_col], errors="coerce")
    out["_x_num"] = pd.to_numeric(out[x_col], errors="coerce")

    # iterate classes
    classes = out[class_col].astype(str).fillna("").str.strip()
    for cls in sorted(set(classes)):
        if not cls or cls.lower() == "nan":
            continue

        idx = out.index[classes == cls]
        if len(idx) < min_class_size:
            continue

        sub = out.loc[idx, ["_x_num", "_ccs_num"]].copy()
        sub = sub[np.isfinite(sub["_x_num"]) & np.isfinite(sub["_ccs_num"])]

        if len(sub) < max(min_points_for_fit, 3):
            continue

        x = sub["_x_num"].to_numpy()
        y = sub["_ccs_num"].to_numpy()

        # Simple linear fit CCS ~ carbons within class
        # Using np.polyfit is fine because we robustify with MAD on residuals
        try:
            slope, intercept = np.polyfit(x, y, deg=1)
        except Exception:
            continue

        y_hat = slope * x + intercept
        resid = y - y_hat

        # Robust scale: MAD
        med = np.nanmedian(resid)
        mad = np.nanmedian(np.abs(resid - med))
        if not np.isfinite(mad) or mad == 0:
            # If MAD collapses (rare), skip filtering for this class
            continue

        # Modified z-like score using MAD
        # (no need to convert to sigma; we use k*MAD threshold directly)
        is_out = np.abs(resid - med) > (mad_k * mad)

        # write back to main df
        out.loc[sub.index, "CCS_fit_residual"] = resid
        out.loc[sub.index, "CCS_fit_mad"] = mad

        if is_out.any():
            outlier_idx = sub.index[is_out]
            out.loc[outlier_idx, "CCS_outlier_flag"] = True
            out.loc[outlier_idx, "CCS_outlier_reason"] = f"CCS outlier in class (|resid-med| > {mad_k}×MAD)"

            # optional per-class plot
            if output_folder and make_plots:
                plot_dir = Path(output_folder) / "debug" / "ccs_outliers"
                plot_dir.mkdir(parents=True, exist_ok=True)

                plt.figure(figsize=(7, 5))
                plt.scatter(x, y, s=18, alpha=0.35, label="All")
                plt.scatter(x[is_out], y[is_out], s=35, alpha=0.9, label="Flagged")

                # regression line (over x-range)
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                ys = slope * xs + intercept
                plt.plot(xs, ys, lw=1)

                plt.xlabel("Total carbons (sum)")
                plt.ylabel("CCS (Å²)")
                plt.title(f"{pol_tag}CCS vs carbons — {cls} (MAD={mad:.2f})")
                plt.tight_layout()

                # label a limited number (avoid unreadable plots)
                lab_idx = list(outlier_idx[:label_max_per_class])
                for j in lab_idx:
                    ann = str(out.at[j, annotation_col]) if annotation_col in out.columns else ""
                    uid = str(out.at[j, id_col]) if id_col in out.columns else str(j)
                    plt.text(
                        float(out.at[j, "_x_num"]) + 0.15,
                        float(out.at[j, "_ccs_num"]),
                        f"{uid} {ann}"[:60],
                        fontsize=7
                    )

                safe_cls = re.sub(r"[^A-Za-z0-9_.-]+", "_", cls)[:80]
                plt.savefig(plot_dir / f"{pol_tag}CCS_outliers_{safe_cls}.png", dpi=100)
                plt.close()

    # cleanup helpers
    out = out.drop(columns=["_ccs_num", "_x_num"], errors="ignore")
    return out

def apply_kendrick_filter(
    df,
    mass_column="Neutral mass",
    subclass_column="Lipid Class",
    kmd_deviation=0.75,
    min_class_size=6,
    output_folder=None,
    pol_tag=""
):
    """
    Filter features based on Kendrick Mass Defect (KMD) consistency within each lipid class.

    Keeps rows where KMD is within ±kmd_deviation of the class median.
    Classes with ≤ min_class_size entries are left unfiltered.
    """
    print(f'\nApplying Kendrick Mass Defect filtering... \n')

    df = df.copy()
    removed_rows = []
    kept_rows = []

    # Normalize adduct text so we can detect 2+ charge states
    if "Adducts" in df.columns:
        df["_Adducts_str"] = (
            df["Adducts"]
            .astype(str)
            .str.strip()
        )
    else:
        df["_Adducts_str"] = ""


    # Ensure required columns exist
    if mass_column not in df.columns:
        raise ValueError(f"Missing mass column: '{mass_column}'")
    if subclass_column not in df.columns:
        raise ValueError(f"Missing subclass column: '{subclass_column}'")

    # Compute Kendrick Mass Defect for all rows
    df["Kendrick Mass"], df["KMD"] = zip(*df[mass_column].astype(float).map(calculate_kendrick_mass_defect))

    # Compute class medians
    kmd_medians = (
        df.groupby(subclass_column)["KMD"]
        .agg(["count", "median"])
        .reset_index()
    )
    kmd_medians = kmd_medians[kmd_medians["count"] > min_class_size]
    median_map = dict(zip(kmd_medians[subclass_column], kmd_medians["median"]))

    # Apply filtering
    for _, row in df.iterrows():
        subclass = row.get(subclass_column)
        kmd = row.get("KMD", np.nan)
        adduct = str(row.get("_Adducts_str", ""))

        # # --- Do NOT filter doubly charged species (e.g. [M+H+H]2+) ---
        # KMDs are calculated based on neutral masses = charges don't matter
        # if "2+" in adduct or "2M" in adduct:
        #     kept_rows.append(row)
        #     continue

        if subclass in median_map:
            deviation = abs(kmd - median_map[subclass])
            if deviation <= kmd_deviation:
                kept_rows.append(row)
            else:
                r = copy.deepcopy(row)
                r["removed_reason"] = f"KMD deviation {deviation:.4f} > {kmd_deviation}"
                removed_rows.append(r)
        else:
            # Keep small subclasses unchanged
            kept_rows.append(row)


    kept_df = pd.DataFrame(kept_rows)
    removed_df = pd.DataFrame(removed_rows)

    print(f"[INFO] Kendrick filter removed {len(removed_df)} features; kept {len(kept_df)}.")
    print(f"[INFO] Median KMDs calculated for {len(median_map)} classes (min size = {min_class_size}).")

    # Optional output logging
    if output_folder:
        debug_dir = Path(output_folder) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        if not removed_df.empty:
            removed_df.to_csv(
                debug_dir / f"{pol_tag}Annotations_Removed_by_KMD.csv",
                index=False,
                encoding="utf-8-sig"
            )

    return kept_df, removed_df

def run_pipeline(input_csv, output_folder, min_score=70, scoring_module="scoring_mammalians", plausibility_module="plausability_filtering_mammalians"):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(input_csv, low_memory=False)

    # Dynamically import scoring & plausibility logic
    scoring = importlib.import_module(scoring_module)
    plausibility = importlib.import_module(plausibility_module)

    print(f'\nFiltering annotations... \n')

    # --- Determine polarity tag from sample columns ---
    sample_cols = [c for c in df.columns if isinstance(c, str) and (c.startswith("P_") or c.startswith("N_"))]

    pol_tag = ""
    if sample_cols:
        has_pos = any(c.startswith("P_") for c in sample_cols)
        has_neg = any(c.startswith("N_") for c in sample_cols)
        if has_pos and not has_neg:
            pol_tag = "Pos_"
        elif has_neg and not has_pos:
            pol_tag = "Neg_"
        else:
            print("***** ERROR: Mixed polarities detected in annotation input! *****", flush=True)
            pol_tag = "Mixed_"

    
    """
    Lipid filtering pipeline with debug printouts of unassigned counts.
    """
    input_path = Path(input_csv)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print('\n----- Starting MS filtering ----- \n')
    # Step 1: Load
    df = pd.read_csv(input_path, low_memory=False)
    print(f'Before filtering and scoring: {len(df)}, unassigned: {count_unassigned(df)}', flush=True)

    # Step 2: Apply scoring
    print(f"[INFO] Applying scoring using {scoring_module}")
    df_scored = scoring.apply_scoring(df, output_folder, pol_tag)
    print(f'After scoring and RT filter: {len(df_scored )}, unassigned: {count_unassigned(df_scored )}', flush=True)

    # ------------------------------------------------------------
    # Step 3: Separate internal standards before filtering
    # ------------------------------------------------------------
    if "Annotation Type" in df_scored.columns:
        is_mask = df_scored["Annotation Type"].astype(str).str.upper().str.strip().eq("IS")
        df_is = df_scored[is_mask].copy()
        df_nonis = df_scored[~is_mask].copy()
        print(f"[INFO] Detected {len(df_is)} internal standards. These will be excluded from filtering steps.", flush=True)
    else:
        df_is = pd.DataFrame()
        df_nonis = df_scored.copy()

    # Step 3: Apply plausibility filter (only to non-IS), but EXEMPT certain high-confidence MetaboScape annotations
    print(f"[INFO] Applying plausibility filtering using {plausibility_module} (excluding internal standards)")
    df_work = df_nonis.copy()

    # Build exemption mask:
    #  - Annotation tier == "High confidence"
    #  - Metaboscape Annotation Status starts with "Lipid Species" OR "Target List"
    tier = df_work.get("Annotation tier", pd.Series([""] * len(df_work), index=df_work.index)).astype(str).str.strip()
    status = df_work.get("Metaboscape Annotation Status", pd.Series([""] * len(df_work), index=df_work.index)).astype(str).str.strip()

    exempt_mask = (
        tier.eq("High confidence")
        & (
            status.str.startswith("Lipid Species", na=False)
            | status.str.startswith("Target List", na=False)
        )
    )

    df_exempt = df_work[exempt_mask].copy()
    df_to_filter = df_work[~exempt_mask].copy()

    print(f"[INFO] Exempting {len(df_exempt)} rows from plausibility filter (High confidence + Lipid Species/Target List).", flush=True)

    df_filtered_rest = plausibility.apply_plausability_filter(df_to_filter, output_folder, pol_tag)

    # Recombine (keep exempt rows unmodified)
    df_filtered = pd.concat([df_filtered_rest, df_exempt], ignore_index=True)

    print(f'After plausibility filter: {len(df_filtered)}, unassigned: {count_unassigned(df_filtered)}', flush=True)
    
    # ------------------------------------------------------------
    # Step 4: Apply Kendrick Mass Defect filter
    # ------------------------------------------------------------
    try:
        df_kmd_kept, df_kmd_removed = apply_kendrick_filter(
            df_filtered,
            mass_column="Neutral mass",
            subclass_column="Lipid Class",
            kmd_deviation=0.75,  # decrease to make it more strict
            output_folder=output_folder,
            pol_tag=pol_tag
        )
        df_filtered = df_kmd_kept
        print(f"[INFO] After Kendrick filter: {len(df_filtered)} kept; {len(df_kmd_removed)} removed.", flush=True)
    except Exception as e:
        print(f"[WARNING] Kendrick filter skipped: {e}", flush=True)


    # Step 5: Collapse duplicates (still only non-IS)
    df_collapsed = collapse_duplicates(df_filtered)
    print(f'After collapse duplicates: {len(df_collapsed)}, unassigned: {count_unassigned(df_collapsed)}', flush=True)

    # Step 6: Apply cutoff to scored rows, keep unassigned (non-IS)
    work = df_collapsed
    if "Annotation" in work.columns:
        ann = work["Annotation"].astype(str).str.strip()
        unassigned = ann.eq("") | ann.eq("nan") | ann.eq("Unassigned") | work["Annotation"].isna()
    else:
        unassigned = pd.Series([True] * len(work), index=work.index)

    df_final_nonis = work[(work["MS Score"] >= min_score) | (unassigned)].reset_index(drop=True)
    # Step 7: Recombine IS with processed lipids BEFORE CCS/RSD/correlation
    df_final = pd.concat([df_final_nonis, df_is], ignore_index=True)
    print(f"[INFO] Recombined table (pre-CCS) has {len(df_final)} total rows (including internal standards).", flush=True)
    print(f'After cutoff (non-IS only): {len(df_final_nonis)}, unassigned: {count_unassigned(df_final_nonis)}', flush=True)

    # ------------------------------------------------------------
    # CCS sanity check: flag CCS-vs-carbons outliers per Lipid Class
    #  - MUST NOT crash if CCS column missing
    #  - Adds: CCS_outlier?, CCS_outlier_score, CCS_outlier_reason
    #  - Plots into output_folder/debug/ccs_by_class (if possible)
    # ------------------------------------------------------------
    try:
        # Accept either CCS header (MetaboScape sometimes uses Å)
        if "CCS (Å²)" in df_final.columns:
            ccs_col = "CCS (Å²)"
        elif "CCS (Å²)" in df_final.columns:
            ccs_col = "CCS (Å²)"
        else:
            ccs_col = None

        if ccs_col is not None:
            # Appendix CCS library (CSV or Excel)
            # Appendix folder lives with the program (not the output folder)
            try:
                program_root = Path(__file__).resolve().parent   # when running as a .py script
            except NameError:
                program_root = Path.cwd()                        # notebooks / interactive runs

            appendix_dir = program_root / "Appendix"

            ref_candidates = []
            if appendix_dir.exists():
                ref_candidates.extend(sorted(appendix_dir.glob("Measured CCS Library.csv")))
                ref_candidates.extend(sorted(appendix_dir.glob("Measured_CCS_Library.*")))
                # if you want to allow any file that contains the name:
                ref_candidates.extend(sorted(appendix_dir.glob("*Measured*CCS*Library*.*")))

            reference_library_path = str(ref_candidates[0]) if ref_candidates else None

            reference_library_path = str(ref_candidates[0]) if len(ref_candidates) > 0 else None
            if reference_library_path:
                print(f"[INFO] Using CCS library: {reference_library_path}", flush=True)
            else:
                print("[INFO] No 'Measured CCS Library' found in Appendix; CCS uses project data only.", flush=True)
                
            df_final = flag_and_plot_ccs_outliers_by_class(
                df_final,
                output_folder=output_folder,
                pol_tag=pol_tag,
                class_col="Lipid Class",
                headgroup_col="Headgroup",
                adduct_col="Adducts",
                carbons_col="Number of carbons in fatty acyls",
                ccs_col=ccs_col,
                annotation_col="Annotation",
                uid_col="UniqueID",
                min_class_size=12,
                mad_z_thresh=10,
                max_labels_per_class=20,
                use_headgroup_for_classes=["HexCer"],
                skip_ccs_classes=[],
                reference_library_path=reference_library_path,
                reference_label="Measured CCS Library"
            )

            # OPTIONAL: eliminate flagged CCS outliers (keep IS)
            if "Annotation Type" in df_final.columns and "CCS_outlier?" in df_final.columns:
                is_is = df_final["Annotation Type"].astype(str).str.upper().str.strip().eq("IS")
                remove_ccs = df_final["CCS_outlier?"].astype(str).str.upper().str.strip().eq("YES") & (~is_is)

                n_removed_ccs = int(remove_ccs.sum())
                if n_removed_ccs > 0:
                    removed_ccs_df = df_final.loc[remove_ccs].copy()
                    removed_ccs_df["removed_reason"] = removed_ccs_df.get("CCS_outlier_reason", "").astype(str)
                    out_csv = Path(output_folder) / "debug" / f"{pol_tag}Annotations_Removed_by_CCS_outliers.csv"
                    removed_ccs_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
                    df_final = df_final.loc[~remove_ccs].copy()
        else:
            # Ensure expected columns exist even if CCS missing
            if "CCS_outlier?" not in df_final.columns:
                df_final["CCS_outlier?"] = ""
            if "CCS_outlier_score" not in df_final.columns:
                df_final["CCS_outlier_score"] = np.nan
            if "CCS_outlier_reason" not in df_final.columns:
                df_final["CCS_outlier_reason"] = "CCS outlier check skipped (no CCS column)"

    except Exception as e:
        # Never block pipeline
        if "CCS_outlier?" not in df_final.columns:
            df_final["CCS_outlier?"] = ""
        if "CCS_outlier_score" not in df_final.columns:
            df_final["CCS_outlier_score"] = np.nan
        if "CCS_outlier_reason" not in df_final.columns:
            df_final["CCS_outlier_reason"] = f"CCS outlier check skipped (error: {e})"
    
    
    # --- Compute RSD QCs (%) and RSD Samples (%) for all features ---
    group_file = Path(output_folder).parent / "sample_groups.csv"
    if group_file.exists():
        group_df = pd.read_csv(group_file, low_memory=False)
        qc_samples = group_df.loc[group_df["Group"].str.upper().str.strip() == "QC", "Sample"].tolist()

        # Build group → sample mapping
        group_map = {
            g.strip(): [s for s in group_df.loc[group_df["Group"] == g, "Sample"].tolist()]
            for g in group_df["Group"].unique()
        }

        # Identify all sample columns
        sample_cols = [c for c in df_final.columns if c.startswith("P_") or c.startswith("N_")]

        # Match QC columns
        qc_cols = []
        for sample in qc_samples:
            if sample in df_final.columns:
                qc_cols.append(sample)
            else:
                qc_cols.extend([c for c in df_final.columns if sample in c])
        qc_cols = list(set(qc_cols))
        print(f"[INFO] Found {len(qc_cols)} QC columns for RSD calculation.", flush = True)

        # Compute per-row RSDs
        rsd_qc_vals, rsd_sample_vals = [], []
        for _, row in df_final.iterrows():
            # QC RSD
            if qc_cols:
                qc_vals = row[qc_cols].astype(float).replace(0, np.nan).dropna()
                rsd_qc = (qc_vals.std(ddof=1) / qc_vals.mean()) * 100 if len(qc_vals) > 1 else np.nan
            else:
                rsd_qc = np.nan

            # Group RSD (across all non-QC samples)
            non_qc_cols = [c for c in sample_cols if c not in qc_cols]
            vals = row[non_qc_cols].astype(float).replace(0, np.nan).dropna()
            rsd_samples = (vals.std(ddof=1) / vals.mean()) * 100 if len(vals) > 1 else np.nan

            rsd_qc_vals.append(rsd_qc)
            rsd_sample_vals.append(rsd_samples)

        df_final["RSD QCs (%)"] = rsd_qc_vals
        df_final["RSD Samples (%)"] = rsd_sample_vals
        print("[INFO] Added RSD QCs (%) and RSD Samples (%) columns to final results.", flush = True)
    else:
        print("[WARNING] sample_groups.csv not found; RSD QCs and Samples left blank.", flush = True)

    def collapse_same_annotation_rt_clusters(
        df: pd.DataFrame,
        sample_cols: list[str],
        qc_cols: Optional[list[str]] = None,
        annotation_col: str = "Annotation",
        class_col: str = "Lipid Class",
        rt_col: str = "RT (min)",
        uid_col: str = "UniqueID",
        ann_type_col: str = "Annotation Type",
        max_rt_window_min: float = 0.3,
        r_thresh: float = 0.90,
        min_nonzero: int = 4,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collapse redundant multi-RT features that share the same annotation (and class)
        by clustering within an RT window using correlation across samples.
        Protects internal standards (Annotation Type == IS).
        Returns (df_kept, decisions_df).
        """
        out = df.copy()

        # Protect IS
        is_is = out.get(ann_type_col, "").astype(str).str.upper().str.strip().eq("IS")
        work = out.loc[~is_is].copy()
        keep = out.loc[is_is].copy()

        # Choose columns for correlation
        use_cols = qc_cols if (qc_cols and len(qc_cols) >= 3) else sample_cols
        X = work[use_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Xlog = np.log10(X + 1.0)

        decisions = []
        keep_uids = set()

        # group key: Annotation + Class (safer than Annotation alone)
        g_ann = work.get(annotation_col, "").astype(str).str.strip()
        g_cls = work.get(class_col, "").astype(str).str.strip()
        group_key = (g_ann + " || " + g_cls).fillna("")

        for g, idx in work.groupby(group_key).groups.items():
            sub = work.loc[idx].copy()
            sub = sub[(sub[annotation_col].astype(str).str.strip() != "") & (sub[annotation_col].astype(str).str.lower().str.strip() != "nan")]
            if sub.shape[0] == 0:
                continue
            if sub.shape[0] == 1:
                u = str(sub.iloc[0][uid_col])
                keep_uids.add(u)
                decisions.append({"GroupKey": g, "KeptUniqueID": u, "DroppedUniqueIDs": "", "Reason": "single"})
                continue

            sub = sub.sort_values(rt_col)
            uids = sub[uid_col].astype(str).tolist()
            rts = pd.to_numeric(sub[rt_col], errors="coerce").fillna(-999).tolist()

            subX = Xlog.loc[sub.index, :]
            nnz = (X.loc[sub.index, :] > 0).sum(axis=1).values

            # union-find
            n = len(uids)
            parent = list(range(n))
            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            def union(a,b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for i in range(n):
                if nnz[i] < min_nonzero:
                    continue
                for j in range(i+1, n):
                    if nnz[j] < min_nonzero:
                        continue
                    if abs(rts[i] - rts[j]) > max_rt_window_min:
                        continue
                    a = subX.iloc[i].values
                    b = subX.iloc[j].values
                    if np.std(a) == 0 or np.std(b) == 0:
                        continue
                    r = np.corrcoef(a, b)[0, 1]
                    if np.isfinite(r) and r >= r_thresh:
                        union(i, j)

            comps = {}
            for i in range(n):
                comps.setdefault(find(i), []).append(i)

            # choose representative per component
            for comp_id, nodes in enumerate(comps.values()):
                nodes_idx = sub.index[nodes]
                med = X.loc[nodes_idx, :].median(axis=1)
                rsd_qc = pd.to_numeric(sub.loc[nodes_idx].get("RSD QCs (%)", np.nan), errors="coerce")
                rsd_s  = pd.to_numeric(sub.loc[nodes_idx].get("RSD Samples (%)", np.nan), errors="coerce")

                score = pd.DataFrame({
                    "rsd_qc": rsd_qc,
                    "rsd_samples": rsd_s,
                    "med": med
                }).sort_values(by=["rsd_qc", "rsd_samples", "med"], ascending=[True, True, False], na_position="last")

                kept_idx = score.index[0]
                kept_uid = str(work.loc[kept_idx, uid_col])
                drop_uids = [str(work.loc[i2, uid_col]) for i2 in nodes_idx if i2 != kept_idx]

                keep_uids.add(kept_uid)
                decisions.append({
                    "GroupKey": g,
                    "ClusterID": comp_id,
                    "KeptUniqueID": kept_uid,
                    "DroppedUniqueIDs": ",".join(drop_uids),
                    "RTs": ",".join([f"{float(work.loc[i2, rt_col]):.3f}" for i2 in nodes_idx]),
                    "Reason": f"corr>= {r_thresh} within {max_rt_window_min} min using {'QC' if (qc_cols and len(qc_cols)>=3) else 'ALL'}"
                })

        kept_work = work[work[uid_col].astype(str).isin(keep_uids)].copy()
        out_kept = pd.concat([kept_work, keep], ignore_index=True)
        decisions_df = pd.DataFrame(decisions)
        return out_kept, decisions_df

    # ------------------------------------------------------------
    # Collapse redundant multi-RT features sharing same Annotation+Class
    # ------------------------------------------------------------
    try:
        # sample columns already defined earlier, but re-derive safely
        sample_cols = [c for c in df_final.columns if isinstance(c, str) and (c.startswith("P_") or c.startswith("N_"))]

        # QC cols from the group file block above if it exists
        qc_cols = qc_cols if "qc_cols" in locals() else None

        df_final, corr_decisions = collapse_same_annotation_rt_clusters(
            df=df_final,
            sample_cols=sample_cols,
            qc_cols=qc_cols,
            max_rt_window_min=0.3,
            r_thresh=0.90,
            min_nonzero=4,
        )
        corr_decisions.to_csv(
            Path(output_folder) / "debug" / f"{pol_tag}Annotations_Collapsed_same_annotation_by_corr.csv",
            index=False, encoding="utf-8-sig"
        )
        print(f"[INFO] Correlation-based annotation collapse complete. Rows now: {len(df_final)}", flush=True)
    except Exception as e:
        print(f"[WARNING] Correlation-based annotation collapse skipped: {e}", flush=True)

    # --- Apply unified column ordering ---
    df_final = reorder_columns(df_final)
    df_scored = reorder_columns(df_scored)

    # Step 6: Save outputs
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    debug_folder = output_path / "debug"
    debug_folder.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_csv)
    scored_name = f"{pol_tag}{input_path.stem}_scored.csv"   # gives raw_ms_search_results_scored.csv
    scored_path = debug_folder / scored_name
    final_path = debug_folder / f"{pol_tag}1-Final_MS_results.csv"

    df_scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    df_final.to_csv(final_path, index=False, encoding="utf-8-sig")
    
    # Step 7: Generate Internal Standards table
    if "Annotation Type" in df_final.columns:
        internal_standards_df = df_final[df_final["Annotation Type"].astype(str).str.upper() == "IS"].copy()
        internal_standards_df = reorder_columns(internal_standards_df)
        if not internal_standards_df.empty:
            internal_standards_path = output_path / f"{pol_tag}Internal_standards.csv"
            internal_standards_df.to_csv(internal_standards_path, index=False, encoding="utf-8-sig")
            print(f"Internal standards table saved to: {internal_standards_path}", flush = True)
        else:
            print("No internal standards detected in 1-Final_MS_results.", flush = True)
            
        # --- Generate internal standard plots ---
        try:
            plot_internal_standards(pol_tag, internal_standards_csv=internal_standards_path, output_folder=output_path)
            print(f"\n ----- Internal standard plots saved to ({output_folder}) ----- \n", flush = True)
        except Exception as e:
            print(f"\n\n ======= Warning: could not generate internal standard plots ({e}) ========\n\n", flush = True)
    
    else:
        print("Warning: 'Annotation Type' column not found; skipping internal standards export.", flush = True)
        
    # ----------------------------------------------
    #                HANDLE ADDUCTS
    # ----------------------------------------------
    
    kept_path, removed_path, summary_path = handle_adducts(
        input_csv=final_path,
        output_folder=output_folder,
        rt_tolerance_seconds=6,
        pol_tag=pol_tag
    )

    # ----------------------------------------------
    #  DEBUG: which adducts were kept vs removed per headgroup
    # ----------------------------------------------
    try:
        plot_adducts_kept_vs_removed_by_headgroup(
            kept_csv=kept_path,
            removed_csv=removed_path,
            output_folder=output_folder,
            pol_tag=pol_tag,
            headgroup_col="Headgroup",
            adduct_col="Adducts",
            min_total_per_headgroup=5,
            max_adducts_per_headgroup=12
        )
    except Exception as e:
        print(f"[WARNING] Adduct-by-headgroup debug plots failed: {e}", flush=True)
        
    removed_plausibility_path = (
        Path(output_folder)
        / "debug"
        / f"{pol_tag}Annotations_Removed_by_plausibility.csv"
    )
    
    removed_kmd_path = (
        Path(output_folder)
        / "debug"
        / f"{pol_tag}Annotations_Removed_by_KMD.csv"
    )
    
    removed_rt_path = (
        Path(output_folder)
        / "debug"
        / f"{pol_tag}Annotations_Removed_by_rt.csv"
    )
    
    removed_plot_root = (
        Path(output_folder)
        / "debug"
        / "annotations_removed"
    )
    removed_plot_root.mkdir(parents=True, exist_ok=True)


    # ----------------------------------------------
    #      PLOT RESULTS (from generate_plots.py)
    # ----------------------------------------------
        
    print("[INFO] Plotting annotation results.", flush = True)
    try:
        plot_results(pol_tag, input_csv = kept_path, output_folder=output_folder, suffix=f"_{pol_tag}before_norm")
    except Exception as e:
        print(f"\n\n ======= Plot results failed due to error {e}. ========\n\n", flush = True)
    try:
        plot_kendrick_mass_vs_defect(input_csv = kept_path, results_folder = output_folder, suffix=f"_{pol_tag}before_norm")
    except Exception as e:
        print(f"\n\n ======= Plot KMD results failed due to error {e}. ========\n\n", flush = True)
    
    # ----------------------------------------------
    #  PLOT ANNOTATIONS REMOVED BY PLAUSIBILITY
    # ----------------------------------------------

    if removed_plausibility_path.exists():
        print(
            f"[INFO] Plotting annotations removed by plausibility filter "
            f"to {removed_plot_root}",
            flush=True
        )

        try:
            plot_results(
                pol_tag,
                input_csv=removed_plausibility_path,
                output_folder=removed_plot_root,
                suffix=f"_{pol_tag}removed_plausibility"
            )
        except Exception as e:
            print(
                f"[WARNING] Plotting plausibility-removed annotations failed: {e}",
                flush=True
            )
            
        try:
            plot_removed_reason_counts(
                input_csv=removed_plausibility_path,
                output_folder=removed_plot_root,
                pol_tag=pol_tag,
                suffix=""
            )
        except Exception as e:
            print(
                f"\n[WARNING] Plotting plausibility removal reasons failed: {e}\n",
                flush=True
            )
                
    else:
        print(
            f"[INFO] No plausibility-removed annotation file found at {removed_plausibility_path}; skipping plots.",
            flush=True
        )
               
    # ----------------------------------------------
    #  PLOT ANNOTATIONS REMOVED BY KMD
    # ----------------------------------------------

    if removed_kmd_path.exists():
        print(
            f"[INFO] Plotting annotations removed by KMD filter "
            f"to {removed_plot_root}",
            flush=True
        )

        try:
            plot_results(
                pol_tag,
                input_csv=removed_kmd_path,
                output_folder=removed_plot_root,
                suffix=f"_{pol_tag}removed_KMD"
            )
        except Exception as e:
            print(
                f"[WARNING] Plotting KMD-removed annotations failed: {e}",
                flush=True
            )

    # ----------------------------------------------
    #  PLOT ANNOTATIONS REMOVED BY RT FILTER
    # ----------------------------------------------

    if removed_rt_path.exists():
        print(
            f"[INFO] Plotting annotations removed by RT filter "
            f"to {removed_plot_root}",
            flush=True
        )

        try:
            plot_results(
                pol_tag,
                input_csv=removed_rt_path,
                output_folder=removed_plot_root,
                suffix=f"_{pol_tag}removed_RT"
            )
        except Exception as e:
            print(
                f"[WARNING] Plotting RT-removed annotations failed: {e}",
                flush=True
            )
    else:
        print(
            "[INFO] No RT-removed annotation file found; skipping RT plots.",
            flush=True
        )

        
    return scored_path, kept_path
