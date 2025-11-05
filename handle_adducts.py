# -------------------------------------------------------------------------
# Post-filtering adduct collapsing for LC-MS lipidomics results
# -------------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path

def handle_adducts(
    input_csv,
    output_folder="results",
    rt_tolerance_seconds=6
):
    """
    Collapse redundant adduct peaks for the same lipid annotation.

    Rules:
      1. Skip all internal standards (Annotation Type == 'IS').
      2. Within ±rt_tolerance_seconds and same Annotation:
         - MS/MS MATCH takes priority over MS MATCH and others.
         - If multiple remain in the same type, keep one with:
             a) Fewest missing values,
             b) Highest mean intensity.
      3. All other redundant peaks are removed and logged.
    """
    print(f'\nHandling adducts...\n')
    input_csv = Path(input_csv)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    debug_folder = output_folder / "debug"
    debug_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    if "Annotation" not in df.columns or "RT (min)" not in df.columns:
        raise ValueError("Input file must contain 'Annotation' and 'RT (min)' columns.")

    # Identify sample columns
    sample_cols = [c for c in df.columns if c.startswith("[POS") or c.startswith("[NEG]")]
    if not sample_cols:
        raise ValueError("No sample columns found. Expected columns starting with [POS or [NEG].")

    # Normalize Annotation Type for consistent comparison
    if "Annotation Type" in df.columns:
        df["Annotation Type norm"] = df["Annotation Type"].astype(str).str.upper().str.strip()
    else:
        df["Annotation Type norm"] = "MS MATCH"

    # Normalize "unknown-like" annotations to NaN (case/space-insensitive)
    _unknown_tokens = {"", "nan", "n/a", "none", "unassigned", "unknown", "no match"}
    ann_norm = (
        df["Annotation"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .map(lambda s: np.nan if s in _unknown_tokens else s)
    )
    df["Annotation_norm"] = ann_norm
        
    # Assign priority: lower = better
    # -----------------------------------------------------------------
    # Compute helper columns before grouping
    # -----------------------------------------------------------------
    def get_priority(x):
        if "MS/MS" in x:
            return 0
        elif "MS" in x:
            return 1
        else:
            return 2

    df["type_priority"] = df["Annotation Type norm"].apply(get_priority)
    df["RT_seconds"] = df["RT (min)"] * 60

    kept_rows = []
    removed_rows = []
    debug_entries = []

    # -------------------------------
    # Exclude internal standards early
    # -------------------------------
    is_mask = df["Annotation Type norm"].eq("IS")
    is_df = df[is_mask].copy()
    df = df[~is_mask].copy()
    if not is_df.empty:
        print(f"[INFO] Skipping {len(is_df)} internal standard features from adduct collapsing.", flush=True)

    # -----------------------------------------------------------------
    # Normalize "unknown-like" annotations to NaN and group
    # -----------------------------------------------------------------
    _unknown_tokens = {"", "nan", "n/a", "none", "unassigned", "unknown", "no match"}
    ann_norm = (
        df["Annotation"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .map(lambda s: np.nan if s in _unknown_tokens else s)
    )
    df["Annotation_norm"] = ann_norm

    # -----------------------------------------------------------------
    # Collapse adducts within annotation, keeping NaN (unknown) group intact
    # -----------------------------------------------------------------
    for ann, group in df.groupby("Annotation_norm", dropna=False):
        if pd.isna(ann):
            # Keep unannotated/unknown rows untouched
            kept_rows.append(group)
            continue

        group_sorted = group.sort_values("RT_seconds")

        while len(group_sorted) > 0:
            ref = group_sorted.iloc[0]
            window_mask = np.abs(group_sorted["RT_seconds"] - ref["RT_seconds"]) <= rt_tolerance_seconds
            window_group = group_sorted[window_mask].copy()

            # Compute missing values and mean intensity
            window_group["missing_count"] = window_group[sample_cols].isna().sum(axis=1)
            window_group["mean_intensity"] = window_group[sample_cols].mean(axis=1, skipna=True)

            # Rank by priority → missing_count → mean_intensity
            best_row = (
                window_group
                .sort_values(["type_priority", "missing_count", "mean_intensity"],
                            ascending=[True, True, False])
                .iloc[0]
            )

            kept_rows.append(best_row.to_frame().T)

            # Identify and log removed rows
            to_remove = window_group.drop(best_row.name)
            if not to_remove.empty:
                removed_rows.append(to_remove)
                removed_list = [
                    f"m/z={r['m/z']:.4f} @RT={r['RT (min)']:.2f}min ({r.get('Annotation Type', '')} {r.get('Adducts', '')})"
                    for _, r in to_remove.iterrows()
                ]
                debug_entries.append({
                    "Annotation": ann,
                    "Kept_mz": best_row["m/z"],
                    "Kept_RT(min)": best_row["RT (min)"],
                    "Kept_Type": best_row.get("Annotation Type", ""),
                    "Kept_Adduct": best_row.get("Adducts", ""),
                    "Kept_missing_values": int(best_row["missing_count"]),
                    "Kept_mean_intensity": best_row["mean_intensity"],
                    "Removed_count": len(to_remove),
                    "Removed_features": "; ".join(removed_list)
                })

            group_sorted = group_sorted.drop(window_group.index)

    # Combine results (add IS back in)
    kept_df = pd.concat(kept_rows, ignore_index=True)
    if not is_df.empty:
        kept_df = pd.concat([kept_df, is_df], ignore_index=True)

    removed_df = pd.concat(removed_rows, ignore_index=True) if removed_rows else pd.DataFrame()

    # Save outputs
    kept_path = debug_folder / "2-Final_annotated_results_adducts_collapsed.csv"
    removed_path = debug_folder / "removed_adducts.csv"
    summary_path = debug_folder / "adduct_collapse_summary.csv"

    kept_df.to_csv(kept_path, index=False, encoding="utf-8-sig")
    if not removed_df.empty:
        removed_df.to_csv(removed_path, index=False, encoding="utf-8-sig")

    if debug_entries:
        pd.DataFrame(debug_entries).to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Summary messages
    print(f"[INFO] Adduct handling complete.", flush=True)
    print(f"[INFO] Kept features: {len(kept_df)}", flush=True)
    print(f"[INFO] Removed adduct candidates: {len(removed_df)}", flush=True)
    print(f"[INFO] Saved collapsed results to: {kept_path}", flush=True)
    if not removed_df.empty:
        print(f"[INFO] Removed adduct list saved to: {removed_path}", flush=True)
    if debug_entries:
        print(f"[INFO] Collapse summary saved to: {summary_path}", flush=True)

    return kept_path, removed_path, summary_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collapse adduct-related duplicate peaks after filtering.")
    parser.add_argument("--input", required=True, help="Path to 1-Final_MS_results.csv")
    parser.add_argument("--out", default="results", help="Output folder")
    parser.add_argument("--tol", type=float, default=10, help="RT tolerance in seconds (default=10)")
    args = parser.parse_args()

    handle_adducts(args.input, args.out, args.tol)
