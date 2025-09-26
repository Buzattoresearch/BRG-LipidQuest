# data_cleansing.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

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
    cdf = cdf.copy()
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
    raise ValueError("Could not find an m/z column. Expected one of: 'm/z meas.', 'm/z', 'm/z measured', 'mz'.")

def apply_data_cleansing(
    df: pd.DataFrame,
    output_folder,
    contaminant_file="Appendix/Contaminants.csv",
    ppm_tolerance=5
):
    """
    Remove rows whose m/z matches a known contaminant (within ppm),
    using the row's own Polarity.

    Returns (cleaned_df, removed_df) and writes CSVs:
      - Removed_contaminants.csv
      - Cleaned_data.csv
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if "Polarity" not in df.columns:
        raise ValueError("Input DataFrame must have a 'Polarity' column.")

    mz_col = _auto_mz_col(df)
    cont = _load_contaminants_by_polarity(contaminant_file)

    # Normalize polarity per row
    pol = df["Polarity"].map(_normalize_polarity)

    # Prepare arrays for vectorized checks
    mz_vals = pd.to_numeric(df[mz_col], errors="coerce").values
    remove_flag = np.zeros(len(df), dtype=bool)
    reasons = np.full(len(df), "", dtype=object)

    # Build quick lambdas for ppm window
    def match_any(mz_array, targets, ppm):
        if not targets:
            return np.zeros_like(mz_array, dtype=bool), np.full(len(mz_array), np.nan)
        # For each target, compute inside-window mask; OR them together
        overall = np.zeros_like(mz_array, dtype=bool)
        which = np.full(len(mz_array), np.nan)  # store first matching target for reason
        for t in targets:
            lower = t * (1 - ppm/1e6)
            upper = t * (1 + ppm/1e6)
            mask = (mz_array >= lower) & (mz_array <= upper)
            # only fill 'which' where we newly matched
            newly = mask & (~overall)
            which[newly] = t
            overall |= mask
        return overall, which

    # Positive rows
    pos_mask = pol == "pos"
    pos_hits, pos_target = match_any(mz_vals, cont["pos"], ppm_tolerance)
    pos_remove = pos_mask.values & pos_hits

    # Negative rows
    neg_mask = pol == "neg"
    neg_hits, neg_target = match_any(mz_vals, cont["neg"], ppm_tolerance)
    neg_remove = neg_mask.values & neg_hits

    remove_flag |= pos_remove | neg_remove

    # Fill reasons
    reasons[pos_remove] = [
        f"contaminant (pos) @ {t:.4f} ppm±{ppm_tolerance}"
        if not np.isnan(t) else "contaminant (pos)"
        for t in pos_target[pos_remove]
    ]
    reasons[neg_remove] = [
        f"contaminant (neg) @ {t:.4f} ppm±{ppm_tolerance}"
        if not np.isnan(t) else "contaminant (neg)"
        for t in neg_target[neg_remove]
    ]

    removed_df = df.loc[remove_flag].copy()
    if not removed_df.empty:
        removed_df["data_cleanup_reason"] = reasons[remove_flag]
    else:
        removed_df = pd.DataFrame(columns=list(df.columns) + ["data_cleanup_reason"])

    cleaned_df = df.loc[~remove_flag].copy()

    # Save
    removed_df.to_csv(output_folder / "Removed_contaminants.csv", index=False, encoding="utf-8-sig")
    cleaned_df.to_csv(output_folder / "Cleaned_data.csv", index=False, encoding="utf-8-sig")

    print(f"Removed {len(removed_df)} rows by contaminant list; kept {len(cleaned_df)} rows.")
    return cleaned_df, removed_df




