# Stats/class_number_carbons_DB.py
# ------------------------------------------------------------
# Stacked bar plots per lipid class:
#   For each class, one plot.
#   X-axis: GROUP
#
# 1) Carbons stack:
#     stack = total carbons (Number of carbons in fatty acyls)
# 2) Double bonds stack:
#     stack = DB equivalents (Double bond equivalents)
#
# Each bar segment height:
#   mean across samples in that group of:
#     (sum of normalized intensities of features in that class with that bin)
#
# Writes:
#   Carbons/
#     PerClass_CSV/<Class>__group_x_carbons_mean.csv
#     PerClass_PNG/<Class>__stacked_carbons.png
#   DoubleBonds/
#     PerClass_CSV/<Class>__group_x_doublebonds_mean.csv
#     PerClass_PNG/<Class>__stacked_doublebonds.png
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from Stats.utils import load_dataset

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
plt.ioff()

def _long_path(p: str) -> str:
    """
    On Windows, allow paths > 260 chars by using the extended-length prefix \\?\
    Must be absolute.
    """
    p = str(p)
    if os.name != "nt":
        return p
    ap = os.path.abspath(p)
    if ap.startswith("\\\\?\\"):
        return ap
    return "\\\\?\\" + ap

def _ensure_dir(p: str) -> str:
    lp = _long_path(p)
    os.makedirs(lp, exist_ok=True)
    return p  # return the normal path string for readability elsewhere

def _sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:."/\\|?*]', "_", str(s))


def _pick_column_ci(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Case-insensitive, strip-aware column picker. Returns first match."""
    if df is None or df.empty:
        return None
    cols = [str(c) for c in df.columns]
    norm = {str(c).strip().lower(): str(c) for c in cols}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm:
            return norm[key]
    return None


def _parse_total_carbons_from_text(s: str) -> float:
    """
    Parse total carbons from strings like:
      'PC 34:1', 'TG 52:3', 'Cer 34:1;O2', 'PE(36:2)', 'SM d42:2'
    Returns np.nan if not found.
    """
    if s is None:
        return np.nan
    txt = str(s)
    m = re.search(r'(\d+)\s*:\s*(\d+)', txt)
    if m:
        try:
            return float(int(m.group(1)))
        except Exception:
            return np.nan
    return np.nan


def _parse_total_double_bonds_from_text(s: str) -> float:
    """
    Parse total double bonds from strings like:
      'PC 34:1', 'TG 52:3', 'Cer 34:1;O2', 'PE(36:2)', 'SM d42:2'
    Returns np.nan if not found.
    """
    if s is None:
        return np.nan
    txt = str(s)
    m = re.search(r'(\d+)\s*:\s*(\d+)', txt)
    if m:
        try:
            return float(int(m.group(2)))
        except Exception:
            return np.nan
    return np.nan


def _order_groups(present: List[str], group_order: Optional[List[str]]) -> List[str]:
    present = [str(g) for g in present]
    if not group_order:
        return present
    gui = [g for g in group_order if g in present]
    rest = [g for g in present if g not in gui]
    return gui + rest


def _write_debug_snapshot(save_dir: str, X: pd.DataFrame, y: pd.Series, feature_meta: pd.DataFrame) -> None:
    try:
        sd = Path(save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "DBG_X_shape.txt").write_text(
            f"X shape: {X.shape}\nX columns (first 30): {list(X.columns[:30])}\n",
            encoding="utf-8",
        )
        (sd / "DBG_y_head.csv").write_text(y.head(50).to_csv(index=True), encoding="utf-8")
        (sd / "DBG_feature_meta_head.csv").write_text(feature_meta.head(50).to_csv(index=True), encoding="utf-8")
        (sd / "DBG_feature_meta_columns.txt").write_text("\n".join(map(str, feature_meta.columns)), encoding="utf-8")
    except Exception:
        pass


def _stacked_per_class(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    feature_meta: pd.DataFrame,
    cls_series: pd.Series,
    value_series: pd.Series,
    value_name: str,                 # "carbons" or "doublebonds"
    csv_dir: str,
    png_dir: str,
    group_order: Optional[List[str]],
    title_suffix: str,               # "carbon-number composition" etc
    y_label: str,
) -> None:
    """
    Make per-class stacked bars for a single binned variable.
    value_series must be numeric (NaN for missing).
    """
    mask = (
        cls_series.notna()
        & (cls_series.astype(str).str.strip() != "")
        & (cls_series.astype(str).str.strip().str.lower() != "nan")
        & value_series.notna()
    )
    if int(mask.sum()) == 0:
        raise ValueError(f"No features remain after filtering for valid class and {value_name}.")

    meta_f = feature_meta.loc[mask].copy()
    meta_f["_Class"] = cls_series.loc[mask].astype(str).str.strip()
    meta_f["_Bin"] = value_series.loc[mask].astype(int)

    classes = sorted(meta_f["_Class"].unique().tolist())

    for cls in classes:
        sub_meta = meta_f[meta_f["_Class"] == cls]
        if sub_meta.empty:
            continue

        feats = sub_meta.index.tolist()
        Xc = X.loc[:, feats]

        bin_map = sub_meta["_Bin"].to_dict()     # feature -> bin int
        bins = sorted(sub_meta["_Bin"].unique().tolist())

        # Per-sample sums per bin
        per_sample_bins = pd.DataFrame(index=Xc.index)
        for b in bins:
            feats_b = [f for f in feats if bin_map.get(f) == b]
            if feats_b:
                per_sample_bins[str(b)] = Xc[feats_b].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            else:
                per_sample_bins[str(b)] = 0.0

        df_bins = per_sample_bins.copy()
        df_bins["Group"] = y.reindex(df_bins.index).astype(str).values
        df_bins = df_bins.dropna(subset=["Group"])

        group_table = df_bins.groupby("Group")[[str(b) for b in bins]].mean()

        present_groups = group_table.index.astype(str).tolist()
        ordered_groups = _order_groups(present_groups, group_order)
        group_table = group_table.reindex(ordered_groups)

        # Write CSV
        safe_cls = _sanitize_filename(cls)
        out_csv = os.path.join(csv_dir, f"{safe_cls}__group_x_{value_name}_mean.csv")
        group_table.to_csv(_long_path(out_csv), index=True)

        # Plot stacked bars
        fig, ax = plt.subplots(figsize=(10.5, 6.8), facecolor="white")
        ax.set_facecolor("white")
        fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.28)

        x = np.arange(len(group_table.index))
        bottoms = np.zeros(len(group_table.index), dtype=float)

        for col in group_table.columns:
            vals = pd.to_numeric(group_table[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottoms, label=str(col))
            bottoms = bottoms + vals

        ax.set_title(f"{cls}: {title_suffix} by group", fontsize=14, pad=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(group_table.index.tolist(), rotation=45, ha="right")

        ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_axisbelow(True)

        ax.legend(title=value_name, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")

        out_png = os.path.join(png_dir, f"{safe_cls}__stacked_{value_name}.png")
        fig.savefig(_long_path(out_png), dpi=100, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


def run_from_stats(
    file_path: str,
    group_file: Optional[str],
    save_dir: str,
    group_order: Optional[List[str]] = None,
    group_colors: Optional[dict] = None,  # not used here (stack colors are per-bin)
    exclude_qc: bool = True,
) -> Dict[str, str]:
    """
    Produce stacked bars per lipid class showing:
      - carbon-number composition by group
      - double-bond composition by group

    Returns a dict of output locations.
    """
    # GUI already passes the final output folder (.../<analysis_type>/<label>).
    out_dir = str(save_dir)
    os.makedirs(_long_path(out_dir), exist_ok=True)

    carb_csv_dir = _ensure_dir(os.path.join(out_dir, "Carbons"))
    carb_png_dir = _ensure_dir(os.path.join(out_dir, "Carbons"))
    db_csv_dir   = _ensure_dir(os.path.join(out_dir, "DoubleBonds"))
    db_png_dir   = _ensure_dir(os.path.join(out_dir, "DoubleBonds"))

    print("[Class carbons] Running class-level stacked bars (carbon-number composition)...", flush=True)

    # Load standardized dataset
    X, y, feature_meta = load_dataset(file_path, group_file)
    if X is None or y is None or feature_meta is None or X.empty or feature_meta.empty:
        raise ValueError("Dataset appears empty or malformed.")

    feature_meta = feature_meta.copy()
    feature_meta.columns = feature_meta.columns.astype(str).str.strip()

    # ---- FORCE FEATURE ID ALIGNMENT ----
    uid_col = _pick_column_ci(feature_meta, ["UniqueID"])
    if uid_col is not None:
        feature_meta["_FeatureID"] = feature_meta[uid_col].astype(str)
        feature_meta = feature_meta.set_index("_FeatureID", drop=True)
    else:
        feature_meta.index = feature_meta.index.astype(str)

    X = X.copy()
    X.columns = X.columns.astype(str)
    X.index = X.index.astype(str)
    y = y.copy()
    y.index = y.index.astype(str)

    common_feats = [c for c in X.columns if c in feature_meta.index]
    X = X.loc[:, common_feats]
    feature_meta = feature_meta.loc[common_feats]

    if exclude_qc:
        qc_mask = ~y.astype(str).str.contains("QC", case=False, na=False)
        X = X.loc[qc_mask]
        y = y.loc[qc_mask]

    # Required: class column
    class_col = _pick_column_ci(feature_meta, ["Lipid Class", "Headgroup"])
    if class_col is None:
        _write_debug_snapshot(out_dir, X, y, feature_meta)
        raise ValueError("Could not find a class column (expected 'Lipid Class' or 'Headgroup').")

    cls_series = feature_meta[class_col].astype(str).str.strip()

    # --- CARBONS series ---
    carb_col = _pick_column_ci(feature_meta, ["Number of carbons in fatty acyls", "Total carbons", "Carbons"])
    carb_primary = pd.to_numeric(feature_meta[carb_col], errors="coerce") if carb_col is not None else pd.Series(np.nan, index=feature_meta.index)
    carb = carb_primary.copy()
    frac_ok = float(carb_primary.notna().mean()) if len(carb_primary) else 0.0
    if frac_ok < 0.05:
        text_col = _pick_column_ci(feature_meta, ["Annotation", "NoAbbrev", "Name", "Lipid"])
        print(f"[Class carbons] Carbon column sparse ({frac_ok:.3f}). Fallback parse from {text_col}.", flush=True)
        if text_col is not None:
            carb_fallback = feature_meta[text_col].apply(_parse_total_carbons_from_text)
            carb = carb_primary.where(carb_primary.notna(), carb_fallback)

    # --- DOUBLE BONDS series ---
    db_col = _pick_column_ci(feature_meta, ["Double bond equivalents", "Double bonds", "DB", "DBE"])
    db_primary = pd.to_numeric(feature_meta[db_col], errors="coerce") if db_col is not None else pd.Series(np.nan, index=feature_meta.index)
    db = db_primary.copy()
    frac_db_ok = float(db_primary.notna().mean()) if len(db_primary) else 0.0
    if frac_db_ok < 0.05:
        text_col = _pick_column_ci(feature_meta, ["Annotation", "NoAbbrev", "Name", "Lipid"])
        print(f"[Class carbons] Double-bond column sparse ({frac_db_ok:.3f}). Fallback parse from {text_col}.", flush=True)
        if text_col is not None:
            db_fallback = feature_meta[text_col].apply(_parse_total_double_bonds_from_text)
            db = db_primary.where(db_primary.notna(), db_fallback)

    # Run the two plot families
    try:
        _stacked_per_class(
            X=X,
            y=y,
            feature_meta=feature_meta,
            cls_series=cls_series,
            value_series=carb,
            value_name="Total carbons",
            csv_dir=carb_csv_dir,
            png_dir=carb_png_dir,
            group_order=group_order,
            title_suffix="carbon-number composition",
            y_label="Mean summed normalized intensity",
        )
    except Exception:
        _write_debug_snapshot(os.path.join(out_dir, "Carbons"), X, y, feature_meta)
        raise

    print("[Class carbons] Running class-level stacked bars (double-bond composition)...", flush=True)

    try:
        _stacked_per_class(
            X=X,
            y=y,
            feature_meta=feature_meta,
            cls_series=cls_series,
            value_series=db,
            value_name="Double bonds",
            csv_dir=db_csv_dir,
            png_dir=db_png_dir,
            group_order=group_order,
            title_suffix="double-bond summed composition",
            y_label="Mean summed normalized intensity",
        )
    except Exception:
        _write_debug_snapshot(os.path.join(out_dir, "DoubleBonds"), X, y, feature_meta)
        raise

    return {
        "out_dir": out_dir,
        "carbons_csv_dir": carb_csv_dir,
        "carbons_png_dir": carb_png_dir,
        "doublebonds_csv_dir": db_csv_dir,
        "doublebonds_png_dir": db_png_dir,
    }