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
    note_text: Optional[str] = None,
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
        fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.34 if note_text else 0.28)

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

        if note_text:
            fig.text(
                0.5,
                0.03,
                note_text,
                ha="center",
                va="bottom",
                fontsize=10,
                color="dimgray",
            )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")

        out_png = os.path.join(png_dir, f"{safe_cls}__stacked_{value_name}.png")
        fig.savefig(_long_path(out_png), dpi=100, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    wts = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = vals.notna() & np.isfinite(vals.to_numpy(dtype=float)) & np.isfinite(wts.to_numpy(dtype=float)) & (wts > 0)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.average(vals.loc[mask].to_numpy(dtype=float), weights=wts.loc[mask].to_numpy(dtype=float)))


def _weighted_linear_regression(x: pd.Series, y: pd.Series, weights: Optional[pd.Series] = None) -> Dict[str, float]:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    w_num = pd.Series(1.0, index=x.index) if weights is None else pd.to_numeric(weights, errors="coerce").fillna(0.0)

    mask = (
        x_num.notna()
        & y_num.notna()
        & np.isfinite(x_num.to_numpy(dtype=float))
        & np.isfinite(y_num.to_numpy(dtype=float))
        & np.isfinite(w_num.to_numpy(dtype=float))
        & (w_num > 0)
    )
    if int(mask.sum()) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": int(mask.sum())}

    x_arr = x_num.loc[mask].to_numpy(dtype=float)
    y_arr = y_num.loc[mask].to_numpy(dtype=float)
    w_arr = w_num.loc[mask].to_numpy(dtype=float)

    if np.unique(x_arr).size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": int(mask.sum())}

    Xmat = np.column_stack([np.ones_like(x_arr), x_arr])
    sqrt_w = np.sqrt(w_arr)
    beta, _, _, _ = np.linalg.lstsq(Xmat * sqrt_w[:, None], y_arr * sqrt_w, rcond=None)
    intercept, slope = beta
    y_hat = Xmat @ beta
    y_bar = np.average(y_arr, weights=w_arr)
    ss_res = np.sum(w_arr * (y_arr - y_hat) ** 2)
    ss_tot = np.sum(w_arr * (y_arr - y_bar) ** 2)
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - (ss_res / ss_tot))

    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n": int(mask.sum())}


def _plot_metric_heatmap(
    table: pd.DataFrame,
    out_png: str,
    out_svg: Optional[str],
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    center_zero: bool = False,
    note_text: Optional[str] = None,
    cbar_labelpad: int = 12,
) -> None:
    if table.empty:
        return

    data = table.astype(float)
    fig, ax = plt.subplots(
        figsize=(
            max(8, 1.0 * len(data.columns) + 3),
            max(6, 0.35 * len(data.index) + 2),
        ),
        facecolor="white",
    )

    vals = data.to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size == 0:
        plt.close(fig)
        return

    if center_zero:
        vmax = float(np.nanmax(np.abs(finite_vals)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))

    im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index.tolist())
    ax.set_title(title, fontsize=14, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, labelpad=cbar_labelpad)

    if note_text:
        fig.subplots_adjust(bottom=0.27)
        fig.text(
            0.5,
            0.005,
            note_text,
            ha="center",
            va="bottom",
            fontsize=10,
            color="dimgray",
        )

    fig.savefig(_long_path(out_png), dpi=120, bbox_inches="tight", pad_inches=0.15)
    if out_svg:
        fig.savefig(_long_path(out_svg), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _plot_2d_enrichment_surface_per_class(
    cls: str,
    class_df: pd.DataFrame,
    ordered_groups: List[str],
    out_png: str,
    out_svg: Optional[str] = None,
    note_text: Optional[str] = None,
) -> None:
    if class_df.empty:
        return

    mats = {}
    vmax = 0.0
    carbons = sorted(class_df["Carbons"].dropna().astype(int).unique().tolist())
    double_bonds = sorted(class_df["DoubleBonds"].dropna().astype(int).unique().tolist())
    if not carbons or not double_bonds:
        return

    for group in ordered_groups:
        sub = class_df[class_df["Group"] == group]
        pivot = (
            sub.pivot_table(
                index="DoubleBonds",
                columns="Carbons",
                values="MeanAbundance",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(index=double_bonds, columns=carbons, fill_value=0.0)
        )
        mats[group] = pivot
        vmax = max(vmax, float(np.nanmax(pivot.to_numpy(dtype=float))) if not pivot.empty else 0.0)

    n_groups = len(ordered_groups)
    ncols = min(3, max(1, n_groups))
    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(9, 4.8 * ncols), max(4.0, 3.8 * nrows)),
        facecolor="white",
        squeeze=False,
    )
    fig.subplots_adjust(top=0.82, bottom=0.30 if note_text else 0.16, hspace=0.68, wspace=0.32)

    for ax, group in zip(axes.ravel(), ordered_groups):
        mat = mats[group]
        im = ax.imshow(
            mat.to_numpy(dtype=float),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=vmax if vmax > 0 else None,
        )
        ax.set_title(str(group), fontsize=15)
        x_positions = np.arange(len(carbons))
        if len(carbons) > 18:
            label_step = 3
        elif len(carbons) > 10:
            label_step = 2
        else:
            label_step = 1
        shown_positions = x_positions[::label_step]
        shown_labels = [carbons[i] for i in shown_positions]
        ax.set_xticks(shown_positions)
        ax.set_xticklabels(shown_labels, rotation=90, ha="center", va="top", fontsize=12)
        ax.set_yticks(np.arange(len(double_bonds)))
        ax.set_yticklabels(double_bonds, fontsize=12)
        ax.set_xlabel("Total carbons", fontsize=13, labelpad=10)
        ax.set_ylabel("Double bond equivalents", fontsize=13, labelpad=12)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xticks(np.arange(-0.5, len(carbons), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(double_bonds), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    for ax in axes.ravel()[n_groups:]:
        ax.axis("off")

    fig.suptitle(f"{cls}: carbon x double-bond enrichment surface", fontsize=18, y=0.99)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    cbar.set_label("Mean summed normalized intensity", labelpad=14, fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    if note_text:
        fig.text(
            0.5,
            0.008,
            note_text,
            ha="center",
            va="bottom",
            fontsize=11,
            color="dimgray",
        )
    fig.savefig(_long_path(out_png), dpi=120, bbox_inches="tight", pad_inches=0.15)
    if out_svg:
        fig.savefig(_long_path(out_svg), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _run_class_trend_analyses(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    feature_meta: pd.DataFrame,
    cls_series: pd.Series,
    carb: pd.Series,
    db: pd.Series,
    out_dir: str,
    group_order: Optional[List[str]],
) -> Dict[str, str]:
    trend_dir = _ensure_dir(os.path.join(out_dir, "TrendStats"))
    surface_dir = _ensure_dir(os.path.join(out_dir, "EnrichmentSurfaces"))
    surface_csv_dir = _ensure_dir(os.path.join(surface_dir, "PerClass_CSV"))
    surface_png_dir = _ensure_dir(os.path.join(surface_dir, "PerClass_PNG"))
    surface_svg_dir = _ensure_dir(os.path.join(surface_dir, "PerClass_SVG"))

    ordered_groups = _order_groups(pd.unique(y.astype(str)).tolist(), group_order)
    classes = sorted([c for c in pd.unique(cls_series.astype(str)) if str(c).strip() and str(c).strip().lower() != "nan"])

    trend_rows = []
    surface_rows = []

    for cls in classes:
        class_mask = cls_series.astype(str).eq(cls)
        feats = [f for f in feature_meta.index[class_mask] if f in X.columns]
        if not feats:
            continue

        class_carb = pd.to_numeric(carb.reindex(feats), errors="coerce")
        class_db = pd.to_numeric(db.reindex(feats), errors="coerce")
        class_ratio = class_db / class_carb.replace(0, np.nan)

        class_surface_rows = []
        for group in ordered_groups:
            sample_mask = y.astype(str).eq(str(group))
            if int(sample_mask.sum()) == 0:
                continue

            mean_abundance = X.loc[sample_mask, feats].apply(pd.to_numeric, errors="coerce").mean(axis=0)
            mean_abundance = pd.to_numeric(mean_abundance, errors="coerce").fillna(0.0)

            mean_carb = _weighted_mean(class_carb, mean_abundance)
            mean_db = _weighted_mean(class_db, mean_abundance)
            mean_ratio = _weighted_mean(class_ratio, mean_abundance)

            carb_bin_df = pd.DataFrame({"x": class_carb, "abundance": mean_abundance}).dropna()
            if not carb_bin_df.empty:
                carb_binned = carb_bin_df.groupby("x", as_index=False).agg(
                    abundance=("abundance", "sum"),
                    feature_count=("abundance", "size"),
                )
                carb_reg = _weighted_linear_regression(
                    carb_binned["x"],
                    carb_binned["abundance"],
                    carb_binned["feature_count"],
                )
            else:
                carb_reg = {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": 0}

            db_bin_df = pd.DataFrame({"x": class_db, "abundance": mean_abundance}).dropna()
            if not db_bin_df.empty:
                db_binned = db_bin_df.groupby("x", as_index=False).agg(
                    abundance=("abundance", "sum"),
                    feature_count=("abundance", "size"),
                )
                db_reg = _weighted_linear_regression(
                    db_binned["x"],
                    db_binned["abundance"],
                    db_binned["feature_count"],
                )
            else:
                db_reg = {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": 0}

            sat_df = pd.DataFrame({"x": class_ratio, "abundance": mean_abundance}).dropna()
            sat_reg = _weighted_linear_regression(
                sat_df["x"],
                sat_df["abundance"],
                sat_df["abundance"].clip(lower=0.0),
            ) if not sat_df.empty else {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": 0}

            trend_rows.append({
                "Class": cls,
                "Group": group,
                "Samples in group": int(sample_mask.sum()),
                "Features in class": int(len(feats)),
                "Weighted mean carbons": mean_carb,
                "Weighted mean double bonds": mean_db,
                "Weighted mean DB_per_carbon": mean_ratio,
                "Abundance_vs_carbons_slope": carb_reg["slope"],
                "Abundance_vs_carbons_intercept": carb_reg["intercept"],
                "Abundance_vs_carbons_r2": carb_reg["r2"],
                "Abundance_vs_carbons_n_bins": carb_reg["n"],
                "Abundance_vs_double_bonds_slope": db_reg["slope"],
                "Abundance_vs_double_bonds_intercept": db_reg["intercept"],
                "Abundance_vs_double_bonds_r2": db_reg["r2"],
                "Abundance_vs_double_bonds_n_bins": db_reg["n"],
                "Abundance_vs_DB_per_carbon_slope": sat_reg["slope"],
                "Abundance_vs_DB_per_carbon_intercept": sat_reg["intercept"],
                "Abundance_vs_DB_per_carbon_r2": sat_reg["r2"],
                "Abundance_vs_DB_per_carbon_n_points": sat_reg["n"],
            })

            surf_df = pd.DataFrame({
                "Class": cls,
                "Group": group,
                "Feature": feats,
                "Carbons": class_carb.reindex(feats).to_numpy(),
                "DoubleBonds": class_db.reindex(feats).to_numpy(),
                "MeanAbundance": mean_abundance.reindex(feats).to_numpy(),
            }).dropna(subset=["Carbons", "DoubleBonds"])
            if not surf_df.empty:
                surf_df["Carbons"] = surf_df["Carbons"].astype(int)
                surf_df["DoubleBonds"] = surf_df["DoubleBonds"].astype(int)
                class_surface_rows.append(surf_df)
                surface_rows.append(surf_df)

        if class_surface_rows:
            cls_surface_df = pd.concat(class_surface_rows, ignore_index=True)
            safe_cls = _sanitize_filename(cls)
            cls_surface_csv = os.path.join(surface_csv_dir, f"{safe_cls}__carbon_x_db_surface.csv")
            cls_surface_df.to_csv(_long_path(cls_surface_csv), index=False)
            cls_surface_png = os.path.join(surface_png_dir, f"{safe_cls}__carbon_x_db_surface.png")
            cls_surface_svg = os.path.join(surface_svg_dir, f"{safe_cls}__carbon_x_db_surface.svg")
            _plot_2d_enrichment_surface_per_class(
                cls,
                cls_surface_df,
                ordered_groups,
                cls_surface_png,
                cls_surface_svg,
                note_text=(
                    "Brighter cells indicate stronger abundance for that carbon/double-bond bin in that group.\n"
                    "Compare hotspot locations across groups to see whether the class shifts toward longer, shorter, more saturated, or more unsaturated species."
                ),
            )

    trend_df = pd.DataFrame(trend_rows)
    if trend_df.empty:
        raise ValueError("No class trend statistics could be computed.")

    trend_df["Class_mean_carbons_all_groups"] = trend_df.groupby("Class")["Weighted mean carbons"].transform("mean")
    trend_df["Class_mean_double_bonds_all_groups"] = trend_df.groupby("Class")["Weighted mean double bonds"].transform("mean")
    trend_df["Class_mean_DB_per_carbon_all_groups"] = trend_df.groupby("Class")["Weighted mean DB_per_carbon"].transform("mean")
    trend_df["Mean chain length shift"] = trend_df["Weighted mean carbons"] - trend_df["Class_mean_carbons_all_groups"]
    trend_df["Mean unsaturation shift"] = trend_df["Weighted mean double bonds"] - trend_df["Class_mean_double_bonds_all_groups"]
    trend_df["Saturation trend shift"] = trend_df["Weighted mean DB_per_carbon"] - trend_df["Class_mean_DB_per_carbon_all_groups"]

    trend_long_csv = os.path.join(trend_dir, "class_group_trend_statistics.csv")
    trend_df.to_csv(_long_path(trend_long_csv), index=False)

    chain_shift = trend_df.pivot(index="Class", columns="Group", values="Mean chain length shift").sort_index()
    unsat_shift = trend_df.pivot(index="Class", columns="Group", values="Mean unsaturation shift").sort_index()
    saturation_table = trend_df.pivot(index="Class", columns="Group", values="Weighted mean DB_per_carbon").sort_index()
    carb_slope = trend_df.pivot(index="Class", columns="Group", values="Abundance_vs_carbons_slope").sort_index()
    db_slope = trend_df.pivot(index="Class", columns="Group", values="Abundance_vs_double_bonds_slope").sort_index()
    sat_slope = trend_df.pivot(index="Class", columns="Group", values="Abundance_vs_DB_per_carbon_slope").sort_index()

    chain_shift_csv = os.path.join(trend_dir, "mean_chain_length_shift_by_class_group.csv")
    unsat_shift_csv = os.path.join(trend_dir, "mean_unsaturation_shift_by_class_group.csv")
    saturation_csv = os.path.join(trend_dir, "saturation_index_by_class_group.csv")
    carb_slope_csv = os.path.join(trend_dir, "abundance_vs_carbons_slope_by_class_group.csv")
    db_slope_csv = os.path.join(trend_dir, "abundance_vs_double_bonds_slope_by_class_group.csv")
    sat_slope_csv = os.path.join(trend_dir, "abundance_vs_saturation_slope_by_class_group.csv")

    chain_shift.to_csv(_long_path(chain_shift_csv), index=True)
    unsat_shift.to_csv(_long_path(unsat_shift_csv), index=True)
    saturation_table.to_csv(_long_path(saturation_csv), index=True)
    carb_slope.to_csv(_long_path(carb_slope_csv), index=True)
    db_slope.to_csv(_long_path(db_slope_csv), index=True)
    sat_slope.to_csv(_long_path(sat_slope_csv), index=True)

    _plot_metric_heatmap(
        chain_shift,
        os.path.join(trend_dir, "mean_chain_length_shift_heatmap.png"),
        os.path.join(trend_dir, "mean_chain_length_shift_heatmap.svg"),
        "Mean shift in chain length per class",
        "Shift in weighted mean carbons",
        cmap="coolwarm",
        center_zero=True,
        note_text=(
            "Positive shift (red) -> this group is enriched in longer-chain lipids for that class\n"
            "Negative shift (blue) -> this group is enriched in shorter-chain lipids for that class\n"
            "Near zero (white/gray) -> little or no chain-length shift relative to the class average"
        ),
    )
    _plot_metric_heatmap(
        unsat_shift,
        os.path.join(trend_dir, "mean_unsaturation_shift_heatmap.png"),
        os.path.join(trend_dir, "mean_unsaturation_shift_heatmap.svg"),
        "Mean shift in unsaturation per class",
        "Shift in weighted mean double bonds",
        cmap="coolwarm",
        center_zero=True,
        note_text=(
            "Positive shift (red) -> this group is enriched in more unsaturated lipids for that class\n"
            "Negative shift (blue) -> this group is enriched in less unsaturated lipids for that class\n"
            "Near zero (white/gray) -> little or no unsaturation shift relative to the class average"
        ),
    )
    _plot_metric_heatmap(
        saturation_table,
        os.path.join(trend_dir, "saturation_index_heatmap.png"),
        os.path.join(trend_dir, "saturation_index_heatmap.svg"),
        "Saturation trend per class",
        "Weighted mean DB/carbon",
        cmap="magma",
        center_zero=False,
        note_text=(
            "Higher values (brighter colors) -> more double bonds per carbon on average within that class\n"
            "Lower values (darker colors) -> fewer double bonds per carbon on average within that class\n"
            "This is an intensity-weighted saturation index, not a regression slope"
        ),
        cbar_labelpad=18,
    )
    _plot_metric_heatmap(
        carb_slope,
        os.path.join(trend_dir, "abundance_vs_carbons_slope_heatmap.png"),
        os.path.join(trend_dir, "abundance_vs_carbons_slope_heatmap.svg"),
        "Weighted regression slope: abundance vs carbons",
        "Slope",
        cmap="coolwarm",
        center_zero=True,
        note_text=(
            "Positive slope (red) -> higher-carbon lipids tend to be more abundant\n"
            "Negative slope (blue) -> higher-carbon lipids tend to be less abundant\n"
            "Near zero (white/gray) -> no systematic abundance relationship with chain length"
        ),
    )
    _plot_metric_heatmap(
        db_slope,
        os.path.join(trend_dir, "abundance_vs_double_bonds_slope_heatmap.png"),
        os.path.join(trend_dir, "abundance_vs_double_bonds_slope_heatmap.svg"),
        "Weighted regression slope: abundance vs double bonds",
        "Slope",
        cmap="coolwarm",
        center_zero=True,
        note_text=(
            "Positive slope (red) -> more unsaturated lipids tend to be more abundant\n"
            "Negative slope (blue) -> more unsaturated lipids tend to be less abundant\n"
            "Near zero (white/gray) -> no systematic abundance relationship with unsaturation"
        ),
    )
    _plot_metric_heatmap(
        sat_slope,
        os.path.join(trend_dir, "abundance_vs_saturation_slope_heatmap.png"),
        os.path.join(trend_dir, "abundance_vs_saturation_slope_heatmap.svg"),
        "Weighted regression slope: abundance vs saturation",
        "Slope",
        cmap="coolwarm",
        center_zero=True,
        note_text=(
            "Positive slope (red) -> lipids with higher saturation metric tend to be more abundant\n"
            "Negative slope (blue) -> lipids with higher saturation metric tend to be less abundant\n"
            "Near zero (white/gray) -> no systematic relationship"
        ),
    )

    surface_long_csv = os.path.join(surface_dir, "all_classes_carbon_x_db_surface_long.csv")
    if surface_rows:
        pd.concat(surface_rows, ignore_index=True).to_csv(_long_path(surface_long_csv), index=False)
    else:
        pd.DataFrame(columns=["Class", "Group", "Feature", "Carbons", "DoubleBonds", "MeanAbundance"]).to_csv(
            _long_path(surface_long_csv), index=False
        )

    return {
        "trend_dir": trend_dir,
        "trend_long_csv": trend_long_csv,
        "chain_shift_csv": chain_shift_csv,
        "unsaturation_shift_csv": unsat_shift_csv,
        "saturation_index_csv": saturation_csv,
        "carbons_slope_csv": carb_slope_csv,
        "doublebonds_slope_csv": db_slope_csv,
        "saturation_slope_csv": sat_slope_csv,
        "surface_dir": surface_dir,
        "surface_long_csv": surface_long_csv,
        "surface_csv_dir": surface_csv_dir,
        "surface_png_dir": surface_png_dir,
        "surface_svg_dir": surface_svg_dir,
    }


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
            note_text=(
                "Each stacked segment shows the contribution of one total-carbon bin within this class.\n"
                "Taller segments mean that chain length contributes more strongly to the class signal in that group."
            ),
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
            value_name="Double bond equivalents",
            csv_dir=db_csv_dir,
            png_dir=db_png_dir,
            group_order=group_order,
            title_suffix="double-bond summed composition",
            y_label="Mean summed normalized intensity",
            note_text=(
                "Each stacked segment shows the contribution of one double-bond bin within this class.\n"
                "Shifts in the stack pattern suggest changes in unsaturation distribution between groups."
            ),
        )
    except Exception:
        _write_debug_snapshot(os.path.join(out_dir, "DoubleBonds"), X, y, feature_meta)
        raise

    trend_outputs = _run_class_trend_analyses(
        X=X,
        y=y,
        feature_meta=feature_meta,
        cls_series=cls_series,
        carb=carb,
        db=db,
        out_dir=out_dir,
        group_order=group_order,
    )

    outputs = {
        "out_dir": out_dir,
        "carbons_csv_dir": carb_csv_dir,
        "carbons_png_dir": carb_png_dir,
        "doublebonds_csv_dir": db_csv_dir,
        "doublebonds_png_dir": db_png_dir,
    }
    outputs.update(trend_outputs)
    return outputs
