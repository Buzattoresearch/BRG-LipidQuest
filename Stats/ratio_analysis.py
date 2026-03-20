from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
from scipy.stats import kruskal

from Stats.figure_style import build_group_palette as _shared_build_group_palette, get_figure_style
from Stats.utils import prepare_output_dir, _CLASS_GROUP_MAP

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
plt.ioff()
sns.set_style("white")


EPSILON = 1e-9

DEFAULT_CLASS_RATIO_DEFS = [
    ("FA", "TG", "FA/TG"),
    ("MG", "TG", "MG/TG"),
    ("DG", "TG", "DG/TG"),
    ("DG", "PC", "DG/PC"),
    ("DG", "PE", "DG/PE"),
    ("DG", "PA", "DG/PA"),
    ("SM", "Cer", "SM/Cer"),
    ("CE", "ST", "CE/ST"),
]

DEFAULT_PRODUCT_RATIO_DEFS = [
    ("PC", "PE", "PC/PE"),
    ("PG", "PE", "PG/PE"),
    ("PA", "PE", "PA/PE"),
    ("PA", "PC", "PA/PC"),
    ("PA", "PG", "PA/PG"),
    ("PS", "PE", "PS/PE"),
    ("CL", "PG", "CL/PG"),
    ("MLCL", "CL", "MLCL/CL"),
    ("DLCL", "CL", "DLCL/CL"),
    ("LPE", "PE", "LPE/PE"),
    ("LPC", "PC", "LPC/PC"),
    ("LPG", "PG", "LPG/PG"),
    ("PEth", "PE", "PEth/PE"),
]


def _load_dataset_preserve_nan(file_path: str, group_file: Optional[str]):
    df = pd.read_csv(file_path, low_memory=False)
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=str), pd.DataFrame()

    meta_cols = [
        "UniqueID", "RT (min)", "m/z", "Polarity", "Annotation",
        "Annotation Type", "Headgroup", "Lipid Class", "Δm/z (mDa)", "Δm/z (ppm)",
        "MS/MS score", "Annotation tier", "mSigma", "Molecular Formula",
        "Plasmenyl?", "Number of carbons in fatty acyls", "Double bond equivalents",
        "Chain type", "PUFA?", "Modifications", "# of modifications", "Oxidized?",
        "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    sample_cols = [c for c in df.columns if c not in meta_cols]

    if group_file is not None and os.path.exists(group_file):
        df_groups = pd.read_csv(group_file, low_memory=False)
        if "Sample" not in df_groups.columns or "Group" not in df_groups.columns:
            raise ValueError(f"[ratios] Invalid group file format: {group_file}")
    else:
        df_groups = pd.DataFrame({"Sample": sample_cols, "Group": "Unknown"})

    df_groups["Sample"] = df_groups["Sample"].astype(str).str.strip()
    df_groups["Group"] = df_groups["Group"].astype(str).str.strip()
    df_cols_lower = {str(c).lower(): c for c in sample_cols}
    matched = [df_cols_lower[s.lower()] for s in df_groups["Sample"] if s.lower() in df_cols_lower]

    if not matched:
        return pd.DataFrame(), pd.Series(dtype=str), pd.DataFrame()

    X = df[matched].T
    X.index.name = "Sample"
    y = df_groups.set_index("Sample").loc[X.index, "Group"]
    feature_meta = df[meta_cols].copy()

    if "UniqueID" in feature_meta.columns:
        X.columns = feature_meta["UniqueID"].astype(str).tolist()
    else:
        X.columns = [f"Feature_{i+1}" for i in range(X.shape[1])]

    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y, feature_meta


def _canon_class(x: str, unknown_policy: str = "append") -> str:
    x = str(x or "").strip()
    if x in _CLASS_GROUP_MAP:
        return _CLASS_GROUP_MAP[x]
    if not x:
        return "Other"
    return "Other" if unknown_policy == "other" else x


def _order_groups(present: List[str], group_order: Optional[List[str]]) -> List[str]:
    present = [str(g) for g in present]
    if not group_order:
        return present
    gui = [g for g in group_order if g in present]
    rest = [g for g in present if g not in gui]
    return gui + rest


def _pick_column_ci(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = [str(c) for c in df.columns]
    norm = {str(c).strip().lower(): str(c) for c in cols}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm:
            return norm[key]
    return None


def _parse_total_carbons_from_text(s: str) -> float:
    if s is None:
        return np.nan
    m = re.search(r"(\d+)\s*:\s*(\d+)", str(s))
    return float(int(m.group(1))) if m else np.nan


def _parse_total_double_bonds_from_text(s: str) -> float:
    if s is None:
        return np.nan
    m = re.search(r"(\d+)\s*:\s*(\d+)", str(s))
    return float(int(m.group(2))) if m else np.nan


def _truthy_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"yes", "y", "true", "1", "plasmenyl", "p"})


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return out
    order = np.argsort(valid.to_numpy(dtype=float))
    ranked = valid.iloc[order]
    n = len(ranked)
    adj = ranked.to_numpy(dtype=float) * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out.loc[ranked.index] = adj
    return out


def _compute_signal(X: pd.DataFrame, feature_mask: pd.Series) -> pd.Series:
    feats = feature_mask.index[feature_mask.fillna(False)].tolist()
    if not feats:
        return pd.Series(0.0, index=X.index)
    return X.loc[:, feats].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)


def _make_ratio(
    X: pd.DataFrame,
    numerator_mask: pd.Series,
    denominator_mask: pd.Series,
    ratio_name: str,
    category: str,
    metadata: Optional[Dict[str, object]] = None,
) -> Optional[pd.DataFrame]:
    num = _compute_signal(X, numerator_mask)
    den = _compute_signal(X, denominator_mask)
    if float(num.fillna(0.0).sum()) <= 0 and float(den.fillna(0.0).sum()) <= 0:
        return None

    raw_ratio = (num + EPSILON) / (den + EPSILON)
    log2_ratio = np.log2(raw_ratio)

    out = pd.DataFrame({
        "Sample": X.index.astype(str),
        "Ratio": ratio_name,
        "Category": category,
        "NumeratorSignal": num.to_numpy(dtype=float),
        "DenominatorSignal": den.to_numpy(dtype=float),
        "RawRatio": raw_ratio.to_numpy(dtype=float),
        "Log2Ratio": log2_ratio.to_numpy(dtype=float),
    })
    if metadata:
        for key, value in metadata.items():
            out[key] = value
    return out


def _compute_ratio_stats(sample_ratios: pd.DataFrame, ordered_groups: List[str]) -> pd.DataFrame:
    rows = []
    for ratio_name, sub in sample_ratios.groupby("Ratio", sort=False):
        vectors = []
        for group in ordered_groups:
            vals = pd.to_numeric(sub.loc[sub["Group"].astype(str) == str(group), "Log2Ratio"], errors="coerce").dropna()
            if len(vals) > 0:
                vectors.append(vals.to_numpy(dtype=float))
        if len(vectors) >= 2:
            try:
                _, p_value = kruskal(*vectors)
            except Exception:
                p_value = np.nan
        else:
            p_value = np.nan

        meta = sub.iloc[0][["Category"]].to_dict()
        rows.append({
            "Ratio": ratio_name,
            "Category": meta["Category"],
            "Kruskal_p_value": p_value,
            "Groups_tested": int(len(vectors)),
        })
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats["FDR_BH"] = _bh_fdr(stats["Kruskal_p_value"])
    return stats


def _ordered_ratio_index(index_like, preferred_order: Optional[List[str]]) -> List[str]:
    present = list(dict.fromkeys(str(x) for x in index_like))
    if not preferred_order:
        return present
    ordered = [ratio_name for ratio_name in preferred_order if ratio_name in present]
    remaining = [ratio_name for ratio_name in present if ratio_name not in ordered]
    return ordered + remaining


def _ratio_sort_rank(category: str, ratio_name: str, custom_ratio_orders: Dict[str, List[str]]) -> int:
    preferred_order = custom_ratio_orders.get(str(category), [])
    if ratio_name in preferred_order:
        return preferred_order.index(ratio_name)
    return len(preferred_order)


def _normalize_ratio_def(item, default_category: str) -> Optional[Dict[str, str]]:
    if isinstance(item, dict):
        numerator = str(item.get("numerator", "")).strip()
        denominator = str(item.get("denominator", "")).strip()
        ratio_name = str(item.get("ratio_name", "")).strip()
        category = str(item.get("category", "")).strip() or default_category
    elif isinstance(item, (list, tuple)) and len(item) >= 3:
        numerator = str(item[0]).strip()
        denominator = str(item[1]).strip()
        ratio_name = str(item[2]).strip()
        category = str(item[3]).strip() if len(item) >= 4 and str(item[3]).strip() else default_category
    else:
        return None
    if not numerator or not denominator or not ratio_name:
        return None
    return {
        "numerator": numerator,
        "denominator": denominator,
        "ratio_name": ratio_name,
        "category": category,
    }


def _normalize_ratio_defs(items, default_category: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for item in items or []:
        ratio_def = _normalize_ratio_def(item, default_category)
        if ratio_def is None:
            continue
        key = (
            ratio_def["numerator"].casefold(),
            ratio_def["denominator"].casefold(),
            ratio_def["ratio_name"].casefold(),
            ratio_def["category"].casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ratio_def)
    return out


def _normalize_ratio_settings(ratio_settings: Optional[dict]) -> Dict[str, object]:
    settings = dict(ratio_settings or {})
    selected_class_ratios = _normalize_ratio_defs(
        settings.get("selected_class_ratios", DEFAULT_CLASS_RATIO_DEFS),
        "Class ratios",
    )
    selected_product_ratios = _normalize_ratio_defs(
        settings.get("selected_product_ratios", DEFAULT_PRODUCT_RATIO_DEFS),
        "Product/substrate-like ratios",
    )
    annotation_ratios = _normalize_ratio_defs(
        settings.get("annotation_ratios", []),
        "Annotation-specific ratios",
    )
    return {
        "include_selected_class_ratios": bool(settings.get("include_selected_class_ratios", True)),
        "include_selected_product_ratios": bool(settings.get("include_selected_product_ratios", True)),
        "include_structural_class_ratios": bool(settings.get("include_structural_class_ratios", True)),
        "include_global_structural_ratios": bool(settings.get("include_global_structural_ratios", True)),
        "selected_class_ratios": selected_class_ratios,
        "selected_product_ratios": selected_product_ratios,
        "annotation_ratios": annotation_ratios,
    }


def get_available_annotation_labels(file_path: str) -> List[str]:
    try:
        df = pd.read_csv(file_path, low_memory=False, usecols=["Annotation"])
    except Exception:
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception:
            return []
    if "Annotation" not in df.columns:
        return []
    annotations = (
        df["Annotation"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    annotations = annotations[annotations.ne("") & ~annotations.str.lower().eq("nan")]
    return sorted(annotations.unique().tolist(), key=str.casefold)


def _plot_ratio_heatmap(
    table: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_png: str,
    out_svg: str,
    title: str,
    note_text: str,
    style: Optional[dict] = None,
    row_labels: Optional[List[str]] = None,
) -> None:
    if table.empty:
        return
    style = style or get_figure_style(False, 100)
    rc = {
        "font.family": style["font_family"],
        "font.size": style.get("base_font_size", style["label_size"]),
        "axes.titlesize": style["title_size"],
        "axes.labelsize": style["label_size"],
        "xtick.labelsize": style["tick_size"],
        "ytick.labelsize": style["tick_size"],
        "legend.fontsize": style["legend_size"],
        "figure.titlesize": style["title_size"],
    }

    with mpl.rc_context(rc=rc):
        fig, ax = plt.subplots(
            figsize=(max(5.5, 0.5 * len(table.columns) + 2.6), max(10, 0.5 * len(table.index) + 3.1)), # (width, height) # this is the size of the whole figure canvas (not the plot)
            facecolor="white",
        )
        vals = table.to_numpy(dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size == 0:
            plt.close(fig)
            return

        data_min = float(np.nanmin(finite_vals))
        data_max = float(np.nanmax(finite_vals))
        if data_max <= 0:
            cmap = plt.get_cmap("Blues_r")
            vmin, vmax = data_min, data_max
        elif data_min >= 0:
            cmap = plt.get_cmap("Reds")
            vmin, vmax = data_min, data_max
        else:
            vmax = float(np.nanmax(np.abs(finite_vals)))
            cmap = style["diverging_cmap"]
            vmin, vmax = -vmax, vmax

        im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_xticklabels(table.columns.tolist(), rotation=45, ha="right", fontsize=style["tick_size"])
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_yticklabels(row_labels or table.index.tolist(), fontsize=style["tick_size"])
        ax.set_xticks(np.arange(-0.5, len(table.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(table.index), 1), minor=True)
        ax.grid(which="minor", color=(1, 1, 1, 0.45), linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(title, fontsize=style["title_size"], pad=12, fontweight="semibold")
        ax.set_xlabel("Group", fontsize=style["label_size"], labelpad=14)
        ax.tick_params(axis="both", labelsize=style["tick_size"])

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.06)
        cbar.set_label("Mean log2 ratio", labelpad=14, fontsize=style["label_size"])
        cbar.ax.tick_params(labelsize=style["tick_size"])

        sig_map = {}
        if not stats_df.empty:
            sig_map = stats_df.set_index("Ratio")["FDR_BH"].to_dict()
        for row_idx, ratio_name in enumerate(table.index.tolist()):
            fdr_val = sig_map.get(ratio_name)
            if pd.notna(fdr_val) and float(fdr_val) < 0.05:
                ax.text(
                    len(table.columns) - 0.5,
                    row_idx,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=style["label_size"],
                    color="black",
                    fontweight="bold",
                )

        fig.subplots_adjust(bottom=0.46)
        fig.text(0.45, -0.02, note_text, ha="center", va="bottom", fontsize=max(style["tick_size"] - 3, 8), color="dimgray")
        fig.savefig(out_png, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.15)
        fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


def _build_group_palette(groups: List[str], group_colors: Optional[dict] = None) -> Dict[str, str]:
    _, palette = _shared_build_group_palette(groups, group_colors=group_colors, group_order=groups)
    return palette


def _plot_ratio_boxplot(
    df: pd.DataFrame,
    ratio_name: str,
    ordered_groups: List[str],
    out_png: str,
    out_svg: str,
    title: str,
    style: Optional[dict] = None,
    group_colors: Optional[dict] = None,
) -> None:
    if df.empty:
        return
    style = style or get_figure_style(False, 100)
    plot_df = df.copy()
    plot_df["Group"] = plot_df["Group"].astype(str)
    plot_df["RawRatio"] = pd.to_numeric(plot_df["RawRatio"], errors="coerce")
    plot_df = plot_df.dropna(subset=["RawRatio"])
    if plot_df.empty:
        return

    order = [g for g in ordered_groups if g in plot_df["Group"].unique().tolist()]
    if not order:
        order = pd.unique(plot_df["Group"]).tolist()
    pal = _build_group_palette(order, group_colors=group_colors)

    with mpl.rc_context({
        "font.family": style["font_family"],
        "font.size": style.get("base_font_size", style["label_size"]),
        "axes.titlesize": style["title_size"],
        "axes.labelsize": style["label_size"],
        "xtick.labelsize": style["tick_size"],
        "ytick.labelsize": style["tick_size"],
    }):
        fig, ax = plt.subplots(
            figsize=(max(6.5, 1.15 * len(order) + 2.4), 6.2),
            facecolor="white",
        )
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        ax.grid(False)

        sns.boxplot(
            data=plot_df,
            x="Group",
            y="RawRatio",
            order=order,
            palette=[pal[g] for g in order],
            showfliers=False,
            linewidth=0.0,
            whiskerprops=dict(color="gray", linewidth=0.6),
            capprops=dict(color="gray", linewidth=0.6),
            medianprops=dict(color="black", linewidth=0.9),
            ax=ax,
        )

        boxes = []
        if getattr(ax, "artists", None):
            boxes = [art for art in ax.artists if isinstance(art, mpatches.PathPatch)]
        if not boxes:
            boxes = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        for patch in boxes:
            fc = patch.get_facecolor()
            try:
                rgba = tuple(fc[0]) if hasattr(fc, "__len__") and len(fc) and hasattr(fc[0], "__len__") else tuple(fc)
            except Exception:
                rgba = mpl.colors.to_rgba(fc)
            patch.set_facecolor((rgba[0], rgba[1], rgba[2], 0.30))
            patch.set_edgecolor((0, 0, 0, 0))
            patch.set_zorder(1)

        sns.stripplot(
            data=plot_df,
            x="Group",
            y="RawRatio",
            order=order,
            palette=[pal[g] for g in order],
            alpha=0.65,
            size=5.0,
            jitter=0.18,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
            zorder=2,
        )

        ax.set_title(title, pad=14, fontweight="semibold")
        # ax.set_xlabel("Group", labelpad=12)
        ax.set_ylabel("Ratio of normalized peak intensities\n× IS concentration", labelpad=12)
        ax.set_xticklabels(order, rotation=45, ha="right")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")

        fig.tight_layout(pad=1.2)
        fig.savefig(out_png, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.1)
        fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def run_from_stats(
    file_path: str,
    group_file: Optional[str],
    save_dir: str,
    group_order: Optional[List[str]] = None,
    group_colors: Optional[dict] = None,
    unknown_policy: str = "append",
    exclude_qc: bool = True,
    dpi: int = 100,
    publication_theme: bool = False,
    ratio_settings: Optional[dict] = None,
) -> Dict[str, str]:
    out_dir = prepare_output_dir(save_dir)
    style = get_figure_style(publication_theme=publication_theme, dpi=dpi)
    settings = _normalize_ratio_settings(ratio_settings)
    print("[Ratios] Running ratio analysis...", flush=True)

    X, y, feature_meta = _load_dataset_preserve_nan(file_path, group_file)
    if X is None or y is None or feature_meta is None or X.empty or feature_meta.empty:
        raise ValueError("Dataset appears empty or malformed.")

    feature_meta = feature_meta.copy()
    feature_meta.columns = feature_meta.columns.astype(str).str.strip()

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
        keep_mask = ~y.astype(str).str.contains("QC", case=False, na=False)
        X = X.loc[keep_mask]
        y = y.loc[keep_mask]

    ordered_groups = _order_groups(pd.unique(y.astype(str)).tolist(), group_order)

    class_col = _pick_column_ci(feature_meta, ["Lipid Class", "Headgroup"])
    if class_col is None:
        raise ValueError("Could not find a class column (expected 'Lipid Class' or 'Headgroup').")
    class_series = feature_meta[class_col].astype(str).map(lambda v: _canon_class(v, unknown_policy=unknown_policy))
    annotation_col = _pick_column_ci(feature_meta, ["Annotation", "NoAbbrev", "Name", "Lipid"])
    annotation_series = (
        feature_meta[annotation_col].astype(str).str.strip()
        if annotation_col
        else pd.Series("", index=feature_meta.index, dtype=str)
    )

    carb_col = _pick_column_ci(feature_meta, ["Number of carbons in fatty acyls", "Total carbons", "Carbons"])
    carb = pd.to_numeric(feature_meta[carb_col], errors="coerce") if carb_col else pd.Series(np.nan, index=feature_meta.index)
    if float(carb.notna().mean()) < 0.05:
        text_col = _pick_column_ci(feature_meta, ["Annotation", "NoAbbrev", "Name", "Lipid"])
        if text_col:
            carb = carb.where(carb.notna(), feature_meta[text_col].apply(_parse_total_carbons_from_text))

    db_col = _pick_column_ci(feature_meta, ["Double bond equivalents", "Double bonds", "DB", "DBE"])
    db = pd.to_numeric(feature_meta[db_col], errors="coerce") if db_col else pd.Series(np.nan, index=feature_meta.index)
    if float(db.notna().mean()) < 0.05:
        text_col = _pick_column_ci(feature_meta, ["Annotation", "NoAbbrev", "Name", "Lipid"])
        if text_col:
            db = db.where(db.notna(), feature_meta[text_col].apply(_parse_total_double_bonds_from_text))

    plas_col = _pick_column_ci(feature_meta, ["Plasmenyl?"])
    plas = _truthy_series(feature_meta[plas_col]) if plas_col else pd.Series(False, index=feature_meta.index)

    class_masks = {cls: class_series.astype(str).eq(cls) for cls in sorted(class_series.dropna().astype(str).unique().tolist())}

    ratio_frames: List[pd.DataFrame] = []

    class_ratio_defs = settings["selected_class_ratios"]
    product_ratio_defs = settings["selected_product_ratios"]
    annotation_ratio_defs = settings["annotation_ratios"]
    class_ratio_order = [item["ratio_name"] for item in class_ratio_defs]
    product_ratio_order = [item["ratio_name"] for item in product_ratio_defs]
    annotation_ratio_order = [item["ratio_name"] for item in annotation_ratio_defs]

    if settings["include_selected_class_ratios"]:
        for ratio_def in class_ratio_defs:
            num_cls = ratio_def["numerator"]
            den_cls = ratio_def["denominator"]
            if num_cls in class_masks and den_cls in class_masks:
                ratio_df = _make_ratio(X, class_masks[num_cls], class_masks[den_cls], ratio_def["ratio_name"], ratio_def["category"])
                if ratio_df is not None:
                    ratio_frames.append(ratio_df)

    if settings["include_selected_product_ratios"]:
        for ratio_def in product_ratio_defs:
            num_cls = ratio_def["numerator"]
            den_cls = ratio_def["denominator"]
            if num_cls in class_masks and den_cls in class_masks:
                ratio_df = _make_ratio(X, class_masks[num_cls], class_masks[den_cls], ratio_def["ratio_name"], ratio_def["category"])
                if ratio_df is not None:
                    ratio_frames.append(ratio_df)

    if annotation_ratio_defs:
        for ratio_def in annotation_ratio_defs:
            num_mask = annotation_series.astype(str).eq(ratio_def["numerator"])
            den_mask = annotation_series.astype(str).eq(ratio_def["denominator"])
            if int(num_mask.sum()) > 0 and int(den_mask.sum()) > 0:
                ratio_df = _make_ratio(
                    X,
                    num_mask,
                    den_mask,
                    ratio_def["ratio_name"],
                    ratio_def["category"],
                    metadata={
                        "NumeratorAnnotation": ratio_def["numerator"],
                        "DenominatorAnnotation": ratio_def["denominator"],
                    },
                )
                if ratio_df is not None:
                    ratio_frames.append(ratio_df)

    if settings["include_structural_class_ratios"]:
        for cls, mask in class_masks.items():
            class_db = db.reindex(mask.index)
            sat_mask = mask & class_db.eq(0)
            unsat_mask = mask & class_db.gt(0)
            if int(sat_mask.sum()) > 0 and int(unsat_mask.sum()) > 0:
                ratio_df = _make_ratio(X, unsat_mask, sat_mask, f"{cls}", "Saturation/desaturation ratios (unsat/sat)")
                if ratio_df is not None:
                    ratio_frames.append(ratio_df)

            class_carb = carb.reindex(mask.index)
            if class_carb.notna().sum() >= 2:
                threshold = float(class_carb[mask].median())
                long_mask = mask & class_carb.gt(threshold)
                short_mask = mask & class_carb.le(threshold)
                if int(long_mask.sum()) > 0 and int(short_mask.sum()) > 0:
                    ratio_df = _make_ratio(
                        X,
                        long_mask,
                        short_mask,
                        f"{cls}",
                        "Elongation indices",
                        metadata={"ThresholdCarbons": threshold},
                    )
                    if ratio_df is not None:
                        ratio_frames.append(ratio_df)

            odd_mask = mask & carb.mod(2).eq(1)
            even_mask = mask & carb.mod(2).eq(0)
            if int(odd_mask.sum()) > 0 and int(even_mask.sum()) > 0:
                ratio_df = _make_ratio(X, odd_mask, even_mask, f"{cls}", "Odd/even chain ratios")
                if ratio_df is not None:
                    ratio_frames.append(ratio_df)

            plas_mask = mask & plas
            non_plas_mask = mask & ~plas
            if int(plas_mask.sum()) > 0 and int(non_plas_mask.sum()) > 0:
                ratio_df = _make_ratio(X, plas_mask, non_plas_mask, f"{cls}: plasmalogen/non-plasmalogen", "Plasmalogen ratios")
                if ratio_df is not None:
                    ratio_frames.append(ratio_df)

    if settings["include_global_structural_ratios"]:
        if int((carb.mod(2).eq(1)).sum()) > 0 and int((carb.mod(2).eq(0)).sum()) > 0:
            ratio_df = _make_ratio(X, carb.mod(2).eq(1), carb.mod(2).eq(0), "Global odd/even", "Odd/even chain ratios")
            if ratio_df is not None:
                ratio_frames.append(ratio_df)

        if plas_col and int(plas.sum()) > 0 and int((~plas).sum()) > 0:
            ratio_df = _make_ratio(X, plas, ~plas, "Global plasmalogen/non-plasmalogen", "Plasmalogen ratios")
            if ratio_df is not None:
                ratio_frames.append(ratio_df)

    if not ratio_frames:
        raise ValueError("No eligible ratios could be computed from this dataset.")

    ratio_df = pd.concat(ratio_frames, ignore_index=True)
    ratio_df["Group"] = y.reindex(ratio_df["Sample"]).astype(str).values

    custom_ratio_orders = {
        "Class ratios": class_ratio_order,
        "Product/substrate-like ratios": product_ratio_order,
        "Annotation-specific ratios": annotation_ratio_order,
    }
    ratio_category_order = list(dict.fromkeys(ratio_df["Category"].astype(str).tolist()))
    category_rank_map = {category: i for i, category in enumerate(ratio_category_order)}
    ratio_df["_category_rank"] = ratio_df["Category"].astype(str).map(category_rank_map).fillna(len(category_rank_map)).astype(int)
    ratio_df["_ratio_rank"] = [
        _ratio_sort_rank(category, ratio_name, custom_ratio_orders)
        for category, ratio_name in zip(ratio_df["Category"].astype(str), ratio_df["Ratio"].astype(str))
    ]
    ratio_df = ratio_df.sort_values(
        ["_category_rank", "_ratio_rank", "Ratio", "Group", "Sample"],
        kind="stable",
    ).reset_index(drop=True)

    stats_df = _compute_ratio_stats(ratio_df, ordered_groups)
    group_means = (
        ratio_df.groupby(["Category", "Ratio", "Group"], sort=False)["Log2Ratio"]
        .mean()
        .reset_index()
    )
    stats_df["_category_rank"] = stats_df["Category"].astype(str).map(category_rank_map).fillna(len(category_rank_map)).astype(int)
    stats_df["_ratio_rank"] = [
        _ratio_sort_rank(category, ratio_name, custom_ratio_orders)
        for category, ratio_name in zip(stats_df["Category"].astype(str), stats_df["Ratio"].astype(str))
    ]
    group_means["_category_rank"] = group_means["Category"].astype(str).map(category_rank_map).fillna(len(category_rank_map)).astype(int)
    group_means["_ratio_rank"] = [
        _ratio_sort_rank(category, ratio_name, custom_ratio_orders)
        for category, ratio_name in zip(group_means["Category"].astype(str), group_means["Ratio"].astype(str))
    ]
    stats_df = stats_df.sort_values(["_category_rank", "_ratio_rank", "Ratio"], kind="stable").reset_index(drop=True)
    group_means = group_means.sort_values(["_category_rank", "_ratio_rank", "Ratio", "Group"], kind="stable").reset_index(drop=True)
    ratio_df = ratio_df.drop(columns=["_category_rank", "_ratio_rank"])
    stats_df = stats_df.drop(columns=["_category_rank", "_ratio_rank"])
    group_means = group_means.drop(columns=["_category_rank", "_ratio_rank"])

    sample_csv = os.path.join(out_dir, "sample_level_ratio_values.csv")
    group_csv = os.path.join(out_dir, "group_mean_log2_ratios.csv")
    stats_csv = os.path.join(out_dir, "ratio_statistics.csv")
    ratio_df.to_csv(sample_csv, index=False)
    group_means.to_csv(group_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)

    figure_paths: Dict[str, str] = {}
    for category, sub in group_means.groupby("Category", sort=False):
        table = sub.pivot(index="Ratio", columns="Group", values="Log2Ratio").reindex(columns=ordered_groups)
        table = table.reindex(_ordered_ratio_index(table.index.tolist(), custom_ratio_orders.get(str(category))))
        cat_stats = stats_df[stats_df["Category"] == category].copy()
        row_labels = None
        if category == "Elongation indices":
            threshold_map = (
                ratio_df.loc[
                    ratio_df["Category"].astype(str) == category,
                    ["Ratio", "ThresholdCarbons"],
                ]
                .dropna(subset=["ThresholdCarbons"])
                .drop_duplicates(subset=["Ratio"])
                .set_index("Ratio")["ThresholdCarbons"]
                .to_dict()
            )
            row_labels = [
                f"{ratio_name} (Thre. {float(threshold_map[ratio_name]):.0f}C)"
                if ratio_name in threshold_map
                else str(ratio_name)
                for ratio_name in table.index.tolist()
            ]
        safe_cat = re.sub(r"[^A-Za-z0-9._-]+", "_", category).strip("_")
        png_path = os.path.join(out_dir, f"{safe_cat}_heatmap.png")
        svg_path = os.path.join(out_dir, f"{safe_cat}_heatmap.svg")
        _plot_ratio_heatmap(
            table,
            cat_stats,
            png_path,
            svg_path,
            title=f"{category} by group",
            note_text=(
                "Positive values (red) -> numerator-enriched relative to denominator\n"
                "Negative values (blue) -> denominator-enriched relative to numerator\n"
                "Near zero -> ratio is near 1 on the log2 scale; * indicates FDR < 0.05 across groups"
            ),
            style=style,
            row_labels=row_labels,
        )
        figure_paths[f"{safe_cat}_png"] = png_path
        figure_paths[f"{safe_cat}_svg"] = svg_path

        boxplot_dir = prepare_output_dir(os.path.join(out_dir, f"{safe_cat}_boxplots"))
        for ratio_name, ratio_sub in ratio_df[ratio_df["Category"].astype(str) == str(category)].groupby("Ratio", sort=False):
            safe_ratio = re.sub(r"[^A-Za-z0-9._-]+", "_", str(ratio_name)).strip("_") or "ratio"
            box_png = os.path.join(boxplot_dir, f"{safe_ratio}.png")
            box_svg = os.path.join(boxplot_dir, f"{safe_ratio}.svg")
            _plot_ratio_boxplot(
                ratio_sub[["Group", "RawRatio"]].copy(),
                ratio_name=str(ratio_name),
                ordered_groups=ordered_groups,
                out_png=box_png,
                out_svg=box_svg,
                title=f"{ratio_name}",
                style=style,
                group_colors=group_colors,
            )
        figure_paths[f"{safe_cat}_boxplot_dir"] = str(boxplot_dir)

    print(f"[Ratios] Completed. Results saved to: {out_dir}", flush=True)
    return {
        "out_dir": str(out_dir),
        "sample_level_ratio_csv": sample_csv,
        "group_mean_ratio_csv": group_csv,
        "ratio_statistics_csv": stats_csv,
        **figure_paths,
    }
