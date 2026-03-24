from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu

from Stats.figure_style import build_group_palette as _shared_build_group_palette, get_figure_style
from Stats.utils import prepare_output_dir, load_dataset

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
plt.ioff()
sns.set_style("white")


DEFAULT_FATTY_ACID_CHAIN_PANEL = [
    "14:0", "14:1",
    "15:0", "15:1",
    "16:0", "16:1",
    "17:0", "17:1",
    "18:0", "18:1", "18:2", "18:3",
    "19:0", "19:1",
    "20:0", "20:1", "20:2", "20:3", "20:4", "20:5", "20:6",
    "21:0", "21:1",
    "22:0", "22:1", "22:2", "22:3", "22:4", "22:5", "22:6",
    "24:0", "24:1",
    "26:0", 
]

DEFAULT_FATTY_ACID_SUM_COMPOSITION_PANEL = [
    "28:0", "28:1",
    "29:0", "29:1",
    "30:0", "30:1", "30:2",
    "31:0", "31:1", "31:2",
    "32:0", "32:1", "32:2", "32:3",
    "33:0", "33:1", "33:2", "33:3",
    "34:0", "34:1", "34:2", "34:3", "34:4",
    "35:0", "35:1", "35:2", "35:3", "35:4",
    "36:0", "36:1", "36:2", "36:3", "36:4", "36:5", "36:6",
    "37:0", "37:1", "37:2", "37:3", "37:4", "37:5", "37:6",
    "38:0", "38:1", "38:2", "38:3", "38:4", "38:5", "38:6", "38:7",
    "39:0", "39:1", "39:2", "39:3", "39:4", "39:5", "39:6", "39:7",
    "40:0", "40:1", "40:2", "40:3", "40:4", "40:5", "40:6", "40:7", "40:8",
]

_MOTIF_RE = re.compile(r"(?<!\d)(\d{1,2}:\d{1,2})(?!\d)")
_LYSO_OR_SINGLE_CHAIN_CLASSES = {
    "FA", "FAHFA", "CAR", "CoA", "CE", "ST", "MG", "LPC", "LPE", "LPG", "LPI",
    "LPA", "LPS", "LPE O-", "LPC O-", "LPG O-", "LPA O-", "LPI O-", "LPS O-",
    "NA", "NAE", "NAT", "NATx", "NAx", "FAL", "FAG", "FOH", "HC", "WE",
}


def _pick_column_ci(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm:
            return norm[key]
    return None


def _build_group_palette(groups_like, group_colors=None, group_order=None):
    return _shared_build_group_palette(groups_like, group_colors=group_colors, group_order=group_order)


def _normalize_class_name(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _extract_fatty_acid_motifs(annotation: object, lipid_class: object) -> List[str]:
    text = str(annotation or "").strip()
    if not text:
        return []

    motifs = _MOTIF_RE.findall(text)
    if not motifs:
        return []

    if len(motifs) >= 2:
        return list(dict.fromkeys(motifs))

    lipid_class_norm = _normalize_class_name(lipid_class)
    if lipid_class_norm in _LYSO_OR_SINGLE_CHAIN_CLASSES:
        return list(dict.fromkeys(motifs))

    # Accept single-chain annotations only when the notation looks explicitly chain-resolved.
    if any(sep in text for sep in ("_", "/", ";")):
        return list(dict.fromkeys(motifs))

    return []


def _compute_motif_masks(feature_meta: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    annotation_col = _pick_column_ci(feature_meta, ["Annotation", "NoAbbrev", "Name", "Lipid"])
    class_col = _pick_column_ci(feature_meta, ["Lipid Class", "Headgroup"])
    if annotation_col is None:
        raise ValueError("Could not find an annotation column for fatty-acid motif parsing.")

    annotations = feature_meta[annotation_col].astype(str)
    lipid_classes = feature_meta[class_col].astype(str) if class_col is not None else pd.Series("", index=feature_meta.index, dtype=str)

    motif_records = []
    motif_masks: dict[str, pd.Series] = {}
    for feature_id in feature_meta.index.astype(str):
        motifs = _extract_fatty_acid_motifs(annotations.loc[feature_id], lipid_classes.loc[feature_id])
        if not motifs:
            continue
        for motif in motifs:
            motif_records.append({"FeatureID": feature_id, "Motif": motif})
            if motif not in motif_masks:
                motif_masks[motif] = pd.Series(False, index=feature_meta.index, dtype=bool)
            motif_masks[motif].loc[feature_id] = True

    motif_map_df = pd.DataFrame(motif_records)
    return motif_masks, motif_map_df


def _motif_sort_key(motif: str) -> tuple[int, int, str]:
    try:
        carbons, double_bonds = motif.split(":", 1)
        return int(carbons), int(double_bonds), motif
    except Exception:
        return (10**9, 10**9, str(motif))


def _select_default_panels(observed_motifs: List[str]) -> tuple[List[str], List[str]]:
    observed = {str(motif) for motif in observed_motifs}
    chain_panel = [motif for motif in DEFAULT_FATTY_ACID_CHAIN_PANEL if motif in observed]
    sum_comp_panel = [motif for motif in DEFAULT_FATTY_ACID_SUM_COMPOSITION_PANEL if motif in observed]
    return chain_panel, sum_comp_panel


def _compute_motif_totals(
    X: pd.DataFrame,
    y: pd.Series,
    motif_masks: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for motif, mask in motif_masks.items():
        feature_ids = mask.index[mask.fillna(False)].tolist()
        if not feature_ids:
            continue
        signal = X.loc[:, feature_ids].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        for sample_name, value in signal.items():
            rows.append({
                "Sample": str(sample_name),
                "Group": str(y.loc[sample_name]),
                "Motif": str(motif),
                "TotalIntensity": float(value) if pd.notna(value) else np.nan,
                "FeatureCount": int(len(feature_ids)),
            })
    return pd.DataFrame(rows)


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


def _compute_motif_stats(sample_totals: pd.DataFrame, ordered_groups: List[str]) -> pd.DataFrame:
    rows = []
    for motif, sub in sample_totals.groupby("Motif", sort=False):
        vectors = []
        for group in ordered_groups:
            vals = pd.to_numeric(
                sub.loc[sub["Group"].astype(str) == str(group), "TotalIntensity"],
                errors="coerce",
            ).dropna()
            if len(vals) > 0:
                vectors.append(vals.to_numpy(dtype=float))
        if len(vectors) >= 2:
            try:
                _, p_value = kruskal(*vectors)
            except Exception:
                p_value = np.nan
        else:
            p_value = np.nan
        rows.append({
            "Motif": str(motif),
            "GroupsTested": int(len(vectors)),
            "Kruskal_p_value": p_value,
        })
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats["FDR_BH"] = _bh_fdr(stats["Kruskal_p_value"])
    return stats


def _compute_motif_pairwise_significance(sample_totals: pd.DataFrame, ordered_groups: List[str]) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    omnibus_records = []
    pairwise_records = []

    for motif, sub in sample_totals.groupby("Motif", sort=False):
        grouped = {}
        for group in ordered_groups:
            vals = pd.to_numeric(
                sub.loc[sub["Group"].astype(str) == str(group), "TotalIntensity"],
                errors="coerce",
            ).dropna()
            if len(vals) > 0:
                grouped[str(group)] = vals.to_numpy(dtype=float)

        if len(grouped) < 2:
            results[str(motif)] = {"omnibus_p": np.nan, "omnibus_fdr": np.nan, "grouped_values": grouped, "pairs": []}
            continue

        try:
            _, omnibus_p = kruskal(*grouped.values())
        except Exception:
            omnibus_p = np.nan
        results[str(motif)] = {"omnibus_p": omnibus_p, "omnibus_fdr": np.nan, "grouped_values": grouped, "pairs": []}
        omnibus_records.append((str(motif), omnibus_p))

        groups = list(grouped.keys())
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                try:
                    _, pval = mannwhitneyu(
                        grouped[g1],
                        grouped[g2],
                        alternative="two-sided",
                        method="asymptotic",
                        use_continuity=False,
                    )
                except Exception:
                    pval = np.nan
                pairwise_records.append((str(motif), g1, g2, pval))

    omnibus_fdr = _bh_fdr(pd.Series({motif: p for motif, p in omnibus_records}, dtype=float))
    pairwise_p = pd.Series(
        [p for _, _, _, p in pairwise_records],
        index=pd.Index(range(len(pairwise_records))),
        dtype=float,
    )
    pairwise_fdr = _bh_fdr(pairwise_p)

    for motif, _ in omnibus_records:
        results[motif]["omnibus_fdr"] = float(omnibus_fdr.get(motif, np.nan))

    for idx, (motif, g1, g2, raw_p) in enumerate(pairwise_records):
        fdr_val = float(pairwise_fdr.loc[idx]) if pd.notna(pairwise_fdr.loc[idx]) else np.nan
        omnibus_p = results[motif]["omnibus_p"]
        omnibus_fdr_val = results[motif]["omnibus_fdr"]
        results[motif]["pairs"].append({
            "Group1": g1,
            "Group2": g2,
            "Raw_p_value": raw_p,
            "FDR_BH": fdr_val,
            "Significant": bool(
                pd.notna(fdr_val)
                and fdr_val < 0.05
                and pd.notna(omnibus_fdr_val)
                and omnibus_fdr_val < 0.05
                and pd.notna(omnibus_p)
            ),
        })

    return results


def _motif_pairwise_significance_to_frame(pairwise_significance: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for motif, payload in pairwise_significance.items():
        omnibus_p = payload.get("omnibus_p", np.nan)
        omnibus_fdr = payload.get("omnibus_fdr", np.nan)
        grouped_values = payload.get("grouped_values", {}) or {}
        pairs = payload.get("pairs", [])
        if not pairs:
            rows.append({
                "Motif": motif,
                "Omnibus_Kruskal_p": omnibus_p,
                "Omnibus_Kruskal_FDR_BH": omnibus_fdr,
                "Group1": "",
                "Group2": "",
                "Raw_p_value": np.nan,
                "FDR_BH": np.nan,
                "Significance": "",
                "RawValues__Group1": "",
                "RawValues__Group2": "",
            })
            continue
        for pair in pairs:
            fdr_val = pair["FDR_BH"]
            if pd.notna(fdr_val) and fdr_val < 0.001:
                stars = "***"
            elif pd.notna(fdr_val) and fdr_val < 0.01:
                stars = "**"
            elif pd.notna(fdr_val) and fdr_val < 0.05:
                stars = "*"
            else:
                stars = ""
            rows.append({
                "Motif": motif,
                "Omnibus_Kruskal_p": omnibus_p,
                "Omnibus_Kruskal_FDR_BH": omnibus_fdr,
                "Group1": pair["Group1"],
                "Group2": pair["Group2"],
                "Raw_p_value": pair["Raw_p_value"],
                "FDR_BH": fdr_val,
                "Significance": stars,
                "RawValues__Group1": "; ".join(f"{float(v):.12g}" for v in grouped_values.get(str(pair["Group1"]), [])),
                "RawValues__Group2": "; ".join(f"{float(v):.12g}" for v in grouped_values.get(str(pair["Group2"]), [])),
            })
    return pd.DataFrame(rows)


def _add_pairwise_brackets(ax, dfp: pd.DataFrame, groups: List[str], sig_pairs: List[dict[str, object]]) -> None:
    if not sig_pairs:
        return

    vals = pd.to_numeric(dfp["TotalIntensity"], errors="coerce").to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return

    significant_pairs = [pair for pair in sig_pairs if pair.get("Significant", False)]
    if not significant_pairs:
        return

    current_ymin, current_ymax = ax.get_ylim()
    data_span = max(float(np.nanmax(finite) - np.nanmin(finite)), 1e-12)
    step = max(data_span * 0.045, abs(current_ymax) * 0.015, 0.015)
    y = current_ymax + step * 0.15
    x_pos = {g: i for i, g in enumerate(groups)}

    for pair in significant_pairs:
        g1, g2, fdr_val = pair["Group1"], pair["Group2"], pair["FDR_BH"]
        if g1 not in x_pos or g2 not in x_pos:
            continue
        x1, x2 = x_pos[g1], x_pos[g2]
        if x1 > x2:
            x1, x2 = x2, x1
        if pd.notna(fdr_val) and fdr_val < 0.001:
            stars = "***"
        elif pd.notna(fdr_val) and fdr_val < 0.01:
            stars = "**"
        else:
            stars = "*"
        bracket_top = y + step * 0.24
        ax.plot([x1, x1, x2, x2], [y, bracket_top, bracket_top, y], color="black", linewidth=1.0, clip_on=False, zorder=5)
        ax.text((x1 + x2) / 2, bracket_top + step * 0.08, stars, ha="center", va="bottom", fontsize=12, fontweight="bold", color="crimson", clip_on=False, zorder=6)
        y += step * 0.65

    ax.set_ylim(current_ymin, y + step * 0.18)


def _heatmap(
    table: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_png: str,
    out_svg: str,
    title: str,
    style: dict,
    y_label: str,
) -> None:
    if table.empty:
        return

    vals = table.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return

    with mpl.rc_context({
        "font.family": style["font_family"],
        "font.size": style.get("base_font_size", style["label_size"]),
        "axes.titlesize": style["title_size"],
        "axes.labelsize": style["label_size"],
        "xtick.labelsize": style["tick_size"],
        "ytick.labelsize": style["tick_size"],
    }):
        fig, ax = plt.subplots(
            figsize=(max(5.5, 1.0 * len(table.columns) + 2.2), max(6.0, 0.48 * len(table.index) + 2.6)),
            facecolor="white",
        )
        data_min = float(np.nanmin(finite))
        data_max = float(np.nanmax(finite))
        if data_max <= 0:
            cmap = plt.get_cmap("Blues_r")
            vmin, vmax = data_min, data_max
        elif data_min >= 0:
            cmap = plt.get_cmap("Reds")
            vmin, vmax = data_min, data_max
        else:
            vmax = float(np.nanmax(np.abs(finite)))
            cmap = style["diverging_cmap"]
            vmin = -vmax

        im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_xticklabels(table.columns.tolist(), rotation=45, ha="right")
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_yticklabels(table.index.tolist())
        ax.set_xlabel("Group", labelpad=12)
        ax.set_ylabel(y_label, labelpad=12)
        ax.set_title(title, pad=12, fontweight="semibold")
        ax.set_xticks(np.arange(-0.5, len(table.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(table.index), 1), minor=True)
        ax.grid(which="minor", color=(1, 1, 1, 0.5), linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        sig_map = {}
        if not stats_df.empty:
            sig_map = stats_df.set_index("Motif")["FDR_BH"].to_dict()
        for row_idx, motif in enumerate(table.index.tolist()):
            fdr_val = sig_map.get(motif)
            if pd.notna(fdr_val) and float(fdr_val) < 0.05:
                ax.text(
                    len(table.columns) - 0.5,
                    row_idx,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=style["label_size"],
                    fontweight="bold",
                    color="black",
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.06)
        cbar.set_label("Mean abundance", labelpad=14)
        cbar.ax.tick_params(labelsize=style["tick_size"])

        fig.tight_layout(pad=1.2)
        fig.savefig(out_png, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.12)
        fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)


def _boxplot(
    df: pd.DataFrame,
    motif: str,
    ordered_groups: List[str],
    group_colors: Optional[dict],
    out_png: str,
    out_svg: str,
    style: dict,
    y_label: str,
    pairwise_significance: Optional[dict[str, object]] = None,
) -> None:
    plot_df = df.copy()
    plot_df["Group"] = plot_df["Group"].astype(str)
    plot_df["TotalIntensity"] = pd.to_numeric(plot_df["TotalIntensity"], errors="coerce")
    plot_df = plot_df.dropna(subset=["TotalIntensity"])
    if plot_df.empty:
        return

    order = [g for g in ordered_groups if g in plot_df["Group"].unique().tolist()]
    if not order:
        order = pd.unique(plot_df["Group"]).tolist()
    _, palette = _build_group_palette(order, group_colors=group_colors, group_order=order)

    fig, ax = plt.subplots(figsize=(6.2, 5.0), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.32)

    sns.boxplot(
        data=plot_df,
        x="Group",
        y="TotalIntensity",
        order=order,
        palette=[palette[g] for g in order],
        showfliers=False,
        width=0.58,
        linewidth=0.0,
        whiskerprops=dict(color="gray", linewidth=0.6),
        capprops=dict(color="gray", linewidth=0.6),
        medianprops=dict(color="black", linewidth=0.75),
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
        patch.set_facecolor((rgba[0], rgba[1], rgba[2], style["box_alpha"]))
        patch.set_edgecolor((0, 0, 0, 0))
        patch.set_zorder(1)

    sns.stripplot(
        data=plot_df,
        x="Group",
        y="TotalIntensity",
        order=order,
        palette=[palette[g] for g in order],
        alpha=0.68,
        size=5.0,
        jitter=0.18,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
        zorder=2,
    )

    ax.set_title(f"{motif}", fontsize=style["title_size"], pad=12, fontweight="semibold")
    ax.set_xlabel(None)
    ax.xaxis.label.set_visible(False)
    ax.set_ylabel(y_label, fontsize=style["label_size"], labelpad=12)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=style["tick_size"])
    ax.tick_params(axis="y", labelsize=style["tick_size"])

    vals = plot_df["TotalIntensity"].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size > 0:
        y_lo = float(np.nanpercentile(finite, 1.0))
        y_hi = float(np.nanpercentile(finite, 99.0))
        if not np.isfinite(y_hi):
            y_hi = float(np.nanmax(finite))
        if not np.isfinite(y_lo):
            y_lo = 0.0
        if not np.isfinite(y_hi) or y_hi <= y_lo:
            delta = abs(y_lo) if y_lo != 0 else 1.0
            y_lo, y_hi = y_lo - 0.25 * delta, y_lo + 0.75 * delta
        margin = 0.05 * max(1e-12, (y_hi - y_lo))
        ax.set_ylim(y_lo - margin, y_hi + margin)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style["line_width"])
        spine.set_color("black")

    _add_pairwise_brackets(ax, plot_df[["Group", "TotalIntensity"]].copy(), order, (pairwise_significance or {}).get("pairs", []))

    fig.tight_layout(pad=1.35)
    fig.savefig(out_png, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def run_from_stats(
    file_path: str,
    group_file: Optional[str],
    save_dir: str,
    group_order: Optional[List[str]] = None,
    group_colors: Optional[dict] = None,
    exclude_qc: bool = True,
    dpi: int = 100,
    publication_theme: bool = False,
    dataset_label: Optional[str] = None,
    **kwargs,
) -> Dict[str, str]:
    out_dir = prepare_output_dir(save_dir)
    style = get_figure_style(publication_theme=publication_theme, dpi=dpi)
    print("[FA motifs] Running fatty-acid motif analysis...", flush=True)

    X, y, feature_meta = load_dataset(file_path, group_file)
    if X.empty or feature_meta.empty:
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

    motif_masks, motif_map_df = _compute_motif_masks(feature_meta)
    if not motif_masks:
        raise ValueError("No chain-resolved fatty-acid motifs could be parsed from the annotations.")

    sample_totals = _compute_motif_totals(X, y, motif_masks)
    if sample_totals.empty:
        raise ValueError("No fatty-acid motif abundances could be computed from this dataset.")

    ordered_groups, _ = _build_group_palette(y.astype(str), group_colors=group_colors, group_order=group_order)
    group_means = (
        sample_totals.groupby(["Motif", "Group"], sort=False)["TotalIntensity"]
        .mean()
        .reset_index()
    )
    stats_df = _compute_motif_stats(sample_totals, ordered_groups)
    pairwise_significance = _compute_motif_pairwise_significance(sample_totals, ordered_groups)
    pairwise_stats_df = _motif_pairwise_significance_to_frame(pairwise_significance)

    motif_summary = (
        sample_totals.groupby("Motif", sort=False)
        .agg(
            FeatureCount=("FeatureCount", "max"),
            SampleCount=("Sample", "nunique"),
            MeanAbundance=("TotalIntensity", "mean"),
            MedianAbundance=("TotalIntensity", "median"),
            MaxAbundance=("TotalIntensity", "max"),
        )
        .reset_index()
    )
    if not stats_df.empty:
        motif_summary = motif_summary.merge(stats_df, on="Motif", how="left")
    motif_summary = motif_summary.sort_values(
        ["MeanAbundance", "FeatureCount", "Motif"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    sample_csv = os.path.join(out_dir, "per_sample_fatty_acid_motif_totals.csv")
    group_csv = os.path.join(out_dir, "per_group_fatty_acid_motif_means.csv")
    summary_csv = os.path.join(out_dir, "fatty_acid_motif_summary.csv")
    pairwise_stats_csv = os.path.join(out_dir, "fatty_acid_motif_pairwise_statistics.csv")
    motif_map_csv = os.path.join(out_dir, "fatty_acid_motif_feature_map.csv")
    sample_totals.to_csv(sample_csv, index=False)
    group_means.to_csv(group_csv, index=False)
    motif_summary.to_csv(summary_csv, index=False)
    pairwise_stats_df.to_csv(pairwise_stats_csv, index=False)
    motif_map_df.to_csv(motif_map_csv, index=False)

    observed_motifs = motif_summary["Motif"].astype(str).tolist()
    top_motifs = observed_motifs[:24]
    chain_panel, sum_comp_panel = _select_default_panels(observed_motifs)
    boxplot_motifs = list(dict.fromkeys(observed_motifs))

    heatmap_table = (
        group_means[group_means["Motif"].astype(str).isin(top_motifs)]
        .pivot(index="Motif", columns="Group", values="TotalIntensity")
        .reindex(index=sorted(top_motifs, key=_motif_sort_key), columns=ordered_groups)
    )
    heatmap_png = os.path.join(out_dir, "fatty_acid_motif_heatmap_top.png")
    heatmap_svg = os.path.join(out_dir, "fatty_acid_motif_heatmap_top.svg")
    _heatmap(
        heatmap_table,
        stats_df,
        heatmap_png,
        heatmap_svg,
        title="Fatty-acid motif abundance by group",
        style=style,
        y_label="Fatty-acid motif",
    )

    panel_outputs: Dict[str, str] = {}
    if chain_panel:
        chain_panel_table = (
            group_means[group_means["Motif"].astype(str).isin(chain_panel)]
            .pivot(index="Motif", columns="Group", values="TotalIntensity")
            .reindex(index=sorted(chain_panel, key=_motif_sort_key), columns=ordered_groups)
        )
        chain_panel_png = os.path.join(out_dir, "fatty_acid_motif_heatmap_chain_panel.png")
        chain_panel_svg = os.path.join(out_dir, "fatty_acid_motif_heatmap_chain_panel.svg")
        _heatmap(
            chain_panel_table,
            stats_df,
            chain_panel_png,
            chain_panel_svg,
            title="Selected fatty-acid chain motifs by group",
            style=style,
            y_label="Fatty-acid motif",
        )
        panel_outputs["chain_panel_heatmap_png"] = chain_panel_png
        panel_outputs["chain_panel_heatmap_svg"] = chain_panel_svg

    if sum_comp_panel:
        sum_panel_table = (
            group_means[group_means["Motif"].astype(str).isin(sum_comp_panel)]
            .pivot(index="Motif", columns="Group", values="TotalIntensity")
            .reindex(index=sorted(sum_comp_panel, key=_motif_sort_key), columns=ordered_groups)
        )
        sum_panel_png = os.path.join(out_dir, "fatty_acid_motif_heatmap_sum_composition_panel.png")
        sum_panel_svg = os.path.join(out_dir, "fatty_acid_motif_heatmap_sum_composition_panel.svg")
        _heatmap(
            sum_panel_table,
            stats_df,
            sum_panel_png,
            sum_panel_svg,
            title="Selected fatty-acid sum compositions by group",
            style=style,
            y_label="Fatty-acid motif",
        )
        panel_outputs["sum_composition_panel_heatmap_png"] = sum_panel_png
        panel_outputs["sum_composition_panel_heatmap_svg"] = sum_panel_svg

    boxplot_dir = prepare_output_dir(os.path.join(out_dir, "FattyAcidMotif_Boxplots"))
    y_label = (
        "Semi-quantitative abundance\n(normalized intensity x IS conc.)"
        if str(dataset_label or "").strip().lower().find("semi-quant") >= 0 or "semi_quant" in str(file_path).lower()
        else "Summed intensity of lipids\ncontaining the fatty-acid motif"
    )
    for motif in boxplot_motifs:
        motif_df = sample_totals.loc[sample_totals["Motif"].astype(str) == str(motif), ["Group", "TotalIntensity"]].copy()
        if motif_df.empty:
            continue
        safe_motif = str(motif).replace(":", "_")
        _boxplot(
            motif_df,
            motif=str(motif),
            ordered_groups=ordered_groups,
            group_colors=group_colors,
            out_png=os.path.join(boxplot_dir, f"{safe_motif}.png"),
            out_svg=os.path.join(boxplot_dir, f"{safe_motif}.svg"),
            style=style,
            y_label=y_label,
            pairwise_significance=pairwise_significance.get(str(motif)),
        )

    print(f"[FA motifs] Completed. Results saved to: {out_dir}", flush=True)
    return {
        "out_dir": str(out_dir),
        "sample_totals_csv": sample_csv,
        "group_means_csv": group_csv,
        "motif_summary_csv": summary_csv,
        "motif_pairwise_statistics_csv": pairwise_stats_csv,
        "motif_feature_map_csv": motif_map_csv,
        "heatmap_top_png": heatmap_png,
        "heatmap_top_svg": heatmap_svg,
        "boxplot_dir": str(boxplot_dir),
        **panel_outputs,
    }
