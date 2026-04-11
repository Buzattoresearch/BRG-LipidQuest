from __future__ import annotations

import os
import re
import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle, Patch

from Stats.figure_style import build_group_palette, get_figure_style
from Stats.utils import prepare_output_dir
from Stats.utils import _CLASS_GROUP_MAP


# ============================================================
# Helpers
# ============================================================

def _canon_class(x: str, unknown_policy: str = "append") -> str:
    x = str(x or "").strip()
    if x in _CLASS_GROUP_MAP:
        return _CLASS_GROUP_MAP[x]
    if not x:
        return "Other"
    return "Other" if unknown_policy == "other" else x


def _order_labels(present: List[str], group_order: Optional[List[str]]) -> List[str]:
    present = [str(g) for g in present]
    if not group_order:
        return present
    gui = [g for g in group_order if g in present]
    rest = [g for g in present if g not in gui]
    return gui + rest


def _build_palette(labels: List[str], group_colors: Optional[dict]) -> Dict[str, str]:
    _, pal = build_group_palette(labels, group_colors=group_colors, group_order=None)
    return pal


def _build_class_colors(classes: List[str]) -> Dict[str, str]:
    colors = []
    for cmap in ("tab20", "tab20b", "tab20c", "Set3", "Pastel2", "Pastel1", "Accent", "Paired"):
        colors += [matplotlib.colors.to_hex(c) for c in plt.get_cmap(cmap).colors]
    out: Dict[str, str] = {}
    for i, cl in enumerate(classes):
        out[cl] = colors[i % len(colors)]
    return out


def _detection_threshold(n_samples: int, min_fraction: float) -> int:
    if n_samples <= 0:
        return 1
    frac_need = int(math.ceil(float(min_fraction) * int(n_samples)))
    return max(1, frac_need)


def _feature_class_column(feature_meta: pd.DataFrame) -> str:
    for cand in ("Lipid Class", "lipid class", "Class", "Lipid_Class"):
        if cand in feature_meta.columns:
            return cand
    raise ValueError("Feature metadata is missing a lipid class column.")


def _feature_metadata_columns(feature_meta: pd.DataFrame) -> List[str]:
    preferred = [
        "UniqueID", "Annotation", "Annotation Type", "Annotation Source", "Headgroup",
        "Lipid Class", "RT (min)", "m/z", "Polarity", "Molecular Formula",
        "MS/MS score", "Annotation tier", "Chain type", "Number of carbons in fatty acyls",
        "Double bond equivalents", "Modifications", "# of modifications", "Oxidized?",
        "PUFA?", "Plasmenyl?", "RSD QCs (%)", "RSD Samples (%)",
    ]
    cols = [c for c in preferred if c in feature_meta.columns]
    if "UniqueID" not in cols and "UniqueID" in feature_meta.columns:
        cols = ["UniqueID"] + cols
    return cols


def _attach_feature_metadata(table: pd.DataFrame, feature_meta: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if out.index.name != "UniqueID":
        out.index.name = "UniqueID"
    out = out.reset_index()

    meta_cols = _feature_metadata_columns(feature_meta)
    if meta_cols:
        meta = feature_meta.loc[:, meta_cols].copy()
        meta["UniqueID"] = meta["UniqueID"].astype(str)
        meta = meta.drop_duplicates(subset=["UniqueID"], keep="first")
        out["UniqueID"] = out["UniqueID"].astype(str)
        overlapping_cols = [c for c in meta_cols if c != "UniqueID" and c in out.columns]
        if overlapping_cols:
            out = out.drop(columns=overlapping_cols, errors="ignore")
        out = out.merge(meta, on="UniqueID", how="left")

        ordered_cols = [c for c in meta_cols if c in out.columns] + [c for c in out.columns if c not in meta_cols]
        out = out.loc[:, ordered_cols]

    return out

def _load_nonimputed_dataset(file_path: str, group_file: Optional[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the stats CSV directly without using Stats.utils.load_dataset,
    so NaN values are preserved exactly for detection logic.

    Returns:
        X: samples x features
        y: group labels aligned to X.index
        feature_meta: feature metadata aligned to X.columns by UniqueID
    """
    if group_file is None:
        raise ValueError("group_file is required for UpSet analysis.")

    df = pd.read_csv(file_path, low_memory=False)
    gdf = pd.read_csv(group_file, low_memory=False)

    if "Sample" not in gdf.columns or "Group" not in gdf.columns:
        raise ValueError("Group file must contain 'Sample' and 'Group' columns.")

    meta_keep = [
        "UniqueID", "RT (min)", "m/z", "Polarity", "Annotation", "Annotation Type",
        "Annotation Source", "Headgroup", "Lipid Class", "Δm/z (mDa)", "Δm/z (ppm)",
        "MS/MS score", "Annotation tier", "mSigma", "Molecular Formula", "Plasmenyl?",
        "Number of carbons in fatty acyls", "Double bond equivalents", "Chain type",
        "PUFA?", "Modifications", "# of modifications", "Oxidized?",
        "RSD QCs (%)", "RSD Samples (%)"
    ]
    meta_cols = [c for c in meta_keep if c in df.columns]

    if "UniqueID" not in df.columns:
        raise ValueError("Stats file is missing 'UniqueID'.")

    sample_cols = [c for c in df.columns if c not in meta_cols]

    # keep only samples present in the group file, preserving dataset order
    gdf = gdf.copy()
    gdf["Sample"] = gdf["Sample"].astype(str).str.strip()
    gdf["Group"] = gdf["Group"].astype(str).str.strip()

    sample_cols = [c for c in sample_cols if str(c).strip() in set(gdf["Sample"])]
    if not sample_cols:
        raise ValueError("No overlapping sample columns between stats file and group file.")

    # feature metadata
    feature_meta = df[meta_cols].copy()
    feature_meta["UniqueID"] = feature_meta["UniqueID"].astype(str).str.strip()

    # intensity matrix: samples x features, preserving NaN
    value_df = df[["UniqueID"] + sample_cols].copy()
    value_df["UniqueID"] = value_df["UniqueID"].astype(str).str.strip()
    value_df = value_df.drop_duplicates(subset=["UniqueID"], keep="first")

    X = value_df.set_index("UniqueID")[sample_cols].transpose()
    X.index.name = "Sample"

    # convert only numerics, preserve NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    # align groups to sample order
    gmap = gdf.drop_duplicates(subset=["Sample"], keep="first").set_index("Sample")["Group"]
    X = X.loc[[s for s in X.index if s in gmap.index]].copy()
    y = gmap.reindex(X.index)

    # align metadata to X columns
    feature_meta = feature_meta.drop_duplicates(subset=["UniqueID"], keep="first")
    feature_meta = feature_meta.set_index("UniqueID").reindex(X.columns).reset_index()

    print(f"[UpSet] Loaded non-imputed dataset directly: X shape = {X.shape}", flush=True)
    print(f"[UpSet] Total NaN values in X = {int(X.isna().sum().sum())}", flush=True)

    return X, y, feature_meta


def _compute_detection_by_group(
    X: pd.DataFrame,
    y: pd.Series,
    groups: List[str],
    min_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      detect_bool: features x groups boolean table
      detect_counts: features x groups count table
    """
    y = y.astype(str)
    count_tables = {}
    bool_tables = {}

    for g in groups:
        mask = y == str(g)
        sub = X.loc[mask]
        n_group = sub.shape[0]
        need = _detection_threshold(n_group, min_fraction=min_fraction)
        counts = sub.notna().sum(axis=0).astype(int)
        count_tables[g] = counts
        bool_tables[g] = counts >= need
        print(f"[UpSet] Group '{g}': n={n_group}, detection threshold={need}", flush=True)

    detect_counts = pd.DataFrame(count_tables).reindex(index=X.columns)
    detect_bool = pd.DataFrame(bool_tables).reindex(index=X.columns).fillna(False)
    detect_bool.index.name = "UniqueID"
    detect_counts.index.name = "UniqueID"
    return detect_bool, detect_counts


def _intersection_key(row: pd.Series, ordered_groups: List[str]) -> str:
    present = [g for g in ordered_groups if bool(row[g])]
    if not present:
        return "None"
    return " & ".join(present)


# ============================================================
# Plotting
# ============================================================

def _plot_upset_only(
    overall_counts: pd.Series,
    intersection_matrix: pd.DataFrame,
    out_png: str,
    out_svg: Optional[str],
    ordered_groups: List[str],
    group_colors: Optional[dict],
    title: str,
    style: Optional[dict] = None,
) -> None:
    if overall_counts.empty:
        raise ValueError("No intersections available to plot.")
    style = style or get_figure_style(False, 100)

    pal_groups = _build_palette(ordered_groups, group_colors)

    n_intersections = len(overall_counts)
    n_groups = len(ordered_groups)
    x = np.arange(len(overall_counts))

    fig = plt.figure(
        figsize=(
            max(14, 1.8 * n_intersections + 6),
            max(6.5, 0.7 * n_groups + 4.5),
        )
    )

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[3.2, max(1.8, 0.6 * n_groups)],
        hspace=0.12,
    )

    # --- top bars ---
    ax_top = fig.add_subplot(gs[0])
    bars = ax_top.bar(x, overall_counts.values, width=0.78, color="#4C78A8")
    ax_top.set_ylabel("Detected features", fontsize=style["label_size"])
    ax_top.set_title(title, fontsize=style["title_size"], pad=12, fontweight="semibold")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="y", labelsize=style["tick_size"])

    ymax = float(np.max(overall_counts.values)) if len(overall_counts.values) else 1.0
    ax_top.set_ylim(0, ymax * 1.12)

    for rect, val in zip(bars, overall_counts.values):
        ax_top.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + ymax * 0.01,
            str(int(val)),
            ha="center",
            va="bottom",
            fontsize=max(style["tick_size"] - 1, 8),
        )

    # --- dot matrix ---
    ax_mid = fig.add_subplot(gs[1], sharex=ax_top)
    ax_mid.set_xlim(-0.5, len(overall_counts) - 0.5)
    ax_mid.set_ylim(-0.5, len(ordered_groups) - 0.5)
    ax_mid.set_yticks(np.arange(len(ordered_groups)))
    ax_mid.set_yticklabels(ordered_groups, fontsize=style["tick_size"])
    ax_mid.set_xticks(x)
    ax_mid.set_xticklabels(
        overall_counts.index.tolist(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=max(style["tick_size"] - 1, 8),
    )
    ax_mid.tick_params(axis="x", which="both", bottom=False, labelbottom=True, pad=8)
    ax_mid.invert_yaxis()
    ax_mid.spines["top"].set_visible(False)
    ax_mid.spines["right"].set_visible(False)
    ax_mid.spines["bottom"].set_visible(False)

    for i, g in enumerate(ordered_groups):
        ax_mid.axhline(i, color="#DDDDDD", linewidth=0.8, zorder=0)

    for j, inter in enumerate(overall_counts.index):
        included = [i for i, g in enumerate(ordered_groups) if bool(intersection_matrix.loc[inter, g])]
        if included:
            ax_mid.plot([j, j], [min(included), max(included)], color="black", linewidth=1.6, zorder=2)
        for i, g in enumerate(ordered_groups):
            filled = bool(intersection_matrix.loc[inter, g])
            fc = pal_groups[g] if filled else "white"
            ec = pal_groups[g] if filled else "#999999"
            ax_mid.scatter(j, i, s=85, facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)

    plt.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.25)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    if out_svg:
        fig.savefig(out_svg, dpi=120, bbox_inches="tight")
    plt.close(fig)

def _plot_intersection_class_heatmap(
    class_counts: pd.DataFrame,
    out_png: str,
    out_svg: Optional[str],
    class_colors: Optional[dict],
    style: Optional[dict] = None,
) -> None:
    if class_counts.empty:
        raise ValueError("No class counts available to plot.")
    style = style or get_figure_style(False, 100)

    n_intersections = len(class_counts.columns)
    n_classes = len(class_counts.index)

    fig, ax = plt.subplots(
        figsize=(
            max(12, 1.4 * n_intersections + 4),
            max(8, 0.42 * n_classes + 3),
        )
    )

    data = class_counts.values.astype(float)
    im = ax.imshow(data, aspect="auto", interpolation="nearest")

    ax.set_yticks(np.arange(len(class_counts.index)))
    ax.set_yticklabels(class_counts.index.tolist(), fontsize=style["tick_size"])

    ax.set_xticks(np.arange(len(class_counts.columns)))
    ax.set_xticklabels(
        class_counts.columns.tolist(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=max(style["tick_size"] - 1, 8),
    )

    ax.set_xlabel("Intersection", fontsize=style["label_size"], labelpad=12)
    ax.set_title("Class-stratified counts within each intersection", fontsize=style["title_size"], pad=12, fontweight="semibold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Number of features", fontsize=style["label_size"])
    cbar.ax.tick_params(labelsize=style["tick_size"])

    plt.subplots_adjust(left=0.12, right=0.82, top=0.90, bottom=0.23)
    fig.savefig(out_png, dpi=style["dpi"], bbox_inches="tight")
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_class_bars(
    class_counts: pd.DataFrame,
    out_png: str,
    out_svg: Optional[str],
    group_colors: Optional[dict],
    title: str,
    style: Optional[dict] = None,
    normalize: bool = False,
) -> None:
    if class_counts.empty:
        raise ValueError("No class counts available to plot.")
    style = style or get_figure_style(False, 100)

    plot_df = class_counts.copy()
    if normalize:
        denom = plot_df.sum(axis=0).replace(0, np.nan)
        plot_df = plot_df.divide(denom, axis=1).fillna(0.0) * 100.0

    classes = plot_df.index.tolist()
    groups = plot_df.columns.tolist()
    x = np.arange(len(classes), dtype=float)
    n_groups = max(len(groups), 1)
    bar_width = min(0.82 / n_groups, 0.24)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_width
    palette = _build_palette(groups, group_colors)

    fig_width = max(11.5, 0.45 * len(classes) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.8), facecolor="white")
    ax.set_facecolor("white")

    for idx, group in enumerate(groups):
        vals = plot_df[group].to_numpy(dtype=float)
        ax.bar(
            x + offsets[idx],
            vals,
            width=bar_width * 0.94,
            label=group,
            color=palette.get(group, "#4C78A8"),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        classes,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=max(style["tick_size"] - 1, 8),
    )
    ax.set_ylabel(
        "Unique features (%)" if normalize else "Unique features",
        fontsize=style["label_size"],
        labelpad=10,
    )
    ax.set_xlabel("Lipid class", fontsize=style["label_size"], labelpad=12)
    ax.set_title(title, fontsize=style["title_size"], pad=12, fontweight="semibold")
    ax.tick_params(axis="y", labelsize=style["tick_size"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ymax = float(np.nanmax(plot_df.to_numpy(dtype=float))) if plot_df.size else 0.0
    if normalize:
        ax.set_ylim(0, max(5.0, min(100.0, ymax * 1.15 if ymax > 0 else 5.0)))
    elif ymax > 0:
        ax.set_ylim(0, ymax * 1.12)

    ax.legend(
        frameon=False,
        fontsize=max(style["tick_size"] - 1, 8),
        ncol=min(4, max(1, len(groups))),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(pad=1.1)
    fig.savefig(out_png, dpi=style["dpi"], bbox_inches="tight")
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def _feature_display_label_map(feature_meta: pd.DataFrame) -> pd.Series:
    meta = feature_meta.copy()
    if "UniqueID" not in meta.columns:
        return pd.Series(dtype=str)
    meta["UniqueID"] = meta["UniqueID"].astype(str)
    if "Annotation" in meta.columns:
        annot = meta["Annotation"].fillna("").astype(str).str.strip()
    else:
        annot = pd.Series("", index=meta.index, dtype=str)
    if "Lipid Class" in meta.columns:
        lipid_class = meta["Lipid Class"].fillna("").astype(str).str.strip()
    else:
        lipid_class = pd.Series("", index=meta.index, dtype=str)

    labels = []
    for uid, ann, cl in zip(meta["UniqueID"], annot, lipid_class):
        if ann:
            label = ann
        elif cl:
            label = f"{cl} | {uid}"
        else:
            label = uid
        labels.append(str(label))
    return pd.Series(labels, index=meta["UniqueID"].astype(str))


def _plot_unique_lipid_tilemap(
    unique_presence: pd.DataFrame,
    feature_labels: pd.Series,
    feature_classes: pd.Series,
    out_png: str,
    out_svg: Optional[str],
    group_colors: Optional[dict],
    title: str,
    style: Optional[dict] = None,
) -> None:
    if unique_presence.empty:
        raise ValueError("No unique lipids available to plot.")
    style = style or get_figure_style(False, 100)

    label_map = feature_labels.astype(str).to_dict()
    class_map = feature_classes.astype(str).fillna("").to_dict()

    ordered_ids = sorted(
        unique_presence.index.astype(str).tolist(),
        key=lambda uid: (class_map.get(uid, ""), label_map.get(uid, uid), uid),
    )
    plot_df = unique_presence.reindex(ordered_ids).copy()
    groups = plot_df.columns.tolist()
    pal = _build_palette(groups, group_colors)

    n_rows = len(plot_df.index)
    fig_height = min(max(5.0, 0.20 * n_rows + 2.2), 28.0)
    fig_width = max(6.5, 1.0 * len(groups) + 4.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")
    ax.set_facecolor("white")

    y_positions = np.arange(n_rows)
    x_positions = np.arange(len(groups))
    ax.scatter(
        np.repeat(x_positions, n_rows),
        np.tile(y_positions, len(groups)),
        s=20,
        facecolor="#F1F1F1",
        edgecolor="#E0E0E0",
        linewidth=0.5,
        zorder=1,
    )

    for x_idx, group in enumerate(groups):
        present_mask = plot_df[group].astype(bool).to_numpy()
        if not np.any(present_mask):
            continue
        ax.scatter(
            np.full(int(np.sum(present_mask)), x_idx, dtype=float),
            y_positions[present_mask],
            s=72,
            facecolor=pal.get(group, "#4C78A8"),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    class_labels = [class_map.get(uid, "") for uid in ordered_ids]
    for idx in range(1, len(class_labels)):
        if class_labels[idx] != class_labels[idx - 1]:
            ax.axhline(idx - 0.5, color="#EAEAEA", linewidth=0.8, zorder=0)

    row_labels = [label_map.get(uid, uid) for uid in ordered_ids]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=style["tick_size"])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=max(style["tick_size"] - 2, 6))
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlabel("Group", fontsize=style["label_size"], labelpad=10)
    ax.set_ylabel("Unique lipid", fontsize=style["label_size"], labelpad=10)
    ax.set_title(title, fontsize=style["title_size"], pad=12, fontweight="semibold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(axis="y", length=0)

    fig.tight_layout(pad=1.1)
    fig.savefig(out_png, dpi=style["dpi"], bbox_inches="tight")
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def _plot_top_unique_lipids(
    summary_df: pd.DataFrame,
    ordered_groups: List[str],
    out_png: str,
    out_svg: Optional[str],
    group_colors: Optional[dict],
    title: str,
    style: Optional[dict] = None,
    top_n: int = 12,
) -> None:
    if summary_df.empty:
        raise ValueError("No unique lipids available to plot.")
    style = style or get_figure_style(False, 100)
    pal = _build_palette(ordered_groups, group_colors)

    groups_to_plot = [g for g in ordered_groups if g in summary_df["Group"].astype(str).unique().tolist()]
    if not groups_to_plot:
        groups_to_plot = pd.unique(summary_df["Group"].astype(str)).tolist()
    n_panels = len(groups_to_plot)
    fig, axes = plt.subplots(
        nrows=n_panels,
        ncols=1,
        figsize=(11.5, max(3.2 * n_panels, 4.2)),
        facecolor="white",
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, group in zip(axes_flat, groups_to_plot):
        sub = summary_df.loc[summary_df["Group"].astype(str) == str(group)].copy()
        sub = sub.sort_values(
            ["DetectionFraction", "DetectionCount", "MeanDetectedIntensity", "FeatureLabel"],
            ascending=[False, False, False, True],
            kind="stable",
        ).head(int(top_n))
        sub = sub.iloc[::-1].reset_index(drop=True)

        vals = pd.to_numeric(sub["DetectionFraction"], errors="coerce").fillna(0.0).to_numpy(dtype=float) * 100.0
        y = np.arange(len(sub))
        ax.barh(
            y,
            vals,
            height=0.68,
            color=pal.get(group, "#4C78A8"),
            edgecolor="none",
            alpha=0.95,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(sub["FeatureLabel"].astype(str).tolist(), fontsize=max(style["tick_size"] - 1, 7))
        ax.set_xlim(0, max(100.0, float(np.nanmax(vals) * 1.12) if len(vals) else 100.0))
        ax.set_xlabel("Detected samples within group (%)", fontsize=style["label_size"])
        ax.set_title(str(group), loc="left", fontsize=style["label_size"], pad=8, fontweight="semibold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#444444")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=style["tick_size"])

        for yi, value, det_n in zip(y, vals, sub["DetectionCount"].tolist()):
            ax.text(
                min(value + 1.2, ax.get_xlim()[1] * 0.98),
                yi,
                f"{int(det_n)}",
                va="center",
                ha="left",
                fontsize=max(style["tick_size"] - 2, 7),
                color="#333333",
            )

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=style["title_size"], y=0.995, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.985], pad=1.1)
    fig.savefig(out_png, dpi=style["dpi"], bbox_inches="tight")
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def _safe_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_") or "group"


def _plot_binary_venn(
    left_label: str,
    right_label: str,
    left_only: int,
    overlap: int,
    right_only: int,
    left_color: str,
    right_color: str,
    out_png: str,
    out_svg: Optional[str],
    title: str,
    style: Optional[dict] = None,
) -> None:
    style = style or get_figure_style(False, 100)
    fig, ax = plt.subplots(figsize=(9.2, 7.2), facecolor="white")
    ax.set_facecolor("#FBFBFC")

    left_center = (0.42, 0.54)
    right_center = (0.58, 0.54)
    radius = 0.24

    left_circle = Circle(
        left_center,
        radius,
        facecolor=matplotlib.colors.to_rgba(left_color, alpha=0.36),
        edgecolor=left_color,
        linewidth=2.2,
    )
    right_circle = Circle(
        right_center,
        radius,
        facecolor=matplotlib.colors.to_rgba(right_color, alpha=0.36),
        edgecolor=right_color,
        linewidth=2.2,
    )
    ax.add_patch(left_circle)
    ax.add_patch(right_circle)

    count_fontsize = max(style["title_size"] + 2, 16)
    count_bbox = dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none", alpha=0.92)
    shared_bbox = dict(
        boxstyle="round,pad=0.28",
        facecolor="#FFF6DD",
        edgecolor="#D8B25A",
        linewidth=0.9,
        alpha=0.98,
    )

    ax.text(
        left_center[0] - 0.11,
        left_center[1],
        f"{left_only}",
        ha="center",
        va="center",
        fontsize=count_fontsize,
        fontweight="bold",
        color="#1A1A1A",
        bbox=count_bbox,
    )
    ax.text(
        0.50,
        0.54,
        f"{overlap}",
        ha="center",
        va="center",
        fontsize=count_fontsize,
        fontweight="bold",
        color="#1A1A1A",
        bbox=shared_bbox,
    )
    ax.text(
        right_center[0] + 0.11,
        right_center[1],
        f"{right_only}",
        ha="center",
        va="center",
        fontsize=count_fontsize,
        fontweight="bold",
        color="#1A1A1A",
        bbox=count_bbox,
    )

    label_bbox_left = dict(
        boxstyle="round,pad=0.42",
        facecolor=matplotlib.colors.to_rgba(left_color, alpha=0.10),
        edgecolor=left_color,
        linewidth=1.2,
    )
    label_bbox_right = dict(
        boxstyle="round,pad=0.42",
        facecolor=matplotlib.colors.to_rgba(right_color, alpha=0.10),
        edgecolor=right_color,
        linewidth=1.2,
    )

    ax.text(
        left_center[0],
        0.17,
        f"{left_label}\nDetected: {left_only + overlap}",
        ha="center",
        va="center",
        fontsize=style["label_size"],
        color="#1A1A1A",
        bbox=label_bbox_left,
    )
    ax.text(
        right_center[0],
        0.17,
        f"{right_label}\nDetected: {right_only + overlap}",
        ha="center",
        va="center",
        fontsize=style["label_size"],
        color="#1A1A1A",
        bbox=label_bbox_right,
    )
    ax.text(
        0.50,
        0.82,
        "Shared",
        ha="center",
        va="center",
        fontsize=max(style["label_size"] - 1, 10),
        color="#6B5A2B",
        fontweight="semibold",
    )

    ax.set_title(title, fontsize=style["title_size"], pad=12, fontweight="semibold")
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.06, 0.90)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.tight_layout(pad=1.0)
    fig.savefig(out_png, dpi=style["dpi"], bbox_inches="tight")
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def _generate_binary_venn_outputs(
    detect_bool: pd.DataFrame,
    ordered_groups: List[str],
    out_dir: str,
    group_colors: Optional[dict],
    min_fraction: float,
    style: Optional[dict],
) -> Tuple[Optional[str], List[str], List[str]]:
    if len(ordered_groups) < 2:
        return None, [], []

    pal_groups = _build_palette(ordered_groups, group_colors)
    venn_dir = prepare_output_dir(os.path.join(out_dir, "binary_venn_diagrams"))

    summary_rows = []
    venn_pngs: List[str] = []
    venn_svgs: List[str] = []

    for left_group, right_group in combinations(ordered_groups, 2):
        left_present = detect_bool[left_group].astype(bool)
        right_present = detect_bool[right_group].astype(bool)

        left_only = int((left_present & ~right_present).sum())
        overlap = int((left_present & right_present).sum())
        right_only = int((~left_present & right_present).sum())
        union_total = left_only + overlap + right_only

        slug = f"{_safe_name(left_group)}__vs__{_safe_name(right_group)}"
        out_png = os.path.join(venn_dir, f"venn_{slug}.png")
        out_svg = os.path.join(venn_dir, f"venn_{slug}.svg")

        _plot_binary_venn(
            left_label=left_group,
            right_label=right_group,
            left_only=left_only,
            overlap=overlap,
            right_only=right_only,
            left_color=pal_groups[left_group],
            right_color=pal_groups[right_group],
            out_png=out_png,
            out_svg=out_svg,
            title=(
                f"Binary Venn: {left_group} vs {right_group}\n"
                f"Detection: >={float(min_fraction):.0%} of group samples"
            ),
            style=style,
        )

        venn_pngs.append(out_png)
        venn_svgs.append(out_svg)
        summary_rows.append({
            "Group A": left_group,
            "Group B": right_group,
            "A only": left_only,
            "Shared": overlap,
            "B only": right_only,
            "Union total": union_total,
            "PNG": out_png,
            "SVG": out_svg,
        })

    summary_csv = os.path.join(venn_dir, "binary_venn_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    return summary_csv, venn_pngs, venn_svgs



# ============================================================
# Public API
# ============================================================

def run_from_stats(
    file_path: str,
    group_file: Optional[str],
    save_dir: str,
    group_order: Optional[List[str]] = None,
    group_colors: Optional[dict] = None,
    min_fraction: float = 0.8,
    top_n_intersections: int = 20,
    max_classes: int = 20,
    unknown_policy: str = "append",
    dpi: int = 100,
    publication_theme: bool = False,
    **kwargs,
) -> Dict[str, str]:
    """
    UpSet plot for non-imputed stats tables.
    Detection is based on missing/non-missing values in each group.
    """
    out_dir = prepare_output_dir(save_dir)
    style = get_figure_style(publication_theme=publication_theme, dpi=dpi)
    print("[UpSet] Running detected/shared feature analysis...", flush=True)

    X, y, feature_meta = _load_nonimputed_dataset(file_path, group_file)
    if X.empty or feature_meta.empty:
        raise ValueError("Dataset appears empty or malformed.")

    groups_present = [str(g) for g in pd.unique(y.astype(str))]
    ordered_groups = _order_labels(groups_present, group_order)

    class_col = _feature_class_column(feature_meta)
    if "UniqueID" not in feature_meta.columns:
        feature_meta = feature_meta.copy()
        feature_meta["UniqueID"] = X.columns.astype(str)

    feature_meta = feature_meta.copy()
    feature_meta[class_col] = feature_meta[class_col].astype(str).map(lambda v: _canon_class(v, unknown_policy=unknown_policy))
    class_map = feature_meta.set_index("UniqueID")[class_col].reindex(X.columns).fillna("Other")
    feature_label_map = _feature_display_label_map(feature_meta)
    group_sample_counts = y.astype(str).value_counts().to_dict()

    detect_bool, detect_counts = _compute_detection_by_group(
        X=X,
        y=y,
        groups=ordered_groups,
        min_fraction=min_fraction,
    )

    detect_bool = detect_bool.copy()
    detect_bool["Intersection"] = detect_bool.apply(lambda r: _intersection_key(r, ordered_groups), axis=1)
    detect_bool["Lipid Class"] = class_map.values
    detect_bool.index.name = "UniqueID"

    detection_csv = os.path.join(out_dir, "feature_detection_matrix_by_group.csv")
    detect_bool_export = _attach_feature_metadata(detect_bool, feature_meta)
    detect_bool_export.to_csv(detection_csv, index=False)

    detect_counts_csv = os.path.join(out_dir, "feature_detection_counts_by_group.csv")
    detect_counts_export = _attach_feature_metadata(detect_counts, feature_meta)
    detect_counts_export.to_csv(detect_counts_csv, index=False)

    present_only = detect_bool[detect_bool["Intersection"] != "None"].copy()
    absent_only = detect_bool[detect_bool["Intersection"] == "None"].copy()

    absent_csv = os.path.join(out_dir, "features_detected_in_no_group.csv")
    absent_export = _attach_feature_metadata(absent_only, feature_meta)
    absent_export.to_csv(absent_csv, index=False)

    unique_group_dir = prepare_output_dir(os.path.join(out_dir, "unique_features_by_group"))
    unique_group_exports: Dict[str, str] = {}
    unique_class_count_tables = {}
    unique_presence_map: Dict[str, pd.Series] = {}
    unique_summary_rows = []
    for group in ordered_groups:
        unique_mask = detect_bool[group].astype(bool)
        for other_group in ordered_groups:
            if other_group == group:
                continue
            unique_mask &= ~detect_bool[other_group].astype(bool)

        unique_features = detect_bool.loc[unique_mask].copy()
        unique_features = unique_features.sort_values(by=["Lipid Class", "Intersection"], kind="stable")
        unique_export = _attach_feature_metadata(unique_features, feature_meta)
        out_csv = os.path.join(unique_group_dir, f"unique_features_{_safe_name(group)}.csv")
        unique_export.to_csv(out_csv, index=False)
        unique_group_exports[f"unique_features_{_safe_name(group)}_csv"] = out_csv
        unique_presence_map[group] = unique_mask.astype(bool)
        unique_class_count_tables[group] = (
            unique_features["Lipid Class"]
            .astype(str)
            .value_counts()
            .sort_values(ascending=False)
        )
        group_n = int(group_sample_counts.get(str(group), 0))
        detected_sub = X.loc[y.astype(str) == str(group), unique_features.index.tolist()].copy() if not unique_features.empty else pd.DataFrame()
        for unique_id in unique_features.index.astype(str).tolist():
            detection_count = int(detect_counts.loc[unique_id, group]) if unique_id in detect_counts.index else 0
            detection_fraction = float(detection_count / group_n) if group_n > 0 else np.nan
            mean_detected = np.nan
            if not detected_sub.empty and unique_id in detected_sub.columns:
                vals = pd.to_numeric(detected_sub[unique_id], errors="coerce").dropna()
                if len(vals) > 0:
                    mean_detected = float(vals.mean())
            unique_summary_rows.append({
                "UniqueID": str(unique_id),
                "Group": str(group),
                "FeatureLabel": str(feature_label_map.get(unique_id, unique_id)),
                "Lipid Class": str(class_map.get(unique_id, "Other")),
                "DetectionCount": detection_count,
                "GroupSampleCount": group_n,
                "DetectionFraction": detection_fraction,
                "MeanDetectedIntensity": mean_detected,
            })

    unique_class_counts = pd.DataFrame(unique_class_count_tables).fillna(0).astype(int)
    unique_class_totals = unique_class_counts.sum(axis=1).sort_values(ascending=False)
    if max_classes and len(unique_class_totals) > int(max_classes):
        keep_classes = unique_class_totals.iloc[: int(max_classes)].index.tolist()
        unique_class_counts = unique_class_counts.reindex(keep_classes)
    else:
        unique_class_counts = unique_class_counts.reindex(unique_class_totals.index.tolist())

    unique_class_counts_csv = os.path.join(out_dir, "unique_feature_class_counts_by_group.csv")
    unique_class_counts.to_csv(unique_class_counts_csv, index=True)

    unique_class_pct = unique_class_counts.divide(unique_class_counts.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0) * 100.0
    unique_class_pct_csv = os.path.join(out_dir, "unique_feature_class_percent_by_group.csv")
    unique_class_pct.to_csv(unique_class_pct_csv, index=True)

    unique_summary_df = pd.DataFrame(unique_summary_rows)
    if not unique_summary_df.empty:
        unique_summary_df = _attach_feature_metadata(unique_summary_df.set_index("UniqueID"), feature_meta)
    unique_summary_csv = os.path.join(out_dir, "unique_feature_detection_summary_by_group.csv")
    unique_summary_df.to_csv(unique_summary_csv, index=False)

    unique_presence_df = pd.DataFrame(unique_presence_map).fillna(False)
    unique_presence_df.index = unique_presence_df.index.astype(str)
    unique_presence_df = unique_presence_df.loc[unique_presence_df.any(axis=1)]
    unique_presence_csv = os.path.join(out_dir, "unique_feature_presence_by_group.csv")
    unique_presence_export = _attach_feature_metadata(unique_presence_df, feature_meta)
    unique_presence_export.to_csv(unique_presence_csv, index=False)

    overall_counts = present_only["Intersection"].value_counts()
    overall_counts = overall_counts.sort_values(ascending=False)
    if top_n_intersections and len(overall_counts) > int(top_n_intersections):
        overall_counts = overall_counts.iloc[: int(top_n_intersections)]

    intersection_order = overall_counts.index.tolist()
    intersection_matrix = pd.DataFrame(
        {
            g: [g in inter.split(" & ") for inter in intersection_order]
            for g in ordered_groups
        },
        index=intersection_order,
    )

    class_counts = (
        present_only[present_only["Intersection"].isin(intersection_order)]
        .groupby(["Lipid Class", "Intersection"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=intersection_order, fill_value=0)
    )
    class_totals = class_counts.sum(axis=1).sort_values(ascending=False)
    if max_classes and len(class_totals) > int(max_classes):
        class_keep = class_totals.iloc[: int(max_classes)].index.tolist()
        class_counts = class_counts.reindex(class_keep)
    else:
        class_counts = class_counts.reindex(class_totals.index.tolist())

    overall_counts_csv = os.path.join(out_dir, "intersection_feature_counts.csv")
    overall_counts.rename("Feature Count").to_csv(overall_counts_csv, index=True)

    class_counts_csv = os.path.join(out_dir, "class_by_intersection_feature_counts.csv")
    class_counts.to_csv(class_counts_csv, index=True)

    class_detection_summary = (
        present_only
        .groupby(["Lipid Class", "Intersection"])
        .size()
        .reset_index(name="Feature Count")
        .sort_values(["Lipid Class", "Feature Count"], ascending=[True, False])
    )
    class_detection_summary_csv = os.path.join(out_dir, "class_intersection_long_table.csv")
    class_detection_summary.to_csv(class_detection_summary_csv, index=False)

    png_path = os.path.join(out_dir, "upset_detected_shared_features.png")
    svg_path = os.path.join(out_dir, "upset_detected_shared_features.svg")

    heatmap_png = os.path.join(out_dir, "upset_class_heatmap.png")
    heatmap_svg = os.path.join(out_dir, "upset_class_heatmap.svg")
    unique_class_bar_png = os.path.join(out_dir, "unique_feature_class_distribution_grouped_bar.png")
    unique_class_bar_svg = os.path.join(out_dir, "unique_feature_class_distribution_grouped_bar.svg")
    unique_class_pct_bar_png = os.path.join(out_dir, "unique_feature_class_distribution_grouped_bar_percent.png")
    unique_class_pct_bar_svg = os.path.join(out_dir, "unique_feature_class_distribution_grouped_bar_percent.svg")
    unique_tile_png = os.path.join(out_dir, "unique_lipid_tilemap_by_group.png")
    unique_tile_svg = os.path.join(out_dir, "unique_lipid_tilemap_by_group.svg")
    unique_top_png = os.path.join(out_dir, "top_unique_lipids_by_group.png")
    unique_top_svg = os.path.join(out_dir, "top_unique_lipids_by_group.svg")

    class_colors = _build_class_colors(class_counts.index.tolist())

    _plot_upset_only(
        overall_counts=overall_counts,
        intersection_matrix=intersection_matrix,
        out_png=png_path,
        out_svg=svg_path,
        ordered_groups=ordered_groups,
        group_colors=group_colors,
        title=(
            f"Detected/shared lipid features across groups\n"
            f"Detection: ≥{float(min_fraction):.0%} of group samples"
        ),
        style=style,
    )

    _plot_intersection_class_heatmap(
        class_counts=class_counts,
        out_png=heatmap_png,
        out_svg=heatmap_svg,
        class_colors=class_colors,
        style=style,
    )

    if not unique_class_counts.empty:
        _plot_grouped_class_bars(
            class_counts=unique_class_counts,
            out_png=unique_class_bar_png,
            out_svg=unique_class_bar_svg,
            group_colors=group_colors,
            title="Class distribution of group-unique lipid features",
            style=style,
            normalize=False,
        )
        _plot_grouped_class_bars(
            class_counts=unique_class_counts,
            out_png=unique_class_pct_bar_png,
            out_svg=unique_class_pct_bar_svg,
            group_colors=group_colors,
            title="Class distribution of group-unique lipid features (%)",
            style=style,
            normalize=True,
        )

    if not unique_presence_df.empty:
        _plot_unique_lipid_tilemap(
            unique_presence=unique_presence_df.reindex(columns=ordered_groups, fill_value=False),
            feature_labels=feature_label_map,
            feature_classes=class_map,
            out_png=unique_tile_png,
            out_svg=unique_tile_svg,
            group_colors=group_colors,
            title="Group-unique lipid detections",
            style=style,
        )

    if not unique_summary_df.empty:
        _plot_top_unique_lipids(
            summary_df=unique_summary_df,
            ordered_groups=ordered_groups,
            out_png=unique_top_png,
            out_svg=unique_top_svg,
            group_colors=group_colors,
            title="Top uniquely detected lipids in each group",
            style=style,
            top_n=12,
        )

    binary_venn_summary_csv, binary_venn_pngs, binary_venn_svgs = _generate_binary_venn_outputs(
        detect_bool=detect_bool[ordered_groups].copy(),
        ordered_groups=ordered_groups,
        out_dir=out_dir,
        group_colors=group_colors,
        min_fraction=min_fraction,
        style=style,
    )

    return {
        "feature_detection_matrix_csv": detection_csv,
        "feature_detection_counts_csv": detect_counts_csv,
        "features_in_no_group_csv": absent_csv,
        "unique_features_by_group_dir": unique_group_dir,
        "unique_feature_class_counts_csv": unique_class_counts_csv,
        "unique_feature_class_percent_csv": unique_class_pct_csv,
        "unique_feature_summary_csv": unique_summary_csv,
        "unique_feature_presence_csv": unique_presence_csv,
        "intersection_counts_csv": overall_counts_csv,
        "class_by_intersection_csv": class_counts_csv,
        "class_intersection_long_csv": class_detection_summary_csv,
        "upset_png": png_path,
        "upset_svg": svg_path,
        "heatmap_png": heatmap_png,
        "heatmap_svg": heatmap_svg,
        "unique_feature_class_bar_png": unique_class_bar_png,
        "unique_feature_class_bar_svg": unique_class_bar_svg,
        "unique_feature_class_percent_bar_png": unique_class_pct_bar_png,
        "unique_feature_class_percent_bar_svg": unique_class_pct_bar_svg,
        "unique_lipid_tilemap_png": unique_tile_png,
        "unique_lipid_tilemap_svg": unique_tile_svg,
        "top_unique_lipids_png": unique_top_png,
        "top_unique_lipids_svg": unique_top_svg,
        "binary_venn_summary_csv": binary_venn_summary_csv or "",
        "binary_venn_pngs": ";".join(binary_venn_pngs),
        "binary_venn_svgs": ";".join(binary_venn_svgs),
        **unique_group_exports,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Detected/shared feature UpSet plot for non-imputed lipidomics stats tables.")
    p.add_argument("--stats_csv", required=True)
    p.add_argument("--group_file", required=False, default=None)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--min_fraction", type=float, default=0.8)
    p.add_argument("--top_n_intersections", type=int, default=20)
    p.add_argument("--max_classes", type=int, default=20)
    args = p.parse_args()

    out = run_from_stats(
        file_path=args.stats_csv,
        group_file=args.group_file,
        save_dir=args.save_dir,
        min_fraction=args.min_fraction,
        top_n_intersections=args.top_n_intersections,
        max_classes=args.max_classes,
    )
    for k, v in out.items():
        print(f"{k}: {v}")
