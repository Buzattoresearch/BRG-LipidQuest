import warnings
from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from Stats.figure_style import get_figure_style
from Stats.pca_analysis import permanova
from Stats.utils import load_dataset, prepare_output_dir

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["mathtext.default"] = "regular"
plt.rcParams["font.size"] = 14
plt.ioff()


def _build_explanation(distance_metric: str, linkage_method: str, coph_corr: float, p_value: float) -> str:
    if p_value < 0.05:
        permanova_line = "PERMANOVA suggests the group labels explain more of the scaled feature-space structure than expected under random label assignment."
    else:
        permanova_line = "PERMANOVA does not show strong evidence that the observed group labels explain the scaled feature-space structure better than chance."

    if np.isnan(coph_corr):
        coph_line = "The cophenetic correlation could not be estimated reliably for this dataset."
    elif coph_corr >= 0.9:
        coph_line = "The dendrogram preserves pairwise sample relationships very well."
    elif coph_corr >= 0.75:
        coph_line = "The dendrogram is a reasonably faithful summary of sample-to-sample distances."
    else:
        coph_line = "The dendrogram is a fairly rough summary of the original sample-to-sample distances."

    text = f"""
Hierarchical clustering interpretation
--------------------------------------
This analysis clusters samples after z-score standardization across lipid features using {linkage_method} linkage and {distance_metric} distance.

How to read the dendrogram
--------------------------
- Samples that merge at lower branch heights are more similar than samples that merge only at higher heights.
- Branch height reflects dissimilarity under the chosen distance metric; longer vertical joins indicate larger differences.
- {coph_line}
- {permanova_line}

Important limitations
---------------------
- The dendrogram depends on preprocessing choices, especially scaling, missing-value handling, and which features are retained.
- Different linkage methods or distance metrics can change cluster shape and merge order.
- The tree imposes a strictly hierarchical structure, even when the underlying biology may be continuous or partially overlapping.
- Apparent clusters in a dendrogram should be confirmed with metadata, replicate behavior, and complementary analyses such as PCA or t-SNE.
"""
    return textwrap.dedent(text).strip() + "\n"


def run_hierarchical_clustering(
    file_path,
    group_file,
    save_dir,
    group_colors=None,
    group_order=None,
    dpi: int = 100,
    publication_theme: bool = False,
):
    file_path = Path(file_path)
    save_dir = prepare_output_dir(save_dir)
    style = get_figure_style(publication_theme=publication_theme, dpi=dpi)
    print(f"[Hierarchical Clustering] Running for: {file_path.name}", flush=True)
    plt.close("all")

    X, y, _feature_meta = load_dataset(file_path, group_file)
    if X.empty or y.empty:
        raise ValueError("Hierarchical clustering could not load a valid dataset.")
    if X.shape[0] < 2:
        raise ValueError("Hierarchical clustering requires at least 2 matched samples.")
    if X.shape[1] < 2:
        raise ValueError("Hierarchical clustering requires at least 2 features.")

    X_scaled = StandardScaler().fit_transform(X)
    dist_condensed = pdist(X_scaled, metric="euclidean")
    distance_matrix = squareform(dist_condensed)
    linkage_method = "ward"
    distance_metric = "euclidean"
    Z = linkage(X_scaled, method=linkage_method, metric=distance_metric)
    coph_corr, _ = cophenet(Z, dist_condensed)
    F_stat, p_value = permanova(distance_matrix, y, permutations=1000)

    leaf_order = dendrogram(Z, no_plot=True, labels=X.index.astype(str).tolist())["ivl"]
    leaf_df = pd.DataFrame(
        {
            "Leaf_Order": np.arange(1, len(leaf_order) + 1),
            "Sample": leaf_order,
        }
    )
    group_lookup = y.astype(str).to_dict()
    leaf_df["Group"] = leaf_df["Sample"].map(group_lookup).fillna("Unknown")
    leaf_df.to_csv(save_dir / "Hierarchical_clustering_leaf_order.csv", index=False, encoding="utf-8-sig")

    params_df = pd.DataFrame(
        [
            {"Parameter": "distance_metric", "Value": distance_metric},
            {"Parameter": "linkage_method", "Value": linkage_method},
            {"Parameter": "n_samples", "Value": X.shape[0]},
            {"Parameter": "n_features", "Value": X.shape[1]},
            {"Parameter": "cophenetic_correlation", "Value": coph_corr},
            {"Parameter": "permanova_F", "Value": F_stat},
            {"Parameter": "permanova_p", "Value": p_value},
        ]
    )
    params_df.to_csv(save_dir / "Hierarchical_clustering_parameters.csv", index=False, encoding="utf-8-sig")
    (save_dir / "Hierarchical_clustering_explanation.txt").write_text(
        _build_explanation(distance_metric, linkage_method, float(coph_corr), float(p_value)),
        encoding="utf-8",
    )

    fig_width = max(9.0, 0.38 * X.shape[0] + 5.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.8))
    d = dendrogram(
        Z,
        labels=X.index.astype(str).tolist(),
        leaf_rotation=90,
        leaf_font_size=max(7, min(style["tick_size"] - 2, 12)),
        color_threshold=0,
        above_threshold_color="#444444",
        ax=ax,
    )

    group_lookup = y.astype(str).to_dict()
    label_to_group = {sample: group_lookup.get(sample, "Unknown") for sample in d["ivl"]}
    present_groups = [label_to_group[sample] for sample in d["ivl"]]
    if group_order:
        ordered_groups = [g for g in group_order if g in present_groups] + [g for g in pd.unique(present_groups).tolist() if g not in group_order]
    else:
        ordered_groups = pd.unique(present_groups).tolist()

    from Stats.pca_analysis import make_distinct_palette
    color_map = make_distinct_palette(ordered_groups, group_colors=group_colors)
    for tick in ax.get_xmajorticklabels():
        sample = tick.get_text()
        tick.set_color(color_map.get(label_to_group.get(sample, "Unknown"), "#444444"))

    ax.set_title("Hierarchical Clustering Dendrogram", fontsize=style["title_size"])
    ax.set_xlabel("Samples", fontsize=style["label_size"], labelpad=12)
    ax.set_ylabel("Linkage Distance", fontsize=style["label_size"], labelpad=12)
    ax.tick_params(axis="x", labelsize=max(style["tick_size"] - 4, 8))
    ax.tick_params(axis="y", labelsize=style["tick_size"])

    if p_value < 0.001:
        p_text = "< 0.001"
    else:
        p_text = f"= {p_value:.3g}"
    stats_text = f"PERMANOVA (1000 perm.): F = {F_stat:.2f}, p {p_text} | COPHENETIC CORR. = {coph_corr:.2f}"
    ax.text(
        0.5,
        -0.24,
        stats_text,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=max(style["tick_size"] - 1, 10),
        color="dimgray",
    )

    legend_handles = []
    for group in ordered_groups:
        legend_handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map.get(group, "#444444"), markersize=8, label=group))
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=style["legend_size"],
            title_fontsize=style["legend_size"],
            frameon=False,
            handletextpad=0.5,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style["line_width"])
        spine.set_color("black")

    fig.subplots_adjust(right=0.84, bottom=0.30)
    fig.savefig(
        save_dir / "Hierarchical_clustering_dendrogram.png",
        dpi=style["dpi"],
        bbox_inches="tight",
        pad_inches=0.3,
        facecolor="white",
    )
    fig.savefig(
        save_dir / "Hierarchical_clustering_dendrogram.svg",
        bbox_inches="tight",
        pad_inches=0.3,
        facecolor="white",
    )
    plt.close(fig)
    print(f"[Hierarchical Clustering] Completed. Results saved to: {save_dir}\n", flush=True)
