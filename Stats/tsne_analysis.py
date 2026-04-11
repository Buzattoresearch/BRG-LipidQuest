import warnings
from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from Stats.figure_style import get_figure_style
from Stats.pca_analysis import (
    BOTTOM_STATS_Y,
    choose_inside_legend_position,
    get_cov_ellipse,
    make_distinct_palette,
    permanova,
    set_plot_limits,
)
from Stats.utils import load_dataset, prepare_output_dir

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["mathtext.default"] = "regular"
plt.rcParams["font.size"] = 14
plt.ioff()

TSNE_RANDOM_STATE = 42


def _choose_perplexity(n_samples: int) -> float:
    if n_samples < 4:
        raise ValueError("t-SNE requires at least 4 samples to produce a stable embedding.")
    upper = max(2.0, float(n_samples - 1) / 3.0)
    return min(30.0, upper)


def _safe_silhouette(scores_2d: np.ndarray, groups: pd.Series) -> float:
    unique_groups = pd.Series(groups).astype(str).nunique()
    if unique_groups < 2 or len(scores_2d) <= unique_groups:
        return float("nan")
    return float(silhouette_score(scores_2d, groups, metric="euclidean"))


def _tsne_explanation_text(
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    p_value: float,
    silhouette: float,
) -> str:
    if np.isnan(silhouette):
        silhouette_line = "Silhouette score could not be computed because the embedding did not contain enough group structure."
    elif silhouette >= 0.5:
        silhouette_line = "The embedding shows relatively strong within-group compactness compared with between-group separation."
    elif silhouette >= 0.2:
        silhouette_line = "The embedding shows some group organization, but the separation is only moderate."
    else:
        silhouette_line = "The embedding shows weak group separation, so clusters should be interpreted cautiously."

    if p_value < 0.05:
        permanova_line = "PERMANOVA on the scaled feature space suggests the group structure is stronger than expected by random label assignment."
    else:
        permanova_line = "PERMANOVA on the scaled feature space does not show strong evidence that the observed grouping exceeds random label assignment."

    limitations = [
        "t-SNE preserves local neighborhoods better than global distances, so spacing between far-apart clusters should not be overinterpreted.",
        "The t-SNE axes are abstract embedding coordinates, not percentages of explained variance like PCA.",
        "Results depend on preprocessing and tuning choices such as perplexity, learning rate, initialization, and random seed.",
        "Different runs or parameter choices can move or reshape clusters even when the same samples are used.",
    ]

    text = f"""
t-SNE interpretation
--------------------
This visualization embeds samples from the standardized feature matrix into two dimensions using t-SNE with random_state={TSNE_RANDOM_STATE}, perplexity={perplexity:.2f}, learning_rate={learning_rate:.2f}, and max_iter={n_iter}.

How to read the figure
----------------------
- Nearby points are more likely to have similar lipid profiles than distant points.
- Confidence ellipses summarize within-group spread only when at least three samples are available in a group.
- {silhouette_line}
- {permanova_line}

Important limitations
---------------------
""" + "\n".join(f"- {line}" for line in limitations) + "\n"
    return textwrap.dedent(text).strip() + "\n"


def run_tsne(
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
    print(f"[t-SNE] Running t-SNE for: {file_path.name}", flush=True)
    plt.close("all")

    X, y, _feature_meta = load_dataset(file_path, group_file)
    if X.empty or y.empty:
        raise ValueError("t-SNE could not load a valid dataset.")
    if X.shape[0] < 4:
        raise ValueError("t-SNE requires at least 4 matched samples.")
    if X.shape[1] < 2:
        raise ValueError("t-SNE requires at least 2 features.")

    X_scaled = StandardScaler().fit_transform(X)
    distance_matrix = squareform(pdist(X_scaled, metric="euclidean"))
    perplexity = _choose_perplexity(X.shape[0])
    learning_rate = max(float(X.shape[0]) / 4.0, 50.0)

    init_components = min(2, X_scaled.shape[0], X_scaled.shape[1])
    if init_components < 2:
        raise ValueError("t-SNE requires at least 2 samples/features to initialize the embedding.")
    pca_init = PCA(n_components=init_components).fit_transform(X_scaled)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=1000,
        init=pca_init,
        random_state=TSNE_RANDOM_STATE,
        metric="euclidean",
        method="barnes_hut",
        angle=0.5,
    )
    scores = tsne.fit_transform(X_scaled)

    F_stat, p_value = permanova(distance_matrix, y, permutations=1000)
    silhouette = _safe_silhouette(scores, y)
    sil_text = "nan" if np.isnan(silhouette) else f"{silhouette:.3f}"
    print(
        f"[t-SNE] PERMANOVA F={F_stat:.3f}, p={p_value:.4g}, silhouette={sil_text}, perplexity={perplexity:.2f}",
        flush=True,
    )

    tsne_df = pd.DataFrame(scores, columns=["tSNE1", "tSNE2"], index=X.index)
    tsne_df["Group"] = y.values
    tsne_df.to_csv(save_dir / "tSNE_scores.csv", index=True, encoding="utf-8-sig")

    param_df = pd.DataFrame(
        [
            {"Parameter": "random_state", "Value": TSNE_RANDOM_STATE},
            {"Parameter": "perplexity", "Value": perplexity},
            {"Parameter": "learning_rate", "Value": learning_rate},
            {"Parameter": "max_iter", "Value": 1000},
            {"Parameter": "metric", "Value": "euclidean"},
            {"Parameter": "init", "Value": "PCA coordinates"},
            {"Parameter": "n_samples", "Value": X.shape[0]},
            {"Parameter": "n_features", "Value": X.shape[1]},
            {"Parameter": "permanova_F", "Value": F_stat},
            {"Parameter": "permanova_p", "Value": p_value},
            {"Parameter": "silhouette_score", "Value": silhouette},
        ]
    )
    param_df.to_csv(save_dir / "tSNE_parameters.csv", index=False, encoding="utf-8-sig")
    (save_dir / "tSNE_explanation_and_limitations.txt").write_text(
        _tsne_explanation_text(perplexity, learning_rate, 1000, p_value, silhouette),
        encoding="utf-8",
    )

    existing = list(tsne_df["Group"].astype(str).unique())
    if group_order:
        groups_order = [g for g in group_order if g in existing] + [g for g in existing if g not in group_order]
    else:
        groups_order = existing
    color_map = make_distinct_palette(groups_order, group_colors=group_colors)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=tsne_df,
        x="tSNE1",
        y="tSNE2",
        hue="Group",
        hue_order=groups_order,
        palette=color_map,
        s=90,
        alpha=0.95,
        edgecolor="black",
        ax=ax,
    )

    ellipses = []
    for group in groups_order:
        data_g = tsne_df.loc[tsne_df["Group"] == group, ["tSNE1", "tSNE2"]]
        if len(data_g) < 3:
            continue
        cov = np.cov(data_g.T)
        center = data_g.mean().values
        color = color_map.get(group, (0.5, 0.5, 0.5))
        try:
            ellipse = get_cov_ellipse(
                cov,
                center,
                nstd=1.96,
                facecolor=color,
                alpha=0.18,
                edgecolor=color,
                linewidth=1,
            )
            ax.add_patch(ellipse)
            ellipses.append(ellipse)
        except Exception:
            pass

    ax.set_xlabel("t-SNE 1", labelpad=12, fontsize=style["label_size"])
    ax.set_ylabel("t-SNE 2", labelpad=12, fontsize=style["label_size"])
    ax.set_title("t-SNE Embedding", fontsize=style["title_size"])
    legend = ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=style["legend_size"],
        title_fontsize=style["legend_size"],
        frameon=False,
        handletextpad=0.5,
    )
    set_plot_limits(ax, tsne_df.rename(columns={"tSNE1": "PC1", "tSNE2": "PC2"}), ellipses=ellipses)

    if p_value < 0.001:
        p_text = "< 0.001"
    else:
        p_text = f"= {p_value:.3g}"
    sil_line = "NA" if np.isnan(silhouette) else f"{silhouette:.2f}"
    bottom_text = f"PERMANOVA (1000 perm.): F = {F_stat:.2f}, p {p_text} | SILHOUETTE SCORE = {sil_line}"
    ax.text(
        0.5,
        BOTTOM_STATS_Y,
        bottom_text,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=style["tick_size"],
        color="dimgray",
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style["line_width"])
        spine.set_color("black")

    ax.set_aspect("auto")
    fig.subplots_adjust(right=0.84, bottom=0.24)
    fig.savefig(save_dir / "tSNE_2D.png", dpi=style["dpi"], bbox_inches="tight", pad_inches=0.3, facecolor="white")
    fig.savefig(save_dir / "tSNE_2D.svg", bbox_inches="tight", pad_inches=0.3, facecolor="white")

    legend.remove()
    inside_loc, inside_anchor = choose_inside_legend_position(
        ax,
        tsne_df.rename(columns={"tSNE1": "PC1", "tSNE2": "PC2"}),
        groups_order,
    )
    ax.legend(
        loc=inside_loc,
        bbox_to_anchor=inside_anchor,
        borderaxespad=0.2,
        fontsize=max(style["legend_size"] - 2, 9),
        title_fontsize=max(style["legend_size"] - 2, 9),
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        handletextpad=0.5,
    )
    fig.subplots_adjust(right=0.96, bottom=0.24)
    fig.savefig(
        save_dir / "tSNE_2D_legend_inside.png",
        dpi=style["dpi"],
        bbox_inches="tight",
        pad_inches=0.3,
        facecolor="white",
    )
    fig.savefig(save_dir / "tSNE_2D_legend_inside.svg", bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=tsne_df,
        x="tSNE1",
        y="tSNE2",
        hue="Group",
        hue_order=groups_order,
        palette=color_map,
        s=90,
        alpha=0.95,
        edgecolor="black",
        ax=ax,
    )

    ellipses = []
    for group in groups_order:
        data_g = tsne_df.loc[tsne_df["Group"] == group, ["tSNE1", "tSNE2"]]
        if len(data_g) < 3:
            continue
        cov = np.cov(data_g.T)
        center = data_g.mean().values
        color = color_map.get(group, (0.5, 0.5, 0.5))
        try:
            ellipse = get_cov_ellipse(
                cov,
                center,
                nstd=1.96,
                facecolor=color,
                alpha=0.18,
                edgecolor=color,
                linewidth=1,
            )
            ax.add_patch(ellipse)
            ellipses.append(ellipse)
        except Exception:
            pass

    for sample_name, row in tsne_df.iterrows():
        ax.text(
            row["tSNE1"] + 0.4,
            row["tSNE2"] + 0.4,
            str(sample_name),
            fontsize=7,
            alpha=0.8,
            color="black",
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("t-SNE 1", labelpad=12, fontsize=style["label_size"])
    ax.set_ylabel("t-SNE 2", labelpad=12, fontsize=style["label_size"])
    ax.set_title("t-SNE Embedding with Labels", fontsize=style["title_size"])
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=style["legend_size"],
        title_fontsize=style["legend_size"],
        frameon=False,
        handletextpad=0.5,
    )
    set_plot_limits(ax, tsne_df.rename(columns={"tSNE1": "PC1", "tSNE2": "PC2"}), ellipses=ellipses)
    ax.text(
        0.5,
        BOTTOM_STATS_Y,
        bottom_text,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=style["tick_size"],
        color="dimgray",
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style["line_width"])
        spine.set_color("black")

    ax.set_aspect("auto")
    fig.subplots_adjust(right=0.84, bottom=0.24)
    fig.savefig(
        save_dir / "tSNE_2D_with_labels.png",
        dpi=style["dpi"],
        bbox_inches="tight",
        pad_inches=0.3,
        facecolor="white",
    )
    fig.savefig(save_dir / "tSNE_2D_with_labels.svg", bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)

    print(f"[t-SNE] Completed. Results saved to: {save_dir}\n", flush=True)
