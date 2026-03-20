
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from Stats.utils import load_dataset, prepare_output_dir
from Stats.figure_style import build_group_palette, get_figure_style
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

import matplotlib as mpl
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["mathtext.default"] = "regular" 

plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")
plt.ioff()

BOTTOM_STATS_Y = -0.24

# ==========================================================
# Helper functions
# ==========================================================
def make_distinct_palette(groups, group_colors=None):
    _, cmap = build_group_palette(groups, group_colors=group_colors, group_order=None)
    return cmap

def get_cov_ellipse(cov, center, nstd=1.96, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)


def get_ellipse_bounds(ellipse):
    """Return axis-aligned bounds for a possibly rotated ellipse."""
    theta = np.deg2rad(ellipse.angle)
    half_width = ellipse.width / 2.0
    half_height = ellipse.height / 2.0

    x_radius = np.sqrt((half_width * np.cos(theta)) ** 2 + (half_height * np.sin(theta)) ** 2)
    y_radius = np.sqrt((half_width * np.sin(theta)) ** 2 + (half_height * np.cos(theta)) ** 2)

    cx, cy = ellipse.center
    return cx - x_radius, cx + x_radius, cy - y_radius, cy + y_radius


def set_plot_limits(ax, data, ellipses=None, pad_fraction=0.08, min_pad=0.5):
    """Fit axes to all score points and any confidence ellipses."""
    x_vals = np.asarray(data["PC1"], dtype=float)
    y_vals = np.asarray(data["PC2"], dtype=float)

    x_min = np.nanmin(x_vals)
    x_max = np.nanmax(x_vals)
    y_min = np.nanmin(y_vals)
    y_max = np.nanmax(y_vals)

    for ellipse in ellipses or []:
        ex_min, ex_max, ey_min, ey_max = get_ellipse_bounds(ellipse)
        x_min = min(x_min, ex_min)
        x_max = max(x_max, ex_max)
        y_min = min(y_min, ey_min)
        y_max = max(y_max, ey_max)

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_pad = max(x_range * pad_fraction, min_pad if x_range == 0 else 0)
    y_pad = max(y_range * pad_fraction, min_pad if y_range == 0 else 0)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")


def set_plot_limits_3d(ax, data, pad_fraction=0.08, min_pad=0.5):
    x_vals = np.asarray(data["PC1"], dtype=float)
    y_vals = np.asarray(data["PC2"], dtype=float)
    z_vals = np.asarray(data["PC3"], dtype=float)

    x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)
    y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
    z_min, z_max = np.nanmin(z_vals), np.nanmax(z_vals)

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_pad = max(x_range * pad_fraction, min_pad if x_range == 0 else 0)
    y_pad = max(y_range * pad_fraction, min_pad if y_range == 0 else 0)
    z_pad = max(z_range * pad_fraction, min_pad if z_range == 0 else 0)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)


def choose_inside_legend_position(ax, data, groups_order):
    """Pick the least crowded corner for an inside legend."""
    if data.empty:
        return "upper right", (0.98, 0.98)

    pts = data.loc[:, ["PC1", "PC2"]].dropna()
    if pts.empty:
        return "upper right", (0.98, 0.98)

    pts_axes = ax.transAxes.inverted().transform(ax.transData.transform(pts.to_numpy(dtype=float)))

    # Roughly estimate legend footprint in axes coordinates.
    legend_height = min(0.11 + 0.045 * max(len(groups_order), 1), 0.48)
    legend_width = 0.20
    margin = 0.03

    candidates = [
        ("upper right", (0.98, 0.98), (1 - legend_width - margin, 1 - legend_height - margin, 1 - margin, 1 - margin)),
        ("upper left", (0.02, 0.98), (margin, 1 - legend_height - margin, legend_width + margin, 1 - margin)),
        ("lower right", (0.98, 0.22), (1 - legend_width - margin, margin + 0.08, 1 - margin, margin + 0.08 + legend_height)),
        ("lower left", (0.02, 0.22), (margin, margin + 0.08, legend_width + margin, margin + 0.08 + legend_height)),
        ("lower center", (0.50, 0.22), (0.50 - legend_width / 2, margin + 0.08, 0.50 + legend_width / 2, margin + 0.08 + legend_height)),
    ]

    best_loc = "upper right"
    best_anchor = (0.98, 0.98)
    best_score = None
    for loc, anchor, (x0, y0, x1, y1) in candidates:
        inside = (
            (pts_axes[:, 0] >= x0) & (pts_axes[:, 0] <= x1) &
            (pts_axes[:, 1] >= y0) & (pts_axes[:, 1] <= y1)
        )
        score = int(np.count_nonzero(inside))
        if best_score is None or score < best_score:
            best_score = score
            best_loc = loc
            best_anchor = anchor

    return best_loc, best_anchor


def plot_pca_scores_3d(pca_df_3d, explained_variance_ratio, save_path, title, group_order=None, group_colors=None, style=None):
    style = style or get_figure_style(False, 100)
    existing = list(pca_df_3d["Group"].astype(str).unique())
    if group_order:
        groups_order = [g for g in group_order if g in existing] + [g for g in existing if g not in group_order]
    else:
        groups_order = existing
    color_map = make_distinct_palette(groups_order, group_colors=group_colors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for group in groups_order:
        data_g = pca_df_3d.loc[pca_df_3d["Group"] == group]
        ax.scatter(
            data_g["PC1"],
            data_g["PC2"],
            data_g["PC3"],
            s=style["marker_size"] * 1.8,
            alpha=0.95,
            color=color_map.get(group, "#808080"),
            edgecolors="black",
            linewidths=0.6,
            label=group,
        )

    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}% Variance)", labelpad=12, fontsize=style["label_size"])
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}% Variance)", labelpad=12, fontsize=style["label_size"])
    ax.set_zlabel(f"PC3 ({explained_variance_ratio[2]*100:.1f}% Variance)", labelpad=12, fontsize=style["label_size"])
    ax.set_title(title, fontsize=style["title_size"], pad=16)
    legend = ax.legend(
        bbox_to_anchor=(1.18, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=style["legend_size"],
        title_fontsize=style["legend_size"],
        frameon=False,
        handletextpad=0.5,
    )
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    set_plot_limits_3d(ax, pca_df_3d)
    fig.subplots_adjust(right=0.72, bottom=0.08)
    fig.savefig(save_path, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.3, facecolor="white")
    fig.savefig(Path(save_path).with_suffix(".svg"), bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)


def permanova(distance_matrix, groups, permutations=1000):
    """PERMANOVA implementation."""
    group_labels = np.array(groups)
    unique_groups = np.unique(group_labels)
    n = len(group_labels)
    num_groups = len(unique_groups)

    total_ss = np.sum(distance_matrix ** 2) / (2 * n)
    within_ss = 0
    for g in unique_groups:
        idx = np.where(group_labels == g)[0]
        sub = distance_matrix[np.ix_(idx, idx)]
        within_ss += np.sum(sub ** 2) / (2 * len(idx))

    between_ss = total_ss - within_ss
    df_between = num_groups - 1
    df_within = n - num_groups
    F_stat = (between_ss / df_between) / (within_ss / df_within) if df_within > 0 else 0

    permuted_F = []
    for _ in range(permutations):
        permuted_labels = np.random.permutation(group_labels)
        wss_perm = 0
        for g in unique_groups:
            idx = np.where(permuted_labels == g)[0]
            sub = distance_matrix[np.ix_(idx, idx)]
            wss_perm += np.sum(sub ** 2) / (2 * len(idx))
        bss_perm = total_ss - wss_perm
        permuted_F.append((bss_perm / df_between) / (wss_perm / df_within))

    p_value = (np.sum(np.array(permuted_F) >= F_stat) + 1.0) / (permutations + 1.0)
    return F_stat, p_value


def detect_outliers_mahalanobis(scores, sample_names, save_dir, alpha=0.05):
    cov = np.cov(scores.T)
    inv_cov = np.linalg.inv(cov)
    mean_centered = scores - np.mean(scores, axis=0)
    m_dist = np.sqrt(np.diag(mean_centered @ inv_cov @ mean_centered.T))
    threshold = np.sqrt(chi2.ppf(1 - alpha, df=scores.shape[1]))
    outliers = m_dist > threshold

    df_out = pd.DataFrame({
        "Sample": sample_names,
        "Mahalanobis_Distance": m_dist,
        "Threshold": threshold,
        "Is_Outlier": outliers
    })
    df_out.to_csv(save_dir / "Mahalanobis_Distances.csv", index=False, encoding="utf-8-sig")
    return outliers


# ==========================================================
# PCA main function
# ==========================================================
def run_pca(file_path, group_file, save_dir, group_colors=None, group_order=None, dpi: int = 100, publication_theme: bool = False):
    print(f"[PCA] Running advanced PCA for: {file_path.name}", flush = True)
    plt.close('all')
    style = get_figure_style(publication_theme=publication_theme, dpi=dpi)
    
    if group_file is not None and os.path.exists(group_file):
        df_groups = pd.read_csv(group_file)
        qc_samples = df_groups.loc[df_groups["Group"].str.upper() == "QC", "Sample"].tolist()
    else:
        qc_samples = []

    # Load dataset (from standardized format)
    X, y, feature_meta = load_dataset(file_path, group_file)
    save_dir = prepare_output_dir(save_dir)

    # Auto-scale
    X_scaled = StandardScaler().fit_transform(X)
    distance_matrix = squareform(pdist(X_scaled, metric="euclidean"))

    # Perform PCA
    n_pcs = min(3, X_scaled.shape[0], X_scaled.shape[1])
    if n_pcs < 2:
        raise ValueError("PCA requires at least 2 samples/features to compute scores plots.")
    pca = PCA(n_components=n_pcs)
    scores_all = pca.fit_transform(X_scaled)
    scores = scores_all[:, :2]

    # PERMANOVA + silhouette
    F_stat, p_value = permanova(distance_matrix, y, permutations=1000)
    silhouette = silhouette_score(scores, y, metric="euclidean")
    print(f"[PCA] PERMANOVA F={F_stat:.3f}, p={p_value:.4g}, silhouette={silhouette:.3f}", flush = True)

    # Prepare DataFrame for plotting
    pca_df = pd.DataFrame(scores, columns=["PC1", "PC2"], index=X.index)
    pca_df["Group"] = y.values

    pca_df_3d = None
    if n_pcs >= 3:
        pca_df_3d = pd.DataFrame(scores_all[:, :3], columns=["PC1", "PC2", "PC3"], index=X.index)
        pca_df_3d["Group"] = y.values

    # Save PCA scores
    pca_df.to_csv(save_dir / "PCA_scores.csv", index=True, encoding="utf-8-sig")
    if pca_df_3d is not None:
        pca_df_3d.to_csv(save_dir / "PCA_scores_3D.csv", index=True, encoding="utf-8-sig")

    # === Plot PCA ===
    plt.figure(figsize=(9, 6))
    existing = list(pca_df["Group"].astype(str).unique())
    if group_order:
        groups_order = [g for g in group_order if g in existing] + [g for g in existing if g not in group_order]
    else:
        groups_order = existing
    color_map = make_distinct_palette(groups_order, group_colors=group_colors)

    ax = sns.scatterplot(
        data=pca_df,
        x="PC1", y="PC2",
        hue="Group",
        hue_order=groups_order,
        palette=color_map,          # dict palette
        s=90, alpha=0.95, edgecolor="black"
    )
    fig = ax.get_figure()

    # Ellipses use the same colors
    ellipses = []
    for group in groups_order:
        data_g = pca_df.loc[pca_df["Group"] == group, ["PC1", "PC2"]]
        if len(data_g) < 3:
            continue
        cov = np.cov(data_g.T)
        center = data_g.mean().values
        color = color_map.get(group, (0.5, 0.5, 0.5))
        try:
            ellipse = get_cov_ellipse(cov, center, nstd=1.96,
                                    facecolor=color, alpha=0.18,
                                    edgecolor=color, linewidth=1)
            ax.add_patch(ellipse)
            ellipses.append(ellipse)
        except Exception:
            pass

    # Labels, title, legend
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", labelpad = 12, fontsize=style["label_size"])
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", labelpad = 12, fontsize=style["label_size"])
    plt.title("PCA Scores", fontsize=style["title_size"])
    legend = ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=style["legend_size"],
        title_fontsize=style["legend_size"],
        frameon=False,
        handletextpad=0.5,
    )
    # ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.8)
    set_plot_limits(ax, pca_df, ellipses=ellipses)

    # === Add bottom statistics text ===
    # (matches original LipidQuest PCA formatting)
    if p_value < 0.001:
        p_text = "< 0.001"
    else:
        p_text = f"= {p_value:.3g}"

    # Centered bottom statistics line (anchored to plot, not figure)
    bottom_text = (
        f"PERMANOVA (1000 perm.): F = {F_stat:.2f}, p {p_text} | "
        f"SILHOUETTE SCORE = {silhouette:.2f}"
    )

    ax.text(
        0.5, BOTTOM_STATS_Y, bottom_text,
        ha="center", va="top",
        transform=ax.transAxes,
        fontsize=style["tick_size"],
        color="dimgray"
    )

    # Add full rectangular border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style["line_width"])
        spine.set_color("black")

    # Save figure
    # plt.tight_layout()
    ax.set_aspect("auto")
    fig.subplots_adjust(right=0.84, bottom=0.24)  # ensures room in interactive view
    fig.savefig(save_dir / "PCA_2D.png", dpi=style["dpi"], bbox_inches="tight", pad_inches=0.3, facecolor="white")
    fig.savefig(save_dir / "PCA_2D.svg", bbox_inches="tight", pad_inches=0.3, facecolor="white")

    legend.remove()
    inside_loc, inside_anchor = choose_inside_legend_position(ax, pca_df, groups_order)
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
    fig.savefig(save_dir / "PCA_2D_legend_inside.png", dpi=style["dpi"], bbox_inches="tight", pad_inches=0.3, facecolor="white")
    fig.savefig(save_dir / "PCA_2D_legend_inside.svg", bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close()

    if pca_df_3d is not None:
        plot_pca_scores_3d(
            pca_df_3d,
            pca.explained_variance_ratio_,
            save_dir / "PCA_3D.png",
            "PCA Scores Plot (3D)",
            group_order=group_order,
            group_colors=group_colors,
            style=style,
        )

    # === PCA with sample labels ===
    plt.figure(figsize=(9, 6))
    existing = list(pca_df["Group"].astype(str).unique())
    if group_order:
        groups_order = [g for g in group_order if g in existing] + [g for g in existing if g not in group_order]
    else:
        groups_order = existing
    color_map = make_distinct_palette(groups_order, group_colors=group_colors)

    ax = sns.scatterplot(
        data=pca_df,
        x="PC1", y="PC2",
        hue="Group",
        hue_order=groups_order,
        palette=color_map,          # dict palette
        s=90, alpha=0.95, edgecolor="black"
    )

    # Ellipses use the same colors
    ellipses = []
    for group in groups_order:
        data_g = pca_df.loc[pca_df["Group"] == group, ["PC1", "PC2"]]
        if len(data_g) < 3:
            continue
        cov = np.cov(data_g.T)
        center = data_g.mean().values
        color = color_map.get(group, (0.5, 0.5, 0.5))
        try:
            ellipse = get_cov_ellipse(cov, center, nstd=1.96,
                                    facecolor=color, alpha=0.18,
                                    edgecolor=color, linewidth=1)
            ax.add_patch(ellipse)
            ellipses.append(ellipse)
        except Exception:
            pass

    # Add sample labels slightly offset from points
    for sample_name, row in pca_df.iterrows():
        ax.text(
            row["PC1"] + 0.4,  # small horizontal offset
            row["PC2"] + 0.4,  # small vertical offset
            str(sample_name),
            fontsize=7,
            alpha=0.8,
            color="black",
            ha="left", va="bottom"
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", labelpad = 12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", labelpad = 12)
    plt.title("PCA Scores with Labels", fontsize=style["title_size"])
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=style["legend_size"],
        title_fontsize=style["legend_size"],
        frameon=False,
        handletextpad=0.5,
    )
    # ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.8)
    set_plot_limits(ax, pca_df, ellipses=ellipses)

    # Bottom stats line (same format)
    if p_value < 0.001:
        p_text = "< 0.001"
    else:
        p_text = f"= {p_value:.3g}"

    bottom_text = (
        f"PERMANOVA (1000 perm.): F = {F_stat:.2f}, p {p_text} | "
        f"SILHOUETTE SCORE = {silhouette:.2f}"
    )

    ax.text(
        0.5, BOTTOM_STATS_Y, bottom_text,
        ha="center", va="top",
        transform=ax.transAxes,
        fontsize=style["tick_size"],
        color="dimgray"
    )
    
    # Add full rectangular border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style["line_width"])
        spine.set_color("black")
        
    # plt.tight_layout()
    ax.set_aspect("auto")
    plt.gcf().subplots_adjust(right=0.84, bottom=0.24)
    plt.savefig(save_dir / "PCA_2D_with_labels.png", dpi=style["dpi"], bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.savefig(save_dir / "PCA_2D_with_labels.svg", bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close()

    # ==============================================================================
    #  LOADINGS
    # ==============================================================================

    loadings = pd.DataFrame(
        pca.components_[:2].T,
        index=X.columns,                # already UniqueIDs from load_dataset()
        columns=["PC1", "PC2"]
    )
    loadings.index.name = "UniqueID"

    # Reset index so UniqueID becomes a column before merging
    loadings_reset = loadings.reset_index()
    loadings_reset["UniqueID"] = loadings_reset["UniqueID"].astype(str)

    # Ensure feature_meta also uses string-based UniqueID
    feature_meta_reset = feature_meta.reset_index() if "UniqueID" in feature_meta.index.names else feature_meta
    feature_meta_reset["UniqueID"] = feature_meta_reset["UniqueID"].astype(str)

    # Merge
    merged_loadings = loadings_reset.merge(
        feature_meta_reset,
        on="UniqueID",
        how="left"
    )

    # Save
    merged_loadings.to_csv(save_dir / "PCA_loadings_with_metadata.csv", index=False, encoding="utf-8-sig")
    loadings_reset.to_csv(save_dir / "PCA_loadings.csv", index=False, encoding="utf-8-sig")

    # === Loadings Plot ===
    plt.figure(figsize=(8, 7))
    ax = sns.scatterplot(x="PC1", y="PC2", data=loadings, s=35, alpha=0.8, edgecolor="none")

    # Add a few labels for top-contributing features
    # Label features with |loading| > threshold (say, 0.035)
    threshold = 0.035
    important = loadings[(loadings["PC1"].abs() > threshold) | (loadings["PC2"].abs() > threshold)]
    for uid, row in important.iterrows():
        plt.text(row["PC1"], row["PC2"], str(uid), fontsize=6, alpha=0.7)

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.6)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.6)
    plt.title("PCA Loadings Plot (Top 15 Features by Magnitude)")
    plt.xlabel("PC1", labelpad = 12)
    plt.ylabel("PC2", labelpad = 12)

    # Adjust axes dynamically for visibility
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    xpad = (xlims[1] - xlims[0]) * 0.1
    ypad = (ylims[1] - ylims[0]) * 0.1
    ax.set_xlim(xlims[0] - xpad, xlims[1] + xpad)
    ax.set_ylim(ylims[0] - ypad, ylims[1] + ypad)

    # Add full rectangular border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # plt.tight_layout()
    plt.savefig(save_dir / "PCA_Loadings.png", dpi=100, bbox_inches="tight", pad_inches=0.2)
    plt.close()


    # === Detect outliers ===
    outliers = detect_outliers_mahalanobis(scores, X.index, save_dir)
    if np.any(outliers):
        found = list(X.index[outliers])
        print(f"[PCA] Outliers detected: {found}", flush=True)
        with open(save_dir / "outliers.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(found))

        # ---------- Re-run PCA without outliers ----------
        X_no = X.loc[~outliers].copy()
        y_no = y.loc[~outliers].reset_index(drop=True)

        if len(X_no) >= 3 and y_no.nunique() >= 1:
            subdir = prepare_output_dir(Path(save_dir) / "Without_outliers")
            X_scaled_no = StandardScaler().fit_transform(X_no)
            dist_no = squareform(pdist(X_scaled_no, metric="euclidean"))

            n_pcs_no = min(3, X_scaled_no.shape[0], X_scaled_no.shape[1])
            if n_pcs_no < 2:
                raise ValueError("PCA without outliers requires at least 2 samples/features.")
            pca_no = PCA(n_components=n_pcs_no)
            scores_no_all = pca_no.fit_transform(X_scaled_no)
            scores_no = scores_no_all[:, :2]

            # stats for the filtered set
            F_no, p_no = permanova(dist_no, y_no, permutations=1000)
            sil_no = silhouette_score(scores_no, y_no, metric="euclidean")

            pca_df_no = pd.DataFrame(scores_no, columns=["PC1", "PC2"], index=X_no.index)
            pca_df_no["Group"] = y_no.values
            pca_df_no.to_csv(subdir / "PCA_scores.csv", index=True, encoding="utf-8-sig")
            pca_df_no_3d = None
            if n_pcs_no >= 3:
                pca_df_no_3d = pd.DataFrame(scores_no_all[:, :3], columns=["PC1", "PC2", "PC3"], index=X_no.index)
                pca_df_no_3d["Group"] = y_no.values
                pca_df_no_3d.to_csv(subdir / "PCA_scores_3D.csv", index=True, encoding="utf-8-sig")

            # ---- plot helper to avoid duplication ----
            def _plot_scores(df, df_3d, pca_model, save_path_png, save_path_labeled_png, stats_tuple):
                Fv, pv, silv = stats_tuple
                existing = list(df["Group"].astype(str).unique())
                if group_order:
                    order = [g for g in group_order if g in existing] + [g for g in existing if g not in group_order]
                else:
                    order = existing
                cmap = make_distinct_palette(order, group_colors=group_colors)

                # plain
                fig, ax = plt.subplots(figsize=(9, 6))
                sns.scatterplot(
                    data=df, x="PC1", y="PC2", hue="Group",
                    hue_order=order, palette=cmap, s=90, alpha=0.95, edgecolor="black", ax=ax
                )
                ellipses = []
                for g in order:
                    dg = df.loc[df["Group"] == g, ["PC1", "PC2"]]
                    if len(dg) >= 3:
                        cov = np.cov(dg.T); ctr = dg.mean().values; col = cmap.get(g, (0.5,0.5,0.5))
                        try:
                            ell = get_cov_ellipse(cov, ctr, nstd=1.96, facecolor=col, alpha=0.18, edgecolor=col, linewidth=1)
                            ax.add_patch(ell)
                            ellipses.append(ell)
                        except Exception:
                            pass
                ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}% Variance)", labelpad=12)
                ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}% Variance)", labelpad=12)
                ax.set_title("PCA Scores Plot (No Outliers)", fontsize=14)
                ax.legend(
                    bbox_to_anchor=(1.02,1),
                    loc="upper left",
                    borderaxespad=0.0,
                    fontsize=12,
                    title_fontsize=12,
                    frameon=False,
                    handletextpad=0.5,
                )
                set_plot_limits(ax, df, ellipses=ellipses)
                p_txt = "< 0.001" if pv < 0.001 else f"= {pv:.3g}"
                ax.text(0.5, BOTTOM_STATS_Y, f"PERMANOVA (1000 perm.): F = {Fv:.2f}, p {p_txt} | SILHOUETTE SCORE = {silv:.2f}",
                        ha="center", va="top", transform=ax.transAxes, fontsize=12, color="dimgray")
                for s in ax.spines.values():
                    s.set_visible(True); s.set_linewidth(1.0); s.set_color("black")
                fig.subplots_adjust(bottom=0.24)
                fig.savefig(save_path_png, dpi=100, bbox_inches="tight", pad_inches=0.3, facecolor="white")
                plt.close(fig)

                if df_3d is not None:
                    plot_pca_scores_3d(
                        df_3d,
                        pca_model.explained_variance_ratio_,
                        save_path_png.parent / "PCA_3D.png",
                        "PCA Scores Plot (No Outliers, 3D)",
                        group_order=group_order,
                        group_colors=group_colors,
                    )

                # with labels
                fig, ax = plt.subplots(figsize=(9, 6))
                sns.scatterplot(
                    data=df, x="PC1", y="PC2", hue="Group",
                    hue_order=order, palette=cmap, s=90, alpha=0.95, edgecolor="black", ax=ax
                )
                ellipses = []
                for g in order:
                    dg = df.loc[df["Group"] == g, ["PC1", "PC2"]]
                    if len(dg) >= 3:
                        cov = np.cov(dg.T); ctr = dg.mean().values; col = cmap.get(g, (0.5,0.5,0.5))
                        try:
                            ell = get_cov_ellipse(cov, ctr, nstd=1.96, facecolor=col, alpha=0.18, edgecolor=col, linewidth=1)
                            ax.add_patch(ell)
                            ellipses.append(ell)
                        except Exception:
                            pass
                for sname, row in df.iterrows():
                    ax.text(row["PC1"]+0.4, row["PC2"]+0.4, str(sname), fontsize=7, alpha=0.8, color="black", ha="left", va="bottom")
                ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}% Variance)", labelpad=12)
                ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}% Variance)", labelpad=12)
                ax.set_title("PCA Scores Plot (No Outliers, With Sample Labels)", fontsize=14)
                ax.legend(
                    bbox_to_anchor=(1.02,1),
                    loc="upper left",
                    borderaxespad=0.0,
                    fontsize=12,
                    title_fontsize=12,
                    frameon=False,
                    handletextpad=0.5,
                )
                set_plot_limits(ax, df, ellipses=ellipses)
                ax.text(0.5, BOTTOM_STATS_Y, f"PERMANOVA (1000 perm.): F = {Fv:.2f}, p {p_txt} | SILHOUETTE SCORE = {silv:.2f}",
                        ha="center", va="top", transform=ax.transAxes, fontsize=12, color="dimgray")
                for s in ax.spines.values():
                    s.set_visible(True); s.set_linewidth(1.0); s.set_color("black")
                fig.subplots_adjust(bottom=0.24)
                fig.savefig(save_path_labeled_png, dpi=100, bbox_inches="tight", pad_inches=0.3, facecolor="white")
                plt.close(fig)

            _plot_scores(
                pca_df_no, pca_df_no_3d, pca_no,
                subdir / "PCA_2D.png",
                subdir / "PCA_2D_with_labels.png",
                (F_no, p_no, sil_no)
            )

            # Loadings (optional for filtered set; keep same format)
            load_no = pd.DataFrame(pca_no.components_[:2].T, index=X_no.columns, columns=["PC1", "PC2"])
            load_no.index.name = "UniqueID"
            load_no.reset_index().to_csv(subdir / "PCA_loadings.csv", index=False, encoding="utf-8-sig")
        else:
            print("[PCA] Outliers found but not enough samples remain to re-run PCA.", flush=True)
    else:
        print("[PCA] No significant outliers detected.", flush=True)

    print(f"[PCA] Completed. Results saved to: {save_dir}\n", flush=True)

