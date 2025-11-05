# TODO: plots are missing the F stats
# TODO: the plots without outliers are not being generated
#TODO: more colours!

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

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


# ==========================================================
# Helper functions
# ==========================================================
def make_distinct_palette(groups):
    """Return a dict {group: rgb} with enough distinct colors.
    - QC (any case) is always black and does not consume a color slot.
    - Supports many classes via tab palettes + HUSL fallback.
    """
    groups = [str(g) for g in groups]
    # preserve order of first appearance
    unique_in_order = list(dict.fromkeys(groups))

    # separate QC from others
    non_qc = [g for g in unique_in_order if g.lower() != "qc"]
    has_qc = any(g.lower() == "qc" for g in unique_in_order)

    n = len(non_qc)
    if n <= 10:
        base = sns.color_palette("tab10", n_colors=n)
    elif n <= 20:
        base = sns.color_palette("tab20", n_colors=n)
    elif n <= 32:
        base = (
            sns.color_palette("tab20", 20)
            + sns.color_palette("tab20b", 20)
            + sns.color_palette("tab20c", 20)
        )[:n]
    else:
        # reasonably distinct for large n
        base = sns.husl_palette(n, s=0.9, l=0.55)

    cmap = {g: base[i] for i, g in enumerate(non_qc)}
    if has_qc:
        # assign black to every QC label variant present
        for g in unique_in_order:
            if g.lower() == "qc":
                cmap[g] = "#000000"
    return cmap

def get_cov_ellipse(cov, center, nstd=1.96, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)


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

    p_value = np.sum(np.array(permuted_F) >= F_stat) / permutations
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
def run_pca(file_path, group_file, save_dir):
    print(f"[PCA] Running advanced PCA for: {file_path.name}", flush = True)

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
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)

    # PERMANOVA + silhouette
    F_stat, p_value = permanova(distance_matrix, y, permutations=1000)
    silhouette = silhouette_score(scores, y, metric="euclidean")
    print(f"[PCA] PERMANOVA F={F_stat:.3f}, p={p_value:.4g}, silhouette={silhouette:.3f}", flush = True)

    # Prepare DataFrame for plotting
    pca_df = pd.DataFrame(scores, columns=["PC1", "PC2"], index=X.index)
    pca_df["Group"] = y.values

    # Save PCA scores
    pca_df.to_csv(save_dir / "PCA_scores.csv", index=True, encoding="utf-8-sig")

    # === Plot PCA ===
    plt.figure(figsize=(9, 6))
    groups_order = list(pca_df["Group"].astype(str).unique())
    color_map = make_distinct_palette(groups_order)

    ax = sns.scatterplot(
        data=pca_df,
        x="PC1", y="PC2",
        hue="Group",
        hue_order=groups_order,
        palette=color_map,          # dict palette
        s=90, alpha=0.95, edgecolor="black"
    )

    # Ellipses use the same colors
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
        except Exception:
            pass

    # Labels, title, legend
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", labelpad = 12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", labelpad = 12)
    plt.title("PCA Scores Plot", fontsize=14)
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=14,
        title_fontsize=14
    )
    # ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.8)
    ax.set_aspect("equal", adjustable="datalim")

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
        0.5, -0.18, bottom_text,
        ha="center", va="top",
        transform=ax.transAxes,
        fontsize=12,
        color="dimgray"
    )

    # === Ensure all points and ellipses are fully visible ===
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # Expand limits by ~10% of the current range to fit ellipses and labels
    xpad = (xlims[1] - xlims[0]) * 0.1
    ypad = (ylims[1] - ylims[0]) * 0.1
    ax.set_xlim(xlims[0] - xpad, xlims[1] + xpad)
    ax.set_ylim(ylims[0] - ypad, ylims[1] + ypad)

    # Save figure
    plt.tight_layout()
    plt.savefig(save_dir / "PCA_2D.png", dpi=300, bbox_inches="tight")
    plt.close()

    # === PCA with sample labels ===
    plt.figure(figsize=(9, 6))
    groups_order = list(pca_df["Group"].astype(str).unique())
    color_map = make_distinct_palette(groups_order)

    ax = sns.scatterplot(
        data=pca_df,
        x="PC1", y="PC2",
        hue="Group",
        hue_order=groups_order,
        palette=color_map,          # dict palette
        s=90, alpha=0.95, edgecolor="black"
    )

    # Ellipses use the same colors
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
    plt.title("PCA Scores Plot (With Sample Labels)", fontsize=14)
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=14,
        title_fontsize=14
    )
    ax.set_aspect("equal", adjustable="datalim")

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
        0.5, -0.18, bottom_text,
        ha="center", va="top",
        transform=ax.transAxes,
        fontsize=12,
        color="dimgray"
    )
    
    # === Ensure all points and ellipses are fully visible ===
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # Expand limits by ~10% of the current range to fit ellipses and labels
    xpad = (xlims[1] - xlims[0]) * 0.1
    ypad = (ylims[1] - ylims[0]) * 0.1
    ax.set_xlim(xlims[0] - xpad, xlims[1] + xpad)
    ax.set_ylim(ylims[0] - ypad, ylims[1] + ypad)

    plt.tight_layout()
    plt.savefig(save_dir / "PCA_2D_with_labels.png", dpi=300, bbox_inches="tight")
    plt.close()

        # === Loadings ===
    loadings = pd.DataFrame(
        pca.components_.T,
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

    plt.tight_layout()
    plt.savefig(save_dir / "PCA_Loadings.png", dpi=300, bbox_inches="tight")
    plt.close()


    # === Detect outliers ===
    outliers = detect_outliers_mahalanobis(scores, X.index, save_dir)
    if np.any(outliers):
        print(f"[PCA] Outliers detected: {list(X.index[outliers])}", flush = True)
        with open(save_dir / "outliers.txt", "w") as f:
            f.write("\n".join(X.index[outliers]))
    else:
        print("[PCA] No significant outliers detected.", flush = True)

    print(f"[PCA] Completed. Results saved to: {save_dir}\n", flush = True)
