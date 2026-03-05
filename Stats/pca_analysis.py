
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
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

import matplotlib as mpl
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
mpl.rcParams["mathtext.default"] = "regular" 

plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")
plt.ioff()

# ==========================================================
# Helper functions
# ==========================================================
def make_distinct_palette(groups, group_colors=None):
    """Return a dict {group: color}.
    - Uses user-specified hex colors when provided in group_colors.
    - QC stays black unless overridden explicitly in group_colors.
    - Falls back to rcParams cycle, then tab palettes/HUSL for coverage.
    """
    groups = [str(g) for g in groups]
    unique_in_order = list(dict.fromkeys(groups))

    # separate QC
    non_qc = [g for g in unique_in_order if g.lower() != "qc"]
    qc_labels = [g for g in unique_in_order if g.lower() == "qc"]

    # base cycle first
    base_cycle = plt.rcParams.get("axes.prop_cycle", None)
    base = []
    if base_cycle:
        base = base_cycle.by_key().get("color", [])

    n = len(non_qc)
    if len(base) < n:
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
            base = sns.husl_palette(n, s=0.9, l=0.55)

    cmap = {}
    # non-QC first
    for i, g in enumerate(non_qc):
        if group_colors and group_colors.get(g):
            cmap[g] = group_colors[g]
        else:
            cmap[g] = base[i % len(base)] if len(base) else "#1f77b4"

    # QC last (black by default unless user overrides)
    for g in qc_labels:
        if group_colors and group_colors.get(g):
            cmap[g] = group_colors[g]
        else:
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
def run_pca(file_path, group_file, save_dir, group_colors=None, group_order=None):
    print(f"[PCA] Running advanced PCA for: {file_path.name}", flush = True)
    plt.close('all')
    
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
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=12,
        title_fontsize=12,
        frameon=False
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

    ax.set_aspect("equal", adjustable="datalim")

    # Add full rectangular border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # Save figure
    # plt.tight_layout()
    fig.subplots_adjust(right=0.80)  # ensures room in interactive view
    fig.savefig(save_dir / "PCA_2D.png", dpi=100, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close()

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
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=12,
        title_fontsize=12,
        frameon=False
    )
    # ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.8)
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

    ax.set_aspect("equal", adjustable="datalim")

    # Add full rectangular border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")
        
    # plt.tight_layout()
    plt.savefig(save_dir / "PCA_2D_with_labels.png", dpi=100, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close()

    # ==============================================================================
    #  LOADINGS
    # ==============================================================================

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

            pca_no = PCA(n_components=2)
            scores_no = pca_no.fit_transform(X_scaled_no)

            # stats for the filtered set
            F_no, p_no = permanova(dist_no, y_no, permutations=1000)
            sil_no = silhouette_score(scores_no, y_no, metric="euclidean")

            pca_df_no = pd.DataFrame(scores_no, columns=["PC1", "PC2"], index=X_no.index)
            pca_df_no["Group"] = y_no.values
            pca_df_no.to_csv(subdir / "PCA_scores.csv", index=True, encoding="utf-8-sig")

            # ---- plot helper to avoid duplication ----
            def _plot_scores(df, pca_model, save_path_png, save_path_labeled_png, stats_tuple):
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
                for g in order:
                    dg = df.loc[df["Group"] == g, ["PC1", "PC2"]]
                    if len(dg) >= 3:
                        cov = np.cov(dg.T); ctr = dg.mean().values; col = cmap.get(g, (0.5,0.5,0.5))
                        try:
                            ell = get_cov_ellipse(cov, ctr, nstd=1.96, facecolor=col, alpha=0.18, edgecolor=col, linewidth=1)
                            ax.add_patch(ell)
                        except Exception:
                            pass
                ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}% Variance)", labelpad=12)
                ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}% Variance)", labelpad=12)
                ax.set_title("PCA Scores Plot (No Outliers)", fontsize=14)
                ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0.0, fontsize=12, title_fontsize=12, frameon=False)
                ax.set_aspect("equal", adjustable="datalim")
                p_txt = "< 0.001" if pv < 0.001 else f"= {pv:.3g}"
                ax.text(0.5, -0.18, f"PERMANOVA (1000 perm.): F = {Fv:.2f}, p {p_txt} | SILHOUETTE SCORE = {silv:.2f}",
                        ha="center", va="top", transform=ax.transAxes, fontsize=12, color="dimgray")
                # pads and border
                xlims = ax.get_xlim(); ylims = ax.get_ylim()
                xpad = (xlims[1]-xlims[0]) * 0.1; ypad = (ylims[1]-ylims[0]) * 0.1
                ax.set_xlim(xlims[0]-xpad, xlims[1]+xpad); ax.set_ylim(ylims[0]-ypad, ylims[1]+ypad)
                for s in ax.spines.values():
                    s.set_visible(True); s.set_linewidth(1.0); s.set_color("black")
                fig.savefig(save_path_png, dpi=100, bbox_inches="tight", pad_inches=0.3, facecolor="white")
                plt.close(fig)

                # with labels
                fig, ax = plt.subplots(figsize=(9, 6))
                sns.scatterplot(
                    data=df, x="PC1", y="PC2", hue="Group",
                    hue_order=order, palette=cmap, s=90, alpha=0.95, edgecolor="black", ax=ax
                )
                for g in order:
                    dg = df.loc[df["Group"] == g, ["PC1", "PC2"]]
                    if len(dg) >= 3:
                        cov = np.cov(dg.T); ctr = dg.mean().values; col = cmap.get(g, (0.5,0.5,0.5))
                        try:
                            ell = get_cov_ellipse(cov, ctr, nstd=1.96, facecolor=col, alpha=0.18, edgecolor=col, linewidth=1)
                            ax.add_patch(ell)
                        except Exception:
                            pass
                for sname, row in df.iterrows():
                    ax.text(row["PC1"]+0.4, row["PC2"]+0.4, str(sname), fontsize=7, alpha=0.8, color="black", ha="left", va="bottom")
                ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}% Variance)", labelpad=12)
                ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}% Variance)", labelpad=12)
                ax.set_title("PCA Scores Plot (No Outliers, With Sample Labels)", fontsize=14)
                ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0.0, fontsize=12, title_fontsize=12, frameon=False)
                ax.set_aspect("equal", adjustable="datalim")
                ax.text(0.5, -0.18, f"PERMANOVA (1000 perm.): F = {Fv:.2f}, p {p_txt} | SILHOUETTE SCORE = {silv:.2f}",
                        ha="center", va="top", transform=ax.transAxes, fontsize=12, color="dimgray")
                xlims = ax.get_xlim(); ylims = ax.get_ylim()
                xpad = (xlims[1]-xlims[0]) * 0.1; ypad = (ylims[1]-ylims[0]) * 0.1
                ax.set_xlim(xlims[0]-xpad, xlims[1]+xpad); ax.set_ylim(ylims[0]-ypad, ylims[1]+ypad)
                for s in ax.spines.values():
                    s.set_visible(True); s.set_linewidth(1.0); s.set_color("black")
                fig.savefig(save_path_labeled_png, dpi=100, bbox_inches="tight", pad_inches=0.3, facecolor="white")
                plt.close(fig)

            _plot_scores(
                pca_df_no, pca_no,
                subdir / "PCA_2D.png",
                subdir / "PCA_2D_with_labels.png",
                (F_no, p_no, sil_no)
            )

            # Loadings (optional for filtered set; keep same format)
            load_no = pd.DataFrame(pca_no.components_.T, index=X_no.columns, columns=["PC1", "PC2"])
            load_no.index.name = "UniqueID"
            load_no.reset_index().to_csv(subdir / "PCA_loadings.csv", index=False, encoding="utf-8-sig")
        else:
            print("[PCA] Outliers found but not enough samples remain to re-run PCA.", flush=True)
    else:
        print("[PCA] No significant outliers detected.", flush=True)

    print(f"[PCA] Completed. Results saved to: {save_dir}\n", flush=True)

