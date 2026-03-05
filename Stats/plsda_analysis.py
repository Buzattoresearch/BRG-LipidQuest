import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, accuracy_score
from matplotlib.patches import Ellipse
from Stats.utils import load_dataset, prepare_output_dir

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

import matplotlib as mpl
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
mpl.rcParams["mathtext.default"] = "regular" 

plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")

# ==========================================================
# Helper functions
# ==========================================================
def make_distinct_palette(groups):
    """Return a dict {group: rgb} with enough distinct colors.
    - QC (any case) is always black and does not consume a color slot.
    - Supports many classes via tab palettes + HUSL fallback.
    """
    groups = [str(g) for g in groups]
    unique_in_order = list(dict.fromkeys(groups))

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
        base = sns.husl_palette(n, s=0.9, l=0.55)

    cmap = {g: base[i] for i, g in enumerate(non_qc)}
    if has_qc:
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


def find_optimal_components(X, y):
    """Find optimal number of components based on CV R²."""
    max_components = min(X.shape[1], len(np.unique(y)) - 1)
    if max_components < 5:
        max_components = 5
    r2_scores = []
    for n in range(2, max_components + 1):
        pls = PLSRegression(n_components=n)
        score = np.mean(cross_val_score(pls, X, y, cv=5, scoring='r2'))
        r2_scores.append(score)
    best = np.argmax(r2_scores) + 2  # ensure minimum 2
    return best, r2_scores[best - 2]


def permutation_test(X, y, pls, n_permutations=100):
    """Permutation test for PLS-DA significance."""
    true_r2 = r2_score(y, pls.predict(X))
    perm_r2_scores = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        pls_perm = PLSRegression(n_components=pls.n_components)
        pls_perm.fit(X, y_perm)
        perm_r2 = r2_score(y_perm, pls_perm.predict(X))
        perm_r2_scores.append(perm_r2)
    p_value = np.sum(np.array(perm_r2_scores) >= true_r2) / n_permutations
    return true_r2, p_value


def calculate_q2_and_accuracy(pls, X, y, n_folds=5):
    """Compute Q² and mean accuracy using K-fold CV.
    - Binary: threshold at 0.5 like before.
    - Multiclass: classify by nearest group centroid in score space.
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    y_true_all, y_pred_cont_all = [], []
    accuracies = []
    n_classes = len(np.unique(y))

    for train_idx, test_idx in kf.split(X):
        pls_cv = PLSRegression(n_components=pls.n_components)
        pls_cv.fit(X[train_idx], y[train_idx])

        # Continuous predictions for Q2
        pred_cont = pls_cv.predict(X[test_idx]).ravel()
        y_true_fold = y[test_idx]

        # Classification for accuracy
        if n_classes == 2:
            y_pred_cls = (pred_cont > 0.5).astype(int)
        else:
            # Project both train and test into score space; use nearest centroid of training groups
            T_train = pls_cv.transform(X[train_idx])[:, :2]
            T_test  = pls_cv.transform(X[test_idx])[:, :2]
            centroids = {c: T_train[y[train_idx] == c].mean(axis=0) for c in np.unique(y)}
            # assign closest centroid
            y_pred_cls = np.array([
                min(centroids.keys(), key=lambda c: np.linalg.norm(t - centroids[c]))
                for t in T_test
            ])

        # Accuracy on class labels
        accuracies.append((y_pred_cls == y_true_fold).mean())

        # Accumulate for Q2
        y_true_all.append(y_true_fold)
        y_pred_cont_all.append(pred_cont)

    y_true_all = np.concatenate(y_true_all)
    y_pred_cont_all = np.concatenate(y_pred_cont_all)
    press = np.sum((y_true_all - y_pred_cont_all) ** 2)
    tss = np.sum((y_true_all - np.mean(y_true_all)) ** 2)
    q2 = 1 - (press / tss) if tss > 0 else np.nan
    return q2, float(np.mean(accuracies))

def calculate_vip(pls, X):
    """Calculate Variable Importance in Projection (VIP) scores."""
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p, h = w.shape
    ssq_y = np.sum(np.square(t @ q.T), axis=0)
    total_ssq_y = np.sum(ssq_y)
    vip_scores = np.zeros(p)
    for i in range(p):
        weight = np.square(w[i, :])
        vip_scores[i] = np.sqrt(p * np.sum(weight * ssq_y) / total_ssq_y)
    return vip_scores


def plot_vip(vip_scores,
             feature_names,
             save_dir,
             filename="VIP_scores_plot.png",
             top_n=None,
             annotations=None,
             clean_labels=False):
    """
    Creates horizontal VIP barplots identical to your original ones.
    - Bars = Viridis gradient with black borders
    - Y-axis = lipid annotations (not UniqueID)
    - Top VIPs at top
    - Auto scales figure height and saves PNG + SVG
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.close('all')
    plt.ioff()
    
    # Convert to DataFrame
    # Convert to DataFrame
    df = pd.DataFrame({"Feature": feature_names, "VIP": np.asarray(vip_scores, dtype=float)})

    # Replace y-labels with annotations if provided
    if annotations is not None and len(annotations) == len(df):
        df["Label"] = pd.Series(annotations, index=df.index).astype(str)
    else:
        df["Label"] = df["Feature"].astype(str)

    # Clean labels optionally
    if clean_labels:
        df["Label"] = df["Label"].str.replace(r"^\d+\|", "", regex=True)

    # Sort ↓ and truncate
    df = df.sort_values("VIP", ascending=False)
    if top_n:
        df = df.head(top_n)

    # === Dynamic but bounded sizing ===
    n = len(df)
    fig_height = min(10.0, max(4.0, 0.28 * n))   # cap height to avoid huge canvases
    fig_width  = 7.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height),
                        constrained_layout=True, facecolor="white")

    colors = sns.color_palette("viridis", n_colors=n)
    ypos = np.arange(len(df))
    ax.barh(ypos, df["VIP"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["Label"])

    if df["Label"].duplicated().any():
        df["Label"] = df["Label"] + df.groupby("Label").cumcount().add(1).astype(str).radd(" (") + ")"

    ax.invert_yaxis()

    ax.set_title("Variable Importance in Projection (VIP) Scores", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("VIP Scores", fontsize=12, labelpad=10)
    ax.set_ylabel("Variables", fontsize=12, labelpad=10)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

    # Add full rectangular border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    fig.savefig(save_dir / filename, dpi=100, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    fig.savefig(save_dir / filename.replace(".png", ".svg"), dpi=100, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)

# ==========================================================
# Main PLS-DA runner
# ==========================================================
def run_plsda(file_path, group_file, save_dir, group_colors=None, group_order=None):
    print(f"[PLS-DA] Running analysis for: {Path(file_path).name}", flush=True)

    # === Load standardized dataset ===
    X, y_labels, feature_meta = load_dataset(file_path, group_file)
    save_dir = prepare_output_dir(save_dir)

    # --- Encode group labels ---
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # --- Standardize data ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Determine optimal number of components ---
    optimal_components, r2_optimal = find_optimal_components(X_scaled, y)
    pls = PLSRegression(n_components=max(2, optimal_components))
    pls.fit(X_scaled, y)

    # --- Explained variance (for first two components) ---
    explained_variance = (np.var(pls.x_scores_, axis=0) / np.var(X_scaled, axis=0).sum()) * 100

    # --- Create score DataFrame ---
    scores = pls.transform(X_scaled)
    plsda_df = pd.DataFrame(scores[:, :2], columns=['Component 1', 'Component 2'], index=X.index)
    plsda_df['Group'] = y_labels.values
    # Save PLS-DA scores for downstream use
    plsda_df.to_csv(save_dir / "PLSDA_scores.csv", index=True, encoding="utf-8-sig")

    # --- Evaluate model performance ---
    r2_true, p_value = permutation_test(X_scaled, y, pls, n_permutations=100)
    q2_value, avg_acc = calculate_q2_and_accuracy(pls, X_scaled, y)

    # ==========================================================
    # SCORES PLOT
    # ==========================================================
    plt.figure(figsize=(9, 6))
    # Derive ordered groups (respect GUI order if provided)
    natural = list(pd.unique(y_labels.astype(str)))
    if group_order:
        groups = [g for g in group_order if g in natural] + [g for g in natural if g not in group_order]
    else:
        groups = natural

    # Colors: user palette takes precedence; fill gaps with distinct palette; QC forced to black
    if isinstance(group_colors, dict) and group_colors:
        color_map = {g: group_colors.get(g) for g in groups}
        # backfill any None with generated colors
        gen_map = make_distinct_palette(groups)
        for g in groups:
            if not color_map.get(g):
                color_map[g] = gen_map[g]
    else:
        color_map = make_distinct_palette(groups)

    ax = sns.scatterplot(
        data=plsda_df,
        x='Component 1', y='Component 2',
        hue='Group',
        hue_order=groups,
        palette=color_map,
        s=100, alpha=0.95, edgecolor='black'
    )


    # 95% confidence ellipses using the same colors
    for group in groups:
        color = color_map[group]
        subset = plsda_df.loc[plsda_df['Group'] == group, ['Component 1', 'Component 2']]
        if len(subset) < 3:
            continue
        cov = np.cov(subset.T)
        center = subset.mean().values
        try:
            ellipse = get_cov_ellipse(
                cov, center, nstd=1.96,
                facecolor=color, edgecolor=color, alpha=0.18, linewidth=1
            )
            ax.add_patch(ellipse)
        except Exception:
            pass

    # Expand limits slightly for clarity
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(xlims[0] - (xlims[1] - xlims[0]) * 0.1,
                xlims[1] + (xlims[1] - xlims[0]) * 0.1)
    ax.set_ylim(ylims[0] - (ylims[1] - ylims[0]) * 0.1,
                ylims[1] + (ylims[1] - ylims[0]) * 0.1)

    ev1 = explained_variance[0] if len(explained_variance) > 0 else 0.0
    ev2 = explained_variance[1] if len(explained_variance) > 1 else 0.0
    plt.xlabel(f"Component 1 ({ev1:.1f}% variance)", labelpad=12)
    plt.ylabel(f"Component 2 ({ev2:.1f}% variance)", labelpad=12)
    plt.title("PLS-DA Scores Plot", fontsize=14)
    # --- Legend (outside right, non-cropped on save) ---
    fig = plt.gcf()
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=12,
        title_fontsize=12,
        borderaxespad=0.0,
        frameon=False,
    )

    # --- Bottom stats line ---
    extra_text = (
        f"5-fold CV | Opt. comps: {optimal_components} | "
        f"R² = {r2_true:.3f} | Q² = {q2_value:.3f} | "
        f"Accuracy = {avg_acc:.3f} | p (100 perm) = {p_value:.3g}"
    )

    ax.text(
        0.5, -0.18, extra_text,
        ha="center", va="top",
        transform=ax.transAxes,
        fontsize=11, color="dimgray"
    )

    ax.set_aspect("equal", adjustable="datalim")

    # Add full rectangular border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # --- Adjust and save properly ---
    fig.subplots_adjust(right=0.80)
    fig.savefig(
        save_dir / "PLSDA_2D.png",
        dpi=100,
        bbox_inches="tight",
        pad_inches=0.3,
        facecolor="white",
    )
    plt.close(fig)


    # ==========================================================
    # VIP SCORES
    # ==========================================================
    vip_scores = calculate_vip(pls, X_scaled)

    # --- Map annotations to UniqueIDs ---
    if isinstance(feature_meta, pd.DataFrame) and "Annotation" in feature_meta.columns:
        feature_meta = feature_meta.reset_index() if "UniqueID" in feature_meta.index.names else feature_meta
        uid_to_annotation = dict(zip(
            feature_meta["UniqueID"].astype(str).str.strip(),
            feature_meta["Annotation"].astype(str).str.strip()
        ))
        annotations = [uid_to_annotation.get(str(uid), str(uid)) for uid in X.columns]
    else:
        annotations = [str(uid) for uid in X.columns]

    # --- Build DataFrame for VIPs ---
    vip_df = pd.DataFrame({
        "UniqueID": X.columns,
        "VIP": vip_scores,
        "Annotation": annotations
    })

    # Merge with full metadata for traceability (robust to index/column types)
    if isinstance(feature_meta, pd.DataFrame):
        # Ensure UniqueID exists as a column
        if "UniqueID" not in feature_meta.columns:
            feature_meta = feature_meta.reset_index()

        # Convert both to string for safe merging
        vip_df["UniqueID"] = vip_df["UniqueID"].astype(str)
        feature_meta["UniqueID"] = feature_meta["UniqueID"].astype(str)

        vip_with_meta = vip_df.merge(feature_meta, on="UniqueID", how="left")
    else:
        vip_with_meta = vip_df.copy()

    # Save VIPs (sorted) with full metadata, plus a minimal table
    vip_with_meta_sorted = vip_with_meta.sort_values("VIP", ascending=False)
    vip_with_meta_sorted.to_csv(save_dir / "PLSDA_VIP_with_metadata.csv", index=False, encoding="utf-8-sig")
    vip_df.sort_values("VIP", ascending=False).to_csv(save_dir / "PLSDA_VIP.csv", index=False, encoding="utf-8-sig")

    # ==========================================================
    # VIP PLOTS
    # ==========================================================
    # plot_vip(vip_scores, X.columns, save_dir,                # Plot VIP scores for all  (not recommended for untargeted analysis)
    #          filename="VIP_scores_plot.png",
    #          top_n=None,
    #          annotations=annotations)

    for topn in [5, 10, 15, 20, 25, 30, 40, 50]:
        plot_vip(vip_scores, X.columns, save_dir,
                 filename=f"VIP_scores_plot_top{topn}.png",
                 top_n=topn,
                 annotations=annotations)

    # === X-loadings with metadata (traceability) ===
    x_load = pd.DataFrame(pls.x_loadings_[:, :2], index=X.columns, columns=["Comp1_loading", "Comp2_loading"])
    x_load.index.name = "UniqueID"
    xl = x_load.reset_index()
    xl["UniqueID"] = xl["UniqueID"].astype(str)

    fm = feature_meta.reset_index() if "UniqueID" in getattr(feature_meta, "index", pd.Index([])).names else feature_meta
    if isinstance(fm, pd.DataFrame) and "UniqueID" in fm.columns:
        fm = fm.copy()
        fm["UniqueID"] = fm["UniqueID"].astype(str)
        xl = xl.merge(fm, on="UniqueID", how="left")

    xl.to_csv(save_dir / "PLSDA_Xloadings_with_metadata.csv", index=False, encoding="utf-8-sig")

    print(f"[PLS-DA] Completed successfully → {save_dir}\n", flush = True)

