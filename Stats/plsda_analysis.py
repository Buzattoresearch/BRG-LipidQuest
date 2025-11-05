import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, accuracy_score
from matplotlib.patches import Ellipse
from Stats.utils import load_dataset, prepare_output_dir

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


# ==========================================================
# Helper functions
# ==========================================================
def make_distinct_palette(groups):
    """Return a dict {group: rgb} with enough distinct colors."""
    groups = list(groups)
    n = len(groups)

    if n <= 10:
        base = sns.color_palette("tab10", n_colors=n)
    elif n <= 20:
        base = sns.color_palette("tab20", n_colors=n)
    elif n <= 32:
        # tab20 + tab20b + tab20c concatenated
        base = (
            sns.color_palette("tab20", 20)
            + sns.color_palette("tab20b", 20)[:6]
            + sns.color_palette("tab20c", 20)[:6]
        )
        base = base[:n]
    else:
        # Arbitrary many, still reasonably distinct
        base = sns.husl_palette(n, s=.9, l=.55)

    return {g: base[i] for i, g in enumerate(groups)}
        
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
    """Compute Q² and mean accuracy using K-fold CV."""
    kf = KFold(n_splits=n_folds, shuffle=True)
    y_true, y_pred_cont = [], []
    accuracies = []

    for train_idx, test_idx in kf.split(X):
        pls_cv = PLSRegression(n_components=pls.n_components)
        pls_cv.fit(X[train_idx], y[train_idx])
        pred = pls_cv.predict(X[test_idx]).ravel()
        y_true.append(y[test_idx])
        y_pred_cont.append(pred)
        y_pred_class = np.where(pred > 0.5, 1, 0)
        accuracies.append(accuracy_score(y[test_idx], y_pred_class))

    y_true = np.concatenate(y_true)
    y_pred_cont = np.concatenate(y_pred_cont)
    press = np.sum((y_true - y_pred_cont) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    q2 = 1 - (press / tss)
    return q2, np.mean(accuracies)


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

    # Convert to DataFrame
    df = pd.DataFrame({
        "Feature": feature_names,
        "VIP": np.asarray(vip_scores, dtype=float)
    })

    # Replace y-labels with annotations if provided
    if annotations is not None and len(annotations) == len(df):
        df["Label"] = annotations
    else:
        df["Label"] = df["Feature"]

    # Optionally remove numeric prefixes
    if clean_labels:
        df["Label"] = df["Label"].str.replace(r"^\d+\|", "", regex=True)

    # Sort descending and truncate
    df = df.sort_values("VIP", ascending=False)
    if top_n:
        df = df.head(top_n)

    # Dynamic figure height (0.4 inch per variable, min 4, max 20)
    n = len(df)
    fig_height = min(20, max(4, 0.4 * n))

    # Plot
    plt.figure(figsize=(7, fig_height))
    colors = sns.color_palette("viridis", n_colors=n)
    bars = plt.barh(df["Label"], df["VIP"], color=colors, edgecolor="black")

    # Flip Y so highest VIP is at top
    plt.gca().invert_yaxis()

    # Aesthetics
    plt.title("Variable Importance in Projection (VIP) Scores", fontsize=14, pad=15, weight="bold")
    plt.xlabel("VIP Scores", fontsize=14, labelpad = 12)
    plt.ylabel("Variables", fontsize=14, labelpad = 12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    
    # Adjust layout and add padding to avoid cropping the title
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # leaves 3% space at top for title
    plt.savefig(save_dir / filename, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(save_dir / filename.replace(".png", ".svg"), dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()

# ==========================================================
# Main PLS-DA runner
# ==========================================================
def run_plsda(file_path, group_file, save_dir):
    print(f"[PLS-DA] Running analysis for: {file_path.name}", flush = True)

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

    # --- Evaluate model performance ---
    r2_true, p_value = permutation_test(X_scaled, y, pls, n_permutations=100)
    q2_value, avg_acc = calculate_q2_and_accuracy(pls, X_scaled, y)

    # ==========================================================
    # SCORES PLOT
    # ==========================================================
    plt.figure(figsize=(9, 6))
    groups = list(pd.unique(y_labels))  # stable order
    color_map = make_distinct_palette(groups)

    ax = sns.scatterplot(
        data=plsda_df,
        x='Component 1', y='Component 2',
        hue='Group',
        palette=color_map,   # dict → stable colors even with many groups
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

    plt.xlabel(f"Component 1 ({explained_variance[0]:.1f}% variance)", labelpad = 12)
    plt.ylabel(f"Component 2 ({explained_variance[1]:.1f}% variance)", labelpad = 12)
    plt.title("PLS-DA Scores Plot", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
              fontsize=14, title_fontsize=14, borderaxespad=0)

    # Add bottom line with stats
    extra_text = (
        f"5-fold CV | Opt. comps: {optimal_components} | "
        f"R² = {r2_true:.3f} | Q² = {q2_value:.3f} | "
        f"Accuracy = {avg_acc:.3f} | p (100 perm) = {p_value:.3f}"
    )
    # Place text centered under the plot area (not tied to figure coords)
    ax.text(
        0.5, -0.18, extra_text,
        ha="center", va="top",
        transform=ax.transAxes,
        fontsize=11, color="dimgray"
    )

    plt.tight_layout()
    plt.savefig(save_dir / "PLSDA_2D.png", dpi=300, bbox_inches="tight")
    plt.close()

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

    print(f"[PLS-DA] Completed successfully → {save_dir}\n", flush = True)

