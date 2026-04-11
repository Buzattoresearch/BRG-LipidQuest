import warnings
from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

RF_RANDOM_STATE = 42


def _build_feature_label_map(feature_meta: pd.DataFrame) -> dict[str, str]:
    if not isinstance(feature_meta, pd.DataFrame) or feature_meta.empty:
        return {}
    fm = feature_meta.reset_index() if "UniqueID" in getattr(feature_meta, "index", pd.Index([])).names else feature_meta.copy()
    if "UniqueID" not in fm.columns:
        return {}
    annotation_col = "Annotation" if "Annotation" in fm.columns else None
    if annotation_col is None:
        return {}
    fm["UniqueID"] = fm["UniqueID"].astype(str).str.strip()
    fm[annotation_col] = fm[annotation_col].astype(str).str.strip()
    fm = fm.drop_duplicates("UniqueID")
    label_map = {}
    for _, row in fm.iterrows():
        unique_id = row["UniqueID"]
        annotation = row[annotation_col]
        if annotation and annotation.lower() not in {"nan", "none", "<na>"}:
            label_map[unique_id] = f"{annotation} | {unique_id}"
        else:
            label_map[unique_id] = unique_id
    return label_map


def _safe_cv_splits(y_encoded: np.ndarray, preferred: int = 5) -> int:
    _, counts = np.unique(y_encoded, return_counts=True)
    min_class = int(counts.min()) if len(counts) else 0
    if min_class < 2:
        raise ValueError("Random forest requires at least 2 samples in every group for stratified cross-validation.")
    return max(2, min(preferred, min_class))


def _permutation_accuracy_test(X: np.ndarray, y: np.ndarray, cv, n_estimators: int, n_permutations: int = 100):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RF_RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict", n_jobs=1)
    true_acc = accuracy_score(y, y_pred)

    perm_scores = []
    rng = np.random.default_rng(RF_RANDOM_STATE)
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_pred = cross_val_predict(model, X, y_perm, cv=cv, method="predict", n_jobs=1)
        perm_scores.append(accuracy_score(y_perm, perm_pred))
    p_value = (np.sum(np.array(perm_scores) >= true_acc) + 1.0) / (n_permutations + 1.0)
    return true_acc, p_value, y_pred


def _build_interpretation(accuracy: float, f1_macro: float, p_value: float) -> str:
    if p_value <= 0.05:
        p_line = "Permutation testing suggests the model performs better than expected from random relabeling."
    else:
        p_line = "Permutation testing does not show strong evidence that the model outperforms random relabeling."

    if accuracy >= 0.8:
        acc_line = "Cross-validated classification accuracy is high for this dataset."
    elif accuracy >= 0.6:
        acc_line = "Cross-validated classification accuracy is moderate and may still be useful for ranking informative features."
    else:
        acc_line = "Cross-validated classification accuracy is limited, so any group-separation claims should be treated cautiously."

    text = f"""
Random forest interpretation
----------------------------
This analysis trains a random forest classifier on the standardized lipid feature matrix and evaluates predictions by stratified cross-validation.

How to read the outputs
-----------------------
- The confusion matrix summarizes how often each group was predicted correctly or confused with another group.
- The ranked feature importance table highlights variables that contributed most to splits in the fitted forest.
- {acc_line}
- Macro F1 complements accuracy by balancing performance across groups of different sizes.
- {p_line}

Important limitations
---------------------
- Feature importance is model-based and does not imply causality.
- Correlated lipid features can split importance across several related variables or make rankings unstable.
- Small sample sizes can make cross-validated estimates noisy, especially for minority groups.
- A model with good predictive performance can still reflect technical structure, batch effects, or preprocessing artifacts rather than biology.
"""
    return textwrap.dedent(text).strip() + "\n"


def run_random_forest(
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
    print(f"[Random Forest] Running analysis for: {file_path.name}", flush=True)
    plt.close("all")

    X, y_labels, feature_meta = load_dataset(file_path, group_file)
    if X.empty or y_labels.empty:
        raise ValueError("Random forest could not load a valid dataset.")
    if y_labels.astype(str).nunique() < 2:
        raise ValueError("Random forest requires at least 2 groups.")
    if X.shape[0] < 4:
        raise ValueError("Random forest requires at least 4 matched samples.")

    le = LabelEncoder()
    y = le.fit_transform(y_labels.astype(str))
    cv_splits = _safe_cv_splits(y, preferred=5)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RF_RANDOM_STATE)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_estimators = 500
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RF_RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_scaled, y)

    cv_accuracy, perm_p_value, y_pred = _permutation_accuracy_test(
        X_scaled,
        y,
        cv=cv,
        n_estimators=n_estimators,
        n_permutations=100,
    )
    macro_f1 = f1_score(y, y_pred, average="macro")
    cm = confusion_matrix(y, y_pred, labels=np.arange(len(le.classes_)))

    distance_matrix = squareform(pdist(X_scaled, metric="euclidean"))
    F_stat, permanova_p = permanova(distance_matrix, y_labels, permutations=1000)

    pred_labels = le.inverse_transform(y_pred)
    pred_df = pd.DataFrame(
        {
            "Sample": X.index.astype(str),
            "True_Group": y_labels.astype(str).values,
            "Predicted_Group": pred_labels,
            "Correct": y_labels.astype(str).values == pred_labels,
        }
    )
    pred_df.to_csv(save_dir / "RandomForest_cross_validated_predictions.csv", index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(
        [
            {"Metric": "cross_validated_accuracy", "Value": cv_accuracy},
            {"Metric": "macro_f1", "Value": macro_f1},
            {"Metric": "permutation_p_value", "Value": perm_p_value},
            {"Metric": "permanova_F", "Value": F_stat},
            {"Metric": "permanova_p_value", "Value": permanova_p},
            {"Metric": "n_estimators", "Value": n_estimators},
            {"Metric": "cv_splits", "Value": cv_splits},
            {"Metric": "n_samples", "Value": X.shape[0]},
            {"Metric": "n_features", "Value": X.shape[1]},
        ]
    )
    metrics_df.to_csv(save_dir / "RandomForest_model_metrics.csv", index=False, encoding="utf-8-sig")

    feature_labels = _build_feature_label_map(feature_meta)
    importance_df = pd.DataFrame(
        {
            "UniqueID": X.columns.astype(str),
            "Importance": rf.feature_importances_,
        }
    )
    importance_df["Annotation_Label"] = importance_df["UniqueID"].map(feature_labels).fillna(importance_df["UniqueID"])
    if isinstance(feature_meta, pd.DataFrame) and not feature_meta.empty:
        fm = feature_meta.reset_index() if "UniqueID" in getattr(feature_meta, "index", pd.Index([])).names else feature_meta.copy()
        if "UniqueID" in fm.columns:
            fm["UniqueID"] = fm["UniqueID"].astype(str)
            importance_df["UniqueID"] = importance_df["UniqueID"].astype(str)
            importance_df = importance_df.merge(fm, on="UniqueID", how="left")
    importance_df = importance_df.sort_values("Importance", ascending=False)
    importance_df.to_csv(save_dir / "RandomForest_feature_importance.csv", index=False, encoding="utf-8-sig")

    (save_dir / "RandomForest_interpretation.txt").write_text(
        _build_interpretation(cv_accuracy, macro_f1, perm_p_value),
        encoding="utf-8",
    )

    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Random Forest Confusion Matrix", fontsize=style["title_size"])
    ax.set_xlabel("Predicted Group", fontsize=style["label_size"], labelpad=10)
    ax.set_ylabel("True Group", fontsize=style["label_size"], labelpad=10)
    stats_text = (
        f"CV accuracy = {cv_accuracy:.2f} | Macro F1 = {macro_f1:.2f} | "
        f"Permutation p {'< 0.001' if perm_p_value < 0.001 else f'= {perm_p_value:.3g}'}"
    )
    ax.text(0.5, -0.18, stats_text, ha="center", va="top", transform=ax.transAxes, fontsize=max(style["tick_size"] - 1, 10), color="dimgray")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style["line_width"])
        spine.set_color("black")
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(save_dir / "RandomForest_confusion_matrix.png", dpi=style["dpi"], bbox_inches="tight", pad_inches=0.25, facecolor="white")
    fig.savefig(save_dir / "RandomForest_confusion_matrix.svg", bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)

    top_n = min(20, len(importance_df))
    top_df = importance_df.head(top_n).iloc[::-1]
    fig_height = max(4.5, 0.33 * top_n)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    ax.barh(top_df["Annotation_Label"], top_df["Importance"], color="#1B6CA8", edgecolor="black", linewidth=0.6)
    ax.set_title("Random Forest Feature Importance", fontsize=style["title_size"])
    ax.set_xlabel("Mean Decrease in Impurity Importance", fontsize=style["label_size"], labelpad=10)
    ax.set_ylabel("Annotation | UniqueID", fontsize=style["label_size"], labelpad=10)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style["line_width"])
        spine.set_color("black")
    fig.savefig(save_dir / "RandomForest_feature_importance_top20.png", dpi=style["dpi"], bbox_inches="tight", pad_inches=0.2, facecolor="white")
    fig.savefig(save_dir / "RandomForest_feature_importance_top20.svg", bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)

    print(
        f"[Random Forest] CV accuracy={cv_accuracy:.3f}, macro F1={macro_f1:.3f}, permutation p={perm_p_value:.4g}",
        flush=True,
    )
    print(f"[Random Forest] Completed. Results saved to: {save_dir}\n", flush=True)
