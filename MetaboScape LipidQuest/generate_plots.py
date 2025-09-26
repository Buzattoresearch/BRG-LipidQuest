import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_results(input_csv, output_folder="results"):
    """
    Generate summary plots from the final filtered results.
    - Counts per lipid class
    - Scatter RT vs m/z
    - Distribution of MS Scores
    """

    df = pd.read_csv(input_csv)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Bar plot of lipid class counts ---
    if "Lipid Class" in df.columns:
        plt.figure(figsize=(10, 6))
        df["Lipid Class"].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
        plt.ylabel("Count")
        plt.xlabel("Lipid Class")
        plt.title("Annotations per Lipid Class")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_folder / "lipid_class_counts.png")
        plt.close()

    # --- Plot 2: Scatter RT vs m/z ---
    if "RT (min)" in df.columns and "m/z" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df["RT (min)"], df["m/z"], alpha=0.6, s=30, c="darkred")
        plt.xlabel("RT (min)")
        plt.ylabel("m/z")
        plt.title("RT vs m/z (filtered results)")
        plt.tight_layout()
        plt.savefig(output_folder / "rt_vs_mz.png")
        plt.close()

    # --- Plot 3: Distribution of MS Scores ---
    if "MS Score" in df.columns:
        plt.figure(figsize=(8, 6))
        df["MS Score"].plot(kind="hist", bins=20, color="darkgreen", edgecolor="black", alpha=0.7)
        plt.xlabel("MS Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of MS Scores")
        plt.tight_layout()
        plt.savefig(output_folder / "ms_score_distribution.png")
        plt.close()

    print(f"Plots saved to: {output_folder}")
