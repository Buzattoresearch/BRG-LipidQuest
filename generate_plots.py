#TODO: Fix plot design

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import os
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*is_sparse is deprecated.*",
    category=FutureWarning
)

# --- Global matplotlib style settings ---
rcParams["font.family"] = "Arial"          # or 'DejaVu Sans', 'Helvetica', 'Calibri', "serif", etc.
rcParams["font.size"] = 14                 # default text size
rcParams["axes.titlesize"] = 14            # plot titles
rcParams["axes.labelsize"] = 14            # axis labels
rcParams["xtick.labelsize"] = 12           # x-axis tick labels
rcParams["ytick.labelsize"] = 12           # y-axis tick labels
rcParams["legend.fontsize"] = 12           # legend text
rcParams["font.weight"] = "normal"         # can be 'bold'
rcParams["axes.titleweight"] = "bold"      # bold titles

def plot_results(input_csv, output_folder="results"):
    """
    Generate summary plots from the final filtered results.
    - Aggregated bar plot: counts per lipid class separated by Annotation Type (MS vs MS/MS)
    - Scatter RT vs m/z
    - Distribution of MS Scores (if available)
    Internal standards (IS) are excluded automatically.
    """
    print(f'\nPlotting annotation results...\n')

    df = pd.read_csv(input_csv, low_memory=False)
    output_folder = Path(output_folder) / "Annotation plots"
    output_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Exclude internal standards (IS)
    # ------------------------------------------------------------------
    # Check across common identifying columns
    exclude_mask = (
        df.astype(str)
        .apply(lambda col: col.str.contains(r"\bIS\b|Internal\s*Standard", case=False, na=False))
        .any(axis=1)
    )
    n_is = exclude_mask.sum()
    if n_is > 0:
        print(f"[INFO] Excluding {n_is} internal standard entries before plotting.", flush=True)
        df = df.loc[~exclude_mask].copy()

    # ------------------------------------------------------------------
    # Plot 1: Aggregated bar plot (Lipid Class × Annotation Type)
    # ------------------------------------------------------------------
    if "Lipid Class" in df.columns and "Annotation Type" in df.columns:
        plt.figure(figsize=(10, 6))

        # Count annotations per lipid class and annotation type
        class_counts = (
            df.groupby(["Lipid Class", "Annotation Type"])
            .size()
            .reset_index(name="Count")
        )

        # Pivot for plotting (Lipid Class × Annotation Type)
        class_pivot = class_counts.pivot(
            index="Lipid Class", columns="Annotation Type", values="Count"
        ).fillna(0)

        # ----------------------------------------------------------
        # Define your preferred class order (custom hierarchy)
        # ----------------------------------------------------------
        preferred_order = [
            "CAR", "CoA", "FA", "FAG", "FAL", "FOH", "HC", "NA", "NAE", "NAT", "WE", "FAHFA",
            "MG", "DG", "TG", "DGDG", "DGMG", "MGDG", "MGMG", "DGTA", "DGTS", "MGTS", "GlcADG", "DGCC", "SQDG", "SQMG",
            "LPA", "PA", "PPA", "LPG", "PG", "CL", "BMP", "LPC", "PC", "PnC", "LPE", "PE", "PnE", "LPS", "PS", "LPI", "PI", "PIM", "PIP", "CDP-DG", "Glc-GP", "GP",
            "Cer", "ACer", "CerP", "GlcCer", "HexCer", "MIPC", "M(IP)2C", "PE-Cer", "CerPE", "PI-Cer", "CerPI", "SCer", "SHexCer", "LSM", "SM", "SPB", "HexSPB", "SPBP", "SulfateHexSPB",
            "CE", "ST", "PK", "PR", "SL", "Other"
        ]

        # Keep only those that exist in your dataset
        ordered_classes = [c for c in preferred_order if c in class_pivot.index]
        unordered_classes = [c for c in class_pivot.index if c not in ordered_classes]

        # Reorder the DataFrame (your preferred order first)
        class_pivot = class_pivot.loc[ordered_classes + unordered_classes]

        # Create stacked bar chart (MS vs MS/MS)
        class_pivot.plot(
            kind="bar",
            stacked=True,
            color=["#1B1B1B", "#838383"],  # MS, MS/MS
            edgecolor="white",
            figsize=(10, 6)
        )

        plt.ylabel("Number of Annotations")
        plt.xlabel("Lipid Class")
        plt.title("Annotations per Lipid Class")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Annotation Type")
        plt.tight_layout()
        plt.savefig(output_folder / "lipid_class_counts_stacked.png", dpi=300)
        plt.savefig(output_folder / "lipid_class_counts_stacked.svg", dpi=300, format="svg")
        plt.close()
    else:
        print("[WARNING] Missing 'Lipid Class' or 'Annotation Type' columns. Skipping bar plot.", flush=True)

    # ------------------------------------------------------------------
    # Plot 2: Scatter RT vs m/z (auto color & shape per lipid class)
    # ------------------------------------------------------------------
    if {"RT (min)", "m/z", "Lipid Class"}.issubset(df.columns):
        plt.figure(figsize=(8, 6))

        # --- Sanitize numeric columns ---
        df["RT (min)"] = pd.to_numeric(df["RT (min)"], errors="coerce")
        df["m/z"] = pd.to_numeric(df["m/z"], errors="coerce")

        # --- Clean Lipid Class entries ---
        df["Lipid Class"] = (
            df["Lipid Class"]
            .astype(str)
            .str.strip()
            .replace(["nan", "NaN", "None"], "")
        )

        # --- Drop unassigned / unknown / empty ---
        df_valid = df[
            ~df["Lipid Class"].str.contains("IS", case=False, na=False)
            & ~df["Lipid Class"].isin(["", "Unassigned", "Unknown", "No match"])
        ].copy()

        # --- Drop rows missing RT or m/z ---
        df_valid = df_valid.dropna(subset=["RT (min)", "m/z"])

        # --- Identify classes ---
        classes = sorted(df_valid["Lipid Class"].unique())
        print(f"[DEBUG] RT–m/z plot classes (excluding unknowns): {classes}")

        if len(classes) == 0:
            print("[WARNING] No valid lipid classes to plot.", flush=True)
        else:
            colors = list(plt.cm.tab20.colors)
            markers = ["o", "s", "D", "^", "v", "P", "X", "*", "p", "h", "<", ">", "d"]

            for i, lipid_class in enumerate(classes):
                group = df_valid[df_valid["Lipid Class"] == lipid_class]
                if group.empty:
                    continue
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                plt.scatter(
                    group["RT (min)"],
                    group["m/z"],
                    s=35, alpha=0.7, color=color, marker=marker,
                    label=lipid_class
                )

            plt.xlabel("RT (min)", fontsize=12, fontweight="bold")
            plt.ylabel("m/z", fontsize=12)
            plt.title("RT vs m/z by Lipid Class", fontsize=14, fontweight="bold")
            plt.legend(
                title="Lipid Class",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=8,
                title_fontsize=9
            )
            plt.tight_layout()
            plt.savefig(output_folder / "rt_vs_mz_by_class.png", dpi=300, bbox_inches="tight")
            plt.savefig(output_folder / "rt_vs_mz_by_class.svg", dpi=300, bbox_inches="tight", format="svg")
            plt.close()
    else:
        print("[WARNING] Missing required columns for RT–m/z plot.", flush=True)

           
def plot_kendrick_mass_vs_defect(input_csv, results_folder):
    """
    Generates a Kendrick Mass vs Kendrick Mass Defect plot by lipid class,
    optimized for readability and balanced layout.
    """
    print(f'\nPlotting Kendrick Mass Defect results...\n')
    df = pd.read_csv(input_csv, low_memory=False)

    # --- Basic filtering ---
    df = df[df["Neutral mass"].notna()]
    df = df[df["Lipid Class"].notna()]
    df = df[df["Annotation Type"].astype(str).str.upper() != "IS"]

    # --- Kendrick Mass calculations ---
    km_ratio = 14.00000 / 14.01565
    df["Kendrick Mass"] = df["Neutral mass"] * km_ratio
    df["Nominal Kendrick Mass"] = df["Kendrick Mass"].round()
    df["Kendrick Defect"] = df["Nominal Kendrick Mass"] - df["Kendrick Mass"]

    # --- Figure setup ---
    plt.figure(figsize=(8, 6))  # landscape layout
    sns.scatterplot(
        data=df,
        x="Kendrick Mass",
        y="Kendrick Defect",
        hue="Lipid Class",
        style="Lipid Class",
        palette="tab20",
        s=45,
        alpha=0.8,
        linewidth=0
    )

    # --- Labels & Title ---
    plt.xlabel("Kendrick Mass", fontweight="bold")
    plt.ylabel("Kendrick Mass Defect")
    plt.title("Kendrick Mass Defect vs Kendrick Mass by Lipid Class", fontweight="bold")

    # --- Legend below the plot, multi-column ---
    handles, labels = plt.gca().get_legend_handles_labels()
    ncol = min(8, max(2, len(labels)//4))
    plt.legend(
        handles=handles,
        labels=labels,
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=ncol,
        fontsize=9,
        title_fontsize=10,
        frameon=False
    )

    # --- Aesthetics ---
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    plt.grid(False)

    # --- Save ---
    out_svg = Path(results_folder) / "Annotation plots" / "kendrick_mass_defect_by_class.svg"
    out_png = Path(results_folder) / "Annotation plots" / "kendrick_mass_defect_by_class.png"

    # Create the parent folder if it doesn't exist
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_svg, bbox_inches="tight", format="svg")
    plt.savefig(out_png, bbox_inches="tight", dpi=400)
    plt.close()
    print(f"[DONE] Saved improved Kendrick plot to:\n  {out_png}")

