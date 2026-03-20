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

# ----------------------------------------------------------
# Define your preferred class order (custom hierarchy)
# ----------------------------------------------------------

preferred_order = [
            "CAR", "CoA", "FA", "FAG", "FAL", "FOH", "HC", "NAx", "NAE", "NAT", "WE", "FAHFA",
            "MG", "DG", "TG", "Hex2DG", "Hex2MG", "HexDG", "HexMG", "DGTA", "DGTS", "MGTS", "GlcADG", "DGCC", "SQDG", "SQMG",
            "LPA", "PA", "PPA", "LPG", "PG", "CL", "MLCL", "DLCL", "BMP", "LPC", "PC", "PnC", "LPE", "PE", "PEth", "PnE", "LPS", "PS", "LPI", "PI", "PIM", "PIP", "CDP-DG", "Glc-GP", "GP",
            "Cer", "ACer", "CerP", "GlcCer", "HexCer", "MIPC", "M(IP)2C", "PE-Cer", "CerPE", "PI-Cer", "CerPI", "SCer", "SHexCer", "LSM", "SM", "SPB", "HexSPB", "SPBP", "SulfateHexSPB",
            "CE", "ST", "PK", "PR", "SL", "Other"
        ]

# ----------------------------------------------------------------------
# Build large deterministic color + marker maps for lipid classes
# ----------------------------------------------------------------------

# Generate a large color palette using seaborn (or matplotlib)
# 80 distinct colors (more than your preferred_order size)
large_palette = sns.color_palette("tab20", 20) \
               + sns.color_palette("Set3", 12) \
               + sns.color_palette("Dark2", 8) \
               + sns.color_palette("Set2", 8) \
               + sns.color_palette("Accent", 8) \
               + sns.color_palette("Paired", 12)

# Flatten to a list of RGB tuples
large_palette = [tuple(c) for c in large_palette]

# Large marker list (≥25 unique symbols)
large_markers = [
    "o",  # circle
    "s",  # square
    "D",  # diamond
    "^",  # triangle up
    "v",  # triangle down
    "<",  # triangle left
    ">",  # triangle right
    "p",  # pentagon
    "h",  # hexagon
    "H",  # rotated hexagon
    "X",  # filled X
    "d",  # thin diamond
    "*",  # star
    "P",  # filled plus
    "8"   # octagon
]

# Build class → color and class → marker mapping
color_map = {}
marker_map = {}

for i, cls in enumerate(preferred_order):
    color_map[cls] = large_palette[i % len(large_palette)]
    marker_map[cls] = large_markers[i % len(large_markers)]
    
# ------------------------------
# Helpers
# ------------------------------

def _safe_filename(text: str) -> str:
    """Make a string safe for filenames."""
    text = str(text).strip()
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace(" ", "_")
    # Keep only common safe chars
    return "".join(ch for ch in text if ch.isalnum() or ch in ("_", "-", ".", "(", ")"))

# ------------------------------
# PLOTS
# ------------------------------

def plot_results(pol_tag, input_csv, output_folder="results", suffix=""):
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

    # --- Determine polarity tag for filenames ---
    sample_cols = [c for c in df.columns if isinstance(c, str) and (c.startswith("P_") or c.startswith("N_"))]
    has_pos = any(c.startswith("P_") for c in sample_cols)
    has_neg = any(c.startswith("N_") for c in sample_cols)

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

    polarity = ""
    if "P" in pol_tag.upper():
        polarity = "positive"  
    elif "N" in pol_tag.upper():
        polarity = "negative"
    elif pol_tag == "":
        polarity = "combined"
        
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

        # Keep only those that exist in your dataset
        ordered_classes = [c for c in preferred_order if c in class_pivot.index]
        unordered_classes = [c for c in class_pivot.index if c not in ordered_classes]

        # Reorder the DataFrame (your preferred order first)
        class_pivot = class_pivot.loc[ordered_classes + unordered_classes]
        
        annotation_order = [
            "MS/MS match",
            "Target list match (MS, RT, CCS)",
            "MS match"
        ]
        
        # Keep only columns that exist
        ordered_annotations = [a for a in annotation_order if a in class_pivot.columns]
        unordered_annotations = [a for a in class_pivot.columns if a not in ordered_annotations]

        # Reorder columns
        class_pivot = class_pivot[ordered_annotations + unordered_annotations]

        # Create stacked bar chart (MS vs MS/MS)
        class_pivot.plot(
            kind="bar",
            stacked=True,
            color=["#136845", "#864646", "#0E4D81"],  # MS/MS, MS, Target
            edgecolor="white",
            figsize=(10, 6)
        )

        # --- Add counts to legend labels ---
        layer_counts = class_pivot.sum(axis=0)  # column sums

        handles, labels = plt.gca().get_legend_handles_labels()

        new_labels = []
        for lbl in labels:
            clean_lbl = lbl.strip()
            if clean_lbl in layer_counts:
                new_labels.append(f"{clean_lbl} ({int(layer_counts[clean_lbl])})")
            else:
                new_labels.append(clean_lbl)

        # Legend with counts
        plt.legend(
            handles=handles,
            labels=new_labels,
            title="Annotation Type")
        plt.ylabel("Number of Annotations")
        plt.xlabel("Lipid Class")
        plt.title(f"Annotations per Lipid Class ({polarity})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_folder / f"{pol_tag}lipid_class_counts_type_stacked{suffix}.png", dpi=100)
        plt.savefig(output_folder / f"{pol_tag}lipid_class_counts_type_stacked{suffix}.svg", dpi=100, format="svg")
        plt.close()

    else:
        print("[WARNING] Missing 'Lipid Class' or 'Annotation Type' columns. Skipping bar plot.", flush=True)
        
    # ------------------------------------------------------------------
    # Plot 2: Aggregated bar plot (Lipid Class × Annotation Tier)
    # ------------------------------------------------------------------
    if "Lipid Class" in df.columns and "Annotation tier" in df.columns:
        plt.figure(figsize=(10, 6))

        # Count annotations per lipid class and annotation type
        class_counts = (
            df.groupby(["Lipid Class", "Annotation tier"])
            .size()
            .reset_index(name="Count")
        )

        # Pivot for plotting (Lipid Class × Annotation tier)
        class_pivot = class_counts.pivot(
            index="Lipid Class", columns="Annotation tier", values="Count"
        ).fillna(0)
        
        # Keep only those that exist in your dataset
        ordered_classes = [c for c in preferred_order if c in class_pivot.index]
        unordered_classes = [c for c in class_pivot.index if c not in ordered_classes]

        # Reorder the DataFrame (your preferred order first)
        class_pivot = class_pivot.loc[ordered_classes + unordered_classes]
        
        annotation_order = [
            "Tier 2",
            "Tier 1",
        ]
        
        # Keep only columns that exist
        ordered_annotations = [a for a in annotation_order if a in class_pivot.columns]
        unordered_annotations = [a for a in class_pivot.columns if a not in ordered_annotations]

        # Reorder columns
        class_pivot = class_pivot[ordered_annotations + unordered_annotations]

        # Create stacked bar chart (MS vs MS/MS)
        class_pivot.plot(
            kind="bar",
            stacked=True,
            color=["#0D4930", "#864646"],  # High, Low
            edgecolor="white",
            figsize=(10, 6)
        )

        # --- Add counts to legend labels ---
        layer_counts = class_pivot.sum(axis=0)  # column sums

        handles, labels = plt.gca().get_legend_handles_labels()

        new_labels = []
        for lbl in labels:
            clean_lbl = lbl.strip()
            if clean_lbl in layer_counts:
                new_labels.append(f"{clean_lbl} ({int(layer_counts[clean_lbl])})")
            else:
                new_labels.append(clean_lbl)

        # Legend with counts
        plt.legend(
            handles=handles,
            labels=new_labels,
            title="Annotation tier")
        plt.ylabel("Number of Annotations")
        plt.xlabel("Lipid Class")
        plt.title(f"Annotations per Lipid Class ({polarity})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_folder / f"{pol_tag}lipid_class_counts_tier_stacked{suffix}.png", dpi=100)
        plt.savefig(output_folder / f"{pol_tag}lipid_class_counts_tier_stacked{suffix}.svg", dpi=100, format="svg")
        plt.close()

    else:
        print("[WARNING] Missing 'Lipid Class' or 'Annotation tier' columns. Skipping bar plot.", flush=True)

    # ------------------------------------------------------------------
    # Plot 2: Scatter RT vs m/z (auto color & shape per lipid class)
    # ------------------------------------------------------------------
    if {"RT (min)", "m/z", "Lipid Class"}.issubset(df.columns):
        plt.figure(figsize=(8, 6))

        # --- Sanitize numeric columns ---
        df["RT (min)"] = pd.to_numeric(df["RT (min)"], errors="coerce")
        df["m/z"] = pd.to_numeric(df["m/z"], errors="coerce")

        # --- Basic filtering and ordering ---
        df = df[df["Annotation Type"].astype(str).str.upper() != "IS"]
        df = df[df["Lipid Class"] != "Other"]
        df = df[df["Annotation"].notna()]
        df["Lipid Class"] = pd.Categorical(df["Lipid Class"], categories=preferred_order, ordered=True)

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

        # ------------------------------------------------------------------
        # Plot 2b: RT vs m/z per class (one file per lipid class)
        # ------------------------------------------------------------------
        per_class_dir = output_folder / "RT_vs_mz_per_class"
        per_class_dir.mkdir(parents=True, exist_ok=True)

        for lipid_class in classes:
            group = df_valid[df_valid["Lipid Class"] == lipid_class].copy()
            if group.empty:
                continue

            plt.figure(figsize=(7, 5))

            color = color_map.get(lipid_class, (0.5, 0.5, 0.5))
            marker = marker_map.get(lipid_class, "o")

            plt.scatter(
                group["RT (min)"],
                group["m/z"],
                s=35,
                alpha=0.75,
                color=color,
                marker=marker,
                edgecolors="none"
            )

            n = int(len(group))
            plt.xlabel("RT (min)", fontsize=12, fontweight="bold")
            plt.ylabel("m/z", fontsize=12)
            plt.title(f"RT vs m/z: {lipid_class} (n={n}, {polarity})", fontsize=13, fontweight="bold")

            plt.tight_layout()

            cls_tag = _safe_filename(lipid_class)
            plt.savefig(per_class_dir / f"{pol_tag}rt_vs_mz_{cls_tag}{suffix}.png", dpi=100, bbox_inches="tight")
            plt.savefig(per_class_dir / f"{pol_tag}rt_vs_mz_{cls_tag}{suffix}.svg", dpi=100, bbox_inches="tight", format="svg")
            plt.close()
            
        if len(classes) == 0:
            print("[WARNING] No valid lipid classes to plot.", flush=True)
        else:
            for lipid_class in classes:
                group = df_valid[df_valid["Lipid Class"] == lipid_class]
                if group.empty:
                    continue

                color = color_map.get(lipid_class, (0.5, 0.5, 0.5))  # fallback grey
                marker = marker_map.get(lipid_class, "o")

                plt.scatter(
                    group["RT (min)"],
                    group["m/z"],
                    s=35, alpha=0.7,
                    color=color,
                    marker=marker,
                    label=lipid_class
                )

            plt.xlabel("RT (min)", fontsize=12, fontweight="bold")
            plt.ylabel("m/z", fontsize=12)
            plt.title(f"RT vs m/z by Lipid Class ({polarity})", fontsize=14, fontweight="bold")
            # --- Legend with counts, excluding classes with n = 0 ---
            handles, labels = plt.gca().get_legend_handles_labels()
            class_counts = df_valid["Lipid Class"].value_counts()
            filtered_handles = []
            filtered_labels = []
            for h, lbl in zip(handles, labels):
                clean = lbl.strip()
                if clean in class_counts and class_counts[clean] > 0:
                    filtered_handles.append(h)
                    filtered_labels.append(f"{clean} ({int(class_counts[clean])})")
                # else → skip zero-count classes
            
            ncol = min(6, max(2, len(filtered_labels)//4))
            plt.legend(
                handles=filtered_handles,
                labels=filtered_labels,
                title="",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=ncol,
                fontsize=8,
                title_fontsize=8,
                frameon=False
            )
            
            plt.tight_layout()
            plt.savefig(output_folder / f"{pol_tag}rt_vs_mz_by_class{suffix}.png", dpi=100, bbox_inches="tight")
            plt.savefig(output_folder / f"{pol_tag}rt_vs_mz_by_class{suffix}.svg", dpi=100, bbox_inches="tight", format="svg")
            plt.close()

    else:
        print("[WARNING] Missing required columns for RT–m/z plot.", flush=True)

           
def plot_kendrick_mass_vs_defect(input_csv, results_folder, suffix=""):
    """
    Generates:
      1) Kendrick Mass vs Kendrick Mass Defect plot by lipid class (one combined plot)
      2) Per-class Kendrick plots (one file per lipid class)
    """
    print(f"\nPlotting Kendrick Mass Defect results...\n")
    df = pd.read_csv(input_csv, low_memory=False)

    # --- Determine polarity tag for filenames ---
    pol_tag = ""
    polarity = "combined"
    if "Pos_" in str(input_csv):
        pol_tag = "Pos_"
        polarity = "positive"
    elif "Neg_" in str(input_csv):
        pol_tag = "Neg_"
        polarity = "negative"

    # --- Basic filtering ---
    required_cols = {"Neutral mass", "Lipid Class", "Annotation Type", "Annotation"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARNING] Missing required columns for Kendrick plot: {sorted(missing)}", flush=True)
        return

    df = df[df["Neutral mass"].notna()]
    df = df[df["Lipid Class"].notna()]
    df = df[df["Annotation Type"].astype(str).str.upper() != "IS"]
    df = df[df["Lipid Class"] != "Other"]
    df = df[df["Annotation"].notna()]

    df["Neutral mass"] = pd.to_numeric(df["Neutral mass"], errors="coerce")
    df = df.dropna(subset=["Neutral mass"])

    # Clean class strings
    df["Lipid Class"] = (
        df["Lipid Class"]
        .astype(str)
        .str.strip()
        .replace(["nan", "NaN", "None"], "")
    )
    df = df[~df["Lipid Class"].isin(["", "Unassigned", "Unknown", "No match"])]

    df["Lipid Class"] = pd.Categorical(df["Lipid Class"], categories=preferred_order, ordered=True)

    # --- Kendrick Mass calculations ---
    km_ratio = 14.00000 / 14.01565
    df["Kendrick Mass"] = df["Neutral mass"] * km_ratio
    df["Nominal Kendrick Mass"] = df["Kendrick Mass"].round()
    df["Kendrick Defect"] = df["Nominal Kendrick Mass"] - df["Kendrick Mass"]

    out_dir = Path(results_folder) / "Annotation plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Combined plot (all classes)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="Kendrick Mass",
        y="Kendrick Defect",
        hue="Lipid Class",
        style="Lipid Class",
        palette=color_map,
        markers=marker_map,
        s=45,
        alpha=0.8,
        linewidth=0,
        size="Lipid Class",
        sizes=(30, 70),
    )

    plt.xlabel("Kendrick Mass", fontweight="bold")
    plt.ylabel("Kendrick Mass Defect")
    plt.title(f"Kendrick Mass Defect vs Kendrick Mass by Lipid Class ({polarity})", fontweight="bold")

    # Legend with counts
    class_counts = df["Lipid Class"].value_counts()
    handles, labels = plt.gca().get_legend_handles_labels()

    # Deduplicate labels, preserve first handle per label
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    handles = list(seen.values())
    labels = list(seen.keys())

    filtered_handles = []
    filtered_labels = []
    for h, lbl in zip(handles, labels):
        clean = str(lbl).strip()
        if clean in class_counts and class_counts[clean] > 0:
            filtered_handles.append(h)
            filtered_labels.append(f"{clean} ({int(class_counts[clean])})")

    ncol = min(6, max(2, len(filtered_labels) // 4))
    plt.legend(
        handles=filtered_handles,
        labels=filtered_labels,
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=ncol,
        fontsize=8,
        title_fontsize=8,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.grid(False)

    out_svg = out_dir / f"{pol_tag}kendrick_mass_defect_by_class{suffix}.svg"
    out_png = out_dir / f"{pol_tag}kendrick_mass_defect_by_class{suffix}.png"
    plt.savefig(out_svg, bbox_inches="tight", format="svg")
    plt.savefig(out_png, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"[DONE] Saved Kendrick plot to:\n  {out_png}")

    # ------------------------------------------------------------------
    # 2) Per-class Kendrick plots
    # ------------------------------------------------------------------
    per_class_dir = out_dir / "Kendrick_per_class"
    per_class_dir.mkdir(parents=True, exist_ok=True)

    classes = [c for c in df["Lipid Class"].dropna().unique().tolist() if str(c).strip() != ""]
    # Keep stable order using preferred_order, then any extras
    classes_ordered = [c for c in preferred_order if c in classes] + [c for c in classes if c not in preferred_order]

    for lipid_class in classes_ordered:
        sub = df[df["Lipid Class"] == lipid_class].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(7, 5))

        color = color_map.get(lipid_class, (0.5, 0.5, 0.5))
        marker = marker_map.get(lipid_class, "o")

        plt.scatter(
            sub["Kendrick Mass"],
            sub["Kendrick Defect"],
            s=45,
            alpha=0.8,
            color=color,
            marker=marker,
            edgecolors="none"
        )

        n = int(len(sub))
        plt.xlabel("Kendrick Mass", fontweight="bold")
        plt.ylabel("Kendrick Mass Defect")
        plt.title(f"Kendrick: {lipid_class} (n={n}, {polarity})", fontweight="bold")

        plt.tight_layout()
        plt.grid(False)

        cls_tag = _safe_filename(lipid_class)
        plt.savefig(per_class_dir / f"{pol_tag}kendrick_{cls_tag}{suffix}.png", dpi=100, bbox_inches="tight")
        plt.savefig(per_class_dir / f"{pol_tag}kendrick_{cls_tag}{suffix}.svg", format="svg", bbox_inches="tight")
        plt.close()

