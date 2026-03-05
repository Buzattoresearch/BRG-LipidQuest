# File: Stats/run_statistics.py
import os
import pandas as pd
from pathlib import Path
from typing import Optional
from Stats.pca_analysis import run_pca
from Stats.plsda_analysis import run_plsda
from Stats.volcano_analysis import run_volcano
from Stats.heatmap_analysis import run_heatmap
from Stats.violinplots import run_violinplots
from Stats.boxplots import run_boxplots
from Stats.correlation_analysis import run_correlation_analysis
from Stats.class_distributions import run_from_stats as run_class_distributions
from Stats.summed_intensity_per_class import run_from_stats as run_class_sums

from Stats.utils import prepare_output_dir

def run_all_statistics(
    output_folder: Path,
    run_with_QCs: bool = True,
    run_without_QCs: bool = True,
    run_highconf: bool = True,
    group_order: Optional[list[str]] = None,
    group_colors: Optional[dict[str, str]] = None,
):

    """
    Master controller for running LipidQuest statistical analyses.
    Replaces old LipidQuest_Stats.py to work with the new GUI output structure.
    """
    output_folder = Path(output_folder)
    stats_dir = output_folder / "statistics"
    # Prefer cleaned sample_groups version if it exists
    group_file = stats_dir / "sample_groups_cleaned.csv"
    if not group_file.exists():
        group_file = output_folder / "sample_groups.csv"

    if not stats_dir.exists():
        raise FileNotFoundError(f"No statistics directory found at: {stats_dir}")

    print(f"\n=== Running Statistics for {output_folder} ===")
    print(f"Using group file: {group_file}")

    # === Define datasets ===
    files_to_run = []
    if run_with_QCs:
        files_to_run.append(stats_dir / "Final_Annotated.csv")
    if run_without_QCs:
        files_to_run.append(stats_dir / "Final_Annotated_Without_QCs.csv")
    if run_highconf:
        files_to_run.append(stats_dir / "Final_Annotated_Without_QCs_HighConf.csv")

    for file_path in files_to_run:
        if not file_path.exists():
            print(f"[Warning] Missing file: {file_path}")
            continue

        dataset_name = file_path.stem
        save_dir = prepare_output_dir(stats_dir / f"Results_{dataset_name}")

        print(f"\n--- Processing dataset: {dataset_name} ---")
        print(f"Saving outputs to: {save_dir}")

        # === Core Analyses ===
        try:
            run_pca(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
        except Exception as e:
            print(f"[PCA Error] {e}")

        try:
            # Skip PLS-DA if dataset contains QCs
            if "Without_QCs" in dataset_name:
                run_plsda(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
            else:
                print(f"[PLS-DA] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[PLS-DA Error] {e}")

        try:
            # Skip Volcano if dataset contains QCs
            if "Without_QCs" in dataset_name:
                run_volcano(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
            else:
                print(f"[Volcano] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Volcano Error] {e}")
            
        try:
            # Skip Heatmap if dataset contains QCs
            if "Without_QCs" in dataset_name:
                run_heatmap(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
            else:
                print(f"[Heatmap] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Heatmap Error] {e}")
         
        try:
            # Skip Violin if dataset contains QCs
            if "Without_QCs" in dataset_name:
                run_boxplots(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
            else:
                print(f"[Box plots] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Boxplot Error] {e}")
                
        try:
            # Skip Violin if dataset contains QCs
            if "Without_QCs" in dataset_name:
                run_violinplots(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
            else:
                print(f"[Violin plots] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Violin Error] {e}")
            
        try:
            # Skip Correlations if dataset contains QCs
            if "Without_QCs" in dataset_name:
                run_correlation_analysis(file_path, group_file, save_dir, group_order=group_order)
            else:
                print(f"[Correlations] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Correlations Error] {e}")
            
        try:
            # Skip Total intensity plots if dataset contains QCs
            if "Without_QCs" in dataset_name:
                run_class_distributions(file_path, group_file, save_dir, group_order=group_order, unknown_policy="append")
            else:
                print(f"[Total intensity] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Total intensity Error] {e}")
            
        try:
            # Skip Summed Intensities per Class if dataset contains QCs
            if "Without_QCs" in dataset_name:
                run_class_sums(file_path, group_file, save_dir, group_order=group_order)
            else:
                print(f"[Summed Intensities per Class] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Summed Intensities per Class Error] {e}")

    print("\n✅ All statistical analyses completed.\n")