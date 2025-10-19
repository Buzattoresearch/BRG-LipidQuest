# File: Stats/run_statistics.py
import os
import pandas as pd
from pathlib import Path
from Stats.pca_analysis import run_pca
from Stats.plsda_analysis import run_plsda
from Stats.volcano_analysis import run_volcano
from Stats.heatmap_analysis import run_heatmap
from Stats.utils import prepare_output_dir

def run_all_statistics(output_folder: Path, run_with_QCs=True, run_without_QCs=True, run_highconf=True):
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
            run_pca(file_path, group_file, save_dir)
        except Exception as e:
            print(f"[PCA Error] {e}")

        try:
            # Skip PLS-DA if dataset contains QCs
            if "Without_QCs" in dataset_name:
                run_plsda(file_path, group_file, save_dir)
            else:
                print(f"[PLS-DA] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[PLS-DA Error] {e}")

        try:
            # Skip Volcano if dataset contains QCs
            if "Without_QCs" in dataset_name:
                run_volcano(file_path, group_file, save_dir)
            else:
                print(f"[Volcano] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Volcano Error] {e}")
            
        try:
            # Skip Heatmap if dataset contains QCs
            if "Without_QCs" in dataset_name:
                run_heatmap(file_path, group_file, save_dir)
            else:
                print(f"[Heatmap] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Heatmap Error] {e}")

    print("\n✅ All statistical analyses completed.\n")