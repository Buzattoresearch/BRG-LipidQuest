# File: Stats/run_statistics.py
from pathlib import Path
from typing import Optional

from Stats.advanced_differential_analysis import run_from_stats as run_advanced_differential_analysis
from Stats.boxplots import run_boxplots
from Stats.class_distributions import run_from_stats as run_class_distributions
from Stats.correlation_analysis import run_correlation_analysis
from Stats.enrichment_analysis import run_from_stats as run_enrichment_analysis
from Stats.heatmap_analysis import run_heatmap
from Stats.pca_analysis import run_pca
from Stats.plsda_analysis import run_plsda
from Stats.ratio_analysis import run_from_stats as run_ratio_analysis
from Stats.summed_intensity_per_class import run_from_stats as run_class_sums
from Stats.upset_plot import run_from_stats as run_upset_plot
from Stats.utils import prepare_output_dir
from Stats.violinplots import run_violinplots
from Stats.volcano_analysis import run_volcano


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
    group_file = stats_dir / "sample_groups_cleaned.csv"
    if not group_file.exists():
        group_file = output_folder / "sample_groups.csv"

    if not stats_dir.exists():
        raise FileNotFoundError(f"No statistics directory found at: {stats_dir}")

    print(f"\n=== Running Statistics for {output_folder} ===")
    print(f"Using group file: {group_file}")

    files_to_run: list[Path] = []
    if run_with_QCs:
        files_to_run.append(stats_dir / "Final_Annotated.csv")
    if run_without_QCs:
        files_to_run.append(stats_dir / "Final_Annotated_Without_QCs.csv")
        files_to_run.append(stats_dir / "Final_Annotated_with_missing_Without_QCs.csv")
    if run_highconf:
        files_to_run.append(stats_dir / "Final_Annotated_Without_QCs_HighConf.csv")
        files_to_run.append(stats_dir / "Final_Annotated_with_missing_Without_QCs_HighConf.csv")

    for file_path in files_to_run:
        if not file_path.exists():
            print(f"[Warning] Missing file: {file_path}")
            continue

        dataset_name = file_path.stem
        save_dir = prepare_output_dir(stats_dir / f"Results_{dataset_name}")

        print(f"\n--- Processing dataset: {dataset_name} ---")
        print(f"Saving outputs to: {save_dir}")

        try:
            run_pca(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
        except Exception as e:
            print(f"[PCA Error] {e}")

        try:
            if "Without_QCs" in dataset_name:
                run_plsda(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
            else:
                print(f"[PLS-DA] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[PLS-DA Error] {e}")

        try:
            if "Without_QCs" in dataset_name:
                run_volcano(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
            else:
                print(f"[Volcano] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Volcano Error] {e}")

        try:
            if "Without_QCs" in dataset_name:
                run_heatmap(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
            else:
                print(f"[Heatmap] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Heatmap Error] {e}")

        try:
            if "Without_QCs" in dataset_name:
                run_boxplots(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
            else:
                print(f"[Box plots] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Boxplot Error] {e}")

        try:
            if "Without_QCs" in dataset_name:
                run_violinplots(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
            else:
                print(f"[Violin plots] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Violin Error] {e}")

        try:
            if "Without_QCs" in dataset_name:
                run_correlation_analysis(file_path, group_file, save_dir, group_order=group_order)
            else:
                print(f"[Correlations] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Correlations Error] {e}")

        try:
            if "Without_QCs" in dataset_name and "with_missing" in dataset_name:
                run_upset_plot(file_path, group_file, save_dir, group_colors=group_colors, group_order=group_order)
            else:
                print(f"[UpSet] Skipped {dataset_name} (requires *_with_missing_Without_QCs dataset).")
        except Exception as e:
            print(f"[UpSet Error] {e}")

        try:
            if "Without_QCs" in dataset_name:
                run_class_distributions(file_path, group_file, save_dir, group_order=group_order, unknown_policy="append")
            else:
                print(f"[Total intensity] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Total intensity Error] {e}")

        try:
            if "Without_QCs" in dataset_name:
                run_class_sums(file_path, group_file, save_dir, group_order=group_order)
            else:
                print(f"[Summed Intensities per Class] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Summed Intensities per Class Error] {e}")

        try:
            if "Without_QCs" in dataset_name:
                run_enrichment_analysis(file_path, group_file, save_dir, group_order=group_order, group_colors=group_colors)
            else:
                print(f"[Enrichment] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Enrichment Error] {e}")

        try:
            if "Without_QCs" in dataset_name:
                run_ratio_analysis(file_path, group_file, save_dir, group_order=group_order, group_colors=group_colors)
            else:
                print(f"[Ratios] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Ratios Error] {e}")

        try:
            if "Without_QCs" in dataset_name:
                run_advanced_differential_analysis(
                    file_path,
                    group_file,
                    save_dir / "Advanced_Differential",
                    group_order=group_order,
                    group_colors=group_colors,
                )
            else:
                print(f"[Advanced Differential] Skipped {dataset_name} (contains QCs).")
        except Exception as e:
            print(f"[Advanced Differential Error] {e}")

    print("\nAll statistical analyses completed.\n")
