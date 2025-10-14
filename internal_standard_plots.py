# TODO: improve plot design. Add flag dor high m/z errors or RSD too high / too low
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_internal_standards(
    internal_standards_csv="Internal_standards.csv",
    output_folder="results"
):
    """
    Generate summary plots for internal standards:
      (1) Combined intensity plot (all standards together)
      (2) Δm/z error plots (Da and ppm) by polarity
      (3) Individual bar plots for each internal standard across all detected sample columns
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    individual_folder = output_folder / "Internal standards"
    individual_folder.mkdir(exist_ok=True)

    # --- Load data ---
    df = pd.read_csv(internal_standards_csv)
    if df.empty:
        print("No internal standards found. Skipping plotting.")
        return

    # --- Clean column names ---
    df.columns = [c.strip() for c in df.columns]

    # --- Dynamically detect sample columns ---
    sample_cols = [c for c in df.columns if str(c).startswith("[POS") or str(c).startswith("[NEG")]
    if not sample_cols:
        print("No sample columns detected (expected columns starting with '[POS' or '[NEG').")
        return
    print(f"Detected {len(sample_cols)} sample columns for plotting.")

    # Melt for grouped plotting
    df_long = df.melt(
        id_vars=["Annotation", "Polarity"],
        value_vars=sample_cols,
        var_name="Sample",
        value_name="Intensity"
    )

    # (1) Combined intensity plot (scatter)
    plt.figure(figsize=(12, 6))
    for name, sub in df_long.groupby("Annotation"):
        plt.scatter(sub["Sample"], sub["Intensity"], label=name, s=40, alpha=0.7)
    plt.xticks(rotation=90, ha="right", fontsize=7)
    plt.ylabel("Intensity")
    plt.title("Internal Standard Intensities (all standards)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(individual_folder / "Internal_standard_intensities.png", dpi=300)
    plt.close()

    # (2) Individual bar plots — one per internal standard
    for name, sub in df_long.groupby("Annotation"):
        plt.figure(figsize=(14, 5))
        plt.bar(sub["Sample"], sub["Intensity"], color="steelblue", edgecolor="black")
        plt.xticks(rotation=90, ha="right", fontsize=6)
        plt.ylabel("Intensity")
        plt.title(f"Internal Standard: {name}")
        plt.tight_layout()

        safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
        plt.savefig(individual_folder / f"{safe_name}_intensity.png", dpi=300)
        plt.close()

    # --- (3) m/z error plots ---
    if not {"Δm/z (mDa)", "Δm/z (ppm)"}.issubset(df.columns):
        print("Δm/z columns not found. Skipping error plots.")
        return

    df["Δm/z (mDa)"] = pd.to_numeric(df["Δm/z (mDa)"], errors="coerce")
    df["Δm/z (ppm)"] = pd.to_numeric(df["Δm/z (ppm)"], errors="coerce")

    for polarity in ["Pos", "Neg"]:
        sub = df[df["Polarity"].astype(str).str.startswith(polarity)]
        if sub.empty:
            continue

        # Δm/z (mDa)
        plt.figure(figsize=(8, 5))
        plt.scatter(sub["Annotation"], sub["Δm/z (mDa)"], s=60, color="darkred", alpha=0.7)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.axhline(0, color="gray", linestyle="--")
        plt.ylabel("Δm/z (mDa)")
        plt.title(f"Internal Standards Δm/z (mDa) — {polarity} mode")
        plt.tight_layout()
        plt.savefig(individual_folder / f"Internal_standards_deltamDa_{polarity}.png", dpi=300)
        plt.close()

        # Δm/z (ppm)
        plt.figure(figsize=(8, 5))
        plt.scatter(sub["Annotation"], sub["Δm/z (ppm)"], s=60, color="darkblue", alpha=0.7)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.axhline(0, color="gray", linestyle="--")
        plt.ylabel("Δm/z (ppm)")
        plt.title(f"Internal Standards Δm/z (ppm) — {polarity} mode")
        plt.tight_layout()
        plt.savefig(individual_folder / f"Internal_standards_deltappm_{polarity}.png", dpi=300)
        plt.close()

    print(f"Internal standard plots saved to: {individual_folder}")

if __name__ == "__main__":
    plot_internal_standards("results/Internal_standards.csv", "results")
