# TODO: improve plot design. Add flag dor high m/z errors or RSD too high / too low
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_internal_standards(pol_tag,
    internal_standards_csv="Internal_standards.csv",
    output_folder="results"
):
    """
    Generate summary plots for internal standards:
      (1) Combined intensity plot (all standards together)
      (2) Δm/z error plots (Da and ppm) by polarity
      (3) Individual bar plots for each internal standard across all detected sample columns
    """
    print(f'\nGenerating internal standard plots for {pol_tag}...\n', flush = True)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if "P" in str(pol_tag):
        polarity = "Pos"
    elif "N" in str(pol_tag):
        polarity = "Neg"
        
    # Normalize path and infer polarity tag from filename (Pos_/Neg_)
    internal_standards_path = Path(output_folder).parent / polarity / f"{pol_tag}Internal_standards.csv"
    print(f'Internal standard file path: {internal_standards_path}', flush = True)
    fname = internal_standards_csv.name

    individual_folder = output_folder / "Internal standards"
    individual_folder.mkdir(exist_ok=True)

    # Separate folders
    main_ion_folder = individual_folder / "Main_ions"
    other_plots_folder = individual_folder / "Other_plots"

    main_ion_folder.mkdir(exist_ok=True)
    other_plots_folder.mkdir(exist_ok=True)

    # --- Load data ---
    df = pd.read_csv(internal_standards_path, low_memory=False)

    if df.empty:
        print("No internal standards found. Skipping plotting.", flush=True)
        return

    # Clean column names immediately (CRITICAL for sample detection)
    df.columns = [c.strip() for c in df.columns]

    # Load group assignments if available
    group_file = Path(output_folder).parent / "sample_groups.csv"

    groups_df = None
    group_map = {}

    if group_file.exists():
        try:
            groups_df = pd.read_csv(group_file)
            if groups_df.empty:
                print("\n ****** Group file is empty. Skipping grouping. *****", flush=True)
                groups_df = None
            else:
                groups_df.columns = [c.strip() for c in groups_df.columns]
                group_map = dict(zip(groups_df["Sample"], groups_df["Group"]))
        except Exception as e:
            print(f"[WARNING] Failed to load sample_groups.csv: {e}", flush=True)
            groups_df = None

    # Detect sample columns AFTER cleaning
    sample_cols = [c for c in df.columns if c.startswith("P_") or c.startswith("N_")]

    if not sample_cols:
        print("No sample columns detected.", flush=True)
        return

    print("\nDETECTED SAMPLE MAPPING:", flush=True)
    for s in sample_cols:
        print(f"{s}: group = '{group_map.get(s, None)}'", flush=True)

    print(f"Detected {len(sample_cols)} sample columns for plotting.")


    # Melt for grouped plotting
    id_cols = ["Annotation", "Polarity"]
    if "UniqueID" in df.columns:
        id_cols.append("UniqueID")
    else:
        print("[WARN] 'UniqueID' not found; filenames will not be UniqueID-specific.", flush = True)

    df_long = df.melt(
        id_vars=id_cols,
        value_vars=sample_cols,
        var_name="Sample",
        value_name="Intensity"
    )

    # Add group information
    if groups_df is not None:
        df_long = df_long.merge(groups_df[["Sample", "Group"]], on="Sample", how="left")
    else:
        df_long["Group"] = "Unknown"

    # -------------------------------------------------------------------
    # Identify the main ion per internal standard
    # -------------------------------------------------------------------

    main_ions = []

    for annotation, sub in df.groupby("Annotation"):
        # split by polarity
        sub_pol = sub[sub["Polarity"].astype(str).str.startswith(polarity)]
        if sub_pol.empty:
            continue
        
        # compute metrics for each UniqueID within this annotation
        for uid, ion in sub_pol.groupby("UniqueID"):
            vals = ion[sample_cols].to_numpy().astype(float).flatten()
            vals = vals[~pd.isna(vals)]

            if len(vals) == 0:
                continue

            median_int = float(pd.Series(vals).median())
            detect_rate = float((vals > 0).sum() / len(vals))
            rsd = float(pd.Series(vals).std() / pd.Series(vals).mean() * 100 if pd.Series(vals).mean() != 0 else 9999)

            main_ions.append({
                "Annotation": annotation,
                "UniqueID": uid,
                "MedianIntensity": median_int,
                "DetectionRate": detect_rate,
                "RSD": rsd,
                "Delta_mDa": float(ion.iloc[0]["Δm/z (mDa)"]),
                "Delta_ppm": float(ion.iloc[0]["Δm/z (ppm)"])
            })

    main_df = pd.DataFrame(main_ions)

    # Rank and select the best ion per internal standard
    main_df = main_df.sort_values(
        by=["Annotation", "MedianIntensity", "DetectionRate", "RSD"],
        ascending=[True, False, False, True]
    )

    best_main = main_df.groupby("Annotation").first().reset_index()

    print("\nSELECTED MAIN IONS PER STANDARD:\n", best_main)

    # Create lookup for main-ion (Annotation, UniqueID) pairs
    main_ion_keys = set(zip(best_main["Annotation"], best_main["UniqueID"]))
    
    # -------------------------------------------------------------------
    # BARPLOTS OF MAIN-ION M/Z ERRORS
    # -------------------------------------------------------------------

    if not best_main.empty:
        # mDa error
        plt.figure(figsize=(12, 5))
        plt.bar(best_main["Annotation"], best_main["Delta_mDa"], color="steelblue", edgecolor="black")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.ylabel("Δm/z (mDa)")
        plt.title(f"Main Ion m/z Error (mDa) — {polarity} mode")
        plt.tight_layout()
        plt.savefig(main_ion_folder / f"{pol_tag}Main_Ions_deltamDa.png", dpi=100)
        plt.close()

        # ppm error
        plt.figure(figsize=(12, 5))
        plt.bar(best_main["Annotation"], best_main["Delta_ppm"], color="darkred", edgecolor="black")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.ylabel("Δm/z (ppm)")
        plt.title(f"Main Ion m/z Error (ppm) — {polarity} mode")
        plt.tight_layout()
        plt.savefig(main_ion_folder / f"{pol_tag}Main_Ions_deltappm.png", dpi=100)
        plt.close()
    else:
        print("No main-ion data available; skipping main-ion m/z error plots.")

    
    # ---------------------------------------
    # Combined intensity plot (scatter)
    # ---------------------------------------
    
    plt.figure(figsize=(12, 6))
    for name, sub in df_long.groupby("Annotation"):
        plt.scatter(sub["Sample"], sub["Intensity"], label=name, s=40, alpha=0.7)
    plt.xticks(rotation=90, ha="right", fontsize=7)
    plt.ylabel("Intensity")
    plt.title("Internal Standard Intensities (all standards)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(other_plots_folder / f"{pol_tag}Internal_standard_intensities.png", dpi=100)
    plt.close()

    # ---------------------------------------
    # Individual bar plots — one per internal standard INSTANCE
    # ---------------------------------------
    
    group_keys = ["Annotation"]
    if "UniqueID" in df_long.columns:
        group_keys.append("UniqueID")

    for keys, sub in df_long.groupby(group_keys):
        # -----------------------------------------------------
        # FORCE QC samples to appear last in barplots and boxplots
        # -----------------------------------------------------
        # Two groups: Non-QC first, then QC
        if "Group" in sub.columns:
            qc_mask = sub["Group"].str.lower() == "qc"
            non_qc = sub.loc[~qc_mask].sort_values("Sample")
            qc_only = sub.loc[qc_mask].sort_values("Sample")
            sub = pd.concat([non_qc, qc_only], axis=0)
        if isinstance(keys, tuple):
            name, uid = keys[0], keys[1] if len(keys) > 1 else None
        else:
            name, uid = keys, None

        plt.figure(figsize=(14, 5))
        # One color per group
        unique_groups = sub["Group"].unique()
        group_colors = dict(zip(unique_groups, plt.cm.Set3(range(len(unique_groups)))))

        bar_colors = [group_colors[g] for g in sub["Group"]]

        plt.bar(
            sub["Sample"],
            sub["Intensity"],
            color=bar_colors,
            edgecolor="black"
        )

        plt.xticks(rotation=90, ha="right", fontsize=6)
        plt.ylabel("Intensity")

        # Optional legend
        handles = [plt.Rectangle((0,0),1,1, color=group_colors[g]) for g in unique_groups]
        plt.legend(handles, unique_groups, title="Group", fontsize=7)

        # Pull metadata from the original internal standards df
        meta = df[(df["Annotation"] == name) & (df["UniqueID"] == uid)]

        if not meta.empty:
            mz = meta.iloc[0].get("m/z")
            rt = meta.iloc[0].get("RT (min)")
        else:
            mz = None
            rt = None

        mz_str = f", m/z {mz:.4f}" if isinstance(mz, (int, float)) else ""
        rt_str = f", RT {rt:.2f} min" if isinstance(rt, (int, float)) else ""

        title = (
            f"Internal Standard: {name}"
            + (f" (UniqueID {uid})" if uid is not None else "")
            + mz_str
            + rt_str
        )
        plt.title(title)

        plt.tight_layout()

        safe_name = "".join(c for c in str(name) if c.isalnum() or c in (" ", "_", "-")).strip()
        suffix = f"_{uid}" if uid is not None else ""
        # Save intensity bar plot depending on whether this is the main ion
        if (name, uid) in main_ion_keys:
            plt.savefig(main_ion_folder / f"{pol_tag}{safe_name}{suffix}_intensity.png", dpi=100)
        else:
            plt.savefig(other_plots_folder / f"{pol_tag}{safe_name}{suffix}_intensity.png", dpi=100)
        plt.close()

        # ------------------------------
        # Group-aware boxplot for each Internal Standard
        # ------------------------------
        if "Group" in sub.columns:
            # Ensure QC is last AND values inside each group preserve the new ordering
            grouped_sub = (
                sub.groupby("Group", sort=False)["Intensity"]
                .agg(list)
            )

            plt.figure(figsize=(8, 4))
            
            # Plot boxplots
            bp = plt.boxplot(
                grouped_sub.tolist(),
                labels=grouped_sub.index,
                vert=True,
                patch_artist=True,
                boxprops=dict(edgecolor='black'),
                capprops=dict(color='black'),
                whiskerprops=dict(color='black'),
            )

            # Fill each box with a distinct color
            colors = plt.cm.Set3(range(len(bp["boxes"])))
            for box, color in zip(bp["boxes"], colors):
                box.set_facecolor(color)
                box.set_edgecolor("black")

            for whisker in bp["whiskers"]:
                whisker.set_color("black")
            for cap in bp["caps"]:
                cap.set_color("black")
            for median in bp["medians"]:
                median.set_color("black")
            plt.ylabel("Intensity")
            plt.xticks(rotation=45, ha="right", fontsize=7)

            bp_title = (
                f"Boxplot: {name}"
                + (f" (UniqueID {uid})" if uid is not None else "")
                + mz_str
                + rt_str
            )

            plt.title(bp_title)
            plt.tight_layout()
            # Save boxplot to the correct folder depending on main-ion status
            if (name, uid) in main_ion_keys:
                plt.savefig(main_ion_folder / f"{pol_tag}{safe_name}{suffix}_boxplot.png", dpi=100)
            else:
                plt.savefig(other_plots_folder / f"{pol_tag}{safe_name}{suffix}_boxplot.png", dpi=100)
            plt.close()
        else:
            print(f"[WARN] No group information available: skipping grouped boxplot for {name}")

    # --- m/z error plots (all) ---
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
        plt.savefig(other_plots_folder / f"{pol_tag}Internal_standards_deltamDa.png", dpi=100)
        plt.close()

        # Δm/z (ppm)
        plt.figure(figsize=(8, 5))
        plt.scatter(sub["Annotation"], sub["Δm/z (ppm)"], s=60, color="darkblue", alpha=0.7)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.axhline(0, color="gray", linestyle="--")
        plt.ylabel("Δm/z (ppm)")
        plt.title(f"Internal Standards Δm/z (ppm) — {polarity} mode")
        plt.tight_layout()
        plt.savefig(other_plots_folder / f"{pol_tag}Internal_standards_deltappm.png", dpi=100)
        plt.close()

    print(f"Internal standard plots saved to: {main_ion_folder} and {other_plots_folder}")

if __name__ == "__main__":
    plot_internal_standards(pol_tag="Pos_", output_folder="results")
