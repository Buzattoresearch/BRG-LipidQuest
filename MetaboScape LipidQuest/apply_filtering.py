import pandas as pd
from pathlib import Path
from scoring_mammalians import apply_scoring
from plausability_filtering_mammalians import apply_plausability_filter


def collapse_duplicates(df):
    """
    Keep only the highest MS Score annotation per UniqueID.
    """
    if "UniqueID" not in df.columns or "MS Score" not in df.columns:
        return df

    df = df.sort_values("MS Score", ascending=False)
    df = df.drop_duplicates(subset=["UniqueID"], keep="first")
    return df.reset_index(drop=True)


def run_pipeline(input_csv="raw_ms_search_results.csv", output_folder="results", min_score=70):
    """
    Lipid filtering pipeline:
    1. Load raw search results
    2. Apply scoring
    3. Filter by biological plausibility (inline here)
    4. Collapse duplicates
    5. Apply minimum score cutoff
    6. Save outputs
    7. Plot results
    """
    input_path = Path(input_csv)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load
    df = pd.read_csv(input_path)
    print(f'Before filtering and scoring: {len(df)}', flush=True)

    # Step 2: Apply scoring (keeps rows with no annotation untouched)
    df = apply_scoring(df, output_folder)
    print(f'After RT filter and scoring: {len(df)}', flush=True)

    # Step 3: Apply plausibility filter (removes biologically unreasonable annotations)
    df = apply_plausability_filter(df, output_folder=output_path, mode="MS")
    print(f'After plausability filter: {len(df)}', flush=True)

    # Step 4: Collapse duplicates
    df = df.sort_values("MS Score", ascending=False).drop_duplicates(subset=["UniqueID"], keep="first")

    # Step 5: Apply cutoff
    df = df[df["MS Score"] >= min_score].reset_index(drop=True)

    # --- Reorder ---
    preferred_order = [
        "UniqueID", "RT (min)", "m/z", "Neutral mass", "Adducts", "Polarity",
        "QC RSD (%)", "Samples RSD (%)", 
        "Annotation", "Headgroup", "Lipid Class", "Molecular Formula",
        "Δm/z (mDa)", "Δm/z (ppm)", "Annotation tier",
        "Annotation Type", "Annotation Source", "Metaboscape Annotation Status", "MS/MS available?", 
        "MS/MS score", "mSigma", 
        "MS Score", "LIPIDMAPS ID (MS matches)", "Matched adduct (MS matches)", "Matched Mass (MS matches)", 
        "Number of carbons in fatty acyls", 
        "Number of carbons in fatty acyl 1", "Number of carbons in fatty acyl 2", "Number of carbons in fatty acyl 3", "Number of carbons in fatty acyl 4", 
        "Double bond equivalents", "Double bonds in fatty acyl 1", "Double bonds in fatty acyl 2", "Double bonds in fatty acyl 3", "Double bonds in fatty acyl 4", 
        "Chain type", "PUFA?",
        "Plasmenyl?", 
        "Modifications", "# of modifications", "Oxidized?", "Carbons / double bond equivalent ratio",
        "Penalty breakdown"
    ]

    # Final order = preferred order + any baseline/extra columns not in preferred
    ordered_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]
    
    # Columns you don't want in the final file
    drop_cols = ["Penalty breakdown", "Internal Standard", "Metaboscape Annotation Status", "Carbons / double bond equivalent ratio"]

    # Build final column list: reorder first, drop unwanted ones
    final_cols = [c for c in ordered_cols if c not in drop_cols] + [
        c for c in remaining_cols if c not in drop_cols
    ]

    # Apply it
    df = df[final_cols]

    # Step 6: Save outputs
    scored_path = output_path / f"{input_path.stem}_scored.csv"
    filtered_path = output_path / "Final_search_results.csv"
    df.to_csv(scored_path, index=False, encoding="utf-8-sig")
    df.to_csv(filtered_path, index=False, encoding="utf-8-sig")

    return scored_path, filtered_path
