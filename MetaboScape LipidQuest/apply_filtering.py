# TODO: add adduct handling
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
    
import pandas as pd
from pathlib import Path
from scoring_mammalians import apply_scoring
from plausability_filtering_mammalians import apply_plausability_filter

def collapse_duplicates(df):
    """
    Collapse duplicates only for annotated rows with a UniqueID.
    Keep the highest MS Score per UniqueID.
    Preserve unassigned rows and MS/MS matches regardless of UniqueID.
    """
    if "UniqueID" not in df.columns or "MS Score" not in df.columns or "Annotation" not in df.columns:
        return df

    ann = df["Annotation"].astype(str).str.strip()
    unassigned_mask = ann.eq("") | ann.eq("nan") | df["Annotation"].isna()

    # Split dataset
    assigned = df[~unassigned_mask].copy()
    unassigned = df[unassigned_mask].copy()

    # Deduplicate assigned rows with valid UniqueID
    with_uid = assigned[assigned["UniqueID"].notna()].copy()
    without_uid = assigned[assigned["UniqueID"].isna()].copy()

    dedup = (
        with_uid.sort_values("MS Score", ascending=False)
        .drop_duplicates(subset=["UniqueID"], keep="first")
    )

    # Recombine: deduped assigned + assigned without UID + all unassigned
    df_out = pd.concat([dedup, without_uid, unassigned], ignore_index=True)
    return df_out


def count_unassigned(df):
    if "Annotation" not in df.columns:
        return 0
    ann = df["Annotation"].astype(str).str.strip()
    return int(
        (ann.eq("") | ann.eq("nan") | ann.eq("Unassigned") | df["Annotation"].isna()).sum()
    )


def run_pipeline(input_csv="raw_ms_search_results.csv", output_folder="results", min_score=70):
    """
    Lipid filtering pipeline with debug printouts of unassigned counts.
    """
    input_path = Path(input_csv)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load
    df = pd.read_csv(input_path)
    print(f'Before filtering and scoring: {len(df)}, unassigned: {count_unassigned(df)}', flush=True)

    # Step 2: Apply scoring
    df = apply_scoring(df, output_folder)
    print(f'After scoring and RT filter: {len(df)}, unassigned: {count_unassigned(df)}', flush=True)

    # Step 3: Apply plausibility filter
    df = apply_plausability_filter(df, output_folder=output_path, mode="MS")
    print(f'After plausibility filter: {len(df)}, unassigned: {count_unassigned(df)}', flush=True)

    # Step 4: Collapse duplicates
    df = collapse_duplicates(df)
    print(f'After collapse duplicates: {len(df)}, unassigned: {count_unassigned(df)}', flush=True)

    # Step 5: Apply cutoff to scored rows, keep unassigned
    if "Annotation" in df.columns:
        ann = df["Annotation"].astype(str).str.strip()
        unassigned = ann.eq("") | ann.eq("nan") | ann.eq("Unassigned") | df["Annotation"].isna()
    else:
        unassigned = pd.Series([True] * len(df), index=df.index)

    df = df[(df["MS Score"] >= min_score) | (unassigned)].reset_index(drop=True)
    print(f'After cutoff: {len(df)}, unassigned: {count_unassigned(df)}', flush=True)

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

    ordered_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]
    
    drop_cols = ["Penalty breakdown", "Internal Standard", "Metaboscape Annotation Status", "Carbons / double bond equivalent ratio"]

    final_cols = [c for c in ordered_cols if c not in drop_cols] + [
        c for c in remaining_cols if c not in drop_cols
    ]

    df = df[final_cols]

    # Step 6: Save outputs
    scored_path = output_path / f"{input_path.stem}_scored.csv"
    filtered_path = output_path / "Final_search_results.csv"
    df.to_csv(scored_path, index=False, encoding="utf-8-sig")
    df.to_csv(filtered_path, index=False, encoding="utf-8-sig")

    return scored_path, filtered_path
