import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_dataset(file_path, group_file):
    """
    Load statistical dataset in the standard LipidQuest format.
    Returns (X, y, feature_meta)
    - X: samples × features (numeric)
    - y: sample group labels (Series)
    - feature_meta: feature-level metadata
    """
    df = pd.read_csv(file_path)
    if df.empty:
        print(f"[load_dataset] ⚠ Empty dataset: {file_path}", flush = True)
        return pd.DataFrame(), pd.Series(dtype=str), pd.DataFrame()

    # Determine sample columns (exclude known metadata)
    meta_cols = [
        "UniqueID", "RT (min)", "m/z", "Polarity", "Annotation",
        "Annotation Type", "Headgroup", "Lipid Class", "Δm/z (mDa)", "Δm/z (ppm)",
        "MS/MS score", "Annotation tier", "mSigma", "Molecular Formula",
        "Plasmenyl?", "Number of carbons in fatty acyls", "Double bond equivalents",
        "Chain type", "PUFA?", "Modifications", "# of modifications", "Oxidized? (remove the RSD columns)"
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    sample_cols = [c for c in df.columns if c not in meta_cols]

    # Load groups
    if group_file is not None and os.path.exists(group_file):
        df_groups = pd.read_csv(group_file)
        if "Sample" not in df_groups.columns or "Group" not in df_groups.columns:
            raise ValueError(f"[load_dataset] Invalid group file format: {group_file}")
    else:
        print("[load_dataset] ⚠ No group file provided — using default 'Unknown' group.", flush = True)
        df_groups = pd.DataFrame({"Sample": sample_cols, "Group": "Unknown"})

    # Normalize sample names (strip spaces, case)
    df_groups["Sample"] = df_groups["Sample"].astype(str).str.strip()
    df_groups["Group"] = df_groups["Group"].astype(str).str.strip()

    # Match samples (case-insensitive)
    df_cols_lower = {c.lower(): c for c in sample_cols}
    matched = []
    for s in df_groups["Sample"]:
        if s.lower() in df_cols_lower:
            matched.append(df_cols_lower[s.lower()])
    
    if "UniqueID" in df.columns:
        df["UniqueID"] = df["UniqueID"].astype(str)
    
    if len(matched) == 0:
        print(f"[load_dataset] ❌ No matching sample names between {file_path.name} and {group_file}.", flush = True)
        print(f"[load_dataset] First few dataset cols: {sample_cols[:5]}", flush = True)
        print(f"[load_dataset] First few group samples: {df_groups['Sample'].head(5).tolist()}", flush = True)
        return pd.DataFrame(), pd.Series(dtype=str), pd.DataFrame()

    # Extract data and groups
    X = df[matched].T
    X.index.name = "Sample"
    y = df_groups.set_index("Sample").loc[X.index, "Group"]

    # Keep feature metadata and link each feature to its UniqueID
    feature_meta = df[meta_cols].copy()

    # Use UniqueID as column names for features
    if "UniqueID" in feature_meta.columns:
        X.columns = feature_meta["UniqueID"].astype(str).tolist()
    else:
        X.columns = [f"Feature_{i+1}" for i in range(X.shape[1])]

    # Replace any non-numeric or NaN
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"[load_dataset] ✅ Loaded {X.shape[0]} samples × {X.shape[1]} features.", flush = True)
    return X, y, feature_meta


def prepare_output_dir(save_dir: Path) -> Path:
    """Ensure output directory exists."""
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
