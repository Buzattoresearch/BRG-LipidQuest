import pandas as pd
from pathlib import Path
import os
import hashlib, re, os


# ============================================================
# CLASS ORDER
# ============================================================

# === Class ordering and mapping ===
_CLASS_ORDER = [
    "CAR", "CoA", "FA", "FAHFA", "FAL, FAG, FOH, HC", "NA, NAE, NAT", "WE",
    "MG", "DG", "TG", "HexDG, HexMG", "DGTA, DGTS, DGCC, GlcADG", "SQDG, SQMG",
    "PA", "LPA", "PC", "LPC", "PE", "LPE", "PG", "LPG", "PI", "PIP", "LPI", "PS, PS-NAc", "LPS",
    "CL", "MLCL", "DLCL", "BMP", "PIM", "GP, Glc-GP",
    "ACer", "Cer", "CerP", "HexCer, GlcCer", "LSM", "MIPC, M(IP)2C", "PE-Cer, PI-Cer",
    "SCer", "SHexCer", "SM", "SPB, HexSPB, SPBP",
    "CE", "ST",
    "PK", "PR", "SL", "Other"
]
_CLASS_ORDER_BACTERIA = [
    #"CoA", 
    "CAR", "FA", "FAL, FOH", 
    "NA, NAE", 
    "WE",
    "MG", "DG", "TG", 
    "SQDG, SQMG",
    "HexDG, HexMG",
    "PA", "LPA", "PC", "LPC", "PE", "LPE", "PEth", "PG", "LPG", "PS, PS-NAc", "LPS",
    "CL", "MLCL", "DLCL", 
    "GP, Glc-GP",
    # "Cer", "HexCer, GlcCer", "MIPC, M(IP)2C",
    # "PK", 
    "PR", "SL", "Other"]

_CLASS_ORDER_MAMMALIAN = ["CAR", "CoA", "FA", "FAHFA", "FAL, FAG, FOH, HC", "NA, NAE, NAT", "WE",
    "MG", "DG", "TG",
    "PA", "LPA", "PC", "LPC", "PE", "LPE", "PEth", "PG", "LPG", "PI", "PIP", "LPI", "PS, PS-NAc", "LPS",
    "CL", "MLCL", "DLCL", "BMP", "GP, Glc-GP",
    "ACer", "Cer", "CerP", "HexCer, GlcCer", "LSM",
    "SCer", "SHexCer", "SM", "SPB, HexSPB, SPBP",
    "CE", "ST",
    "PR", "Other"]

_CLASS_ORDER_FUNGI = ["CAR", "CoA", "FA", "FAHFA", "FAL, FAG, FOH, HC", "NA, NAE, NAT", "WE",
    "MG", "DG", "TG",
    "PA", "LPA", "PC", "LPC", "PE", "LPE", "PEth", "PG", "LPG", "PI", "PIP", "LPI", "PS, PS-NAc", "LPS",
    "CL", "MLCL", "DLCL", 
    "BMP", "GP, Glc-GP",
    "ACer", "Cer", "CerP", "PE-Cer", "PI-Cer", "HexCer, GlcCer", "LSM",
    "MIPC",
    "SCer", "SHexCer", "SM", "SPB, HexSPB, SPBP",
    "CE", "ST",
    "PR", "Other"]

_CLASS_GROUP_MAP = {
    "CAR":"CAR","Car":"CAR","CoA":"CoA", "FA":"FA", "FAHFA":"FAHFA",
    "FAL":"FAL, FAG, FOH, HC","FAG":"FAL, FAG, FOH, HC","FOH":"FAL, FAG, FOH, HC","HC":"FAL, FAG, FOH, HC",
    "FAL":"FAL, FOH","FOH":"FAL, FOH",
    "NA":"NA, NAE, NAT","NAx":"NA, NAE, NAT", "NAE":"NA, NAE, NAT","NAT":"NA, NAE, NAT",
    "NA":"NA, NAE","NAx":"NA, NAE", "NAE":"NA, NAE",
    "WE":"WE",
    "MG":"MG", "DG":"DG", "TG":"TG","TG O-":"TG",
    "Hex2DG":"HexDG, HexMG","HexDG":"HexDG, HexMG","Hex2MG":"HexDG, HexMG","HexMG":"HexDG, HexMG",
    "DGTA":"DGTA, DGTS, DGCC, GlcADG","DGTS":"DGTA, DGTS, DGCC, GlcADG","DGCC":"DGTA, DGTS, DGCC, GlcADG","GlcADG":"DGTA, DGTS, DGCC, GlcADG","G":"DGTA, DGTS, DGCC, GlcADG",
    "SQDG":"SQDG, SQMG","SQMG":"SQDG, SQMG",
    "PA O-":"PA","PA":"PA","PPA":"PA",
    "LPA O-":"LPA","LPA":"LPA",
    "PC O-":"PC","PC":"PC","PnC":"PC",
    "LPC O-":"LPC","LPC":"LPC",
    "PE O-":"PE","PE":"PE","PnE":"PE",
    "LPE O-":"LPE","LPE":"LPE",
    "PEth": "PEth",
    "PG O-":"PG","PG":"PG",
    "LPG O-":"LPG","LPG":"LPG",
    "PI O-":"PI","PI":"PI",
    "LPI O-":"LPI","LPI":"LPI",
    "PS":"PS, PS-NAc","PS-NAc":"PS, PS-NAc","PS ":"PS, PS-NAc",
    "LPS O-":"LPS","LPS":"LPS",
    "CL":"CL", 
    "MLCL":"MLCL", 
    "DLCL":"DLCL",
    "BMP":"BMP","LBPA":"BMP",
    "PIM":"PIM",
    "GP":"GP, Glc-GP","Glc-GP":"GP, Glc-GP","CDP-DG":"GP, Glc-GP","PT":"GP, Glc-GP","LPT":"GP, Glc-GP",
    "ACer":"ACer","AC":"ACer",
    "C":"Cer","Cer":"Cer",
    "CerP":"CerP",
    "HexCer":"HexCer, GlcCer","GlcCer":"HexCer, GlcCer","H":"HexCer, GlcCer",
    "LSM":"LSM",
    "MIPC":"MIPC, M(IP)2C","M(IP)2C":"MIPC, M(IP)2C","IPC":"MIPC, M(IP)2C",
    "PE-Cer":"PE-Cer, PI-Cer","PI-Cer":"PE-Cer, PI-Cer","CerPE":"PE-Cer, PI-Cer","CerPI":"PE-Cer, PI-Cer",
    "SCer":"SCer", "SM":"SM", 
    "HexSPB":"SPB, HexSPB, SPBP","SPB":"SPB, HexSPB, SPBP","SPBP":"SPB, HexSPB, SPBP",   
    "CE":"CE", "ST":"ST","SFE":"ST", 
    "PK":"PK", "PR":"PR", "SL":"SL",
    "N":"Other","": "Other"
}

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
        "Chain type", "PUFA?", "Modifications", "# of modifications", "Oxidized?", 
        "CCS (Å²)", "Mob. 1/K0", "ΔCCS [%]",
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
