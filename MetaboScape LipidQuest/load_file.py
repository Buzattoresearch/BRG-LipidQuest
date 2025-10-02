import re
import pandas as pd
from pathlib import Path
from lipid_utils import (
    parse_fatty_acyls, classify_chain_type, classify_pufa,
    extract_modifications, count_modifications, is_oxidized)
from data_cleansing import apply_data_cleansing

def read_metaboscape_table(path, sheet_name=None, index_col=None):
    path = Path(path)
    xls = pd.ExcelFile(path)
    s = sheet_name if sheet_name is not None else xls.sheet_names[0]

    df = pd.read_excel(xls, sheet_name=s, index_col=index_col, header=0)
    df.insert(0, "UniqueID", range(1, len(df) + 1))

    # Drop unwanted columns if they exist
    for col in ["Flags", "Boxplot", "AQ"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def load_headgroup_to_class(mapping_path):
    """Load the Headgroup_to_class mapping file and return a dict {headgroup: subclass}."""
    df_map = pd.read_csv(mapping_path, encoding="latin1")
    mapping = {}

    for _, row in df_map.iterrows():
        lipid_class = str(row["Lipid Class"]).strip()
        for col in df_map.columns[1:]:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                mapping[str(val).strip()] = lipid_class
    return mapping


def sanitize_file(path, output_folder=None):
    path = Path(path)
    df = read_metaboscape_table(path)

    # Add Headgroup
    if "Name" in df.columns:
        df["Headgroup"] = df["Name"].fillna("").apply(
            lambda x: str(x).split(" ")[0] if str(x).strip() else ""
        )

    # Remove isotope/deuterium labels
    df["Headgroup"] = df["Headgroup"].str.replace(r"\[D\d+\]", "", regex=True).str.strip()

    # Load mapping from Appendix
    program_folder = Path(__file__).resolve().parent
    mapping_path = program_folder / "Appendix" / "Headgroup_to_class.csv"
    headgroup_map = load_headgroup_to_class(mapping_path)

    # Map Headgroup → Lipid Class
    df["Lipid Class"] = df["Headgroup"].map(headgroup_map)
    df["Lipid Class"] = df["Lipid Class"].fillna("Other")
    df.loc[df["Headgroup"].isna() | (df["Headgroup"].str.strip() == ""), "Lipid Class"] = ""

    # Handle plasmalogens
    mask = df["Name"].str.contains(r"\bO-", na=False)
    df.loc[mask, "Headgroup"] = df.loc[mask, "Headgroup"] + " O-"
    df["Plasmenyl?"] = df["Name"].str.contains(r"\bO-", na=False).map({True: "Yes", False: "No"})

    def assign_annotation_type(name, msms_score):
        # IS has priority
        if isinstance(name, str) and re.search(r"\[D\d+\]", name):
            return "IS"
        # Otherwise, MS/MS match if score > 0
        if pd.notna(msms_score) and float(msms_score) > 0:
            return "MS/MS match"
        return ""

    df["Annotation Type"] = df.apply(
        lambda row: assign_annotation_type(row.get("Name", ""), row.get("MS/MS score", "")),
        axis=1
    )

    # --- Polarity ---
    def detect_polarity(ions):
        if not isinstance(ions, str) or not ions.strip():
            return ""
        first_ion = ions.split(",")[0].strip()
        if "]+" in first_ion:
            return "Pos"
        if "]-" in first_ion:
            return "Neg"
        return ""
    df["Polarity"] = df["Ions"].apply(detect_polarity)

    # Parse fatty acyls
    fa_info = df["Name"].apply(parse_fatty_acyls)
    for i in range(4):
        df[f"Number of carbons in fatty acyl {i+1}"] = fa_info.apply(lambda x: x[i][0] if len(x) > i else "")
        df[f"Double bonds in fatty acyl {i+1}"] = fa_info.apply(lambda x: x[i][1] if len(x) > i else "")

    df["Number of carbons in fatty acyls"] = fa_info.apply(lambda x: sum(c for c, _ in x) if x else "")
    df["Double bond equivalents"] = fa_info.apply(lambda x: sum(d for _, d in x) if x else "")

    # Derived annotations
    def _classify_chain_type(row):
        vals = [
            row.get("Number of carbons in fatty acyls"),
            row.get("Number of carbons in first fatty acyl"),
            row.get("Number of carbons in second fatty acyl"),
        ]
        vals = [v for v in vals if pd.notna(v) and v != ""]
        if not vals:
            return ""
        if any(int(v) % 2 != 0 for v in vals):
            return "odd"
        return "even"

    df["Chain type"] = df.apply(_classify_chain_type, axis=1)
    
    df["PUFA?"] = df["Double bond equivalents"].apply(classify_pufa)
    df["Modifications"] = df["Name"].apply(extract_modifications)
    df["# of modifications"] = df["Modifications"].apply(count_modifications)

    # Ratio safely
    df["Carbons / double bond equivalent ratio"] = (
        pd.to_numeric(df["Number of carbons in fatty acyls"], errors="coerce")
        / pd.to_numeric(df["Double bond equivalents"], errors="coerce")
    )
    df["Carbons / double bond equivalent ratio"] = (
        df["Carbons / double bond equivalent ratio"]
        .replace([float("inf"), -float("inf")], pd.NA)
        .fillna("")
    )

    # Oxidized?
    df["Oxidized?"] = df.apply(
        lambda row: is_oxidized(row.get("Modifications", ""), row.get("Lipid Class", "")), axis=1
    )

    # --- Annotation tier ---
    df["Annotation tier"] = df["MS/MS score"].apply(
        lambda x: "High confidence" if pd.notna(x) and x > 400 else ""
    )
    
    # Apply data cleansing 
    df, removed_df = apply_data_cleansing(df, output_folder, contaminant_file="Appendix/Contaminants.csv")

    # --- Reorder ---
    preferred_order = [
        "UniqueID", "RT [min]", "m/z meas.", "M meas.", "Ions", "Polarity",
        "Internal Standard", "QC RSD [%]", "Samples RSD [%]", "MS/MS", "Name",
        "Annotation Type", "Annotations", "Annotation Source", "Headgroup", "Lipid Class",
        "Δm/z [mDa]", "Δm/z [ppm]", "MS/MS score", "Annotation tier", "mSigma",
        "Molecular Formula", "Plasmenyl?", "Number of carbons in first fatty acyl",
        "Number of carbons in second fatty acyl", "Number of carbons in fatty acyls",
        "Double bond equivalents", "Chain type", "PUFA?", "Modifications",
        "# of modifications", "Oxidized?", "Carbons / double bond equivalent ratio",
    ]
    ordered_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    # Rename
    rename_map = {
        "RT [min]": "RT (min)", "m/z meas.": "m/z", "M meas.": "Neutral mass",
        "Ions": "Adducts", "QC RSD [%]": "QC RSD (%)", "Samples RSD [%]": "Sample RSD (%)",
        "MS/MS": "MS/MS available?", "Name": "Annotation", "Annotations": "Metaboscape Annotation Status", 
        "Δm/z [mDa]": "Δm/z (mDa)", "Δm/z [ppm]": "Δm/z (ppm)"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Save
    if output_folder:
        output_folder = Path(output_folder)
        if output_folder.suffix.lower() == ".csv":
            output_path = output_folder
        else:
            output_folder.mkdir(parents=True, exist_ok=True)
            output_path = output_folder / f"{path.stem}_sanitized.csv"
            output_path_cleansed = output_folder / f"{path.stem}_removed_contaminants.csv"
            output_path_noise = output_folder / f"{path.stem}_removed_noise.csv"
    else:
        output_path = path.with_name(f"{path.stem}_sanitized.csv")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    removed_df.to_csv(output_path_cleansed, index=False, encoding="utf-8-sig")
    return output_path, df
