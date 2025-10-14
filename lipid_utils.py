import re
import pandas as pd

# ---------------- Fatty acyl parsing ----------------
def parse_fatty_acyls(name):
    """Return list of (carbons, double bonds) from a lipid name like 'CL 18:1_18:2_16:0_16:1'."""
    if not isinstance(name, str):
        return []
    return [(int(c), int(d)) for c, d in re.findall(r"(\d+):(\d+)", name)]


# ---------------- Chain type ----------------
def classify_chain_type(total_c, c1, c2):
    vals = [v for v in [total_c, c1, c2] if pd.notna(v) and v != ""]
    if not vals:
        return ""
    if any(int(v) % 2 != 0 for v in vals):
        return "odd"
    return "even"


# ---------------- PUFA ----------------
def classify_pufa(dbe):
    if pd.isna(dbe) or dbe == "" or dbe == 0:
        return "No"
    return "Yes" if dbe >= 2 else "No"


# ---------------- Modifications ----------------
def extract_modifications(name):
    if not isinstance(name, str):
        return ""
    parts = name.split(";", 1)
    if len(parts) < 2:
        return ""
    after = parts[1]
    if "/" in after:
        return ""
    return after.strip()


def count_modifications(mod_str):
    if not isinstance(mod_str, str) or not mod_str:
        return 0
    total = 0
    chunks = re.split(r"[;,\s]+", mod_str)
    for chunk in chunks:
        if not chunk:
            continue
        matches = re.findall(r"(\d*)([A-Za-z]+)(\d*)", chunk)
        for pre, token, post in matches:
            count = 1
            if pre.isdigit():
                count = int(pre)
            if post.isdigit():
                count = int(post)
            total += count
    return total


# ---------------- Oxidation ----------------
def is_oxidized(mods, lipid_class):
    mods = str(mods).upper()
    lipid_class = str(lipid_class)
    glyco_classes = {"HexCer", "LacCer", "GlcCer", "GalCer", "SM", "Cer", "ACer"}
    if not mods or "/" in mods:
        return "No"
    if lipid_class in glyco_classes and (mods.strip() in {"O2", "O3"}):
        return "No"
    if re.search(r"\bO\d*\b", mods) or "OH" in mods or "OOH" in mods or "OXO" in mods:
        return "Yes"
    return "No"
