#TODO: add scoring system, check performance

# -----------------------------
# BACTERIA
# -----------------------------

import os
import pandas as pd
from collections import Counter
from pathlib import Path

def apply_plausability_filter(df, output_folder, mode="MS", row_type="Annotation", rt_field="RT (min)"):
    """
    Apply organism plausibility filters to a pandas DataFrame.
    Saves removed rows (with reason) to CSV and returns filtered DataFrame.
    """
    print('\n\n -----------  APPLYING FILTERS FOR BACTERIA --------- \n\n')
    
    removal_reasons = Counter()
    removed_rows = []

    def add_removed_row(row, reason):
        row_copy = row.copy()
        row_copy["removed_reason"] = reason
        removed_rows.append(row_copy)
        removal_reasons[reason] += 1

    print(f"\nStarting plausibility filter. Number of matches: {len(df)}")

    rows_to_drop = []
    for idx, row in df.iterrows():
        val = str(row.get(row_type, ""))

        # Skip metadata or unidentified rows
        if not str(row.get(rt_field, "")).replace(".", "", 1).isdigit():
            continue
        if "Name" in row and not row["Name"]:
            continue

        remove_flag = False

        if [row["Polarity"]] == "Neg" and (row["Lipid Class"] in ["TG", "CE", "MG", "DG", "WE", "HC"]):  #Lipids not detected in negative ionization
            remove_flag = True
            (row, "Implausible polarity")

        if [row["Polarity"]] == "Pos" and (row["Lipid Class"] in ["FA"]):  #Lipids not detected in positive ionization
            remove_flag = True
            (row, "Implausible polarity")
                
        # Implausible classes
        if val.startswith(("ACer ", "Car ", "CE ", "Cer", "CerP ", "DGCC ", "DGDG ", "DGMG ", "MGMG ", "MGDG ", "DGTA ", "DGTS ", "MGTS ", "FAG ", "FAHFA ", "GlcADG ", 
                           "GlcCer", "Glc-GP ", "HC ", "HexCer", "HexSPB ", "PS-NAc ","LSM ", "MIPC ", "M(IP)2C ", "IPC ", "NAT ", "PIP ", "PnC ", "PnE ", "SCer ", "SHexCer ", 
                           "PI ", "PS ", "SM ", "SPB ", "SPBP ", "ST ", "SulfateHexSPB ", "PE-Cer ", "PI-Cer ", "NAE ", "BMP ", "PIM ", "LPS ", "WE ", "NAT ", 
                           "FAL ", "FOH ", "LNAPE ", "NAPE ",  "PT ", "LPT ", "PEth ", 'PEtOH ', 'PMeOH ', 'PE-NMe', 'PE-NMe2', 'CPA', 'FAL ', 'FOH ')): 
            remove_flag = True
            add_removed_row(row, "Implausible classes")  
                
        # === Impossible plasmalogens ===
        if (" O-" in val or # Plasmalogens and ether-linked lipids in general are not found in Gram-negatives. They can only be found in Gram-positives and archea.
            "DG O-" in val or "DG dO-" in val or " dO-" in val or "MG O-" in val or
            "DGDG O-" in val or "DGMG O-" in val or "MGDG O-" in val or "MGMG O" in val or
            "PG O-" in val or "PA O-" in val or "PI O-" in val or "PS O-" in val or
            "PE-NMe2 O-" in val or ("O-" in val and ";" in val) or
            "O-14:1" in val or "O-15:" in val or "O-17:" in val or "O-19:" in val or
            "O-21:" in val or "O-23:" in val or "O-25:" in val or "O-26:" in val or
            "O-27:" in val or "O-28:" in val or "O-29:" in val or "O-31:" in val or
            "O-33:" in val or "O-35:" in val or "O-37:" in val or "O-39:" in val or
            "O-41:" in val or "O-44:7" in val or "O-43:" in val or "O-46:7" in val or
            "O-45:" in val or "O-47:" in val or "O-48:" in val or "O-49:" in val or
            "O-51:" in val or "O-53:" in val or "O-55:" in val or "O-57:" in val or
            "O-59:" in val or "O-61:" in val or "O-63:" in val or "O-65:" in val or
            "O-66:" in val or "O-67:" in val or "O-68:" in val or "O-69:" in val or
            "O-70:" in val or "O-71:" in val or "O-72:" in val or "O-73:" in val or
            "O-75:" in val or "O-77:" in val or "O-79:" in val or "O-81:" in val or
            "CPA O-" in val or "Glc-GP O-" in val):
            remove_flag = True
            add_removed_row(row, "Impossible plasmalogens")

        #  === Contaminants, strange chain lengths ===
        if ("contaminant" in val or
            "DEET" in val or "marine" in val or "fungi" in val):
            remove_flag = True
            add_removed_row(row, "Contaminants")
            
        #Odd FAs
        if ("PR" not in val and 
            "PK" not in val and 
            "SL" not in val and 
            "ST" not in val and (
                " 2:" in val or " 3:" in val or " 4:" in val or
                " 5:" in val or " 6:" in val or " 7:" in val or
                " 8:" in val or " 9:" in val or " 11:" in val or
                " 13:" in val or                
                "_3:" in val or "_4:" in val or "_5:" in val or
                "_6:" in val or "_7:" in val or "_8:" in val or
                "_9:" in val or "_11:" in val or "_13:" in val or
                ":4" in val or ":5" in val or ":6" in val or                        # PUFA are not found in bacteria
                ":7" in val or ":8" in val or ":9" in val or                        # PUFA are not found in bacteria
                ":10" in val or ":11" in val or ":12" in val or                     # PUFA are not found in bacteria
                ":13" in val or ":14" in val or ":15" in val or                     # PUFA are not found in bacteria
                "12:0_12:" in val or "12:0_14:" in val or "12:0_15:" in val or      # odd combinations of FAs
                "12:0_13:" in val or "12:0_17:" in val or "12:0_19:" in val or      
                "12:0_20:" in val or "13:0_12:" in val or "13:0_13:" in val or      
                "13:0_15:" in val or "13:0_17:" in val or "13:0_19:" in val or      
                "13:0_20:" in val or                                                    
                "14:0;O" in val or "14:1;O" in val or
                "14:0_14:" in val or "14:0_15:" in val or "14:0_17:" in val or 
                "14:0_19:" in val or "14:0_20:" in val or 
                "14:2" in val or "14:3" in val or  "14:4" in val or
                "15:1;O" in val or "15:1_12:" in val or "15:1_13:" in val or
                "15:1_15:" in val or "15:1_17:" in val or "15:1_19:" in val or
                "15:1_20:" in val or "15:1_21:" in val or "15:1_22:" in val or
                "15:2" in val or "15:3" in val or "15:4" in val or
                "16:2" in val or "16:3" in val or "16:4" in val or
                "17:0_12:" in val or "17:0_13:" in val or "17:0_15:" in val or
                "17:0_17:" in val or "17:0_19:" in val or "17:0_21:" in val or
                "17:1;O" in val or "17:1_12:" in val or "17:1_13:" in val or
                "17:1_15:" in val or "17:1_17:" in val or "17:1_19:" in val or "17:1_21:" in val or
                "17:2" in val or "17:3" in val or "17:4" in val or
                "19:0;O" in val or 
                "19:0_12:" in val or "19:0_13:" in val or "19:0_14:" in val or
                "19:0_15:" in val or "19:0_17:" in val or "19:0_19:" in val or
                "19:0_21:" in val or
                "19:1;O" in val or
                "19:1_12:" in val or "19:1_13:" in val or "19:1_14:" in val or
                "19:1_15:" in val or "19:1_17:" in val or "19:1_19:" in val or
                "19:1_21:" in val or
                "19:2" in val or "19:3" in val or
                "20:1;O3" in val or "20:2;O" in val or "20:3" in val or
                "21:0_12:" in val or "21:0_13:" in val or "21:0_14:" in val or
                "21:0_15:" in val or "21:0_17:" in val or "21:0_19:" in val or
                "21:0_21:" in val or
                "21:1;O" in val or "21:2" in val or "21:3" in val or
                "22:" in val or                                                         # only found in eukaryotic organisms
                "23:" in val or
                "24:3" in val or "24:5" in val or "24:7" in val or "24:7;O3" in val or
                "25:" in val or
                "26:" in val or                                                         # only found in eukaryotic organisms
                "27:" in val or
                "28:2" in val or "28:3" in val or "28:4" in val or
                "29:" in val or
                "31:1;O" in val or "31:2;O" in val or
                "32:0;O2" in val or "32:2;O2" in val or
                "33:5" in val or "33:6;O" in val or
                "36:1;O" in val or
                "37:" in val or
                "39:" in val or
                "41:" in val or
                "42:5;O3" in val or
                "43:" in val or "43:0;O3" in val or
                "44:9" in val or
                "45:" in val or
                "46:4;O3" in val or
                "47:" in val or
                "48:0" in val or "48:3;O" in val or
                "49:" in val or "49:3;O" in val or
                "50:11" in val or
                "51:" in val or
                "52:13" in val or
                "53:" in val or "55:" in val or "57:" in val or
                "59:" in val or
                "66:8" in val
                )):
            remove_flag = True
            add_removed_row(row, "Odd FAs")

        # Lipids with more than 11 extra oxygens    
        if any(f";O{i}" in val for i in range(11, 100)):
            remove_flag = True
            add_removed_row(row, "Lipids with more than 11 extra oxygens ")
        
        # Lipids with more than 14 double bonds
        if any(f":{i}" in val for i in range(14, 100)):
            remove_flag = True
            add_removed_row(row, "Lipids with more than 14 double bonds") 
                
        # Rare fatty acyls
        if val.startswith(("FOH ", "FAL ", "HC ", "FAG ", "SFE ")): 
            if (any(f";O{i}" in val for i in range(6, 100)) or
                any(f"{i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59))
                ):
                remove_flag = True
                add_removed_row(row, "Rare FA with too many extra oxygens or OCFA")                
                        
        # Sphingolipids with too many extra oxygens, too many or too few carbons, or odd carbon numbers
        if val.startswith(("Cer ", "SM ", "HexCer ", "Hex2Cer ", "SPB ", "SPBP ", "SHexCer ", "CerP ")): 
            if (any(f";O{i}" in val for i in range(5, 11)) or
                any(f":{i}" in val for i in range(4,100)) or
                any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or
                any(f"_{i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60)) or
                any(f"/{i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 57, 59))
                ):
                remove_flag = True
                add_removed_row(row, "Odd sphingolipids")           
            
        # AcylCeramides with too many extra oxygens, too many carbons, or odd carbon numbers
        if val.startswith(("ACer ")):
            if (any(f";O{i}" in val for i in range(4, 7)) or
                any(f" {i}:" in val for i in range(57, 100)) or
                any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or
                any(f"/{i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60)) or
                any(f"_{i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60))
                ):
                remove_flag = True
                add_removed_row(row, "Odd acylCeramides")
                        
        # HexCer
        if val.startswith(("HexCer ", "Hex2Cer ", "Hex(2)Cer ", 'CerP')):
            if (any(f";O{i}" in val for i in range(4, 100)) or
            any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 12, 13, 15, 17, 19, 21, 23, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51)) or
            any(f" {i}:" in val for i in range (51, 100)) or
            any(f" :{i}" in val for i in range(8, 100)) or
            any(f"/{i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25)) or
            any(f"_{i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25)) or
            any(f"_{i}:" in val for i in range(26, 100)) or
            any(f"/{i}:" in val for i in range(26, 100))
            ):
                remove_flag = True
                add_removed_row(row, "Odd HexCer")
                            
        #Acylglycerols and fatty acyls with too many oxygens, nitrogens, or double bonds, OR too few carbons (too small)                     
        if val.startswith(("Car ", "CAR ", "CoA ", "COA ", "Coa ", "FAL ", "FOH ", "FAG ", "MG ", "TG ", "DG ")): 
            if (any(f";O{i}" in val for i in range(3, 100)) or
                any(f";N{i}" in val for i in range(3, 100)) or
                any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 13, 19, 21, 23, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51)) or
                any(f"_{i}:" in val for i in (3, 5, 7, 9, 11, 13, 19, 21, 23, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51)) or
                any(f"/{i}:" in val for i in (3, 5, 7, 9, 11, 13, 19, 21, 23, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51))
                ):
                remove_flag = True
                add_removed_row(row, "Odd acylGlycerols or FA")
                
        #Car (Short- and medium-chain acylcarnitines (≤C14) could be encountered in bacteria, but generally only if they are exposed to external carnitine or supplemented media (e.g., in host environments or specialized growth media. Long- and very-long-chain acylcarnitines (≥C18) are not plausible bacterial products)                  
        if val.startswith(("Car ", "CoA ")): 
            if (any(f" {i}:" in val for i in range(1, 11)) or  
                any(f" {i}:" in val for i in range(21, 100)) or  
                any(f";O{i}" in val for i in range(1, 100)) or  
                any(f":{i}" in val for i in range(3, 100))
                ):
                remove_flag = True
                add_removed_row(row, "Odd Car or CoA")
                
        #AFree fatty acyls with too many oxygens, nitrogens, or double bonds, OR too few carbons (too small)                     
        if val.startswith("FA "): 
            if (any(f";O{i}" in val for i in range(4, 100)) or
                any(f";N{i}" in val for i in range(4, 100)) or
                any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 13, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51)) or
                any(f"_{i}:" in val for i in (3, 5, 7, 9, 11, 13, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51)) or
                any(f"/{i}:" in val for i in (3, 5, 7, 9, 11, 13, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51))
                ):
                remove_flag = True
                add_removed_row(row, "Odd free FA")
            
        #NAs or NAEs with too many or too few carbons, too many double bonds, and too many nitrogens or oxygens            
        if val.startswith(("NA ", "NAE ", "NAT ")):
            if (any(f" {i}:" in val for i in range(27, 100)) or  
                any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27)) or  
                any(f":{i}" in val for i in range(6, 100)) or
                any(f";O{i}" in val for i in range(3, 100)) or
                any(f";N{i}:" in val for i in range(4, 100))                
                ):
                remove_flag = True
                add_removed_row(row, "Odd NA, NAE, or NAT")
                            
        #FAHFA, WE with too many or too few carbons, too many double bonds, and too many nitrogens or oxygens            
        if val.startswith(("FAHFA ", "WE ")):
            if (any(f" {i}:" in val for i in range(49, 100)) or  
                any(f":{i}" in val for i in range(6, 100)) or
                any(f";O{i}" in val for i in range(4, 100)) or
                any(f";N{i}:" in val for i in range(4, 100))                
                ):
                remove_flag = True
                add_removed_row(row, "Odd FAHFA or WE")
                
        #Hydroxylated CEs           
        if val.startswith(("CE ")):
            if (any(f";O{i}" in val for i in range(1, 100)) or
                any(f" {i}:" in val for i in (19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or
                any(f" {i}" in val for i in ("16:3", "24:5", "24:1;O3", "24:2;O3", "24:3;O"))               
                ):
                remove_flag = True
                add_removed_row(row, "Odd CE")
                        
        # Lipids with one fatty acyl and too few or too many carbons or too many double bonds
        if val.startswith(("LPA ", "LPE ", "LPC ", "LPI ", "LPS ", "LPG ", "LPT ", "MG ", "FA ", "FOH ", "FAL ", "FAG ", "CE ", "ST ", "SFE ", "Car ", "CAR ")):
            if (any(f" {i}:" in val for i in (1, 3, 5, 7, 9, 11, 13, 25, 27, 28, 29, 31, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or
                any(f" {i}:" in val for i in range(30, 100)) or  
                any(f" {i}:" in val for i in range(1, 13)) or
                any(f" O-{i}:" in val for i in range(1, 13)) or
                any(f" O-{i}:" in val for i in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28, 29, 31, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or  
                any(f" O-{i}:" in val for i in range(30, 100)) or
                any(f":{i}" in val for i in range(10, 100))                 
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with one fatty acyl and too few or too many carbons or too many double bonds")
                        
        # Lipids with 2 fatty acyls (or more) and too few carbons
        if val.startswith(("FAHFA ", "WE ", "PA ", "PE ", "PE-NMe ", "PE-NMe2 ", "PnE ", "PC ", "PI ", "PT ", "PIP ", "PIP2 ", "PS ", "PG ", "DG ", "SM ", "Cer ", "TG ", "CerP ", "PE-Cer ", "PI-Cer ", "BMP ", "HBMP ", "WE ")):
            if (any(f" O-{i}:" in val for i in range(0, 14)) or  
                any(f" O-{i}:" in val for i in (1, 3, 5, 7, 9, 11, 13, 19, 21, 23, 25, 27, 28, 29, 31, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or    
                any(f"/{i}:" in val for i in range(0, 14)) or  
                any(f" {i}:" in val for i in range(0, 14)) or 
                any(f" {i}:" in val for i in (1, 3, 5, 7, 9, 11, 13, 25, 27, 28, 29, 31, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or  
                any(f" {i}:" in val for i in range(26, 29)) or
                any(f"_{i}:" in val for i in range(0, 14))
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with 2 fatty acyls or more and too few carbons") 
                        
        # Lipids with 2 fatty acyls and too many carbons
        if val.startswith(("FAHFA ", "WE ", "PA ", "PE ", "PE-NMe ", "PE-NMe2 ", "PnE ", "PC ", "PI ", "PT ", "PIP ", "PIP2 ", "PS ", "PG ", "DG ", "SM ", "Cer ", "CerP ", "PE-Cer ", "PI-Cer ", "IPC ", "BMP ", "HBMP ", "WE ")):
            if (any(f" {i}:" in val for i in range(50, 100)) or 
                any(f" {i}:" in val for i in (13, 27, 28, 29, 30, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81)) or 
                any(f" O-{i}:" in val for i in range(48, 100)) or
                any(f";O{i}" in val for i in range(3, 100)) or
                any(f":{i}" in val for i in range(10, 100)) or
                any(f"/{i}:" in val for i in range(27, 100)) or
                any(f"_{i}:" in val for i in range(27, 100))      
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with 2 fatty acyls and too many carbons")     

        # PIPs
        if val.startswith(("PIP ", "PIP2 ")):
            if (any(f" {i}:" in val for i in range(50, 100)) or 
                any(f" {i}:" in val for i in range(1, 14)) or 
                any(f" {i}:" in val for i in (13, 15, 17, 19, 21, 23, 25, 27, 28, 29, 30, 31, 33, 35, 37, 41, 43, 45, 47, 49, 51)) or 
                any(f";O{i}" in val for i in range(2, 100)) or
                any(f":{i}" in val for i in range(8, 100)) or
                any(f"/{i}:" in val for i in range(27, 100)) or
                any(f"_{i}:" in val for i in range(27, 100))      
                ):
                remove_flag = True
                add_removed_row(row, "Unreasonable PIP")    
                
        if val.startswith(("DG ")):
            if (any(f" :{i}" in val for i in range(6, 200)) or      
                any(f" {i}:" in val for i in range(45, 100)) or
                any(f" {i}:" in val for i in range(27, 31)) or
                any(f" {i}:" in val for i in (13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89))    
                ):
                remove_flag = True
                add_removed_row(row, "Unreasonable DG")                  

        # Lipids with 3 fatty acyls and too few carbons
        if val.startswith(("TG ", "MLCL ")):
            if (any(f" {i}:" in val for i in range(27, 43)) or   
                any(f" {i}:" in val for i in range(2, 11)) or   
                any(f" {i}:" in val for i in (13, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75)) or   
                any(f"/{i}:" in val for i in range(2, 11)) or
                any(f"_{i}:" in val for i in range(2, 11)) or
                any(f"_{i}:" in val for i in (13, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75)) or 
                any(f":{i}" in val for i in range(10, 50)) or
                any(f";O{i}" in val for i in range(4, 20))       
                ):
                remove_flag = True 
                add_removed_row(row, "Lipids with 3 fatty acyls and too few carbons")
                                
        # Lipids with 3 fatty acyls and too many carbons
        if val.startswith(("TG ", "MLCL ")):
            if (any(f" {i}:" in val for i in range(70, 200)) or      
                any(f"/{i}:" in val for i in range(27, 130)) or
                any(f"_{i}:" in val for i in range(27, 130)) or
                any(f" O-{i}:" in val for i in (13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91))       
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with 3 fatty acyls and too many carbons") 
                
        # Lipids with 4 fatty acyls and too few carbons
        if val.startswith(("CL ")):
            if (any(f" {i}:" in val for i in range(2, 11)) or     
                any(f"/{i}:" in val for i in range(2, 15)) or
                any(f"_{i}:" in val for i in range(2, 15)) or  
                any(f":{i}" in val for i in range(10, 50)) or
                any(f";O{i}" in val for i in range(3,20)) or
                any(f" {i}:" in val for i in (15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89))      
                ):
                remove_flag = True 
                add_removed_row(row, "Lipids with 4 fatty acyls and too few carbons")
                
        if val.startswith(("CL ")):
            if (any(f" {i}:" in val for i in range(78, 200)) or      
                any(f"/{i}:" in val for i in range(27, 100)) or
                any(f"_{i}:" in val for i in range(27, 100))   
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with 4 fatty acyls and too many carbons") 
                            
        # Sterols / Steroids
        if (val.startswith(("ST ", "SFE ")) and
            ((f";HexNAc" in val) or      
            (f";GlcA" in val) or
            (f";G" in val) or
            any(f";O{i}" in val for i in range(5, 100)) or
            any(f" {i}:" in val for i in range(1, 17)) or
            any(f" {i}:" in val for i in range(28, 100)) or 
            any(f" {i}:" in val for i in (22, 23, 25, 26, 28))    
            )):
            remove_flag = True
            add_removed_row(row, "Sterols / Steroids") 
                        
        # Sterols / Steroids - special cases 
        patterns = [
            "ST 18:4;O5;Hex",
            "ST 18:5;O;HexNAc",
            "ST 19:0;O4;HexNAc",
            "ST 19:0;O7;G",
            "ST 19:2;O;HexNAc",
            "ST 19:2;O2;HexNAc",
            "ST 19:1;O2;S",
            "ST 20:2;O;S ",
            "ST 20:2;O2;S ",
            "ST 20:1;O8;Hex", 
            "ST 20:1;O3;Hex",
            "ST 20:3;O2;S",
            "ST 24:8;O4",
            "ST 21:2;O4",
            "ST 21:3;O2",
            "ST 22:1;O2;S",
            "ST 22:1;O4;HexNAc",
            "ST 22:5;O4;S",
            "ST 22:5;O5;G",
            "ST 22:2;O3;HexNAc",
            "ST 22:5;O;HexNAc",
            "ST 23:1;O4;HexNAc",
            "ST 23:2;O3;G",
            "ST 23:6;O5;G"
            "ST 22:6;O7;HexNAc",
            "ST 23:2;O2;HexNAc",
            "ST 23:5;O;HexNAc",
            "ST 24:2;O2;HexNAc",
            "ST 24:1;O;Hex",
            "ST 24:0;O4",
            "ST 27:2;O4",
            "ST 27:3;O;F2",
            "ST 27:3;O2;F3",
            "ST 27:3;O3;F6",
            "ST 25:0;O8;G",
            "ST 26:4;O8;Hex",
            "ST 27:4;O",
            "ST 27:0;O7;G",
            "ST 27:1;O4",
            "ST 27:7;O6;GlcA",
            "ST 28:2;O4",
            "ST 29:3;O2;F2",
            "ST 29:5;O8;GlcA",
            "ST 29:6;O5;G",
            "ST 29:7;O7;GlcA",
            "ST 29:7;O8;GlcA"
            ]
        if any(pattern in val for pattern in patterns):
            remove_flag = True 
            add_removed_row(row, "Sterols / Steroids - special cases")
            
        # PK, PR and others - special cases 
        patterns = [
            "PK 26:9;O6",
                "PK 34:7;O17",
                "PK 32:8;O21",
                "PK 42:1;O8;N3",
                "FAHFA 26:0;O",
                "FAHFA 36:0;O",
                "PE-NMe 34:1",
                "PE-NMe 36:0",
                "PE-Cer 40:0;O5",
                "PE-Cer 42:1;O5",
                ";O2/22:0;O",
                ";O2/22:1;O",
                ";O2/24:0;O",
                ";O2/26:0;O",
                ";O2/26:1",
                ";O2/24:2",
                ";O3/16:0;O",
                ";O3/16:1;O",
                ";O3/16:2;O",
                ";O3/12:0;O",
                ";O3/12:1;O",
                ";O3/18:1;O",
                ";O3/18:2;O",
                ";O3/18:3;O",
                ";O3/22:0;O",
                ";O3/26:0;O",
                ";O3/24:0;O",
                ";O3/20:0;O",
                "20:0;O2/20:0;O"
                "PC 12:",
                "PC 13:",
                "PC 14:",
                "PC 15:",
                "PC 17:",
                "PC 19:",
                "PC 20:",
                "PC 21:",
                "PC 30:2;O",
                "PC 32:2;O",
                "PC 32:1;O",
                "PC 30:0;O",
                "MG 12:",
                "MG 13:",
                "MG 15:",
                "MG 17:",
                "MG 19:",
                "MG 20:",
                "MG 21:",
                "Car ",
                "CPA "
                "CAR ",
                'PnC ',
                'PnE ',
                'PS-NAc ',
                "CerP 46:4;O3",
                "CerP 46:2;O3",
                "WE 22:",
                "HETE",
                "Pelargonidin",
                "Thermozeaxanthin",
                "Citronellal",
                "glucopyranosyl)-3-keto-(1,27R)-octacosanediol",
                "dihydrostilbene",
                "hydroxy-leukotriene",
                "trihydroxydammar",
                "Hexanoyloxyisomytiloxanthin",
                "Hydroxymytiloxanthin",
                "tetrahydrospheroidene",
                "keto-gamma-carotene glucoside hexadecanoate",
                "dihydroxydammar",
                "demethylmenaquinone",
                "farnesol",
                "Hydroxy-4-(methylthio)butanoic acid",
                "Isopentenyldehydrosaproxanthin",
                "Dibromo-2-n-butylacrylic acid",
                "Diapocaroten-4'-al-4-oic acid",
                "arabinopyranosyl undecaprenyl phosphate",
                "Chloro-2-hydroxyaurone",
                "resorcinol",
                "isolongifol-5-ene",
                "Acetamidovalerate",
                "Chloroapigenin", 
                "Methylsalicylic acid",
                "oxogeraniol",
                "Acinospesigenin A",
                "phytofluene",
                "Armillaripin",
                "Cryptoxanthin",
                "bhas#30",
                "Janohigenin",
                "Neodidymellioic acid",
                "Phytyl phosphate",
                "Violaxanthin",
                "Rubraflavone",
                "Plastochromanol",
                "Coenzyme Q9",
                "Spinochalcone",
                "Coenzyme Q10",
                "Ginsenoside",
                "Rh2",
                "Glisoprenin",
                "Annotemoyin",
                "Aliso",
                "Ganocasurarin",
                "Anhydroeschscholtzxanthin",
                "Dieposabadelin",
                "Torularhodin",
                "Plastochromenol",
                "Pectenolone",
                "carotene",
                "capsorubin",
                "xanthin",
                "decaprenyl phosphate",
                "phosphosulfocholine",
                "epoxyphylloquinone",
                "quinone",
                "Phytofluene",
                "Phyto",
                "phyto",
                "Acacic acid",
                "Chinensen",
                "Emmotin",
                "Alcyopterosin",
                "3S-hydroxy-4R-methyl-2S-(n-eicos-11'-yn-19'-enyl)butanolide",
                "6-(13-hydroxytetradecyl)benzene-1,2,4-triol",
                "bacteriohopane", # not found in E. coli or NG
                "Ficaprenol",
                "hydroxychroman",
                "Brahucin",
                "abietadienal",
                "phytin",
                "trihydroxy-urs-12-en-28-oic acid",
                "Sinulobatin",
                "Retinal",
                "retinal",
                "retinol",
                "Gingerol",
                "Sclareo",
                "isopropyl-4-methyloxepan-2-one",
                "palmitoylglycerone 3-phosphate",
                "Phytenoic Acid",
                "Trehalose 6,6'-dipalmitate",
                "Argutenol",
                "Dolabrin",
                "Kaempferol",
                "aminaribioside",
                "Phytanic acid",
                "squalene",
                "Squalene",
                "isolongifol",
                "Cardanol",
                "Pileadimenthenol",
                "Triacetin",
                "Demethylspheroidene",
                "Dihydroisopentenyldehydrorhodopin",
                "Scalarin",
                "dioxaspiro[5.5]undecane",
                "dioxaspiro",
                "Dehydrovomifoliol",
                "ascr#",
                "mifoliol",
                "Dimethoxystilbene",
                "Oxophytani",
                "phytanic",
                "Adipostatin",
                "Brachyanone",
                "Squafosacin",
                "ACer",
                "Acer",
                "Phytenic",
                "trehalose",
                "Decaprenol",
                "Dolichoic",
                "dolichol",
                "Sugeonyl",
                "Dimethoxystilbene",
                "Docosadienyl glycerone-3-phosphate",
                "hopane",
                "Annopurpuricin",
                "Bombiprenone",
                "Brachyanone",
                "Oblatone",
                "Ustusa",
                "Drimenol",
                "Citronellyl",
                "isoprenoid",
                "rhamnosyl",
                "helminthosporol",
                "docosadienyl glycerone-3-phosphate",
                "Chloro-8E,10E-undecadien-1-ol",
                "Dimethoxystilbene",
                "Africananol",
                "Crannenol",
                "Linalyl acetate",
                "linoleyl glycerone-3-phosphate",
                "henicos-14-en-1-yl)phenol",
                "henicos-14-en-1-yl)phenol",
                "dimethyl-22-oxavitamin D3",
                "trihydroxyvitamin D3",
                "hydroxypentyl)vitamin D3",
                "hydroxyhexyl)vitamin D3",
                "hydroxybutoxy)vitamin D3",
                "-epivitamin D3",
                "-oxavitamin D3",
                "-norvitamin D3",
                "Hydroxygeminivitamin D3",
                "-cyclovitamin D3",
                "dihydroxyvitamin D3",
                "didehydrovitamin D3",
                "dihydrovitamin D3",
                "Hydroxyvitamin D5",
                "Oceanin",
                "tetradecene-8,10,12-triyne",
                "Campesterol ester",
                "xyloside",
                "10/10/10-gly",
                "Serradiol",
                "Ophiobolin T",
                "Sitosterol",
                "22:1(5Z)(9Me,13Me,17Me,21Me)",
                "6-Cl-iso 22:1 delta4",
                "AC2SGL",
                "As-PL",
                "Brassicasteryl",
                "Campestanyl",
                "FMC-5",
                "mLPA",
                "cyanolipid",
            ]
        
        if any(pattern in val for pattern in patterns):
            remove_flag = True 
            add_removed_row(row, "PK, PR and others - special cases")

        if remove_flag:
            rows_to_drop.append(idx)

   
    # Save dropped rows into debug
    if removed_rows:
        debug_folder = Path(output_folder) / "debug"
        debug_folder.mkdir(parents=True, exist_ok=True)
        dropped_path = debug_folder / f"Removed_by_plausibility_{mode}.csv"
        pd.DataFrame(removed_rows).to_csv(dropped_path, index=False, encoding="utf-8-sig")
        
    if removed_rows:
        dropped_path = debug_folder  / f"Removed_by_plausibility_{mode}.csv"
        pd.DataFrame(removed_rows).to_csv(dropped_path, index=False, encoding="utf-8-sig")

    # Drop them from the DataFrame
    if rows_to_drop:
        df = df.drop(rows_to_drop)

    print(f"Selected {len(rows_to_drop)} peaks to remove by plausibility filter ({mode}).", flush=True)
    print("Removal reasons summary:", dict(removal_reasons), flush=True)
    print("Matches left:", len(df), flush=True)

    return df