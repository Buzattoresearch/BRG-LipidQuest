#TODO: add scoring system

# -----------------------------
# BACTERIA
# -----------------------------

import os
import pandas as pd
from collections import Counter
from pathlib import Path

def apply_plausability_filter(df, output_folder, pol_tag, mode="MS", row_type="Annotation", rt_field="RT (min)"):
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

        polarity = str(row.get("Polarity", "")).strip()
        lipid_class = str(row.get("Lipid Class", "")).strip()

        # Lipids not detected in negative ionization
        if polarity.startswith("Neg") and lipid_class in ["TG", "CE", "MG", "DG", "WE", "HC"]:
            remove_flag = True
            add_removed_row(row, "Implausible polarity")

        # Lipids not detected in positive ionization
        if polarity.startswith("Pos") and lipid_class in ["FA"]:
            remove_flag = True
            add_removed_row(row, "Implausible polarity")

        # Implausible classes
        if val.startswith(("NoAbbrev:",
                           "ACer ", "Car ", "CE ", "Cer", "CerP ", "DGCC ", "Hex2MG ", "HexMG ", "DGTA ", "DGTS ", "MGTS ", "FAG ", "FAHFA ", "GlcADG ", 
                           "GlcCer", "HC ", "HexCer ", "HexCer", "Hex-Hex", "Hex2Cer ", "HexSPB ", "MIPC ", "M(IP)2C ", "IPC ", "NAT ", "PIP ", "PIM ", "PIM1 ", "PIM2 ", "PIM3 ", "PIM4 ", "PIM5 ", "PIM6 ", "PnC ", "PnE ", "SCer ", "SHexCer ", 
                           "PI ", "LSM ", "SPB ", "SPBP ", "ST ", "SulfateHexSPB ", "PE-Cer ", "CerPE ", "PI-Cer ", "CerPI ", "BMP ", "LPS ", "WE ", "NAT ", 
                           "Am-Hex-PE ", "PK ", "PT ")): 
            # ACer = Acyl ceramides (acylated ceramides) (human, animals)
            # Car = Acyl carnitines (human, animals)
            # CE = cholesteryl ester, esterified sterols (Human, animal)
            # Cer = ceramides (Human, animal, plant)
            # CerP = Ceramide phosphates (Human, animals)
            # DGCC = Diacylglyceryl-carboxyhydroxymethylcholine (algae)
            # Hex2DG = Digalactosyl diacylglycerol (Plants, algae)
            # Hex2MG = Digalactosyl monoacylglycerol (Plants, algae)
            # HexDG = Monogalactosyl diacylglycerol (Plants, algae)
            # HexMG = Monogalactosyl monoacylglycerol  (plants)
            # DGTA = Betaine lipids (diacylglyceryl-hydroxymethyl-trimethyl-alanine) (Algae, bacteria)
            # DGTS = Betaine lipids (diacylglyceryl-trimethyl-homoserine) (Algae, some plants, some bacteria)
            # MGTS = Betaine lipids (monoacylglyceryl-trimethyl-homoserine) (Algae, some plants, some bacteria)
            # FAG = Fatty glycosides (Plants, microbes)
            # FAHFA = Fatty acid esters of hydroxy fatty acids (Human, animal)
            # GlcADG = Glucuronosyldiacylglycerol (Cyanobacteria)
            # GlcCer = Glucosylceramides (human, animnal, plant)
            # HC = Hydrocarbons
            # HexCer = Hexosylceramides (complex glycosphingolipids - cerebrosides, galactosylceramides, gangliosides, etc) (human, animal, plant)
            # HexSPB = Hexosyl sphingoid bases (human, animal)
            # PS-NAc = Diacylglycerophosphoserines (diacylglycerophospho-N-acyl-serine) (animals but very rare)
            # # M(IP)2C =  Ceramide phosphoinositols (fungi, protozoa)
            # MIPC =  Ceramide phosphoinositols (fungi, protozoa)
            # IPC =  Ceramide phosphoinositols (fungi, protozoa)
            # NAT = N-acyl taurines (human, animal)
            # PIP = Phosphatidylinositol phosphates (phosphoinositides)
            # PIM = Phosphatidylinositol mannosides (mycobacteria)
            # PnC =  Phosphonocholines (Marine organisms, bacteria)
            # PnE =  Phosphonoethanolamines (Marine organisms, bacteria)
            # SCer = Sulfoceramides (human, animal)
            # SHexCer = Sulfatides (human, animal)
            # PI = Phosphatidylinositols (Animals, plants, fungi, algae, mycobacteria, cyanobacteria)
            # SM = Sphingomyelins (human, animal)
            # LSM = lysosphingomyelins (human, animal)
            # SPB = sphingoid bases (human, animal)
            # SPBP = Sphingoid bases 1-phosphate (human, animal)
            # ST = sterols 
            # SulfateHexSPB =  Sulfohexosyl sphingoid bases (Marine organisms - rare)
            # PE-Cer, CerPE =  Ceramide phosphatidylethanolamines (Invertebrates, marine organisms)
            # PI-Cer, CerPI =  Ceramide phosphatidylinositols (Invertebrates, marine organisms)
            # NAE = Acyl ethanolamines (human, animal)
            # BMP = Bismonophosphates (bis(monoacylglycero)phosphates) (human, animal)
            # LPS = Lysophosphatidylserines (human, animal)
            # WE = Waxes (Plants, animals)
            # NAT = Acyl taurines (Human, animal)
            # FOH = Fatty alcohols (animals, plants, fungi, algae, protists, some bacteria - actinobacteria, marine bacteria, some Streptomyces spp., cyanobacteria, some some Shewanella and Vibrio species, Planctomycetes, Myxobacteria)
            # LNAPE =  N-acyl-lysophosphatidylethanolamine
            # NAPE =  N-acyl-phosphatidylethanolamine (exist in E. coli)
            # PEtH or PEtOH =  Diacylglycerophosphoethanols (Phosphatidylethanol) (not physiological; marker of ethanol exposure in organisms that contain phospholipase D)
            # PMeOH =  Diacylglycerophosphomethanols (Phosphatidylmethanol) (not physiological; marker of ethanol exposure in organisms that contain phospholipase D)
            # PE-NMe = diacylglycerophospho-N-methylethanolamine, also known as PMME - phosphatidyl-N-monomethylethanolamine (Bacteria; Phosphatidylethanolamine methylation pathway (PE → PMME → PDME → PC))
            # PE-NMe2 =  diacylglycerophospho-N,N-dimethylethanolamine, also known as PDME - phosphatidyl-N,N-dimethylethanolamine (Bacteria, algae, very rare in animals; Phosphatidylethanolamine methylation pathway (PE → PMME → PDME → PC))
            # CPA = Cyclic phosphatidic acids (human, animal)
            # FAL = Fatty aldehydes (animals, plants, algae, fungi, some bacteria - cyanobacteria, marine bacteria, Actinobacteria)
            # PK =  Polyketides (Microbes, plants)
            # Am-Hex-PE =  Phosphatidylethanolamine glycans (Protozoa, parasites)
            # LSQMG = LysoSulfoquinovosyl monoacylglycerols (sulfolipids) (Plants, algae)
            # SQMG = Sulfoquinovosyl monoacylglycerols (sulfolipids) (plants, algae)
            # PE-N[FA] =  Diacylglycerophosphoethanolamines (diacylglycerophospho-N-acyl-ethanolamine; same as NAPE) (Animals, plants, some fungi)
            # PS-NAc =  Diacylglycerophosphoserines (diacylglycerophospho-N-acyl-serine) (Animals, but very rare, trace amounts)
            # SL as a headgroup = sulfonolipid (sulfonic sphingoid bases) (Bacteroidetes)

            remove_flag = True
            add_removed_row(row, "Implausible classes")  

        # ------------------------------------------
        # OPTIONAL FOR SPECIFIC BACTERIA
        # ------------------------------------------
        # FOR E. coli
        if val.startswith(("NoAbbrev:",
                           "Hex2DG ", "Hex2MG ", "HexMG ", "HexDG ", "Hex(", "HexCer ", "WD ", "WE ")): 
                remove_flag = True
                add_removed_row(row, "Implausible E. coli classes")  
        if val.startswith(("PC 28:", "LPC 26:", "LPE 26:", "LPC 24:", "LPE 24")): 
                remove_flag = True
                add_removed_row(row, "Implausible E. coli lipids")
                
        # === Impossible plasmalogens ===
        if (" O-" in val or # Plasmalogens and ether-linked lipids in general are not found in Gram-negatives. They can only be found in Gram-positives and archea.
            "DG O-" in val or "DG dO-" in val or " dO-" in val or "MG O-" in val or
            "Hex2DG O-" in val or "Hex2MG O-" in val or "HexDG O-" in val or "HexMG O" in val or
            "PE-NMe2 O-" in val or ("O-" in val and ";" in val) or
            "O-14:1" in val or 
            "O-23:" in val or "O-25:" in val or
            "O-27:" in val or
            "O-35:" in val or "O-37:" in val or "O-39:" in val or
            "O-41:" in val or "O-44:7" in val or "O-43:" in val or "O-46:7" in val or
            "O-45:" in val or "O-47:" in val or "O-48:" in val or "O-49:" in val or
            "O-51:" in val or "O-53:" in val or "O-55:" in val or "O-57:" in val or
            "O-59:" in val or "O-61:" in val or "O-63:" in val or "O-65:" in val or
            "O-66:" in val or "O-67:" in val or "O-68:" in val or "O-69:" in val or
            "O-70:" in val or "O-71:" in val or "O-72:" in val or "O-73:" in val or
            "O-75:" in val or "O-77:" in val or "O-79:" in val or "O-81:" in val or
            "CPA O-" in val or "Glc-GP O-" in val or
            ("O-12:" in val and "_" not in val and "/" not in val and "L" not in val) or ("O-13:" in val and "_" not in val and "/" not in val and "L" not in val) or ("O-14:" in val and "_" not in val and "/" not in val and "L" not in val) or
            ("O-15:" in val and "_" not in val and "/" not in val and "L" not in val) or ("O-16:" in val and "_" not in val and "/" not in val and "L" not in val) or ("O-17:" in val and "_" not in val and "/" not in val and "L" not in val) or
            ("O-18" in val and "_" not in val and "/" not in val and "L" not in val) or ("O-19" in val and "_" not in val and "/" not in val and "L" not in val) or ("O-20" in val and "_" not in val and "/" not in val and "L" not in val) or
            ("O-21" in val and "_" not in val and "/" not in val and "L" not in val) or ("O-22" in val and "_" not in val and "/" not in val and "L" not in val) or ("O-23" in val and "_" not in val and "/" not in val and "L" not in val) or
            ("O-24" in val and "_" not in val and "/" not in val and "L" not in val) or ("O-25" in val and "_" not in val and "/" not in val and "L" not in val) or ("O-26" in val and "_" not in val and "/" not in val and "L" not in val)):
            remove_flag = True
            add_removed_row(row, "Impossible plasmalogens")

        #  === Contaminants, strange chain lengths ===
        if ("contaminant" in val or
            "DEET" in val or "marine" in val or "fungi" in val or "plant" in val or "insect" in val or "phyto" in val):
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
                "_9:" in val or "_11:" in val or "_13:" in val or                   # PUFA are not found in bacteria
                ":11" in val or ":12" in val or                                     # PUFA are not found in bacteria
                ":13" in val or ":14" in val or ":15" in val or                     # PUFA are not found in bacteria
                
                "12:0;O" in val or
                "12:0_12:" in val or                                                # odd combinations of FAs
                "12:0_13:" in val or "12:0_19:" in val or      
                "12:0_20:" in val or 
                
                "13:0_12:" in val or "13:0_13:" in val or      
                "13:0_15:" in val or "13:0_17:" in val or "13:0_19:" in val or      
                "13:0_20:" in val or   
                
                "14:0;O2" in val or "14:0;O3" in val or "14:0;O4" in val or                                                      
                "14:2" in val or "14:3" in val or  "14:4" in val or
                
                "13:0;O" in val or 
                  
                "14:0;O2" in val or "14:0;O3" in val or "14:0;O4" in val or                                                 
                "14:1;O2" in val or "14:1;O3" in val or "14:1;O4" in val or  
                "14:2" in val or "14:3" in val or  "14:4" in val or
                
                "15:1_13:" in val or
                "15:1_21:" in val or "15:1_22:" in val or
                
                "15:2;O" in val or "15:3;O" in val or "15:4;O" in val or 
                "15:2" in val or "15:3" in val or "15:4" in val or
                "16:4" in val or "16:5" in val or "16:6" in val or
                
                "17:0_13:" in val or
                "17:0_21:" in val or
                "17:1_12:" in val or "17:1_13:" in val or
                "17:3" in val or "17:4" in val or
                
                "18:5" in val or "18:6" in val or
                
                "19:0_12:" in val or "19:0_13:" in val or
                "19:1_12:" in val or "19:1_13:" in val or
                "19:3" in val or
                
                "21:0_12:" in val or "21:0_13:" in val or "21:0_14:" in val or
                "21:0_15:" in val or "21:0_17:" in val or "21:0_19:" in val or
                "21:0_21:" in val or
                "21:2" in val or "21:3" in val or
                "21:4" in val or "21:5" in val or "21:6" in val or
                
                "22:2" in val or "22:3" in val or  "22:4" in val or                       # only found in eukaryotic organisms
                "22:5" in val or "22:6" in val or 
                
                "23:" in val or
                
                "24:1;O" in val or "24:2;O" in val or 
                "24:3" in val or "24:4" in val or "24:5" in val or "24:6" in val or "24:7" in val or
                
                "25:" in val or
                
                ("26:" in val and "_" in val) or ("26:" in val and "/" in val) or
                "26:3" in val or "26:4" in val or "26:5" in val or
                "26:6" in val or "26:7" in val or "26:8" in val or
                "26:1;O" in val or "26:2;O" in val or "26:3;O" in val or 
                
                ("27:" in val and "_" in val) or ("27:" in val and "/" in val) or 
                
                "28:3" in val or "28:4" in val or
                
                "29:3" in val or "29:4" in val or "29:5" in val or
                
                "30:0_" in val or "31:0_" in val or "32:0_" in val or 
                "33:0_" in val or "34:0_" in val or "35:0_" in val or 
                "36:0_" in val or "37:0_" in val or "38:0_" in val or
                "39:0_" in val or "40:0_" in val or "41:0_" in val or
                "_27:" in val or "_28:" in val or "_29:" in val or
                "_30:" in val or "_31:" in val or "_32:" in val or 
                "_33:" in val or "_34:" in val or "_35:" in val or 
                "_36:" in val or "_37:" in val or "_38:" in val or
                "_39:" in val or "_40:" in val or "_41:" in val or
                "_42:" in val or "_43:" in val or "_44:" in val or
                
                "30:6" in val or "30:7" in val or "30:8" in val or
                
                "31:6" in val or "31:7" in val or "31:8" in val or
                
                "32:6" in val or "32:7" in val or "32:8" in val or
                
                "33:6" in val or "33:7" in val or "33:8" in val or
                
                "34:6" in val or "34:7" in val or "34:8" in val or
                
                "35:6" in val or "35:7" in val or "35:8" in val or
                
                "36:6" in val or "36:7" in val or "36:8" in val or
                
                "37:6" in val or "37:7" in val or "37:8" in val or

                "39:" in val or
                
                "41:" in val
        )):
            remove_flag = True
            add_removed_row(row, "Odd FAs")        
            
        #Odd FAs - Part II
        if ("PR" not in val and 
            "PK" not in val and 
            "SL" not in val and 
            "ST" not in val and (
                "44:6" in val or "44:7" in val or "44:8" in val or
                "44:9" in val or "44:10" in val or "44:11" in val or
                
                "45:" in val or
                
                "46:4" in val or "46:5" in val or
                "46:6" in val or "46:7" in val or "46:8" in val or
                "46:9" in val or "46:10" in val or "46:11" in val or
                
                "47:4" in val or "47:5" in val or
                "47:6" in val or "47:7" in val or "47:8" in val or
                "47:9" in val or "47:10" in val or "47:11" in val or
                
                "48:6" in val or "48:7" in val or "48:8" in val or
                "48:9" in val or "48:10" in val or "48:11" in val or
                
                "49:6" in val or "49:7" in val or "49:8" in val or
                "49:9" in val or "49:10" in val or "49:11" in val or
                
                "50:6" in val or "50:7" in val or "50:8" in val or
                "50:9" in val or "50:10" in val or "50:11" in val or
                "50:12" in val or "50:13" in val or "50:14" in val or              
                
                "51:4" in val or "51:5" in val or
                "51:6" in val or "51:7" in val or "51:8" in val or
                "51:9" in val or "51:10" in val or "51:11" in val or
                "51:12" in val or "51:13" in val or "51:14" in val or   
                
                "52:6" in val or "52:7" in val or "52:8" in val or
                "52:9" in val or "52:10" in val or "52:11" in val or
                "52:12" in val or "52:13" in val or "52:14" in val or 
                
                "53:6" in val or "53:7" in val or "53:8" in val or
                "53:9" in val or "53:10" in val or "53:11" in val or
                "53:12" in val or "53:13" in val or "53:14" in val or 
                
                "54:9" in val or "54:10" in val or "54:11" in val or
                "54:12" in val or "54:13" in val or "54:14" in val or 
                
                "55:6" in val or "55:7" in val or "55:8" in val or
                "55:9" in val or "55:10" in val or "55:11" in val or
                "55:12" in val or "55:13" in val or "55:14" in val or 
                
                "56:9" in val or "56:10" in val or "56:11" in val or
                "56:12" in val or "56:13" in val or "56:14" in val or
                
                "57:6" in val or "57:7" in val or "57:8" in val or
                "57:9" in val or "57:10" in val or "57:11" in val or
                "57:12" in val or "57:13" in val or "57:14" in val or 
                
                "58:9" in val or "58:10" in val or "58:11" in val or
                "58:12" in val or "58:13" in val or "58:14" in val or
                
                "59:6" in val or "59:7" in val or "59:8" in val or
                "59:9" in val or "59:10" in val or "59:11" in val or
                "59:12" in val or "59:13" in val or "59:14" in val or 
                
                "60:9" in val or "60:10" in val or "60:11" in val or
                "60:12" in val or "60:13" in val or "60:14" in val or
                
                "61:" in val or
                "62:" in val or
                "63:" in val or
                
                "64:6" in val or "64:7" in val or "64:8" in val or
                "64:9" in val or "64:10" in val or "64:11" in val or
                "64:12" in val or "64:13" in val or "64:14" in val or 
                
                "65:6" in val or "65:7" in val or "65:8" in val or
                "65:9" in val or "65:10" in val or "65:11" in val or
                "65:12" in val or "65:13" in val or "65:14" in val or 
                
                "65:6" in val or "65:7" in val or "65:8" in val or
                "65:9" in val or "65:10" in val or "65:11" in val or
                "65:12" in val or "65:13" in val or "65:14" in val or 
                
                "66:6" in val or "66:7" in val or "66:8" in val or
                "66:9" in val or "66:10" in val or "66:11" in val or
                "66:12" in val or "66:13" in val or "66:14" in val or 
                
                "67:6" in val or "67:7" in val or "67:8" in val or
                "67:9" in val or "67:10" in val or "67:11" in val or
                "67:12" in val or "67:13" in val or "67:14" in val or 
                
                "68:6" in val or "68:7" in val or "68:8" in val or
                "68:9" in val or "68:10" in val or "68:11" in val or
                "68:12" in val or "68:13" in val or "68:14" in val or 
                
                "69:6" in val or "69:7" in val or "69:8" in val or
                "69:9" in val or "69:10" in val or "69:11" in val or
                "69:12" in val or "69:13" in val or "69:14" in val or 
                
                "70:9" in val or "70:10" in val or "70:11" in val or
                "70:12" in val or "70:13" in val or "70:14" in val or 
                
                "71:6" in val or "71:7" in val or "71:8" in val or
                "71:9" in val or "71:10" in val or "71:11" in val or
                "71:12" in val or "71:13" in val or "71:14" in val or 
                
                "72:9" in val or "72:10" in val or "72:11" in val or
                "72:12" in val or "72:13" in val or "72:14" in val or 
                
                "73:6" in val or "73:7" in val or "73:8" in val or
                "73:9" in val or "73:10" in val or "73:11" in val or
                "73:12" in val or "73:13" in val or "73:14" in val or 
                
                "74:12" in val or "74:13" in val or "74:14" in val or 
                
                "75:9" in val or "75:10" in val or "75:11" in val or
                "75:12" in val or "75:13" in val or "75:14" in val or 
                
                "76:12" in val or "76:13" in val or "76:14" in val or 
                
                "77:12" in val or "77:13" in val or "77:14" in val or 
                
                "78:12" in val or "78:13" in val or "78:14" in val or 
                
                "79:12" in val or "79:13" in val or "79:14" in val or 
                
                "80:12" in val or "80:13" in val or "80:14" in val
            )):
            remove_flag = True
            add_removed_row(row, "Odd FAs")

        #Odd FAs - Part III (sum compositions)
        if ((not val.startswith(("LPA ", "LPC ", "LPE ", "LPG ", "LPI ", "LPS ", "LPT ", "LPE-NMe ", "LPE-NMe2 "))) and
            ("_" not in val and "/" not in val) and # only sum composition level
            ("PC " in val or "PE " in val or "PA " in val or "PG " in val or "PS " in val or "PI " in val or "PIP " in val or "BMP " in val or "LBPA " in val or "CL " in val or 
            "SM " in val or "Cer " in val or
            "DG " in val or "FAHFA " in val or "WE " in val         
            ) and (
                "10:" in val or "11:" in val or "12:" in val or
                "13:" in val or "14:" in val or "15:" in val or
                "16:" in val or "17:" in val or "18:" in val or
                "19:" in val or "20:" in val or "21:" in val or
                "22:" in val or "23:" in val or "24:" in val or 
                "25:" in val or "26:" in val
                )):
            remove_flag = True
            add_removed_row(row, "Odd FAs - summed compositions for 2 fatty acyls")
            
        #Odd FAs - Part IV (sum compositions - 3 fatty acyls)
        if ((not val.startswith(("LPA ", "LPC ", "LPE ", "LPG ", "LPI ", "LPS ", "LPT ", "LPE-NMe ", "LPE-NMe2 "))) and
            ("_" not in val and "/" not in val) and # only sum composition level
            ("TG " in val  or "ACer " in val  or "ACER " in val or "Acer " in val         
            ) and (
                "10:" in val or "11:" in val or "12:" in val or
                "13:" in val or "14:" in val or "15:" in val or
                "16:" in val or "17:" in val or "18:" in val or
                "19:" in val or "20:" in val or "21:" in val or
                "22:" in val or "23:" in val or "24:" in val
                )):
            remove_flag = True
            add_removed_row(row, "Odd FAs - summed compositions for 3 fatty acyls")

        # Lipids with more than 11 extra oxygens    
        if any(f";O{i}" in val for i in range(11, 100)):
            remove_flag = True
            add_removed_row(row, "Lipids with more than 11 extra oxygens ")
        
        # Lipids with more than 14 double bonds
        if any(f":{i}" in val for i in range(14, 100)):
            remove_flag = True
            add_removed_row(row, "Lipids with more than 14 double bonds") 
                
        # Rare fatty acyls
        if val.startswith(("FOH ", "FAL ", "HC ", "FAG ", "SFE ", "NAx ", "NA ", "NAE ")): 
            if (any(f";O{i}" in val for i in range(6, 100)) or
                any(f"{i}:" in val for i in (3, 5, 7, 9, 11, 13, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59))
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
        if val.startswith(("ACer ", "Acer ", "ACER ")):
            if (any(f";O{i}" in val for i in range(4, 7)) or
                any(f" {i}:" in val for i in range(57, 100)) or
                any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or
                any(f"/{i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60)) or
                any(f"_{i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60))
                ):
                remove_flag = True
                add_removed_row(row, "Odd acylCeramides")
                        
        # HexCer
        if val.startswith(("HexCer ", "Hex2Cer ", "Hex(2)Cer ", 'CerP ', "SHexCer")):  # oxygens from sugars are not included in abbreviations
            if (any(f";O{i}" in val for i in range(4, 100)) or
            any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 12, 13, 15, 17, 19, 21, 23, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47)) or
            any(f" {i}:" in val for i in range (47, 100)) or
            any(f" :{i}" in val for i in range(5, 100)) or
            any(f"/{i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25)) or
            any(f"_{i}:" in val for i in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25)) or
            any(f"_{i}:" in val for i in range(26, 100)) or
            any(f"/{i}:" in val for i in range(26, 100))
            ):
                remove_flag = True
                add_removed_row(row, "Odd HexCer")
                            
        #Acylglycerols and fatty acyls with too many oxygens, nitrogens, or double bonds, OR too few carbons (too small)                     
        if val.startswith(("Car ", "CAR ", "CoA ", "COA ", "Coa ", "FAL ", "FOH ", "FAG ", "MG ", "DG ")): 
            if (any(f";O{i}" in val for i in range(3, 100)) or
                any(f";N{i}" in val for i in range(3, 100)) or
                any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 13, 25, 27, 39, 41, 43, 45, 47, 49, 51)) or
                any(f"_{i}:" in val for i in (3, 5, 7, 9, 11, 13, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51)) or
                any(f"/{i}:" in val for i in (3, 5, 7, 9, 11, 13, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51))
                ):
                remove_flag = True
                add_removed_row(row, "Odd acylGlycerols or FA")
                
        #Car (Short- and medium-chain acylcarnitines (≤C14) could be encountered in bacteria, but generally only if they are exposed to external carnitine or supplemented media (e.g., in host environments or specialized growth media. Long- and very-long-chain acylcarnitines (≥C18) are not plausible bacterial products)                  
        if val.startswith(("Car ", "CoA ", "CAR ")): 
            if (any(f" {i}:" in val for i in range(1, 11)) or  
                any(f" {i}:" in val for i in range(21, 100)) or  
                any(f";O{i}" in val for i in range(1, 100)) or  
                any(f":{i}" in val for i in range(3, 100))
                ):
                remove_flag = True
                add_removed_row(row, "Odd Car or CoA")
                
        #AFree fatty acyls with too many oxygens, nitrogens, or double bonds, OR too few carbons (too small)                     
        if val.startswith(("FA", "FAL", "FOH", "HC", "FAG")): 
            if (any(f";O{i}" in val for i in range(5, 100)) or
                any(f";N{i}" in val for i in range(5, 100)) or
                any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51)) or
                any(f"_{i}:" in val for i in (3, 5, 7, 9, 11, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51)) or
                any(f"/{i}:" in val for i in (3, 5, 7, 9, 11, 25, 27, 28, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51))
                ):
                remove_flag = True
                add_removed_row(row, "Odd free FA")
            
        #NAs or NAEs with too many or too few carbons, too many double bonds, and too many nitrogens or oxygens            
        if val.startswith(("NA ", "NAx ", "NAE ", "NAT ")):
            if (any(f" {i}:" in val for i in range(27, 100)) or  
                any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 13, 21, 23, 25, 26, 27, 28, 29)) or  
                any(f":{i}" in val for i in range(4, 100)) or
                any(f";O{i}" in val for i in range(3, 100)) or
                any(f";N{i}:" in val for i in range(4, 100))                
                ):
                remove_flag = True
                add_removed_row(row, "Odd NA, NAE, or NAT")
                            
        #FAHFA, WE with too many or too few carbons, too many double bonds, and too many nitrogens or oxygens            
        if val.startswith(("FAHFA ", "WE ")):
            if (any(f" {i}:" in val for i in range(49, 100)) or  
                any(f":{i}" in val for i in range(5, 100)) or
                any(f";O{i}" in val for i in range(4, 100)) or
                any(f";N{i}:" in val for i in range(4, 100))                
                ):
                remove_flag = True
                add_removed_row(row, "Odd FAHFA or WE")
                
        #Hydroxylated CEs           
        if val.startswith(("CE ")):
            if (any(f";O{i}" in val for i in range(1, 100)) or
                any(f" {i}:" in val for i in (21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or
                any(f" {i}" in val for i in ("24:5", "24:1;O3", "24:2;O3", "24:3;O"))               
                ):
                remove_flag = True
                add_removed_row(row, "Odd CE")
                        
        # Lipids with one fatty acyl and too few or too many carbons or too many double bonds
        if val.startswith(("LPA ", "LPE ", "LPC ", "LPI ", "LPS ", "LPG ", "LPT ", "MG ", "FA ", "FOH ", "FAL ", "FAG ", "CE ", "SFE ", "Car ", "CAR ")):
            if (any(f" {i}:" in val for i in (1, 3, 5, 7, 9, 11, 13, 25, 27, 28, 29, 31, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or
                any(f" {i}:" in val for i in range(30, 100)) or  
                any(f" {i}:" in val for i in range(1, 11)) or
                any(f" O-{i}:" in val for i in range(1, 11)) or
                any(f" O-{i}:" in val for i in (1, 3, 5, 7, 9, 11, 13, 21, 23, 25, 27, 28, 29, 31, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or  
                any(f" O-{i}:" in val for i in range(30, 100)) or
                any(f":{i}" in val for i in range(3, 100))                 
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with one fatty acyl and too few or too many carbons or too many double bonds")
                        
        # Lipids with 2 fatty acyls (or more) and too few carbons
        if val.startswith(("FAHFA ", "WE ", "PA ", "PE ", "PE-NMe ", "PE-NMe2 ", "PnE ", "PC ", "PI ", "PEth", "PT ", "PIP", "PS ", "PG ", "DG ", "SM ", "Cer ", "CerP ", "PE-Cer ", "PI-Cer ", "BMP ", "HBMP ", "WE ", "PIM")):
            if (any(f" O-{i}:" in val for i in range(0, 11)) or  
                any(f" O-{i}:" in val for i in (1, 3, 5, 7, 9, 11, 13, 21, 23, 25, 27, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or    
                any(f"/{i}:" in val for i in range(0, 11)) or  
                any(f" {i}:" in val for i in range(0, 11)) or 
                any(f" {i}:" in val for i in (1, 3, 5, 7, 9, 11, 13, 43, 45, 47, 49, 51, 53, 55, 57, 59)) or  
                any(f"_{i}:" in val for i in range(0, 11))
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with 2 fatty acyls or more and too few carbons") 
                
        
                        
        # Lipids with 2 fatty acyls and too many carbons
        if val.startswith(("FAHFA ", "WE ", "PE-NMe ", "PE-NMe2 ", "PEth", "PnE ", "PC ", "PI ", "PT ", "PEth", "PIP", "DG ", "SQDG ", "SM ", "Cer ", "CerP ", "PE-Cer ", "PI-Cer ", "IPC ", "BMP ", "HBMP ", "WE ", "PIM")):
            if (any(f" {i}:" in val for i in range(44, 100)) or 
                any(f" {i}:" in val for i in (13, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81)) or 
                any(f" O-{i}:" in val for i in range(48, 100)) or
                any(f";O{i}" in val for i in range(3, 100)) or
                any(f":{i}" in val for i in range(5, 100)) or
                any(f"/{i}:" in val for i in range(27, 100)) or
                any(f"_{i}:" in val for i in range(27, 100))      
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with 2 fatty acyls and too many carbons")  
                
        # Lipids with 2 fatty acyls and too many carbons
        if val.startswith(("PE ", "PE-N[FA]", "NAPE ", "PS ", "PG ", "PA ")):
            if (any(f" {i}:" in val for i in range(62, 100)) or 
                any(f" {i}:" in val for i in (13, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81)) or 
                any(f" O-{i}:" in val for i in range(48, 100)) or
                any(f";O{i}" in val for i in range(3, 100)) or
                any(f":{i}" in val for i in range(5, 100)) or
                any(f"/{i}:" in val for i in range(27, 100)) or
                any(f"_{i}:" in val for i in range(27, 100))      
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with 2 or 3 fatty acyls (specific phospholipids) and too many carbons")  
        
        # Lipids with 2 fatty acyls and odd chains
        if val.startswith(("FAHFA ", "WE ", "PE-NMe ", "PE-NMe2 ", "PE-N[FA]", "PnE ", "PEth", "PI ", "PT ", "PIP", "PS ", "SM ", "IPC ", "BMP ", "HBMP ", "WE ", "PIM")):
            if (any(f";O{i}" in val for i in range(2, 100)) or
                any(f" {i}:" in val for i in (27, 29, 37, 39, 41, 43, 45, 46, 47, 48, 49, 50)) or 
                any(f":{i}" in val for i in range(5, 100)) or  
                any(f"_{i}:" in val for i in (11, 12, 13, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50))  
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with 2 fatty acyls and odd chains") 
                           
        #WE with odd compositions            
        if val.startswith(("WE ", "WD ")):
            if (any(f" {i}:" in val for i in range(49, 100)) or  
                any(f" {i}:" in val for i in (3, 5, 7, 9, 11, 13, 21, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49))               
                ):
                remove_flag = True
                add_removed_row(row, "Odd WE")
                
        # PIPs
        if val.startswith(("PIP ", "PIP1 ", "PIP2 ", "PIP3 ", "PIP4 ", "PIP5 ")):
            if (any(f" {i}:" in val for i in range(50, 100)) or 
                any(f" {i}:" in val for i in range(1, 14)) or 
                any(f" {i}:" in val for i in (13, 21, 23, 25, 27, 28, 29, 31, 33, 35, 37, 41, 43, 45, 47, 49, 51)) or 
                any(f";O{i}" in val for i in range(2, 100)) or
                any(f":{i}" in val for i in range(5, 100)) or
                any(f"/{i}:" in val for i in range(27, 100)) or
                any(f"_{i}:" in val for i in range(27, 100))      
                ):
                remove_flag = True
                add_removed_row(row, "Unreasonable PIP")   
                
        if val.startswith(("DG ", "SQDG ", "HexDG ", "Hex2DG ")):
            if (any(f" :{i}" in val for i in range(4, 200)) or      
                any(f" {i}:" in val for i in range(45, 100)) or
                any(f" {i}:" in val for i in (13, 21, 23, 25, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89))    
                ):
                remove_flag = True
                add_removed_row(row, "Unreasonable DG")                  

        # Lipids with 3 fatty acyls and too few carbons
        if val.startswith(("TG ", "MLCL ")):
            if (any(f" {i}:" in val for i in range(27, 29)) or      
                any(f" {i}:" in val for i in (13, 21, 23, 25, 27, 29, 41, 59, 61, 63, 65, 67, 69, 71, 73, 75)) or   
                any(f"/{i}:" in val for i in range(2, 11)) or
                any(f"_{i}:" in val for i in range(2, 11)) or
                any(f"_{i}:" in val for i in (13, 21, 23, 25, 27, 29, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75)) or 
                any(f"/{i}:" in val for i in (13, 21, 23, 25, 27, 29, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75)) or 
                any(f":{i}" in val for i in range(6, 50))   
                ):
                remove_flag = True 
                add_removed_row(row, "Lipids with 3 fatty acyls and too few carbons")
                                
        # Lipids with 3 fatty acyls and too many carbons
        if val.startswith(("TG ")):
            if (any(f" {i}:" in val for i in range(70, 200)) or    
                any(f" {i}:" in val for i in range(27, 42)) or       
                any(f"/{i}:" in val for i in range(27, 130)) or
                any(f"_{i}:" in val for i in range(27, 130)) or
                any(f" O-{i}:" in val for i in (13, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91)) or
                any(f";O{i}" in val for i in range(2, 20))      
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with 3 fatty acyls and too many carbons") 
                
        # Lipids with 3 fatty acyls and too many carbons
        if val.startswith(("MLCL ")):
            if (any(f" {i}:" in val for i in range(70, 200)) or      
                any(f"/{i}:" in val for i in range(27, 130)) or
                any(f"_{i}:" in val for i in range(27, 130)) or
                any(f" O-{i}:" in val for i in (13, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41))       
                ):
                remove_flag = True
                add_removed_row(row, "Lipids with 3 fatty acyls and too many carbons") 

        # Lipids with 4 fatty acyls and too few carbons
        if val.startswith(("CL ")):
            if (any(f" {i}:" in val for i in range(2, 11)) or     
                any(f"/{i}:" in val for i in range(2, 13)) or
                any(f"_{i}:" in val for i in range(2, 13)) or  
                any(f":{i}" in val for i in range(9, 50)) or
                any(f";O{i}" in val for i in range(3,20)) or
                any(f" {i}:" in val for i in (23, 25))      
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
        if (val.startswith(("ST ")) and
            ((f";Hex" in val) or      
            (f";GlcA" in val) or
            any(f":{i}" in val for i in range(6, 100)) or
            any(f";O{i}" in val for i in range(8, 100)) or
            any(f" {i}:" in val for i in range(1, 17)) or
            any(f" {i}:" in val for i in range(29, 100)) or
            any(f" {i}:" in val for i in (22, 23, 25, 28)) or 
            ((f";Gly" in val) and any(f" {i}:" in val for i in (18, 19, 20, 21, 27))) or
            ((f";Tau" in val) and any(f" {i}:" in val for i in (18, 19, 20, 21, 27))) or
            f";Leu" in val
            )):
            remove_flag = True
            add_removed_row(row, "Sterols / Steroids") 
                        
        # Sterols / Steroids - special cases 
        patterns = [
            "ST 18:3",
            "ST 18:5;O2;Gly",
            "ST 19:0;O7;G",
            "ST 19:1;O2;S",
            "ST 19:1;O2;Gly",
            "ST 19:3;O5;Tau",
            "ST 19:5",
            "ST 20:0;O3;Gly",
            "ST 20:0;O7;Tau",
            "ST 20:2;O;S ",
            "ST 20:2;O2;S ",
            "ST 20:3;O2;S",
            "ST 20:3;O3;Tau",
            "ST 20:4;O5;Tau",
            "ST 20:5",
            "ST 21:2;O4",
            "ST 21:3;O2",
            "ST 21:0;O3;Gly",
            "ST 21:2;O2;Gly",
            "ST 21:2;O2;Tau",
            "ST 21:4;O5;Gly",
            "ST 21:4;O7;Gly",
            "ST 21:5",
            "ST 21:6",
            "ST 24:0;O4",
            "ST 24:2;O7;Tau",
            "ST 24:4;O",
            "ST 24:6",
            "ST 24:8;O4",
            "ST 24:4;O3;Gly",
            "ST 26:2;O4;Gly",
            "ST 26:5",
            "ST 26:6",
            "ST 26:7",
            "ST 27:2;O4",
            "ST 27:3;O;F2",
            "ST 27:3;O2;F3",
            "ST 27:3;O3;F6",
            "ST 27:3;O4;Gly",
            "ST 27:6;O7;Gly",
            "ST 27:6;O6;Tau",
            "ST 27:4;O",
            "ST 27:0;O7;G",
            "ST 27:1;O4",
            "ST 27:5",
            "ST 27:6",
            "ST 27:7",
            "ST 27:8",
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
                'PnC ',
                'PnE ',
                "CerP 46:4;O3",
                "CerP 46:2;O3",
                "WE 22:",
                "HETE",
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
        dropped_path = debug_folder / f"{pol_tag}Annotations_Removed_by_plausibility.csv"
        pd.DataFrame(removed_rows).to_csv(dropped_path, index=False, encoding="utf-8-sig")

    # Drop them from the DataFrame
    if rows_to_drop:
        df = df.drop(rows_to_drop)

    print(f"Selected {len(rows_to_drop)} peaks to remove by plausibility filter ({mode}).", flush=True)
    print("Removal reasons summary:", dict(removal_reasons), flush=True)
    print("Matches left:", len(df), flush=True)

    return df