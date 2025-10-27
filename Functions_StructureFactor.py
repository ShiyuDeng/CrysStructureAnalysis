#!/usr/bin/env python3
#### all functions related to structure factor calculations and plotting
import pandas as pd

def calc_d_hexagonal(h, k, l, a, c):
    """Calculates the d-spacing for hexagonal crystal systems.
    1/d**2 = 4/3 * (h**2 + h*k + k**2) + 1**2/c**2
    """
    import math
    tmp = 4/3 * (h**2 + h*k + k**2)/a**2 + l**2/c**2
    d = 1/math.sqrt(tmp)
    return d

##############  READ files #####################
def parse_reflection_fcf(filepath, tag, a, c):
    """
    Parses the reflection output from CrysAlisPro (.fcf) 
    returns a DataFrame with columns:
    h, k, l, f2_calc, f2_meas, d_spacing
    """

    print(f"Read data from {filepath}:\n")

    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()

        in_reflection_block = False
        for line in lines:
            # Check for the start of the reflection data block
            if line.strip().startswith('_refln_index_h'):
                in_reflection_block = True
                continue
        
            # Process reflection data lines
            if in_reflection_block:
                if not line.strip() or line.strip().startswith('#'):
                    continue  # Skip empty lines or comments
                parts = line.strip().split()
                if len(parts) == 7:
                    h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                    fc = float(parts[3])
                    fo = float(parts[4])
                    # add a columne of calculated d-spacing based on hkl
                    d_spacing = calc_d_hexagonal(h, k, l, a, c)

                    data.append({'h': h, 
                                 'k': k, 
                                 'l': l, 
                                 'f2_calc': fc, 
                                 'f2_meas': fo, 
                                 'd': d_spacing,
                                 'tag': tag})
                else:
                    print(f"Skipping line due to unexpected format: {line.strip()}")
                    continue

    df = pd.DataFrame(data)
    #sort by d_spacing from largest to smallest
    df.sort_values(by='d', inplace=True, ascending=False)

    print(f"Parsed {len(df)} reflections from {filepath}\n")
    if df.empty:
        print("No valid reflections found in the file.")
        return df
    else:
        print(df)

    return df

def parse_hkl_vesta(filepath, tag, a, c):
    """
    Parse VESTA output for 
    (h, k, l) indices, 
    d spacing, 
    Structure factor |F|,
    Intensity I.
    """

    print(f"Read data from {filepath}:\n")
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            # Skip header or empty/comment lines
            if line.strip() == "" or line.startswith("h") or line.startswith("//"):
                continue
            parts = line.split()

            if len(parts) == 10:
                h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                d_direct = float(parts[3])
                StructureFactor = float(parts[6])
                Intensity = float(parts[8])
                d_calc = calc_d_hexagonal(h, k, l, a, c)

                data.append({'h': h, 
                        'k': k, 
                        'l': l, 
                        'd_direct': d_direct,
                        'd_calc': d_calc,
                        'StructureFactor': StructureFactor,
                        'Intensity': Intensity,
                        'tag': tag}) 
            else:
                print(f"Skipping line due to unexpected format: {line.strip()}")
                continue

    df = pd.DataFrame(data)
    
    print(f"Parsed {len(df)} reflections from {filepath}\n")
    if df.empty:
        print("No valid reflections found in the file.")
        return df
    else:
        print(df)

    return df

########## END : Read files ###########################


def pad_with_zeros(x, y):
    if len(y) > len(x):
        x += [0.0] * (len(y) - len(x))
    return x, y

############ Analysis: compare hkl ######
def compare_hkl(df1, df2, outfile):

    tag1 = df1['tag'].iloc[0] if 'tag' in df1.columns else 'DataFrame1'
    tag2 = df2['tag'].iloc[0] if 'tag' in df2.columns else 'DataFrame2'

    hkl1 = set(tuple(row) for row in df1[['h', 'k', 'l']].values)
    hkl2 = set(tuple(row) for row in df2[['h', 'k', 'l']].values)

    matches = hkl1 & hkl2
    only_in_1 = hkl1 - hkl2
    only_in_2 = hkl2 - hkl1

    with open(outfile, "w") as f:
        f.write(f"Comparison of (h k l) indices between {tag1} and {tag2}\n")
        f.write(f"Total (h k l) in {tag1}: {len(hkl1)}\n")
        f.write(f"Total (h k l) in {tag2}: {len(hkl2)}\n")
        f.write(f"Matches: {len(matches)}\n")
        f.write(f"Only in {tag1}: {len(only_in_1)}\n")
        f.write(f"Only in {tag2}: {len(only_in_2)}\n\n")

        if matches:
            f.write("\nMatched (h k l) indices:\n")
            for hkl in sorted(matches):
                f.write(f"{hkl}\n")
        if only_in_1:
            f.write(f"\n(h k l) indices only in {tag1}:\n")
            for hkl in sorted(only_in_1):
                f.write(f"{hkl}\n")
        if only_in_2:
            f.write(f"\n(h k l) indices only in {tag2}:\n")
            for hkl in sorted(only_in_2):
                f.write(f"{hkl}\n")
#####################################################


############# Analysis: compare in d-spacing #######
def compare_d_spacing(df1, df2, outfile, tolerance=0.0001):
    """
    Compare d-spacings between two datasets and find matches.
    Groups FCF reflections by d-spacing to avoid duplicate VESTA matches.
    
    Args:
        df1: exp. data, from parse_reflection_fcf (with columns: h, k, l, f2_calc, f2_meas, d)
        df2: model, from parse_hkl_vesta (with columns: h, k, l, d_direct, d_calc, StructureFactor, Intensity)
        outfile: Output file path to save results
        tolerance: Tolerance for d-spacing matching (default: 0.0001 Å)
    
    Returns:
        results: DataFrame with matched d-spacings and associated data
    """
    # Group FCF reflections by d-spacing (rounded to handle floating point precision)
    df1_grouped = df1.groupby(df1['d'].round(6))
    
    results = []
    d_spacing_groups = []
    
    # Sort groups by d-spacing in descending order
    sorted_groups = sorted(df1_grouped, key=lambda x: x[0], reverse=True)
    
    for d_spacing, group in sorted_groups:
        # Get all FCF reflections with this d-spacing
        fcf_reflections = []
        for _, row in group.iterrows():
            fcf_reflections.append({
                'h': row['h'], 'k': row['k'], 'l': row['l'],
                'd': row['d'], 'f2_calc': row['f2_calc'], 'f2_meas': row['f2_meas']
            })
        
        # Find VESTA matches for this d-spacing (use the first reflection's d-spacing as reference)
        reference_d = fcf_reflections[0]['d']
        vesta_matches = []
        
        for _, row2 in df2.iterrows():
            if abs(reference_d - row2['d_calc']) <= tolerance:
                vesta_matches.append({
                    'h': row2['h'], 'k': row2['k'], 'l': row2['l'],
                    'd_calc': row2['d_calc'], 'StructureFactor': row2['StructureFactor'], 'Intensity': row2['Intensity'],
                    'd_difference': abs(reference_d - row2['d_calc'])
                })
        
        # Sort VESTA matches by d_difference
        vesta_matches.sort(key=lambda x: x['d_difference'])
        
        # Store the grouped information
        d_spacing_groups.append({
            'fcf_reflections': fcf_reflections,
            'vesta_matches': vesta_matches,
            'd_spacing': reference_d
        })
        
        # Create results entries for individual row tracking (for compatibility)
        for fcf_refl in fcf_reflections:
            if vesta_matches:
                for i, vesta_match in enumerate(vesta_matches):
                    result_entry = {
                        'df1_h': fcf_refl['h'],
                        'df1_k': fcf_refl['k'],
                        'df1_l': fcf_refl['l'],
                        'df1_d': fcf_refl['d'],
                        'df1_f2_meas': fcf_refl['f2_meas'],
                        'df1_f2_calc': fcf_refl['f2_calc'],
                        'df2_h': vesta_match['h'],
                        'df2_k': vesta_match['k'],
                        'df2_l': vesta_match['l'],
                        'df2_d_calc': vesta_match['d_calc'],
                        'df2_Intensity': vesta_match['Intensity'],
                        'd_difference': vesta_match['d_difference'],
                        'match_rank': i + 1
                    }
                    results.append(result_entry)
            else:
                # No matches found
                result_entry = {
                    'df1_h': fcf_refl['h'],
                    'df1_k': fcf_refl['k'],
                    'df1_l': fcf_refl['l'],
                    'df1_d': fcf_refl['d'],
                    'df1_f2_meas': fcf_refl['f2_meas'],
                    'df1_f2_calc': fcf_refl['f2_calc'],
                    'df2_h': None,
                    'df2_k': None,
                    'df2_l': None,
                    'df2_d_calc': None,
                    'df2_Intensity': None,
                    'd_difference': None,
                    'match_rank': 0
                }
                results.append(result_entry)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to file
    print(f"\n=== Comparing d-spacing {df1['tag'].iloc[0]} vs {df2['tag'].iloc[0]} with tolerance {tolerance}")
    with open(outfile, 'w') as f:
        # Print header and summary
        f.write("FCF vs VESTA d-spacing Comparison\n")
        f.write(f"{df1['tag'].iloc[0]} vs {df2['tag'].iloc[0]}\n")
        f.write("="*80 + "\n")
        f.write(f"Tolerance used: {tolerance} Å\n")
        f.write(f"Total reflections in {df1['tag'].iloc[0]}: {len(df1)}\n")
        f.write(f"Total reflections in {df2['tag'].iloc[0]}: {len(df2)}\n")
        f.write(f"Unique d-spacing groups in {df1['tag'].iloc[0]}: {len(d_spacing_groups)}\n")
        f.write(f"d-spacing groups without matches: {len([g for g in d_spacing_groups if not g['vesta_matches']])}\n")
        f.write("\n")
        
        # Write df1 in tabular format
        f.write("="*80 + "\n")
        f.write(f"Dataset 1: {df1['tag'].iloc[0]}\n")
        f.write("="*80 + "\n")
        f.write(f"{'(h,k,l)':<12} {'d_spacing':<12} {'F²_calc':<12} {'F²_meas':<12}\n")
        f.write("-" * 48 + "\n")
        for _, row in df1.iterrows():
            hkl = f"({int(row['h'])},{int(row['k'])},{int(row['l'])})"
            f.write(f"{hkl:<12} {row['d']:<12.4f} {row['f2_calc']:<12.2f} {row['f2_meas']:<12.2f}\n")
        
        f.write("\n")
        
        # Write df2 in tabular format
        f.write("="*80 + "\n")
        f.write(f"Dataset 2: {df2['tag'].iloc[0]}\n")
        f.write("="*80 + "\n")
        f.write(f"{'(h,k,l)':<12} {'d_spacing':<12} {'StructFactor':<12} {'Intensity':<12}\n")
        f.write("-" * 48 + "\n")
        for _, row in df2.iterrows():
            hkl = f"({int(row['h'])},{int(row['k'])},{int(row['l'])})"
            f.write(f"{hkl:<12} {row['d_calc']:<12.4f} {row['StructureFactor']:<12.2f} {row['Intensity']:<12.2f}\n")
        
        f.write("\n")
        
        # Write comparison results using grouped format
        f.write("="*80 + "\n")
        f.write("FCF vs VESTA d-spacing Comparison Results (Grouped by d-spacing):\n")
        f.write(f"FCF:{df1['tag'].iloc[0]}\n")
        f.write(f"VESTA:{df2['tag'].iloc[0]}\n")
        f.write("="*80 + "\n")
        
        # Process each d-spacing group
        for group in d_spacing_groups:
            fcf_reflections = group['fcf_reflections']
            vesta_matches = group['vesta_matches']
            d_spacing = group['d_spacing']
            
            # Write d-spacing header
            f.write(f"\nd-spacing: {d_spacing:.4f} Å\n")
            f.write("-" * 30 + "\n")
            
            # Write FCF reflections for this d-spacing
            f.write("FCF reflections:\n")
            for refl in fcf_reflections:
                hkl = f"({int(refl['h'])},{int(refl['k'])},{int(refl['l'])})"
                f.write(f"  {hkl:<12} F²_calc: {refl['f2_calc']:<8.2f} F²_meas: {refl['f2_meas']:<8.2f}\n")
            
            # Write VESTA matches for this d-spacing
            if vesta_matches:
                f.write("VESTA matches:\n")
                for match in vesta_matches:
                    hkl = f"({int(match['h'])},{int(match['k'])},{int(match['l'])})"
                    f.write(f"  {hkl:<12} StructFactor: {match['StructureFactor']:<8.2f} Intensity: {match['Intensity']:<8.2f}\n")
            else:
                f.write("VESTA matches: None\n")
            
            f.write("\n")
    
    print(f"d-spacing comparison results saved to: {outfile}")
    
    return results_df
##############################


############# Analysis: compare FCF files in d-spacing #######
def compare_fcf_in_dspacing(df1, df2, outfile, tolerance=0.0001):
    """
    Compare d-spacings between two FCF datasets and find matches.
    Groups FCF1 reflections by d-spacing to avoid duplicate FCF2 matches.
    
    Args:
        df1: FCF data 1, from parse_reflection_fcf (with columns: h, k, l, f2_calc, f2_meas, d)
        df2: FCF data 2, from parse_reflection_fcf (with columns: h, k, l, f2_calc, f2_meas, d)
        outfile: Output file path to save results
        tolerance: Tolerance for d-spacing matching (default: 0.0001 Å)
    
    Returns:
        results: DataFrame with matched d-spacings and associated data
    """
    # Group FCF1 reflections by d-spacing (rounded to handle floating point precision)
    df1_grouped = df1.groupby(df1['d'].round(6))
    
    results = []
    d_spacing_groups = []
    
    # Sort groups by d-spacing in descending order
    sorted_groups = sorted(df1_grouped, key=lambda x: x[0], reverse=True)
    
    for d_spacing, group in sorted_groups:
        # Get all FCF1 reflections with this d-spacing
        fcf1_reflections = []
        for _, row in group.iterrows():
            fcf1_reflections.append({
                'h': row['h'], 'k': row['k'], 'l': row['l'],
                'd': row['d'], 'f2_calc': row['f2_calc'], 'f2_meas': row['f2_meas']
            })
        
        # Find FCF2 matches for this d-spacing (use the first reflection's d-spacing as reference)
        reference_d = fcf1_reflections[0]['d']
        fcf2_matches = []
        
        for _, row2 in df2.iterrows():
            if abs(reference_d - row2['d']) <= tolerance:
                fcf2_matches.append({
                    'h': row2['h'], 'k': row2['k'], 'l': row2['l'],
                    'd': row2['d'], 'f2_calc': row2['f2_calc'], 'f2_meas': row2['f2_meas'],
                    'd_difference': abs(reference_d - row2['d'])
                })
        
        # Sort FCF2 matches by d_difference
        fcf2_matches.sort(key=lambda x: x['d_difference'])
        
        # Store the grouped information
        d_spacing_groups.append({
            'fcf1_reflections': fcf1_reflections,
            'fcf2_matches': fcf2_matches,
            'd_spacing': reference_d
        })
        
        # Create results entries for individual row tracking (for compatibility)
        for fcf1_refl in fcf1_reflections:
            if fcf2_matches:
                for i, fcf2_match in enumerate(fcf2_matches):
                    result_entry = {
                        'df1_h': fcf1_refl['h'],
                        'df1_k': fcf1_refl['k'],
                        'df1_l': fcf1_refl['l'],
                        'df1_d': fcf1_refl['d'],
                        'df1_f2_meas': fcf1_refl['f2_meas'],
                        'df1_f2_calc': fcf1_refl['f2_calc'],
                        'df2_h': fcf2_match['h'],
                        'df2_k': fcf2_match['k'],
                        'df2_l': fcf2_match['l'],
                        'df2_d': fcf2_match['d'],
                        'df2_f2_calc': fcf2_match['f2_calc'],
                        'df2_f2_meas': fcf2_match['f2_meas'],
                        'match_rank': i + 1
                    }
                    results.append(result_entry)
            else:
                # No matches found
                result_entry = {
                    'df1_h': fcf1_refl['h'],
                    'df1_k': fcf1_refl['k'],
                    'df1_l': fcf1_refl['l'],
                    'df1_d': fcf1_refl['d'],
                    'df1_f2_meas': fcf1_refl['f2_meas'],
                    'df1_f2_calc': fcf1_refl['f2_calc'],
                    'df2_h': None,
                    'df2_k': None,
                    'df2_l': None,
                    'df2_d': None,
                    'df2_f2_calc': None,
                    'df2_f2_meas': None,
                    'match_rank': 0
                }
                results.append(result_entry)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to file
    print(f"\n=== Comparing d-spacing {df1['tag'].iloc[0]} vs {df2['tag'].iloc[0]} with tolerance {tolerance}")
    with open(outfile, 'w') as f:
        # Print header and summary
        f.write("FCF d-spacing Comparison\n")
        f.write(f"{df1['tag'].iloc[0]} vs {df2['tag'].iloc[0]}\n")
        f.write("="*80 + "\n")
        f.write(f"Tolerance used: {tolerance} Å\n")
        f.write(f"Total reflections in {df1['tag'].iloc[0]}: {len(df1)}\n")
        f.write(f"Total reflections in {df2['tag'].iloc[0]}: {len(df2)}\n")
        f.write(f"Unique d-spacing groups in {df1['tag'].iloc[0]}: {len(d_spacing_groups)}\n")
        f.write(f"d-spacing groups without matches: {len([g for g in d_spacing_groups if not g['fcf2_matches']])}\n")
        f.write("\n")
        
        # Write df1 in tabular format
        f.write("="*80 + "\n")
        f.write(f"Dataset 1: {df1['tag'].iloc[0]}\n")
        f.write("="*80 + "\n")
        f.write(f"{'(h,k,l)':<12} {'d_spacing':<12} {'F²_calc':<12} {'F²_meas':<12}\n")
        f.write("-" * 48 + "\n")
        for _, row in df1.iterrows():
            hkl = f"({int(row['h'])},{int(row['k'])},{int(row['l'])})"
            f.write(f"{hkl:<12} {row['d']:<12.4f} {row['f2_calc']:<12.2f} {row['f2_meas']:<12.2f}\n")
        
        f.write("\n")
        
        # Write df2 in tabular format
        f.write("="*80 + "\n")
        f.write(f"Dataset 2: {df2['tag'].iloc[0]}\n")
        f.write("="*80 + "\n")
        f.write(f"{'(h,k,l)':<12} {'d_spacing':<12} {'F²_calc':<12} {'F²_meas':<12}\n")
        f.write("-" * 48 + "\n")
        for _, row in df2.iterrows():
            hkl = f"({int(row['h'])},{int(row['k'])},{int(row['l'])})"
            f.write(f"{hkl:<12} {row['d']:<12.4f} {row['f2_calc']:<12.2f} {row['f2_meas']:<12.2f}\n")
        
        f.write("\n")
        
        # Write comparison results using grouped format
        f.write("="*80 + "\n")
        f.write("FCF d-spacing Comparison Results (Grouped by d-spacing):\n")
        f.write(f"FCF1:{df1['tag'].iloc[0]}\n")
        f.write(f"FCF2:{df2['tag'].iloc[0]}\n")
        f.write("="*80 + "\n")
        
        # Process each d-spacing group
        for group in d_spacing_groups:
            fcf1_reflections = group['fcf1_reflections']
            fcf2_matches = group['fcf2_matches']
            d_spacing = group['d_spacing']
            
            # Write d-spacing header
            f.write(f"\nd-spacing: {d_spacing:.4f} Å\n")
            f.write("-" * 30 + "\n")
            
            # Write FCF1 reflections for this d-spacing
            f.write("FCF1 reflections:\n")
            for refl in fcf1_reflections:
                hkl = f"({int(refl['h'])},{int(refl['k'])},{int(refl['l'])})"
                f.write(f"  {hkl:<12} F²_calc: {refl['f2_calc']:<8.2f} F²_meas: {refl['f2_meas']:<8.2f}\n")
            
            # Write FCF2 matches for this d-spacing
            if fcf2_matches:
                f.write("FCF2 matches:\n")
                for match in fcf2_matches:
                    hkl = f"({int(match['h'])},{int(match['k'])},{int(match['l'])})"
                    f.write(f"  {hkl:<12} F²_calc: {match['f2_calc']:<8.2f} F²_meas: {match['f2_meas']:<8.2f}\n")
            else:
                f.write("FCF2 matches: None\n")
            
            f.write("\n")
    
    print(f"FCF d-spacing comparison results saved to: {outfile}")
    
    return results_df
##############################


############# PLOTTING FUNCTIONS #############
def plot_f2(fc, fo, file_title, outfile, P=0, weights=None, min_val=None, max_val=None):
    """
    Plots F^2_calc vs F^2_meas.
    Reference for R factor: http://pd.chem.ucl.ac.uk/pdnn/refine1/rfacs.htm
    R = Σ|F_o² - F_c²| / ΣF_o²
    χ² = (Rwp / Rexp)²
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    fc = np.array(fc)
    fo = np.array(fo)

    # Default weights (equal weighting)
    if weights is None:
        weights = np.ones_like(fo)
    else:
        weights = np.array(weights, dtype=float)

    # define as a function 
    N = len(fo)

    # R factor
    numerator = np.sum(np.abs(fo - fc))
    denominator = np.sum(np.abs(fo))
    R_factor = numerator / denominator if denominator != 0 else np.nan

    # --- Rwp, Rexp, χ² ---
    diff_sq = (np.sqrt(fo) - np.sqrt(fc)) ** 2
    Rwp = np.sqrt(np.sum(weights * diff_sq) / np.sum(weights * fo))
    Rexp= np.sqrt((N - P ) / np.sum(weights * fo)) if N > P else np.nan
    chi2 = (Rwp / Rexp) ** 2 if Rexp != 0 else np.nan

    text_str = (
        f"$R = {R_factor:.4f}$\n"
        f"$R_{{wp}} = {Rwp:.4f}$\n"
        f"$R_{{exp}} = {Rexp:.4f}$\n"
        f"$\\chi^2 = {chi2:.2f}$"
    )
    print(f"For {file_title}:\n{text_str}")

    plt.figure(figsize=(6, 5))
    plt.scatter(fc, fo, alpha=0.6, edgecolor='k', s=20)

    if min_val is None:
        min_val = min(min(fc), min(fo))
    if max_val is None:
        max_val = 1.05 * max(max(fc), max(fo))
    # min_val, max_val = 0, 70000

    plt.text(
        0.05, 0.65, 
        text_str,
        transform=plt.gca().transAxes, 
        fontsize=12, 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    plt.xlabel(f"F² (calculated)")
    plt.ylabel(f"F² (observed)")
    plt.title(f"{file_title}")

    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved to: {outfile}")
