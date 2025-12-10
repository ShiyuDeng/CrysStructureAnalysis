#!/usr/bin/env python3
#### all functions related to structure factor calculations and plotting
import pandas as pd
import numpy as np

def calc_d_hexagonal(h, k, l, a, c):
    """Calculates the d-spacing for hexagonal crystal systems.
    1/d**2 = 4/3 * (h**2 + h*k + k**2) + 1**2/c**2
    """
    tmp = 4/3 * (h**2 + h*k + k**2)/a**2 + l**2/c**2
    d = 1/np.sqrt(tmp)
    return d

##############  READ files #####################
def parse_reflection_hkl(filepath, tag, a=None, c=None):
    """
    Parses the SHELX HKL file format from CrysAlisPro (.hkl)
    Format: h k l F² sigma (5 columns)
    Cell parameters are at the end of the file in CELL line.

    If a and c are not provided, they will be extracted from the file.

    Returns a DataFrame with columns:
    h, k, l, f2_meas, sigma_f2_meas, d_calc, tag

    Note: HKL files don't contain F²_calc, only measured values.
    """
    ldebug = False
    print(f"Read HKL data from {filepath}:\n")

    data = []
    cell_params = {}

    with open(filepath, 'r') as file:
        lines = file.readlines()

    # First pass: extract cell parameters from CELL line if a,c not provided
    if a is None or c is None:
        for line in lines:
            if line.strip().startswith('CELL'):
                # CELL wavelength a b c alpha beta gamma
                parts = line.strip().split()
                if len(parts) >= 7:
                    cell_params['wavelength'] = float(parts[1])
                    cell_params['a'] = float(parts[2])
                    cell_params['b'] = float(parts[3])
                    cell_params['c'] = float(parts[4])
                    cell_params['alpha'] = float(parts[5])
                    cell_params['beta'] = float(parts[6])
                    cell_params['gamma'] = float(parts[7])
                    print(f"Extracted cell parameters from file:")
                    print(f"  a={cell_params['a']:.4f}, b={cell_params['b']:.4f}, c={cell_params['c']:.4f}")
                    print(f"  alpha={cell_params['alpha']:.2f}, beta={cell_params['beta']:.2f}, gamma={cell_params['gamma']:.2f}")
                    if a is None:
                        a = cell_params['a']
                    if c is None:
                        c = cell_params['c']
                break

    if a is None or c is None:
        print("ERROR: Could not determine cell parameters. Please provide a and c.")
        return pd.DataFrame()

    # Second pass: read reflection data
    for line in lines:
        line_stripped = line.strip()

        # Skip empty lines
        if not line_stripped:
            continue

        # Skip SHELX command lines (start with letters)
        if line_stripped[0].isalpha():
            continue

        # Skip comment lines
        if line_stripped.startswith('#'):
            continue

        parts = line_stripped.split()

        # HKL data lines have 5 numeric columns: h k l F² sigma
        if len(parts) >= 5:
            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                f2 = float(parts[3])
                sigma = float(parts[4])

                # Skip the origin reflection (0,0,0)
                if h == 0 and k == 0 and l == 0:
                    continue

                # Calculate d-spacing
                d_calc = calc_d_hexagonal(h, k, l, a, c)

                data.append({
                    'h': h, 'k': k, 'l': l,
                    'f2_meas': f2,
                    'sigma_f2_meas': sigma,
                    'd_calc': d_calc,
                    'tag': tag
                })
            except (ValueError, IndexError):
                # Not a valid data line, skip it
                if ldebug:
                    print(f"DEBUG: Skipping line: {line_stripped}")
                continue

    df = pd.DataFrame(data)
    if ldebug:
        print(f"DEBUG: df read\n{format(df)}")

    # Sort by d_spacing from largest to smallest
    if not df.empty:
        df.sort_values(by='d_calc', inplace=True, ascending=False)

    if ldebug:
        print(f"DEBUG: after sorting by d, df\n{format(df)}")

    print(f"Parsed {len(df)} reflections from {filepath}\n")
    if df.empty:
        print("No valid reflections found in the file.")
    else:
        print(df)

    return df


def parse_reflection_fcf(filepath, tag, a, c):
    """
    Parses the reflection output from CrysAlisPro (.fcf) 
    returns a DataFrame with columns:
    h, k, l, f2_calc, f2_meas, d_spacing
    """
    ldebug=False
    ### add: local_loop=False
    # if local_loop:
    #     REQUIRED_COLUMNS = {
    #         'h': '_refln_index_h',
    #         'k': '_refln_index_k',
    #         'l': '_refln_index_l',
    #         'f2_calc': '_refln_F_squared_calc',
    #         'f2_meas': '_refln_F_squared_meas',
    #         'sigma_f2_meas': '_refln_F_squared_sigma',
    #     }
    #     column_indices = {} 
    #     in_loop_header = False

    print(f"Read data from {filepath}:\n")
    in_reflection_block = False
    data = []
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Check for the start of the reflection data block
            if line.strip().startswith('_refln_index_h'):
                in_reflection_block = True
                continue

            ### for future development and debug: more robust reading ###
            # if line.strip().startswith('loop_'):
            #     in_loop_header = True
            #     in_reflection_block = False
            #     column_indices = {}
            
            #     if in_loop_header:
            #         # Check if the line is one of the required column tags
            #         is_column_header = False
            #         for output_key, tag_name in REQUIRED_COLUMNS.items():
            #             if line.startswith(tag_name):
            #                 # The index of the column in the data line is its order in the header list
            #                 column_indices[output_key] = len(column_indices)
            #                 is_column_header = True
            #                 break
                    
            #         if not is_column_header:
            #             if not line or line.startswith('#') or line.startswith('_'):
            #                 in_loop_header = False
            #             else:
            #                 in_loop_header = False
                        
            #             # Check the required columns
            #             if not in_loop_header: # Header reading stopped
            #                 if len(column_indices) == len(REQUIRED_COLUMNS):
            #                     in_reflection_block = True
            #                 else:
            #                     continue
                            
            if in_reflection_block:
                if not line.strip() or line.strip().startswith('#'):
                    continue  # Skip empty lines or comments
                parts = line.strip().split()

                if len(parts) >= 6:
                    h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                    fc = float(parts[3])
                    fo = float(parts[4])
                    f2_sigma = float(parts[5])

                    # if local_loop:
                    #     h = int(parts[column_indices['h']])
                    #     k = int(parts[column_indices['k']])
                    #     l = int(parts[column_indices['l']])
                    #     fc = float(parts[column_indices['f2_calc']])
                    #     fo = float(parts[column_indices['f2_meas']])
                    #     f2_sigma = float(parts[column_indices['sigma_f2_meas']])

                    # add a columne of calculated d-spacing based on hkl
                    d_calc = calc_d_hexagonal(h, k, l, a, c)

                    data.append({
                        'h': h, 'k': k, 'l': l, 
                        'f2_calc': fc, 
                        'f2_meas': fo, 
                        'sigma_f2_meas': f2_sigma, 
                        'd_calc': d_calc,
                        'tag': tag
                    })
                else:
                    print(f"WARNING: check line format! {line.strip()}")
                    exit

    df = pd.DataFrame(data)
    if ldebug:
        print(f"DEBUG: df read\n{format(df)}")

    #sort by d_spacing from largest to smallest
    df.sort_values(by='d_calc', inplace=True, ascending=False)
    if ldebug: 
        print(f"DEBUG: after sorting by d, df\n{format(df)}")

    print(f"Parsed {len(df)} reflections from {filepath}\n")
    if df.empty:
        print("No valid reflections found in the file.")
        return df
    else:
        print(df)

    return df
########################################################


def parse_hkl_vesta(filepath, tag, a, c):
    """
    Parse VESTA output robustly using column names.
    Automatically handles header and extracts (h, k, l, |F|, I) by name.
    """
    print(f"Reading data from {filepath}...")
    
    try:
        df = pd.read_fwf(filepath, sep=r'\s+', header=0)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return pd.DataFrame()

    # Rename columns for easy access, handling common variations in VESTA output
    df.rename(columns={
        'd (Å)': 'd_direct',
        '|F|': 'StructureFactor',
        'I': 'Intensity'
    }, inplace=True)
    
    # 3. Select and ensure presence of required columns
    required_cols = ['h', 'k', 'l', 'd_direct', 'StructureFactor', 'Intensity']
    
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: File is missing required columns: {missing}. Available columns: {list(df.columns)}")
        return pd.DataFrame()

    # 4. Filter the DataFrame to include only the required columns
    df = df[required_cols]

    # 5. Calculate d_calc and add tag
    # Use pandas vectorization for performance
    df['d_calc'] = calc_d_hexagonal(df['h'], df['k'], df['l'], a, c)
    df['tag'] = tag

    print(f"Parsed {len(df)} reflections from {filepath}")
    
    if df.empty:
        print("No valid reflections found after parsing.")
    else:
        print(df.head()) # Print the first few rows

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
    df1_grouped = df1.groupby(df1['d_calc'].round(6))
    
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
                'd_calc': row['d_calc'], 'f2_calc': row['f2_calc'], 'f2_meas': row['f2_meas']
            })
        
        # Find VESTA matches for this d-spacing (use the first reflection's d-spacing as reference)
        reference_d = fcf_reflections[0]['d_calc']
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
                        'df1_d': fcf_refl['d_calc'],
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
                    'df1_d': fcf_refl['d_calc'],
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
            f.write(f"{hkl:<12} {row['d_calc']:<12.4f} {row['f2_calc']:<12.2f} {row['f2_meas']:<12.2f}\n")
        
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
    # add filter to avoid negative values under sqrt
    fo = np.clip(fo, a_min=0, a_max=None)
    fc = np.clip(fc, a_min=0, a_max=None)   

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
    plt.close()
    print(f"Plot saved to: {outfile}")


########### HKL general condition for hexagonal cell ########
def HKL_hexagonal_conditions(h,k,l):
    """ check if the (h,k,l) meets hexagonal conditions:
    -h + k + l = 3n, n: integer
    return True if conditions are met, else False
    """
    cond1 = (-h + k + l) % 3 == 0
    return cond1 
#############################################################


#################      Friedel Pairs analysis  ##############
def analyze_friedel_pairs(df_in, outfile, condition='None'):
    """ check the intensity of (h,h,k) vs (-h,-k,-l) pairs in the df_in
    analysis results in outfile
    """

    with open(outfile, 'w') as f:
        f.write("Friedel Pairs Analysis\n")
        f.write("="*50 + "\n")

        if condition == 'hexagonal':
            f.write("Applying hexagonal conditions for (h,k,l)")
            df_filtered = df_in[df_in.apply(lambda row: HKL_hexagonal_conditions(row['h'], row['k'], row['l']), axis=1)]
        else:
            df_filtered = df_in

        f.write(f"Total reflections to be analyzed: {len(df_filtered)}\n\n")

        # Create a set to track processed pairs
        processed_pairs = set()

        for _, row in df_filtered.iterrows():
            h, k, l = row['h'], row['k'], row['l']
            friedel_hkl = (-h, -k, -l)

            if (h, k, l) in processed_pairs or friedel_hkl in processed_pairs:
                continue  # Skip already processed pairs

            # Find the Friedel pair in the DataFrame
            pair_row = df_filtered[
                (df_filtered['h'] == friedel_hkl[0]) & 
                (df_filtered['k'] == friedel_hkl[1]) & 
                (df_filtered['l'] == friedel_hkl[2])
            ]

            if not pair_row.empty:
                f2_meas_1 = row['f2_meas']
                f2_meas_2 = pair_row.iloc[0]['f2_meas']
                ratio = f2_meas_1 / f2_meas_2 if f2_meas_2 != 0 else np.nan

                f.write(f"Pair: ({h}, {k}, {l}) and ({friedel_hkl[0]}, {friedel_hkl[1]}, {friedel_hkl[2]})\n")
                f.write(f"  F²_meas 1: {f2_meas_1:.2f}\n")
                f.write(f"  F²_meas 2: {f2_meas_2:.2f}\n")
                f.write(f"  Ratio (1/2): {ratio:.4f}\n\n")

                processed_pairs.add((h, k, l))
                processed_pairs.add(friedel_hkl)

    print(f"Friedel pair analysis completed. Results saved to {outfile}.")

    ### print out the top 10 largest deviations
    df_friedel = prepare_friedel_data(df_in, condition=condition)
    df_friedel['Abs_Delta_I_obs'] = df_friedel['Delta_I_obs'].abs()
    df_friedel_sorted = df_friedel.sort_values(by='Abs_Delta_I_obs', ascending=False)
    print("\nTop 10 Friedel pairs with largest intensity differences:")
    print(df_friedel_sorted[['h', 'k', 'l', 'I_obs_hkl', 'I_obs_hkl_inv', 'Delta_I_obs', 'Z_score']].head(10))  

        
################# END - Friedel Pairs analysis ##############


def is_primary_pair(row):
        hkl = row['hkl']
        inv = (-row['h'], -row['k'], -row['l'])
        # Exclude self-pair (e.g., origin) where hkl == inv
        if hkl == inv:
            return False
        # Keep only one orientation to avoid double counting
        return hkl > inv

#################       Friedel Pairs analysis (Statistical)  ##############
def prepare_friedel_data(df_in, condition='None'):
    """ 
    Prepares a clean DataFrame containing one row per Friedel pair, 
    calculating the observed intensity difference and its statistical significance.
    """
    ldebug=False

    if condition == 'hexagonal':
        print("Applying hexagonal conditions for (h,k,l)")
        df_filtered = df_in[df_in.apply(lambda row: HKL_hexagonal_conditions(row['h'], row['k'], row['l']), axis=1)].copy()
    else:
        df_filtered = df_in.copy() 

    # 1. Create the HKL key for matching
    df_filtered['hkl'] = df_filtered.apply(lambda row: (row['h'], row['k'], row['l']), axis=1)
    df_filtered['hkl_inv'] = df_filtered.apply(lambda row: (-row['h'], -row['k'], -row['l']), axis=1)

    # 2. Rename columns for reflection 1 (hkl)
    df1 = df_filtered[['hkl', 'h', 'k', 'l', 'f2_meas', 'sigma_f2_meas']].rename(
        columns={'f2_meas': 'I_obs_hkl', 'sigma_f2_meas': 'sigma_hkl'}
    )
    
    # 3. Rename columns for reflection 2 (hkl_inv)
    df2 = df_filtered[['hkl_inv', 'f2_meas', 'sigma_f2_meas']].rename(
        columns={'hkl_inv': 'hkl', 'f2_meas': 'I_obs_hkl_inv', 'sigma_f2_meas': 'sigma_hkl_inv'}
    )

    # 4. Merge to pair up the reflections
    # The merge is performed on the primary hkl index and the inverted hkl_inv index
    # We only keep pairs where (hkl) is lexicographically smaller than (-h -k -l) 
    # to ensure each pair is counted only once.
    df_merged = pd.merge(
        df1, 
        df2, 
        on='hkl', 
        how='inner'
    )
    if ldebug:
        print(df_merged.head())

    df_pairs = df_merged[df_merged.apply(is_primary_pair, axis=1)].copy()
    
    # 5. Calculate Statistical Metrics
    df_pairs['Delta_I_obs'] = df_pairs['I_obs_hkl'] - df_pairs['I_obs_hkl_inv']
    
    # Statistical Significance (e.s.d. of the difference)
    # sigma(Delta I) = sqrt( sigma(I_hkl)^2 + sigma(I_inv)^2 )
    df_pairs['sigma_Delta_I_obs'] = np.sqrt(
        df_pairs['sigma_hkl']**2 + df_pairs['sigma_hkl_inv']**2
    )
    
    # Z-score (Statistical Significance)
    # This is the number of standard deviations the difference is away from zero.
    df_pairs['Z_score'] = df_pairs['Delta_I_obs'] / df_pairs['sigma_Delta_I_obs']
    
    print(f"Total unique Friedel pairs found: {len(df_pairs)}")
    
    return df_pairs[['h', 'k', 'l', 'I_obs_hkl', 'I_obs_hkl_inv', 
                     'sigma_hkl', 'sigma_hkl_inv', 'Delta_I_obs', 
                     'sigma_Delta_I_obs', 'Z_score']]

################# END - Friedel Pairs analysis (Statistical) ##############


##########################  PLOT  #################################
def plot_friedel_analysis(df_pairs, output_dir='./friedel_plots', label=None):
    """
    Generates the two critical plots for proving broken inversion symmetry:
    1. I(hkl) vs I(-h-k-l) Scatter Plot - deviation from y=x line indicates non-centrosymmetry.
    2. Distribution of Statistical Significance (Z-score)：
    - show a distribution significantly wider and/or less centered than the expected Gaussian, proving the deviations are physical, not noise.

    Args:
        df_pairs: DataFrame with Friedel pair data
        output_dir: Directory to save plots (default: './friedel_plots')
        label: Optional prefix for output filenames (e.g., '012_FePSe3' -> '012_FePSe3_Intensity_Pairs.png')
    """
    import os
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    try:
        import seaborn as sns
    except Exception:
        sns = None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # set fontsize for both plots
    plt.rcParams.update({'font.size': 14})

    # --- PLOT 1: I(hkl) vs I(-h-k-l) Scatter Plot ---
    fig, ax = plt.subplots(figsize=(8,6))
    # Raw data in scatter plot (log scale)
    ax.scatter(df_pairs['I_obs_hkl'], df_pairs['I_obs_hkl_inv'], 
               marker='x', s=20, alpha=0.8, color="#0032f9") 
    
    # Plot the y=x line (Centrosymmetric condition)
    I_max = 2 * max(df_pairs['I_obs_hkl'].max(), df_pairs['I_obs_hkl_inv'].max())
    ax.plot([0, I_max], [0, I_max], 'r--', alpha=0.50,
            label=r'$I(hkl) = I(\bar{h}\bar{k}\bar{l})$')
    ax.set_xlim(1, I_max)
    ax.set_ylim(1, I_max)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel(r'$I_{\text{obs}}(hkl)$ [a.u.]')
    ax.set_ylabel(r'$I_{\text{obs}}(\bar{h}\bar{k}\bar{l})$ [a.u.]')
    if label:
        ax.set_title(f'{label}\nObserved Friedel Pair Intensities')
    else:
        ax.set_title('Observed Friedel Pair Intensities')
    ax.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    
    # Construct filename with optional label prefix
    if label:
        plot1_name = f'{label}_Intensity_Pairs.png'
    else:
        plot1_name = 'Intensity_Pairs.png'
    plot1_path = os.path.join(output_dir, plot1_name)
    fig.savefig(plot1_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot 1 saved to {plot1_path}")
    
    # --- PLOT 2: Distribution of Statistical Significance (Z-score) ---
    fig, ax = plt.subplots(figsize=(8,6))
    df_valid = df_pairs.dropna(subset=['Z_score'])
    
    # Plot the histogram of the Z-score
    if (sns is not None):
        sns.histplot(df_valid['Z_score'], bins=60, kde=True, ax=ax, color='#ff7f0e')
    else:
        ax.hist(df_valid['Z_score'].values, bins=60, color='#ff7f0e', alpha=0.8)
    
    # Add the expected normal distribution for reference (Centrosymmetric, random noise, Gaussian)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 200)
    gaussian_pdf = norm.pdf(x, 0, 1)
    scale_factor = ax.get_ylim()[1] / gaussian_pdf.max()
    ax.plot(x, gaussian_pdf * scale_factor * 0.9, 'k--', linewidth=1.25, 
            label='Expected Noise\n(Gaussian, $\sigma$=1)')
    
    bound=max(abs(xmin), abs(xmax))
    ax.set_xlim(-bound, bound)

    ax.set_xlabel(r'$Z_{\text{obs}} = \frac{\Delta I_{\text{obs}}}{\sigma(\Delta I_{\text{obs}})}$')
    ax.set_ylabel('Count (Number of Friedel Pairs)')
    if label:
        ax.set_title(f'{label}\nStatistical Significance of Anomalous Signal')
    else:
        ax.set_title('Statistical Significance of Anomalous Signal')
    ax.legend(loc='upper right')

    # Construct filename with optional label prefix
    if label:
        plot2_name = f'{label}_Z_score_distribution.png'
    else:
        plot2_name = 'Z_score_distribution.png'
    plot2_path = os.path.join(output_dir, plot2_name)
    fig.savefig(plot2_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot 2 saved to {plot2_path}")

