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
    
    Args:
        df1: exp. data, from parse_reflection_fcf (with columns: h, k, l, f2_calc, f2_meas, d)
        df2: model, from parse_hkl_vesta (with columns: h, k, l, d_direct, d_calc, StructureFactor, Intensity)
        outfile: Output file path to save results
        tolerance: Tolerance for d-spacing matching (default: 0.01 Å)
    
    Returns:
        results: DataFrame with matched d-spacings and associated data
    """
    results = []
    
    # Iterate through each row in df1
    for idx1, row1 in df1.iterrows():
        d1 = row1['d']
        h1, k1, l1 = row1['h'], row1['k'], row1['l']
        f2_meas = row1['f2_meas']
        f2_calc = row1['f2_calc']
        
        # Find all matching d-spacings in df2 within tolerance
        matches_in_df2 = []
        for idx2, row2 in df2.iterrows():
            d2 = row2['d_calc']
            if abs(d1 - d2) <= tolerance:
                matches_in_df2.append({
                    'df2_h': row2['h'],
                    'df2_k': row2['k'],
                    'df2_l': row2['l'],
                    'df2_d_calc': d2,
                    'df2_Intensity': row2['Intensity'],
                    'd_difference': abs(d1 - d2)
                })
        
        if matches_in_df2:
            matches_in_df2.sort(key=lambda x: x['d_difference'])
            
            for i, match in enumerate(matches_in_df2):
                result_entry = {
                    'df1_h': h1,
                    'df1_k': k1,
                    'df1_l': l1,
                    'df1_d': d1,
                    'df1_f2_meas': f2_meas,
                    'df1_f2_calc': f2_calc,
                    'df2_h': match['df2_h'],
                    'df2_k': match['df2_k'],
                    'df2_l': match['df2_l'],
                    'df2_d_calc': match['df2_d_calc'],
                    'df2_Intensity': match['df2_Intensity'],
                    'd_difference': match['d_difference'],
                    'match_rank': i + 1 
                }
                results.append(result_entry)
        else:
            # No matches found - still record the df1 entry
            result_entry = {
                'df1_h': h1,
                'df1_k': k1,
                'df1_l': l1,
                'df1_d': d1,
                'df1_f2_meas': f2_meas,
                'df1_f2_calc': f2_calc,
                'df2_h': None,
                'df2_k': None,
                'df2_l': None,
                'df2_d_calc': None,
                'df2_Intensity': None,
                'd_difference': None,
                'match_rank': 0  # 0 indicates no match found
            }
            results.append(result_entry)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to file
    with open(outfile, 'w') as f:
        # Print out df1 and df2 in tabular format first
        
        f.write("d-spacing Comparison\n")
        f.write(f"{df1['tag'].iloc[0]} vs {df2['tag'].iloc[0]}\n")
        f.write("="*80 + "\n")
        f.write(f"Tolerance used: {tolerance} Å\n")
        f.write(f"Total reflections in df1: {len(df1)}\n")
        f.write(f"Total reflections in df2: {len(df2)}\n")
        f.write(f"Total matches found: {len(results_df[results_df['match_rank'] > 0])}\n")
        f.write(f"Reflections in df1 without matches: {len(results_df[results_df['match_rank'] == 0])}\n")
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
        
        # Write comparison results
        f.write("="*80 + "\n")
        f.write("d-spacing Comparison Results:\n")
        f.write("="*80 + "\n")
        
        # Write column headers with left alignment
        f.write(f"{'d_spacing':<10} {'exp. (h,k,l)':<15} {'exp. F^2_meas':<15} {'exp. F^2_calc':<15} {'VESTA (h,k,l)':<15} {'Simulated Intensity':<15}\n")
        f.write("-" * 78 + "\n")
        
        # Group results by df1 entries to avoid repetition
        current_df1_key = None
        for _, row in results_df.iterrows():
            df1_key = (int(row['df1_h']), int(row['df1_k']), int(row['df1_l']))
            df1_hkl = f"({df1_key[0]},{df1_key[1]},{df1_key[2]})"
            
            # Only show df1 info once per unique reflection
            if df1_key != current_df1_key:
                current_df1_key = df1_key
                df1_d = f"{row['df1_d']:.4f}"
                df1_f2_meas = f"{row['df1_f2_meas']:.2f}"
                df1_f2_calc = f"{row['df1_f2_calc']:.2f}"
                
                if row['match_rank'] > 0:
                    df2_hkl = f"({int(row['df2_h'])},{int(row['df2_k'])},{int(row['df2_l'])})"
                    f.write(f"{df1_d:<10} {df1_hkl:<15} {df1_f2_meas:<15} {df1_f2_calc:<15} {df2_hkl:<15} {row['df2_Intensity']:<12.2f}\n")
                else:
                    f.write(f"{df1_d:<10} {df1_hkl:<15} {df1_f2_meas:<15} {df1_f2_calc:<15}{'No match':<15} {'':<10} {'':<12}\n")
            else:
                # Additional matches for the same df1 reflection - show only df2 info
                if row['match_rank'] > 0:
                    df2_hkl = f"({int(row['df2_h'])},{int(row['df2_k'])},{int(row['df2_l'])})"
                    f.write(f"{'':<10} {'':<15} {'':<15} {'':<15} {df2_hkl:<15} {row['df2_Intensity']:<12.2f}\n")
    
    print(f"d-spacing comparison results saved to: {outfile}")
    
    return results_df
##############################


############# PLOTTING FUNCTIONS #############
def plot_f2(fc, fo, file_title, outfile):
    """
    Plots F^2_calc vs F^2_meas.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.scatter(fc, fo, alpha=0.6, edgecolor='k', s=20)
    min_val = min(min(fc), min(fo))
    max_val = max(max(fc), max(fo))

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    plt.xlabel(f"F² (calculated)")
    plt.ylabel(f"F² (observed)")
    plt.title(f"{file_title}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved to: {outfile}")