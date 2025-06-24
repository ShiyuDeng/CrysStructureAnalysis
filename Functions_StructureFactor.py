#!/usr/bin/env python3
#### all functions related to structure factor calculations and plotting
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd

def calc_d_hexagonal(h, k, l, a, c):
    """Calculates the d-spacing for hexagonal crystal systems.
    1/d**2 = 4/3 * (h**2 + h*k + k**2) + 1**2/c**2
    """
    import math
    tmp = 4/3 * (h**2 + h*k + k**2)/a**2 + l**2/c**2
    print(tmp)
    d = 1/math.sqrt(tmp)
    return d

##############  READ files #####################
def parse_reflection_fcf(filepath, a, c):
    """
    Parses the reflection output from CrysAlisPro (.fcf) 
    returns a DataFrame with columns:
    h, k, l, f2_calc, f2_meas, d_spacing
    """

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
                if len(parts) < 7:
                    continue  # Skip malformed lines
                try:
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
                                 'd': d_spacing})
                except ValueError:
                    continue 
                else:
                    print(".fcf output format error: CHECK!")         
    df = pd.DataFrame(data)
    return df

def parse_hkl_vesta(filename, a, c):
    """
    Parse VESTA output for 
    (h, k, l) indices, 
    d spacing, 
    Structure factor |F|,
    Intensity I.
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Skip header or empty/comment lines
            if line.strip() == "" or line.startswith("h") or line.startswith("//"):
                continue
            parts = line.split()

            if len(parts) >= 10:
                try:
                    h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                    d_direct = float(parts[3])
                    StructureFactor = float(parts[6])
                    Intensity = float(parts[7])
                    d_calc = calc_d_hexagonal(h, k, l, a, c)

                    data.append({'h': h, 
                            'k': k, 
                            'l': l, 
                            'd_direct': d_direct,
                            'd_calc': d_calc,
                            'StructureFactor': StructureFactor,
                            'Intensity': Intensity}) 

                except ValueError:
                    continue
            else:
                print("VESTA output format error: CHECK!")
                exit(1)

    return data

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
def compare_d(df1, df2, outfile):
# Use d-spacing from df1 and df2 as references to compare
# if df1['d'] = df2['d_calc']
# store df1['h,k,l'], df1['d'], df1['f2_meas'], df1['d_calc'] - df2['h,k,l'], df2['Intensity'] into a results DataFrame
#                                                             - df2['h,k,l'], df2['Intensity']
#                                                             - ..... (all d_spacing matches in df2)

    return results
##############################


############# PLOTTING FUNCTIONS #############
def plot_f2(fc, fo, file_title, outfile):
    """
    Plots F^2_calc vs F^2_meas.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(fc, fo, alpha=0.6, edgecolor='k', s=20)
    min_val = min(min(fc), min(fo))
    max_val = max(max(fc), max(fo))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    plt.xlabel(f"F² (calculated)")
    plt.ylabel(f"F² (observed)")
    plt.title('%s' % file_title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()
    print(f"Plot saved to {outfile}")