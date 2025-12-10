import os
import sys
from Functions_StructureFactor import parse_reflection_fcf, parse_reflection_hkl, analyze_friedel_pairs, prepare_friedel_data, plot_friedel_analysis
import matplotlib.pyplot as plt
import seaborn as sns

# usage:
# python3 FriedelPair.py -in <input_file> -condition <hexagonal> -out <output_file>
#
# Supported input file formats:
#   .fcf - CrysAlisPro FCF format (CIF-style, has both F²_calc and F²_meas)
#   .hkl - SHELX HKL format (simple h k l F² sigma, cell params at end)

# Default hexagonal lattice parameters (used if not specified and not found in file)
# Set to None to extract from file when possible
DEFAULT_a = 6.0155
DEFAULT_c = 15.3415
a = None
c = None


### add -h information
if '-h' in sys.argv or '--help' in sys.argv:
    print("Usage: python3 FriedelPair.py -in <input_file> -condition <hexagonal> -out <output_file>")
    print("")
    print("  -in        : Path to the input reflection file (.fcf or .hkl format).")
    print("  -condition : Crystal system condition (currently only 'hexagonal' is supported).")
    print("  -out       : Path to the output analysis file (default: friedel_pair_analysis_output.txt).")
    print("  -a         : (Optional) Lattice parameter 'a' in Angstroms.")
    print("  -c         : (Optional) Lattice parameter 'c' in Angstroms.")
    print("  -label     : (Optional) Label prefix for output plot files.")
    print("")
    print("Supported formats:")
    print("  .fcf  - CrysAlisPro FCF file (CIF-style format)")
    print("  .hkl  - SHELX HKL file (h k l F² sigma format)")
    sys.exit(0)

# read the file path from command line arguments
if '-in' in sys.argv:
    input_file = sys.argv[sys.argv.index('-in') + 1]
else:
    print("Error: Input file not specified. Use -in <input_file>")
    sys.exit(1)

if '-out' in sys.argv:
    output_file = sys.argv[sys.argv.index('-out') + 1]
else:
    output_file = 'friedel_pair_analysis_output.txt'  # default output file

if '-condition' in sys.argv:
    condition = sys.argv[sys.argv.index('-condition') + 1]
    if condition != 'hexagonal':
        print("Error: Currently only 'hexagonal' condition is supported.")
        sys.exit(1)
else:
    print("Warning: Condition not specified. Using default 'hexagonal'.")
    condition = 'hexagonal'  # default

# Optional lattice parameters (will override defaults or file-extracted values)
if '-a' in sys.argv:
    a = float(sys.argv[sys.argv.index('-a') + 1])
if '-c' in sys.argv:
    c = float(sys.argv[sys.argv.index('-c') + 1])

# Optional label for output plot files
if '-label' in sys.argv:
    label = sys.argv[sys.argv.index('-label') + 1]
else:
    label = None

# Detect file type based on extension
def get_file_type(filepath):
    """Detect file type from extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.fcf':
        return 'fcf'
    elif ext == '.hkl':
        return 'hkl'
    else:
        # Try to detect from content
        with open(filepath, 'r') as f:
            first_lines = f.read(500)
            if '_refln_index_h' in first_lines or 'loop_' in first_lines:
                return 'fcf'
            elif 'CELL' in first_lines or 'HKLF' in first_lines:
                return 'hkl'
        return 'unknown'

file_type = get_file_type(input_file)
print(f"Detected file type: {file_type}")     



### MAIN PROGRAM ###

def main():
    global a, c

    # Parse the input file based on detected file type
    if file_type == 'fcf':
        # FCF files: use provided values or defaults
        a_use = a if a is not None else DEFAULT_a
        c_use = c if c is not None else DEFAULT_c
        print(f"Using lattice parameters: a={a_use:.4f}, c={c_use:.4f}")
        df = parse_reflection_fcf(input_file, tag='Friedel Pair Analysis', a=a_use, c=c_use)
    elif file_type == 'hkl':
        # For HKL files, lattice parameters can be extracted from file if not provided
        # Pass None to let parser extract from file, or use provided values
        df = parse_reflection_hkl(input_file, tag='Friedel Pair Analysis', a=a, c=c)
    else:
        print(f"Error: Unknown file type for {input_file}")
        print("Supported formats: .fcf (CrysAlisPro FCF) or .hkl (SHELX HKL)")
        sys.exit(1)

    if df.empty:
        print("Error: No data loaded from input file.")
        sys.exit(1)

    print(f"Loaded {len(df)} reflections from: {input_file}")

    analyze_friedel_pairs(df, outfile=output_file, condition=condition)

    df_pairs = prepare_friedel_data(df, condition=condition)

    if df_pairs.empty:
        print("Analysis complete, but no unique Friedel pairs found. Check HKL filter or data.")
        return

    # 3. Generate the plots (Now called from Functions_StructureFactor)
    plot_friedel_analysis(df_pairs, label=label)
    print("\n--- Plot Generation Complete ---")

if __name__ == "__main__":
    main()