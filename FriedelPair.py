import os
import sys
from Functions_StructureFactor import parse_reflection_fcf, analyze_friedel_pairs, prepare_friedel_data, plot_friedel_analysis
import matplotlib.pyplot as plt
import seaborn as sns

# usage:
# python3 FriedelPair.py -in <input_file> -condition <hexagonal> -out <output_file>

# hexagonal lattice parameters
a = 6.0155
c = 15.3415

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
    print("Error: Condition not specified. Use -condition <hexagonal>")
    condition = 'hexagonal' #default
    sys.exit(1)     

### MAIN PROGRAM ###

def main():
    df = parse_reflection_fcf(input_file, tag='Friedel Pair Analysis', a=a, c=c)
    print(f"Loaded data from: {input_file}")
    # print(format(df.head()))

    analyze_friedel_pairs(df, outfile=output_file, condition=condition)

    df_pairs = prepare_friedel_data(df, condition=condition)
    
    if df_pairs.empty:
        print("Analysis complete, but no unique Friedel pairs found. Check HKL filter or data.")
        return

    # 3. Generate the plots (Now called from Functions_StructureFactor)
    plot_friedel_analysis(df_pairs)
    print("\n--- Plot Generation Complete ---")

if __name__ == "__main__":
    main()