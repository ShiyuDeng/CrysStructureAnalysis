import os
import sys
import importlib.util
from Functions_StructureFactor import * 

def load_input_file(input_file):
    """Dynamically load input file as a module"""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location("input_module", input_file)
    input_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(input_module)
    return input_module

def print_help():
    print("""
Usage: python3 main_StructureFactor.py -in <input_file> [OPTIONS]

Required:
  -in <input_file>     Specify the input file (Python module with configuration)

Options:
  -compare_fcf         Compare .fcf files
  -compare_vesta_hkl   Compare VESTA HKL files  
  -compare_fcf_vesta   Compare FCF and VESTA HKL
  -compare_d           Compare d-spacing between FCF and VESTA
  -plot_fcf         Plot F² (calculated) vs F² (observed) for FCF files
  -h, --help           Show this help message and exit

Examples:
  python3 main_StructureFactor.py -in case_study/input.py -compare_d
  python3 main_StructureFactor.py -in my_config.py -compare_fcf -compare_vesta_hkl
  
If no specific options are given, all available comparisons will be run.
""")

def main(input_file, compare_fcf=False, plot_fcf=False, compare_vesta_hkl=False, compare_fcf_vesta=False, compare_d=False):
    
    # Load input configuration
    inp = load_input_file(input_file)
    print(f"Loaded configuration from: {input_file}")
    print(f"Crystal system: a={inp.a}, c={inp.c}")
    
    # Read data files
    print("Loading data files...")
    df_fcf_r3 = parse_reflection_fcf(inp.fcf_r3,inp.tag1_fcf, inp.a, inp.c)
    df_fcf_rm3 = parse_reflection_fcf(inp.fcf_rm3, inp.tag2_fcf, inp.a, inp.c)

    df_vesta_r3 = parse_hkl_vesta(inp.vesta_file1,inp.tag1_vesta, inp.a, inp.c)
    df_vesta_rm3 = parse_hkl_vesta(inp.vesta_file2, inp.tag2_vesta, inp.a, inp.c)

    print("\nStarting analysis...")
    # Perform comparisons across hkl idex
    if compare_fcf:
        compare_hkl(df_fcf_r3, df_fcf_rm3, inp.out_fcf)
        print(f"\n=== Comparing FCF files: {inp.fcf_r3} and {inp.fcf_rm3} is done.")

    if compare_vesta_hkl:
        print(f"\n=== Comparing VESTA output: {inp.vesta_file1} and {inp.vesta_file2}")
        compare_hkl(df_vesta_r3, df_vesta_rm3, inp.out_vesta)

    if compare_fcf_vesta:
        print(f"\n=== Comparing FCF and VESTA HKL: {inp.fcf_r3} and {inp.vesta_file1}")
        compare_hkl(df_fcf_r3, df_vesta_r3, f"{inp.out_fcf_vesta}_hkl")

    # perform d-spacing comparison
    if compare_d:
        tolerance = getattr(inp, 'tolerance', 0.0001) 
        for sg in ['R3', 'Rm3']:
            if sg == 'R3':
                df_fcf = df_fcf_r3
                df_vesta = df_vesta_r3
            else:
                df_fcf = df_fcf_rm3
                df_vesta = df_vesta_rm3
            compare_d_spacing(df_fcf, df_vesta, 
                              outfile=f"{inp.out_fcf_vesta}_dspacing_{sg}", tolerance=tolerance)
            
        compare_fcf_in_dspacing(df_fcf_rm3, df_fcf_r3, 
                                outfile=f"{inp.out_fcf_vesta}_dspacing_fcf_R3_Rm3", tolerance=tolerance)

    if plot_fcf:
        #### future update: all for x,y to be read as inputs, and set title accordingly
        # R3 plot: x = calc (R3), y = meas (R3)
        plot_f2(df_fcf_r3['f2_calc'], df_fcf_r3['f2_meas'], inp.title1, inp.save1, inp.P) # inp.min_val, inp.max_val)
        # R-3 plot: x = calc (R-3), y = meas (R-3)
        # Fixed: should use f2_meas from the same dataset for y-axis
        plot_f2(df_fcf_rm3['f2_calc'], df_fcf_rm3['f2_meas'], inp.title2, inp.save2, inp.P)#, inp.min_val, inp.max_val)
        ###################################

if __name__ == '__main__':
    args = sys.argv[1:]
    
    if not args or '-h' in args or '--help' in args:
        print_help()
        sys.exit(0)
    
    # Parse input file
    if '-in' not in args:
        print("Error: Input file must be specified with -in option")
        print_help()
        sys.exit(1)
    
    try:
        input_file_index = args.index('-in') + 1
        input_file = args[input_file_index]
    except (ValueError, IndexError):
        print("Error: No input file specified after -in")
        print_help()
        sys.exit(1)
    
    # Parse comparison options
    compare_fcf = '-compare_fcf' in args
    compare_vesta_hkl = '-compare_vesta_hkl' in args
    compare_fcf_vesta = '-compare_fcf_vesta' in args
    compare_d = '-compare_d' in args
    plot_fcf= '-plot_fcf' in args 

    main(input_file, compare_fcf, plot_fcf, compare_vesta_hkl, compare_fcf_vesta, compare_d)
