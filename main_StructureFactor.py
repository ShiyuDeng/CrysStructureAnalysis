#import argparse
import sys
from Functions_StructureFactor import * 

import input as inp

def print_help():
    print("""
Usage: python3 main_StructureFactor.py [OPTIONS]

Options:
  -compare_fcf         Compare .fcf files (needs inp.r3, inp.rm3, inp.tag1_fcf, inp.tag2_fcf, inp.out_fcf, inp.title1, inp.title2, inp.save1, inp.save2)
  -compare_vesta_hkl   Compare VESTA HKL files (needs inp.file1, inp.tag1, inp.file2, inp.tag2, inp.out)
  -compare_fcf_vesta   Compare FCF and VESTA HKL (needs inp.r3, inp.file1, inp.tag1_fcf, inp.tag1, inp.out_fcf_vesta)
  -compare_d           Compare d-spacing (needs inp.r3, inp.file1, inp.tag1_fcf, inp.tag1, inp.out_fcf_vesta)
  -h, --help           Show this help message and exit

If no options are given, all available comparisons will be run (if the required inputs are present).
""")

def main(compare_fcf=True, compare_vesta_hkl=True, compare_fcf_vesta=True):
        
    # Read information
    df_fcf_r3 = parse_reflection_fcf(inp.fcf_r3, inp.a, inp.c)
    df_fcf_r3['tag'] = inp.tag1_fcf

    df_fcf_rm3 = parse_reflection_fcf(inp.fcf_rm3, inp.a, inp.c)
    df_fcf_rm3['tag'] = inp.tag2_fcf

    df_vesta_r3 = parse_hkl_vesta(inp.vesta_file1, inp.a, inp.c)
    df_vesta_r3['tag'] = inp.tag1_vesta

    df_vesta_rm3 = parse_hkl_vesta(inp.vesta_file2, inp.a, inp.c)
    df_vesta_rm3['tag'] = inp.tag2_vesta

    if compare_fcf:
        print(f"Comparing FCF files: {inp.fcf_r3} and {inp.fcf_rm3}")
        compare_hkl(df_fcf_r3, df_fcf_rm3, inp.out_fcf)

        ####  PLOT ########################
        #### future update: all for x,y to be read as inputs, and set title accordingly
        # R3 plot: x = calc (R3), y = meas (R3)
        plot_f2(df_fcf_r3['f2_calc'], df_fcf_r3['f2_calc_meas'], inp.title1, inp.save1)
        # R-3 plot: x = calc (R-3), y = meas (R3)
        f2_calc_rm3, f2_meas_r3 = pad_with_zeros(f2_calc_rm3, f2_meas_r3)
        plot_f2(df_fcf_rm3['f2_calc'], df_fcf_r3['f2_calc_meas'], inp.title2, inp.save2)
        ###################################

    if compare_vesta_hkl:
        print(f"Comparing VESTA output: {inp.vesta_file1} and {inp.vesta_file2}")
        compare_hkl(df_vesta_r3, df_vesta_rm3, inp.out_vesta)

    if compare_fcf_vesta:
        print(f"Comparing FCF and VESTA HKL: {inp.fcf_r3} and {inp.vesta_file1}")
        compare_hkl(df_vesta_r3, df_fcf_r3, inp.out_fcf_vesta)

    if compare_d:
        print(f"Comparing with d-spacing as the reference: {inp.fcf_r3} and {inp.vesta_file1}")
        compare_d_spacing(df_fcf_r3, df_vesta_r3, inp.out_fcf_vesta)

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args or '-h' in args or '--help' in args:
        print_help()

    compare_fcf = '-compare_fcf' in args
    compare_vesta_hkl = '-compare_vesta_hkl' in args
    compare_fcf_vesta = '-compare_fcf_vesta' in args
    compare_d = '-compare_d' in args

    main(compare_fcf, compare_vesta_hkl, compare_fcf_vesta, compare_d)
