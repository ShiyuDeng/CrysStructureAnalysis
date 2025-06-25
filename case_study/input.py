# hexagonal lattice parameters
a = 6.0155
c = 15.3415
gamma = 120.0

### compare_fcf
fcf_r3='015_ST4_FePSe3_P7_10p0_0_15.10.2023_R3_2.fcf'
fcf_rm3='015_ST4_FePSe3_P7_10p0_0_R-3.fcf'
tag1_fcf='exp.10 GPa, R3 model, fcf output'
tag2_fcf='exp.10 GPa, R-3 model, fcf output'
out_fcf='out_compare_fcf_hkl'

### if plot fcf
save1='R3_10GPa_fcf.png'
save2='Rm3_10GPa_fcf.png'
title1='10 GPa, x = calc (R3), y = meas (R3)'
title2='10 GPa, x = calc (R-3), y = meas (R-3)'


### compare_vesta_hkl
vesta_file1='015_ST4_FePSe3_P7_10p0_0_15.10.2023_R3_2_StructureFactor_VESTA.txt'
tag1_vesta='R3 model, VESTA output'
vesta_file2='015_ST4_FePSe3_P7_10p0_0_15.10.2023_R3_2_modifiedR-3_StrctureFactor_VESTA.txt'
tag2_vesta='R-3 model, VESTA output'
out_vesta='out_compare_vesta_hkl'

### compare_fcf_vesta
out_fcf_vesta='out_compare_fcf_vesta_hkl'



