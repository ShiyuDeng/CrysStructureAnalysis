# Crystal Structure Factor Analysis

A Python script for analyzing and comparing crystal structure factors from experimental and theoretical sources.

## Overview

This tool provides analysis of crystal structure data by comparing:
- **Experimental data** from CrysAlisPro (.fcf files)
- **Theoretical models** from structure factor calculations (e.g. by VESTA)

Comparisons can be made in
- **d-spacing** suggested (To calculate d interactively: `python3 calc_d_spacing.py`)
- **Miller indices (hkl)** (Multiplcity is not reflected in experimental output, only one set of Miller indices will be returned in CrysAlisPro output.)



### Usage

```bash
# Get help and see all available options
python3 main_StructureFactor.py -h

# Run all available comparisons
python3 main_StructureFactor.py -in case_study/input.py

# Run d-spacing comparison with specific input file
python3 main_StructureFactor.py -in case_study/input.py -compare_d

# Run specific comparisons
python3 main_StructureFactor.py -in input.py -compare_fcf -compare_vesta_hkl
```

### Available Comparison Types

| Option | Description |
|--------|-------------|
| `-compare_fcf` | Compare two .fcf files (different crystal models) |
| `-compare_vesta_hkl` | Compare two VESTA structure factor outputs |
| `-compare_fcf_vesta` | Compare experimental (.fcf) vs theoretical (VESTA) data |
| `-compare_d` | Match reflections by d-spacing between datasets |

### Input File Structure


##### 1. CrysAlisPro Files (.fcf)
- Standard reflection output format
- Contains h, k, l indices and F² values (calculated and measured)

##### 2. VESTA Structure Factor Files (.txt)
- Generated from VESTA's structure factor calculation
- Contains h, k, l indices, d-spacing, structure factors, and intensities

##### 3. Python input file (e.g., `input.py`) with user-defined configuration:
example:
```python
# Crystal lattice parameters (hexagonal system)
a = 6.0155  # Å
c = 15.3415  # Å
gamma = 120.0  # degrees

# Data files
fcf_r3 = 'path/to/R3_model.fcf'
fcf_rm3 = 'path/to/R-3_model.fcf'
vesta_file1 = 'path/to/R3_VESTA_output.txt'
vesta_file2 = 'path/to/R-3_VESTA_output.txt'

# File descriptions
tag1_fcf = 'Experimental data, R3 space group'
tag2_fcf = 'Experimental data, R-3 space group'
tag1_vesta = 'Theoretical model, R3'
tag2_vesta = 'Theoretical model, R-3'

# Output filenames
out_fcf = 'fcf_comparison_results'
out_vesta = 'vesta_comparison_results'
out_fcf_vesta = 'fcf_vesta_comparison'

# Plotting parameters
title1 = 'R3 model: F²(calc) vs F²(obs)'
title2 = 'R-3 model: F²(calc) vs F²(obs)'
save1 = 'R3_structure_factors.png'
save2 = 'R-3_structure_factors.png'

# Analysis parameters (optional)
tolerance = 0.01  # d-spacing matching tolerance in Å
```

### Output Files

The tool generates several types of output:

1. **Comparison Reports** (`.txt`): Detailed statistical analysis of matches and differences
2. **Plots** (`.png`): Structure factor correlation plots
3. **CSV Files** (optional): Detailed numerical results for further analysis


----------
### Case Studies

#### [FePSe₃ Synchrotron Single Crystal Data under pressure]
- **Data Source**: Single crystal synchrotron data, I19 beamline, Diamond Light Source (UK)
- **Pressure**: 10 GPa
- **Space Groups**: $R3$ vs  $R\overline{3}$
- **Files**:
  - `*.fcf`: CrysAlisPro refinement output
  - `*_VESTA.txt`: Theoretical structure factors from VESTA models

**Example Analysis**:
```bash
python3 main_StructureFactor.py -in case_study/input.py -compare_d
```

### Installation

1. Clone or download this repository
2. Ensure Python 3.x is installed with required packages:
   ```bash
   pip install pandas numpy matplotlib
   ```
3. Place your data files in appropriate directories
4. Create your input file
5. Run the analysis


## Contributing

For questions, suggestions, or collaboration opportunities, please contact:
📧 **dengs@ill.fr**

## License

This project is developed for academic research purposes.

---

*Developed for crystal structure analysis in high-pressure crystallography studies*