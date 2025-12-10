# Crystal Structure Factor Analysis

A Python toolkit for analyzing and comparing crystal structure factors from experimental and theoretical sources, including Friedel pair analysis for detecting broken inversion symmetry.

## Overview

This tool provides analysis of crystal structure data by:

### 1. Structure Factor Comparison
Compare experimental vs theoretical structure factors:
- **Experimental data** from CrysAlisPro (.fcf files)
- **Theoretical models** from structure factor calculations (e.g. by VESTA)

Comparisons can be made in:
- **d-spacing** suggested (To calculate d interactively: `python3 calc_d_spacing.py`)
- **Miller indices (hkl)** (Multiplicity is not reflected in experimental output, only one set of Miller indices will be returned in CrysAlisPro output.)

### 2. Friedel Pair Analysis
Analyze Friedel pairs I(hkl) vs I(-h,-k,-l) to detect:
- **Broken inversion symmetry** (non-centrosymmetric structures)
- **Anomalous scattering effects**
- Statistical significance of intensity differences



## Usage

### Structure Factor Comparison

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

### Friedel Pair Analysis

Analyze Friedel pairs to test for centrosymmetry:

```bash
# Get help
python3 FriedelPair.py -h

# Analyze HKL file (cell parameters auto-extracted from file)
python3 FriedelPair.py -in data.hkl -condition hexagonal -label "sample_name"

# Analyze FCF file (specify lattice parameters)
python3 FriedelPair.py -in data.fcf -condition hexagonal -a 6.0 -c 15.3 -label "sample_name"

# Full example
python3 FriedelPair.py -in ./case_study/012_ST4_FePSe3_P4_4p4_05.12.2025_P1.hkl \
    -condition hexagonal \
    -out ./012_friedel_analysis.txt \
    -label "012_P4_4p4GPa_P1"
```

#### Friedel Pair Options

| Option | Description |
|--------|-------------|
| `-in` | Input reflection file (.fcf or .hkl format) |
| `-condition` | Crystal system condition (currently `hexagonal`) |
| `-out` | Output analysis text file |
| `-label` | Label for plot titles and output filenames |
| `-a` | (Optional) Lattice parameter a in Angstroms |
| `-c` | (Optional) Lattice parameter c in Angstroms |

#### Output

The Friedel pair analysis generates:
1. **Text report**: Detailed pair-by-pair analysis with intensity ratios
2. **Intensity plot** (`*_Intensity_Pairs.png`): I(hkl) vs I(-h-k-l) scatter plot
3. **Z-score distribution** (`*_Z_score_distribution.png`): Statistical significance histogram

**Interpretation**:
- Points deviating from y=x line indicate broken inversion symmetry
- Z-score distribution wider than Gaussian noise indicates real anomalous signal

### Available Comparison Types

| Option | Description |
|--------|-------------|
| `-compare_fcf` | Compare two .fcf files (different crystal models) |
| `-compare_vesta_hkl` | Compare two VESTA structure factor outputs |
| `-compare_fcf_vesta` | Compare experimental (.fcf) vs theoretical (VESTA) data |
| `-compare_d` | Match reflections by d-spacing between datasets |

### Supported File Formats

#### 1. CrysAlisPro FCF Files (.fcf)
- CIF-style reflection output format
- Contains: h, k, l, F²_calc, F²_meas, sigma, status flag
- Cell parameters in header

#### 2. SHELX HKL Files (.hkl)
- Simple numeric format from CrysAlisPro
- Contains: h, k, l, F², sigma (5 columns)
- Cell parameters at end of file in `CELL` line

#### 3. VESTA Structure Factor Files (.txt)
- Generated from VESTA's structure factor calculation
- Contains h, k, l indices, d-spacing, structure factors, and intensities

#### 4. Python Input File (e.g., `input.py`)
User-defined configuration for structure factor comparison:
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
- **Space Groups**: $R3$ vs  $R\overline{3}$
- **Files**:
  - `*.fcf`: CrysAlisPro refinement output
  - `*_VESTA.txt`: Theoretical structure factors from VESTA models

**Example Analysis**:
```bash
python3 main_StructureFactor.py -in case_study/input.py -compare_d
```

## Installation

1. Clone or download this repository:
   ```bash
   git clone https://github.com/your-username/CrysStructureAnalysis.git
   cd CrysStructureAnalysis
   ```

2. Install required Python packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy
   ```

3. Run analysis on your data files

## Contributing

For questions, suggestions, or collaboration opportunities, please contact:
- dengs@ill.fr or sd864@cantab.ac.uk

## License

This project is licensed under the **GNU General Public License v2.0** (GPL-2.0).

See the [LICENSE](LICENSE) file for details.

---

*Developed for crystal structure analysis in high-pressure crystallography studies.*