# Cinay et al., Carnival N2O Code Instructions
## To Install the Necessary Python Packages

```bash
conda env create -f environment.yml
conda activate carnival_env
```

## Input Files
All input files are placed in `inputs/`. None of them are included in this repository.

- `DataS1.xlsx`: Carnival observations (one sheet per year, UTC), provided as Data S1 with the manuscript.
- `stilt_GAL_WRF_{year}_baseWRF_every30min_1hrBack.nc`: STILT footprints for 2024, 2025, and 2026. You can obtain the files upon request or by following the methods section for running STILT.
- `GAL202403_N2O_CO.nc`, `GAL202502_N2O_CO.nc`, `GAL202603_N2O_CO.nc`: Non-carnival observations for plot_fig02.py, available from https://www.bco-dmo.org/dataset/917743.
- `202402.csv`, `202503.csv`, `202602.csv`: Meteorological station data for plot_fig04.py. These are available upon request and can be found at https://weathergsc.usfq.edu.ec/.
- `basemaps/sancristobal_osm_z13.tif`: Offline OpenStreetMap basemap for plot_fig05.py (included).

plot_fig05.py also requires WRF outputs that are too large to include here. They can be made available upon request or run following the instructions in Table S1. Set `config_wrf_path` at the top of the script to their location.

## To Run Bayesian Inversions Using SLURM
```bash
sbatch runInversion.sh
```
The inversion results are written to `outputs/inversionResults_{year}.nc`. Once the submitted jobs are completed, you can run the plotting scripts.

## To Plot the Figures in The Main Manuscript Using SLURM
```bash
sbatch plot.sh
```
The figures are written to `figures/`.

| Script | Figure |
|---|---|
| plot_fig01.py | N2O mole fractions with CO enhancement |
| plot_fig02.py | Diurnal cycles |
| plot_fig03.py | N2O vs. CO |
| plot_fig04.py | Wind direction variability |
| plot_fig05.py | WRF wind and STILT footprint maps |
| plot_fig06.py | Inversion-based emissions |
| plot_fig07.py | 2026 forward model |
