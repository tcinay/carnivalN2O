# Cinay et al., Carnival N2O Code Instructions
## To Install the Necessary Python Packages 
 
```bash
conda env create -f environment.yml
conda activate carnival_env
```
Running inversions requires STILT output. You can obtain the files upon request or by following the methods section for running STILT. Observational data can be obtained from Data S1 provided with the manuscript.

## To Run Bayesian Inversions Using SLURM
```bash
sbatch runInversions.sh
```
Once the submitted jobs are completed, you can run the plotting script. 
It is important to note that plot_fig03.py requires WRF outputs that are too large to include here. They can be made available upon request or run following the instructions in Table S1. You can download non-carnival observations for plot_fig02.py from https://www.bco-dmo.org/dataset/917743.

## To Plot the Figures in The Main Manuscript Using SLURM
```bash
sbatch plot.sh
```