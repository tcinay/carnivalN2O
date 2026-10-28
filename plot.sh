#!/bin/bash
#SBATCH --job-name=carnival_plotting
#SBATCH --output=logs/carnival_plotting.out
#SBATCH --error=logs/carnival_plotting.out
#SBATCH -n 1
#SBATCH -c 4

module load anaconda/2023b
source activate carnival_env

mkdir -p logs figures

/home/gridsan/tcinay/.conda/envs/carnival_env/bin/python plot_fig01.py
/home/gridsan/tcinay/.conda/envs/carnival_env/bin/python plot_fig02.py
/home/gridsan/tcinay/.conda/envs/carnival_env/bin/python plot_fig03.py
/home/gridsan/tcinay/.conda/envs/carnival_env/bin/python plot_fig04.py
/home/gridsan/tcinay/.conda/envs/carnival_env/bin/python plot_fig05.py
/home/gridsan/tcinay/.conda/envs/carnival_env/bin/python plot_fig06.py
/home/gridsan/tcinay/.conda/envs/carnival_env/bin/python plot_fig07.py
