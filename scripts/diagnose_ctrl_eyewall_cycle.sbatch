#!/bin/bash
#SBATCH --job-name=ctrl_erc
#SBATCH --partition=gpu-part
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=output/ctrl_eyewall_cycle_25N_66_88h/slurm-%j.out
#SBATCH --error=output/ctrl_eyewall_cycle_25N_66_88h/slurm-%j.err
set -euo pipefail
cd /data1/home/zhangyx/project/TC_dynamic
mkdir -p output/ctrl_eyewall_cycle_25N_66_88h
export OMP_NUM_THREADS=4
/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python scripts/diagnose_ctrl_eyewall_cycle.py
