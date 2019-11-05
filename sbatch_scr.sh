#!/bin/bash
#SBATCH --output=/misc/vlgscratch4/BrunaGroup/rj1408/dynamic_nn/models/static_gcn/btcotc/def/train_logs.out
#SBATCH --error=/misc/vlgscratch4/BrunaGroup/rj1408/dynamic_nn/models/static_gcn/btcotc/def/train_logs.err
#SBATCH --job-name=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=12GB
#SBATCH --mail-type=END
#SBATCH --mail-user=rj1408@nyu.edu

module purge

eval "$(conda shell.bash hook)"
conda activate dgl_env
srun python3 dyngraph_exps/staticgcn_edgereg.py \
    --out-path /misc/vlgscratch4/BrunaGroup/rj1408/dynamic_nn/models/static_gcn/btcotc/def/ \
 --data-path /misc/vlgscratch4/BrunaGroup/rj1408/dynamic_nn/data/btcotc/soc-sign-bitcoinotc.csv \
 --learning-rate 0.1  --n-epochs 200 --stpsize 50