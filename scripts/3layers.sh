#!/bin/bash
#SBATCH --output=/misc/vlgscratch4/BrunaGroup/rj1408/dynamic_nn/models/static_gcn/btcotc/eye_exps/3layers/train_logs.out
#SBATCH --error=/misc/vlgscratch4/BrunaGroup/rj1408/dynamic_nn/models/static_gcn/btcotc/eye_exps/3layers/train_logs.err
#SBATCH --job-name=3layers
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
srun python3 staticgcn_edgereg_oh.py \
    --out-path /misc/vlgscratch4/BrunaGroup/rj1408/dynamic_nn/models/static_gcn/btcotc/eye_exps/3layers/ \
 --data-path /misc/vlgscratch4/BrunaGroup/rj1408/dynamic_nn/data/btcotc/soc-sign-bitcoinotc.csv \
 --learning-rate 0.01  --n-epochs 100 --stpsize 20 --node-dim 128 --n-layers 3
