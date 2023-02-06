#!/bin/bash --login
#$ -cwd
#SBATCH --job-name=sk_u_net
#SBATCH --out=base_model.out.%J
#SBATCH --err=base_model.err.%J
#SBATCH -p gpu
#SBATCH --mem-per-cpu=30G
#SBATCH -n 1
#SBATCH --gres=gpu:1

for ii in `seq 1 6`; do 
python sk-net-inference.py $ii
done
