#!/bin/bash
#SBATCH --job-name=distancemap_gen
#SBATCH --output=/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/dense_net_3d_segmentation-1-dice-tumor--106/functions/losses/out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=6000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --nodelist=res-hpc-gpu02
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/lkeb-hpc/syousefi/Programs/cuda9.0/cuda/lib64/
source /exports/lkeb-hpc/syousefi/TF17_lo01/bin/activate
echo "on Hostname = $(hostname)"
echo "on GPU      = $CUDA_VISIBLE_DEVICES"
echo
echo "@ $(date)"
echo
python /exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/dense_net_3d_segmentation-1-dice-tumor--106/functions/losses/distance_map.py --where_to_run Cluster
