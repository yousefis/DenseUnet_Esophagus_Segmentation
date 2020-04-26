#!/bin/bash
#SBATCH --job-name=RF13531_20200323_140825
#SBATCH --output=/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/23432_0.75_4-train1-03222020_180/result/output.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=9200
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
python /exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/dense_net_3d_segmentation-1-dice-tumor--106/test_network_newdata.py --where_to_run Cluster
