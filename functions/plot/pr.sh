#!/bin/bash
#SBATCH --job-name=tst_att
#SBATCH --output=/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-04172020_140/result_vali/output1.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4200
#SBATCH --partition=LKEBgpu
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --nodelist=res-hpc-lkeb05
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/lkeb-hpc/syousefi/Programs/cuda9.0/cuda/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/software/NVIDIA/cuda-9.0/extras/CUPTI/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/software/NVIDIA/cuda-9.0/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/software/NVIDIA/cudnn-9.0/lib64/
source /exports/lkeb-hpc/syousefi/Programs/TF112/bin/activate
echo "on Hostname = $(hostname)"
echo "on GPU      = $CUDA_VISIBLE_DEVICES"
echo
echo "@ $(date)"
echo
python /exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/dense_net_3d_segmentation-1-dice-tumor--106/functions/plot/PR_curve.py --where_to_run Cluster
