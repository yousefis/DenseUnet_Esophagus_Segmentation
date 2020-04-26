

def job_script(setting, job_name=None, script_address=None, job_output_file=None):
    text = '#!/bin/bash \n'
    text = text + '#SBATCH --job-name=' + job_name + '\n'
    text = text + '#SBATCH --output=' + job_output_file + '\n'
    text = text + '#SBATCH --ntasks=1 \n'
    text = text + '#SBATCH --cpus-per-task=' + str(setting['cluster_NumberOfCPU']) + '\n'
    text = text + '#SBATCH --mem-per-cpu=' + str(setting['cluster_MemPerCPU']) + '\n'
    text = text + '#SBATCH --partition=' + setting['cluster_Partition'] + '\n'
    text = text + '#SBATCH --gres=gpu:1 \n'
    text = text + '#SBATCH --time=0 \n'
    if setting['cluster_NodeList'] is not None:
        text = text + '#SBATCH --nodelist='+setting['cluster_NodeList']+' \n'

    text = text + 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/lkeb-hpc/syousefi/Programs/cuda9.0/cuda/lib64/' '\n'
    text = text + 'source ' + setting['cluster_venv_slurm'] + '\n'
    # text = text + 'module load cuda9' + '\n'

    text = text + 'echo "on Hostname = $(hostname)"' '\n'
    text = text + 'echo "on GPU      = $CUDA_VISIBLE_DEVICES"' '\n'
    text = text + 'echo' '\n'
    text = text + 'echo "@ $(date)"' '\n'
    text = text + 'echo' '\n'

    text = text + 'python ' + script_address + ' --where_to_run Cluster '
    if setting['cluster_Partition'] == 'cpu':
        text = text + '--only_generate_image  True '
    elif setting['never_generate_image']:
        text = text + '--never_generate_image  True '
    text = text + '\n'
    return text
