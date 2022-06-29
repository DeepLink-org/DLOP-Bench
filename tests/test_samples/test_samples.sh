# Run Parrots Samples
source pat_latest
export PYTHONPATH=$PWD:$PYTHONPATH
FRAMEWORK=parrots srun -p pat_dev --gres=gpu:1 python ./tests/test_samples/test_samples.py -t test_parrots
conda deactivate
unset PYTHONPATH

# Run Torch Samples
source activate
source deactivate
source pt1.4v1
export PYTHONPATH=$PWD:$PYTHONPATH
FRAMEWORK=torch srun -p pat_dev --gres=gpu:1 python ./tests/test_samples/test_samples.py -t test_torch
FRAMEWORK=torch srun -p pat_dev --gres=gpu:1 python ./tests/test_samples/test_samples.py -t compare_results
conda deactivate
unset PYTHONPATH