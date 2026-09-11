#!/bin/bash
#SBATCH --partition=helios
#SBATCH --qos=devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=pd1_training
#SBATCH --output=pd1_training_%j.log

module load apps/miniconda/4.7.12
source activate polylung
unset PYTHONHOME
unset PYTHONPATH

which python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi
free -g

python scripts/pd1_train_swin_lunghist700.py \
  --data-dir data/lunghist700_binary \
  --epochs 8 \
  --batch-size 8 \
  --num-workers 0 \
  --lr 1e-4 \
  --published-baseline 0.0000 \
  --pretrained \
  --output evidence/public/pd1/pd1_metrics_gpu_py37.json
