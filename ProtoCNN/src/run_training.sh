pwd

eval "$(conda shell.bash hook)"
conda activate prototype_learning
# nvidia-smi

CUDA_VISIBLE_DEVICES=3 python3 train_custom.py

conda deactivate