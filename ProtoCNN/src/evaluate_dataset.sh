pwd

eval "$(conda shell.bash hook)"
conda activate prototype_learning
# nvidia-smi

echo "Starting with Normal"
CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
    --dataset_dir="../../datasets/imdb_dataset" \
    --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/imdb_dataset/softmax-epoch_0=00-val_loss_0=0.4311-val_acc_0=0.8279.ckpt" \
    --name "normal" \
    --num_labels=2

echo "Starting dynamic"
CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
    --dataset_dir="../../datasets/imdb_dataset" \
    --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/imdb_dataset/without_dynamic-epoch_0=00-val_loss_0=0.4503-val_acc_0=0.8255.ckpt" \
    --name "without_dynamic" \
    --num_labels=2

echo "Starting with linear"
CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
    --dataset_dir="../../datasets/imdb_dataset" \
    --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/imdb_dataset/with_linear-epoch_0=00-val_loss_0=0.4160-val_acc_0=0.8353.ckpt" \
    --name "with_linear" \
    --num_labels=2

echo "Starting without clustering"
CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
    --dataset_dir="../../datasets/imdb_dataset" \
    --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/imdb_dataset/without_clustering_loss-epoch_0=00-val_loss_0=0.4965-val_acc_0=0.8158.ckpt" \
    --name "without_clustering" \
    --num_labels=2

echo "Starting without separation"
CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
    --dataset_dir="../../datasets/imdb_dataset" \
    --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/imdb_dataset/without_separation_loss-epoch_0=00-val_loss_0=0.4898-val_acc_0=0.7970.ckpt" \
    --name "without_separation" \
    --num_labels=2

conda deactivate
