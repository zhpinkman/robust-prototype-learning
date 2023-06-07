pwd

eval "$(conda shell.bash hook)"
conda activate prototype_learning
# nvidia-smi

echo "Starting dynamic"
CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
--dataset_dir="/scratch/darshan/prototype-learning/robust-prototype-learning/datasets/imdb_dataset" \
--model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/imdb_dataset/without_dynamic-epoch_0=00-val_loss_0=0.4006-val_acc_0=0.8364.ckpt" \
--num_labels=2

echo "Starting with linear"
CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
--dataset_dir="/scratch/darshan/prototype-learning/robust-prototype-learning/datasets/imdb_dataset" \
--model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/imdb_dataset/with_linear-epoch_0=00-val_loss_0=0.3869-val_acc_0=0.8473.ckpt" \
--num_labels=2


echo "Starting without clustering"
CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
--dataset_dir="/scratch/darshan/prototype-learning/robust-prototype-learning/datasets/imdb_dataset" \
--model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/imdb_dataset/without_clustering_loss-epoch_0=00-val_loss_0=0.4709-val_acc_0=0.8288.ckpt" \
--num_labels=2

echo "Starting without separation"
CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
--dataset_dir="/scratch/darshan/prototype-learning/robust-prototype-learning/datasets/imdb_dataset" \
--model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/imdb_dataset/without_separation_loss-epoch_0=00-val_loss_0=0.4027-val_acc_0=0.8398.ckpt" \
--num_labels=2

conda deactivate