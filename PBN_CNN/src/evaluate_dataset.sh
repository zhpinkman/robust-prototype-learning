pwd

eval "$(conda shell.bash hook)"
conda activate prototype_learning
# nvidia-smi

echo "Starting with Normal"
CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
    --dataset_dir="../../datasets/imdb_dataset" \
    --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/imdb_dataset/bigger_softmax-epoch_0=00-val_loss_0=0.4354-val_acc_0=0.8422.ckpt" \
    --name "normal" \
    --num_labels=2

# echo "Starting dynamic"
# CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
#     --dataset_dir="../../datasets/dbpedia_dataset" \
#     --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/dbpedia_dataset/without_dynamic-epoch_0=00-val_loss_0=0.2951-val_acc_0=0.9484.ckpt" \
#     --name "without_dynamic" \
#     --num_labels=9

# echo "Starting with linear"
# CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
#     --dataset_dir="../../datasets/dbpedia_dataset" \
#     --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/dbpedia_dataset/with_linear-epoch_0=00-val_loss_0=0.2940-val_acc_0=0.9361.ckpt" \
#     --name "with_linear" \
#     --num_labels=9

# echo "Starting without clustering"
# CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
#     --dataset_dir="../../datasets/dbpedia_dataset" \
#     --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/dbpedia_dataset/without_clustering_loss-epoch_0=00-val_loss_0=0.6541-val_acc_0=0.7835.ckpt" \
#     --name "without_clustering" \
#     --num_labels=9

# echo "Starting without separation"
# CUDA_VISIBLE_DEVICES=3,4,5,6 python evaluate_dataset.py \
#     --dataset_dir="../../datasets/dbpedia_dataset" \
#     --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/dbpedia_dataset/without_separation_loss-epoch_0=00-val_loss_0=0.3985-val_acc_0=0.9016.ckpt" \
#     --name "without_separation" \
#     --num_labels=9

conda deactivate
