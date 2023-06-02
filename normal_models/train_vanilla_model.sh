################################ Training ################################

dataset="logical_fallacy"
num_labels=13

echo "Mode" $1
model_checkpoint="bert-base-uncased"

if [ "$1" = "train" ]; then

    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 python vanilla_model.py \
        --mode train \
        --batch_size 256 \
        --max_length 64 \
        --logging_steps 10 \
        --num_epochs 20 \
        --data_dir "../datasets/${dataset}_dataset" \
        --num_labels ${num_labels} \
        --model_dir "models/${dataset}_${model_checkpoint}" \
        --model_checkpoint "${model_checkpoint}"

################################ Testing ################################

else

    TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=2 python vanilla_model.py \
        --mode test \
        --batch_size 256 \
        --max_length 64 \
        --logging_steps 50 \
        --data_dir "../datasets/${dataset}_dataset" \
        --num_labels ${num_labels} \
        --model_dir "models/${dataset}_${model_checkpoint}"

fi
