################################ Training ################################

dataset="ag_news"
num_labels=4

echo "Mode" $1

if [ "$1" = "train" ]; then

    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python vanilla_model.py \
        --mode train \
        --batch_size 256 \
        --max_length 64 \
        --logging_steps 50 \
        --data_dir "../datasets/${dataset}_dataset" \
        --num_labels ${num_labels} \
        --model_dir "models/${dataset}"

################################ Testing ################################

else

    TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=2 python vanilla_model.py \
        --mode test \
        --batch_size 256 \
        --max_length 64 \
        --logging_steps 50 \
        --data_dir "../datasets/${dataset}_dataset" \
        --num_labels ${num_labels} \
        --model_dir "models/${dataset}"

fi
