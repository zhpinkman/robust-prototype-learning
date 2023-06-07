################################ Training ################################

dataset="imdb"
num_labels=2

echo "Mode" $1
for model_checkpoint in "funnel-transformer/small-base"; do

    if [ "$1" = "train" ]; then

        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2,3,4,5 python vanilla_model.py \
            --mode train \
            --batch_size 32 \
            --max_length 512 \
            --logging_steps 50 \
            --num_epochs 1 \
            --data_dir "../datasets/${dataset}_dataset" \
            --num_labels ${num_labels} \
            --model_dir "models/${dataset}_${model_checkpoint}" \
            --model_checkpoint "${model_checkpoint}"

    ################################ Testing ################################

    else

        TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=2,3,4,5 python vanilla_model.py \
            --mode test \
            --batch_size 32 \
            --max_length 512 \
            --data_dir "../datasets/${dataset}_dataset" \
            --num_labels ${num_labels} \
            --model_dir "models/${dataset}_${model_checkpoint}"

    fi
done
