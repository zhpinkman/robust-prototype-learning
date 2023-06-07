################################ Training ################################

dataset="dbpedia"
num_labels=9

echo "Mode" $1
for model_checkpoint in "bert-base-uncased" "roberta-base" "distilbert-base-uncased"; do

    if [ "$1" = "train" ]; then

        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3,4,5,6 python vanilla_model.py \
            --mode train \
            --batch_size 32 \
            --max_length 256 \
            --logging_steps 200 \
            --num_epochs 0.2 \
            --data_dir "../datasets/${dataset}_dataset" \
            --num_labels ${num_labels} \
            --model_dir "models/${dataset}_${model_checkpoint}" \
            --model_checkpoint "${model_checkpoint}"

    ################################ Testing ################################

    else

        TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=3,4,5,6,7 python vanilla_model.py \
            --mode test \
            --batch_size 32 \
            --max_length 64 \
            --logging_steps 50 \
            --data_dir "../datasets/${dataset}_dataset" \
            --num_labels ${num_labels} \
            --model_dir "models/${dataset}_${model_checkpoint}"

    fi
done
