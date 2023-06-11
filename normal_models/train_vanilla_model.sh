################################ Training ################################

dataset="dbpedia"

echo "Dataset" ${dataset}

echo "Mode" $1
# for model_checkpoint in "prajjwal1/bert-small" "funnel-transformer/small-base"; do
for model_checkpoint in "bert-base-uncased" "distilbert-base-uncased" "roberta-base"; do
    echo "Model checkpoint" ${model_checkpoint}
    if [ "$1" = "train" ]; then

        CUDA_VISIBLE_DEVICES=4,5 python vanilla_model.py \
            --mode train \
            --batch_size 4 \
            --logging_steps 400 \
            --num_epochs 0.08 \
            --dataset ${dataset} \
            --data_dir "../datasets/${dataset}_dataset" \
            --model_dir "models/${dataset}_${model_checkpoint}" \
            --model_checkpoint "${model_checkpoint}"

    ################################ Testing ################################

    else

        WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=4,5 python vanilla_model.py \
            --mode test \
            --batch_size 64 \
            --dataset ${dataset} \
            --data_dir "../datasets/${dataset}_dataset" \
            --model_dir "models/${dataset}_${model_checkpoint}"

    fi
    echo "-----------------------------------------------------------------"
done
