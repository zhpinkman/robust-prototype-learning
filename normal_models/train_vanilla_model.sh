################################ Training ################################

dataset="dbpedia"
num_labels=9

echo "Mode" $1
for model_checkpoint in "prajjwal1/bert-small"; do

    if [ "$1" = "train" ]; then

        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2,3,4,5,6 python vanilla_model.py \
            --mode train \
            --batch_size 4 \
            --logging_steps 400 \
            --num_epochs 0.08 \
            --dataset ${dataset} \
            --data_dir "../datasets/${dataset}_dataset" \
            --num_labels ${num_labels} \
            --model_dir "models/${dataset}_${model_checkpoint}" \
            --model_checkpoint "${model_checkpoint}"

    ################################ Testing ################################

    else

        TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=2,3,4,5,6 python vanilla_model.py \
            --mode test \
            --batch_size 4 \
            --dataset ${dataset} \
            --data_dir "../datasets/${dataset}_dataset" \
            --num_labels ${num_labels} \
            --model_dir "models/${dataset}_${model_checkpoint}"

    fi
done
