################################ Training ################################

# Model checkpoints can be chosen from the following list
# as well as any other checkpoint from the HuggingFace library
# "ModelTC/bart-base-mnli"
# "google/electra-base-discriminator"
# "prajjwal1/bert-medium"
# "funnel-transformer/small-base"
# "prajjwal1/bert-tiny"
# "bert-base-uncased"
# "distilbert-base-uncased"
# "roberta-base"

dataset=$2
echo "Dataset" ${dataset}
echo "Mode" $1
for model_checkpoint in "ModelTC/bart-base-mnli" "google/electra-base-discriminator" "prajjwal1/bert-medium" "prajjwal1/bert-small"; do
    echo "Model checkpoint" ${model_checkpoint}
    if [ "$1" = "train" ]; then

        WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=${3} python vanilla_model.py \
            --mode train \
            --batch_size $4 \
            --logging_steps 100 \
            --num_epochs $5 \
            --dataset ${dataset} \
            --data_dir "../datasets/${dataset}_dataset" \
            --model_dir "models/${dataset}_${model_checkpoint}" \
            --model_checkpoint "${model_checkpoint}"

    ################################ Testing ################################

    else

        WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=${3} python vanilla_model.py \
            --mode test \
            --batch_size 64 \
            --dataset ${dataset} \
            --data_dir "../datasets/${dataset}_dataset" \
            --model_dir "models/${dataset}_${model_checkpoint}"

    fi
    echo "-----------------------------------------------------------------"
done
