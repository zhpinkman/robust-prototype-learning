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

if [ "$dataset" = "imdb" ]; then
    batch_size=16
elif [ "$dataset" = "dbpedia" ]; then
    batch_size=16
else
    batch_size=256
fi

for model_checkpoint in "ModelTC/bart-base-mnli" "google/electra-base-discriminator" "prajjwal1/bert-medium"; do
    # for model_checkpoint in "bert-base-uncased" "distilbert-base-uncased" "roberta-base"; do
    # for model_checkpoint in "prajjwal1/bert-medium"; do
    echo "Model checkpoint" ${model_checkpoint}
    if [ "$1" = "train" ]; then

        TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=${3} python vanilla_model.py \
            --mode train \
            --batch_size ${batch_size} \
            --logging_steps 20 \
            --num_epochs 5 \
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
