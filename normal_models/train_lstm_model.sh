################################ Training ################################

# for dataset in "imdb" "ag_news" "dbpedia"; do
dataset=$2
echo "Dataset" ${dataset}

echo "Mode" $1
if [ "$1" = "train" ]; then

    CUDA_VISIBLE_DEVICES=$3 python lstm_model.py \
        --mode train \
        --batch_size 256 \
        --num_epochs 10 \
        --dataset ${dataset} \
        --model_dir "models/lstm_${dataset}/"

################################ Testing ################################

else

    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=6,7 python lstm_model.py \
        --mode test \
        --batch_size 256 \
        --dataset ${dataset} \
        --model_dir "models/lstm_${dataset}/"

fi
echo "-----------------------------------------------------------------"
# done
