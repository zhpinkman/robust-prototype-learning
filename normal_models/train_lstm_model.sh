################################ Training ################################

for dataset in "imdb" "ag_news" "dbpedia"; do

    echo "Dataset" ${dataset}

    echo "Mode" $1
    if [ "$1" = "train" ]; then

        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=4,5 python lstm_model.py \
            --mode train \
            --batch_size 256 \
            --num_epochs 3 \
            --dataset ${dataset} \
            --model_dir "models/lstm_${dataset}/"

    ################################ Testing ################################

    else

        TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=4,5 python lstm_model.py \
            --mode test \
            --batch_size 256 \
            --dataset ${dataset} \
            --model_dir "models/lstm_${dataset}/"

    fi
    echo "-----------------------------------------------------------------"
done
